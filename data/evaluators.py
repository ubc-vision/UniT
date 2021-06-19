import os
import sys
import time
import json
import torch
import logging
import datetime
import tempfile
import detectron2.utils.comm as comm
import numpy as np
from collections import defaultdict, OrderedDict
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator
)

from detectron2.evaluation.pascal_voc_evaluation import voc_eval
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from .datasets.voc.register_voc import RegisterVOC
from .datasets.coco.register_coco import RegisterCOCO
import multiprocessing as mp
from pathlib import Path
import xml.etree.ElementTree as ET
from functools import lru_cache
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation.evaluator import inference_context
from detectron2.evaluation.evaluator import DatasetEvaluator
import pickle
import copy 

class PascalVOCEvaluator(PascalVOCDetectionEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.
    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        super(PascalVOCEvaluator, self).__init__(dataset_name)
    
    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

        self._logger.info("Class-wise breakdown of AP at 0.5 IoU")
        for cls_name, AP in zip(self._class_names, aps[50]):
            self._logger.info("{} --> {}".format(cls_name, AP))
        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        return ret

class PascalVOCDetectionWeakEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.
    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, cfg=None):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        if cfg is not None:
            self._novel_classes = cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID
            self._base_classes = cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID
        else:
            self._novel_classes = []
            self._base_classes = []

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        dirname = Path("tmp/dets")
        dirname.mkdir(parents=True, exist_ok=True)
        res_file_template = os.path.join(dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        pool = mp.Pool(10)
        
        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])

            with open(res_file_template.format(cls_name), "w") as f:
                f.write("\n".join(lines))

            args = []
            for thresh in range(50, 100, 5):
                args.append([
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    thresh / 100.0,
                    self._is_2007
                ])
            results = pool.starmap(voc_eval, args)
            for thresh, result in zip(range(50, 100, 5), results):
                rec, prec, ap = result
                aps[thresh].append(ap * 100)
        pool.close()
        pool.join()
        self._logger.info("Class-wise breakdown of AP at 0.5 IoU")
        idx = 0
        novel_mean = 0.0
        for cls_name, AP in zip(self._class_names, aps[50]):
            self._logger.info("{} --> {}".format(cls_name, AP))
            if idx in self._novel_classes:
                novel_mean += AP
            idx += 1
        if len(self._novel_classes) > 0: 
            novel_mean = novel_mean / len(self._novel_classes)
        else:
            novel_mean = None
        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75], "novel_mean": novel_mean}

        # with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        #     res_file_template = os.path.join(dirname, "{}.txt")

        #     aps = defaultdict(list)  # iou -> ap per class
        #     for cls_id, cls_name in enumerate(self._class_names):
        #         lines = predictions.get(cls_id, [""])

        #         with open(res_file_template.format(cls_name), "w") as f:
        #             f.write("\n".join(lines))

        #         for thresh in range(50, 100, 5):
        #             rec, prec, ap = voc_eval(
        #                 res_file_template,
        #                 self._anno_file_template,
        #                 self._image_set_path,
        #                 cls_name,
        #                 ovthresh=thresh / 100.0,
        #                 use_07_metric=self._is_2007,
        #             )
        #             aps[thresh].append(ap * 100)

        # ret = OrderedDict()
        # mAP = {iou: np.mean(x) for iou, x in aps.items()}
        # ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        return ret

class COCOEvaluatorWeakEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super(COCOEvaluatorWeakEvaluator, self).__init__(dataset_name, cfg, distributed, output_dir)
        self._file_counter = 0
        self._logger = logging.getLogger(__name__)
        if cfg is not None:
            self._novel_classes = cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID
            self._base_classes = cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID
        else:
            self._novel_classes = []
            self._base_classes = []

    def _summarize(self, params, eval_obj, ap=1, iouThr=None, areaRng="all", maxDets=100, novel=True):
        p = params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            s = eval_obj["precision"]
            if iouThr is not None:
                t = np.where(np.abs(iouThr - p.iouThrs) < 0.001)[0]
                s = s[t]
            if novel:
                s = s[:, :, self._novel_classes, aind, mind]
            else:
                s = s[:, :, self._base_classes, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval_obj["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if novel:
                s = s[:, self._novel_classes, aind, mind]
            else:
                s = s[:, self._base_classes, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s


    def _summarizeDets(self, params, eval_obj, novel=True):
        stats = np.zeros((12,))
        stats[0] = self._summarize(params, eval_obj, 1, novel=novel)
        stats[1] = self._summarize(params, eval_obj, 1, iouThr=0.5, maxDets=params.maxDets[2], novel=novel)
        stats[2] = self._summarize(params, eval_obj, 1, iouThr=0.75, maxDets=params.maxDets[2], novel=novel)
        stats[3] = self._summarize(params, eval_obj, 1, areaRng="small", maxDets=params.maxDets[2], novel=novel)
        stats[4] = self._summarize(params, eval_obj, 1, areaRng="medium", maxDets=params.maxDets[2], novel=novel)
        stats[5] = self._summarize(params, eval_obj, 1, areaRng="large", maxDets=params.maxDets[2], novel=novel)
        stats[6] = self._summarize(params, eval_obj, 0, maxDets=params.maxDets[0], novel=novel)
        stats[7] = self._summarize(params, eval_obj, 0, maxDets=params.maxDets[1], novel=novel)
        stats[8] = self._summarize(params, eval_obj, 0, maxDets=params.maxDets[2], novel=novel)
        stats[9] = self._summarize(params, eval_obj, 0, areaRng="small", maxDets=params.maxDets[2], novel=novel)
        stats[10] = self._summarize(params, eval_obj, 0, areaRng="medium", maxDets=params.maxDets[2], novel=novel)
        stats[11] = self._summarize(params, eval_obj, 0, areaRng="large", maxDets=params.maxDets[2], novel=novel)

        stats = stats * 100

        self._logger.info("AP: {:.2f}".format(stats[0]))
        self._logger.info("AP 0.5: {:.2f}".format(stats[1]))
        self._logger.info("AP 0.75: {:.2f}".format(stats[2]))
        self._logger.info("AP S/M/L: {:.2f} / {:.2f} / {:.2f}".format(stats[3], stats[4], stats[5]))
        self._logger.info("R 1/10/100: {:.2f} / {:.2f} / {:.2f}".format(stats[6], stats[7], stats[8]))
        self._logger.info("R S/M/L: {:.2f} / {:.2f} / {:.2f}".format(stats[9], stats[10], stats[11]))
        return stats

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}
        eval_obj = coco_eval.eval
        params = coco_eval.eval['params']
        self._logger.info("Evaluation results for novel classes on {}: \n".format(iou_type))
        stats = self._summarizeDets(params, eval_obj)
        # self._logger.info("Evaluation results for base classes on {}: \n".format(iou_type))
        # stats = self._summarizeDets(params, eval_obj, novel=False)
        # all_files = [x for x in os.listdir(self._output_dir) if '.pkl' in x]
        # if len(all_files) > 0:
        #     file_num = sorted([int(x.split("_")[-1].split(".")[0]) for x in all_files])[-1] + 1
        # else:
        #     file_num = 0
        # file_path = os.path.join(self._output_dir, "coco_obj_"+str(file_num)+".pkl")
        # with open(file_path, 'wb') as f:
        #     pickle.dump({'eval':coco_eval, 'class_names':class_names}, f)
        self._file_counter += 1
        ret = super(COCOEvaluatorWeakEvaluator, self)._derive_coco_results(coco_eval, iou_type, class_names)
        ret['novel_mean'] = stats[1]
        return ret


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap