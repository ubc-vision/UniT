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
from .datasets.coco_note.register_coco_note import RegisterCOCONote
from .datasets.coco_dock.register_coco_dock import RegisterCOCODOCK
from .evaluators import PascalVOCEvaluator, PascalVOCDetectionWeakEvaluator, COCOEvaluatorWeakEvaluator


def register_datasets(args_data):
    if args_data.DATASETS.FEWSHOT.TYPE == 'VOC':
        register_voc_instance = RegisterVOC(
            args_data.DATASETS.FEWSHOT.SPLIT_ID, args_data.DATASETS.FEWSHOT.NUM_SHOTS)
        register_voc_instance._register_datasets()
    elif args_data.DATASETS.FEWSHOT.TYPE == 'COCO':
        register_voc_instance = RegisterCOCO(4, args_data.DATASETS.FEWSHOT.NUM_SHOTS)
        register_voc_instance._register_datasets()
    elif args_data.DATASETS.FEWSHOT.TYPE == 'VOC2007':
        register_voc_instance = RegisterVOC(
            args_data.DATASETS.FEWSHOT.SPLIT_ID, args_data.DATASETS.FEWSHOT.NUM_SHOTS, filter_07_base=True)
        register_voc_instance._register_datasets()
    elif args_data.DATASETS.FEWSHOT.TYPE == 'COCO_NOTE':
        register_voc_instance = RegisterCOCONote(args_data.DATASETS.FEWSHOT.NUM_SHOTS)
        register_voc_instance._register_datasets()
    elif args_data.DATASETS.FEWSHOT.TYPE == 'COCO_DOCK':
        register_voc_instance = RegisterCOCODOCK(args_data.DATASETS.FEWSHOT.NUM_SHOTS)
        register_voc_instance._register_datasets()
    else:
        return ValueError("Dataset: {} not recognized".format(args_data['type']))
        
def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        print (dataset_name)
        evaluator_list.append(COCOEvaluatorWeakEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionWeakEvaluator(dataset_name, cfg=cfg)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)



def inference_on_dataset_meta(model, data_loader, att_vecs_support, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, att_vecs_support)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def include_dependencies(path_json):
    with open(path_json, 'r') as fp:
        f = json.load(fp)
    for dependency in f['dependencies']:
        print(" > Including dependency: {}".format(dependency))
        sys.path.append(dependency)