"""
Register VOC base-training routine
ref: register-base-training-voc.ipynb in `fsod/`
"""

import os
import copy
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import yaml
from ...pipelines_adaptor.voc.base_training import BaseTrainFSODAdaptor

PATH_CONFIG_BASE_TRAINING = '../../pipelines_adaptor/voc/config_base_training.yaml'
DATA_ROOT_DIR = '../../data_utils/data/VOCdevkit/VOC2007'
# assert(os.path.isfile(PATH_CONFIG_BASE_TRAINING))

class BaseTrainData:
    """
    Registers data for base-training phase
    """
    def __init__(self, novel_id=0, filter_07_base=False):
        self.novel_id = novel_id
        self.filter_07_base = filter_07_base
        with open(os.path.abspath(__file__ + "/../" + PATH_CONFIG_BASE_TRAINING), 'r') as f:
            self.data_options = yaml.safe_load(f)
        self.base_train_adaptor_instance = BaseTrainFSODAdaptor(
            novel_id=self.novel_id, path_config=os.path.abspath(__file__ + "/../" + PATH_CONFIG_BASE_TRAINING))

        self.base_train_adaptor_instance = copy.deepcopy(self.base_train_adaptor_instance)

    def _load_datasets(self, ):
        # Query: Train
        self.base_training_query_train = self.get_query_set_base_training_train(
            self.base_train_adaptor_instance)

        # Support: Train + Validation
        self.base_training_support = self.get_support_set_base_training(
                                            self.base_train_adaptor_instance)

        # Query: Validation
        self.base_training_query_val = self.get_query_set_base_training_val(
                                            self.base_train_adaptor_instance)

    def _register_datasets(self, ):
        # Query: Train
        DatasetCatalog.register(
            "voc_base_training_query_train",
            lambda: self.get_query_set_base_training_train(self.base_train_adaptor_instance)
        )
        MetadataCatalog.get("voc_base_training_query_train").set(
            thing_classes=self.base_train_adaptor_instance.cfg.voc_classes,
            evaluator_type='pascal_voc')

        # Support: Train + Validation
        DatasetCatalog.register(
            "voc_base_training_support",
            lambda: self.get_support_set_base_training(self.base_train_adaptor_instance)
        )
        MetadataCatalog.get("voc_base_training_support").set(
            thing_classes=self.base_train_adaptor_instance.cfg.voc_classes,
            evaluator_type='pascal_voc')

        # Query: Validation
        DatasetCatalog.register(
            "voc_base_training_query_val",
            lambda: self.get_query_set_base_training_val(self.base_train_adaptor_instance)
        )
        MetadataCatalog.get("voc_base_training_query_val").set(
            thing_classes=self.base_train_adaptor_instance.cfg.voc_classes,
            evaluator_type='pascal_voc',
            dirname=self.data_options['data_root']+'/VOCdevkit/VOC2007',
            year=2007,
            split='test')

    def get_query_set_base_training_train(self, base_pipe_instance):
        """
        Query: Train
        """
        dataset_dicts = []
        lab2int = self.get_label2int_from_list(base_pipe_instance.cfg.voc_classes)

        for path_img in base_pipe_instance.query_dataset_train.lines:
            path_img = path_img.strip()
            if self.filter_07_base:
                if 'VOC2007' not in path_img:
                    continue
            annot_txt_path = base_pipe_instance.query_dataset_train.get_labpath(path_img)
            annot_xml_path = os.path.splitext(annot_txt_path)[0].replace('labels', 'Annotations') + '.xml'
            assert(os.path.isfile(annot_xml_path))

            tree = ET.parse(annot_xml_path)

            record = {
                "file_name": path_img,
                "image_id": os.path.basename(path_img).split('.')[0],
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            
    #         obj_annos_detectron = REL_fetch_bbox_from_darknet_preprocess(annot_txt_path)
            obj_annos_detectron = self.ABS_fetch_bbox_from_XML_detectron(
                        annot_xml_path, lab2int, base_pipe_instance.cfg.base_ids)

            record["annotations"] = obj_annos_detectron
            if len(obj_annos_detectron) > 0:
                dataset_dicts.append(record)

        return dataset_dicts

    def get_query_set_base_training_val(self, base_pipe_instance):
        """
        Query: Validation
        !! Similar to the Train counterpart. Might refactor later.
        """
        dataset_dicts = []
        lab2int = self.get_label2int_from_list(base_pipe_instance.cfg.voc_classes)

        for path_img in base_pipe_instance.query_dataset_val.lines:
            path_img = path_img.strip()
            if self.filter_07_base:
                if 'VOC2007' not in path_img:
                    continue
            annot_txt_path = base_pipe_instance.query_dataset_val.get_labpath(path_img)
            annot_xml_path = os.path.splitext(annot_txt_path)[0].replace('labels', 'Annotations') + '.xml'
            assert(os.path.isfile(annot_xml_path))

            tree = ET.parse(annot_xml_path)

            record = {
                "file_name": path_img,
                "image_id": os.path.basename(path_img).split('.')[0],
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }

        #         obj_annos_detectron = REL_fetch_bbox_from_darknet_preprocess(annot_txt_path)
            obj_annos_detectron = self.ABS_fetch_bbox_from_XML_detectron(
                    annot_xml_path, lab2int, base_pipe_instance.cfg.base_ids)

            record["annotations"] = obj_annos_detectron
            if len(obj_annos_detectron) > 0:
                dataset_dicts.append(record)

        return dataset_dicts

    def get_support_set_base_training(self, base_pipe_instance):
        """
        Support: Train + Validation
        All samples in the class-based bins have at least one of their respective class's bbox 
        """
        dataset_dicts_label_wise = {e: [] for e in base_pipe_instance.cfg.base_ids}
        lab2int = self.get_label2int_from_list(base_pipe_instance.cfg.voc_classes)

        assert(len(base_pipe_instance.support_set.metalines) == len(base_pipe_instance.cfg.base_ids))

        for i, class_id in enumerate(base_pipe_instance.cfg.base_ids):
        #     print(i, class_id)
        #     print("Number of images before class-wise object filtering: {}"
        #           .format(len(base_pipe_instance.support_set.metalines[i])))

            for path_img in base_pipe_instance.support_set.metalines[i]:
                path_img = path_img.strip()
                if self.filter_07_base:
                    if 'VOC2007' not in path_img:
                        continue
                annot_txt_path = base_pipe_instance.query_dataset_train.get_labpath(path_img)
                annot_xml_path = os.path.splitext(annot_txt_path)[0].replace('labels', 'Annotations') + '.xml'
                assert(os.path.isfile(annot_xml_path))

                tree = ET.parse(annot_xml_path)

                record = {
                    "file_name": path_img,
                    "image_id": os.path.basename(path_img).split('.')[0],
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }

                obj_annos_detectron = self.ABS_fetch_bbox_from_XML_detectron(
                                                annot_xml_path, lab2int, [class_id])

                record["annotations"] = obj_annos_detectron
                if len(obj_annos_detectron) > 0:
                    dataset_dicts_label_wise[class_id].append(record)
        return dataset_dicts_label_wise

    def ABS_fetch_bbox_from_XML_detectron(self, annot_xml_path, lab2int, class_ids_to_include):
        """
        taken from: detectron2/data/datasets/pascal_voc.py
        """
        tree = ET.parse(annot_xml_path)
        obj_annos_detectron = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            
            category_id = lab2int[cls]
            if category_id in class_ids_to_include:
                obj_annos_detectron.append({
                    "category_id": category_id,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS
                })
        return obj_annos_detectron

    def REL_fetch_bbox_from_darknet_preprocess(self, annot_txt_path):
        """
        Uses old version of darknet voc_label.py file which ignores the fact that
        VOC's coordinates starts at (1, 1)
        """
        obj_annos_np = np.loadtxt(annot_txt_path)
        obj_annos_np = np.reshape(obj_annos_np, (-1, 5))

        obj_annos_detectron = []
        for obj_anno in obj_annos_np:

            category_id = int(obj_anno[0])
            bbox = list(obj_anno[1:])

            if category_id in self.base_train_adaptor_instance.cfg.base_ids:
                obj_annos_detectron.append({
                    "category_id": category_id,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_REL,
                })
            else:
                pass
        return obj_annos_detectron

    def get_label2int_from_list(self, labels_list):
        lab2int = {e: i for i, e in enumerate(labels_list)}
        return lab2int