"""
Register COCO base-training routine
"""

import os
import copy
import pickle

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json

from ...pipelines_adaptor.coco.base_training import BaseTrainFSODAdaptor

PATH_CONFIG_BASE_TRAINING = '/h/skhandel/FewshotDetection/WSASOD/data/pipelines_adaptor/coco/config_base_training.yaml'
# assert(os.path.isfile(PATH_CONFIG_BASE_TRAINING))


class BaseTrainData:
    """
    Registers data for base-training phase
    """
    def __init__(self, novel_id=3):
        self.novel_id = novel_id

        self.base_train_adaptor_instance = self._fetch_the_copy_of_instance_if_tmp_stored()

        self.base_train_adaptor_instance = copy.deepcopy(self.base_train_adaptor_instance)

        self._register_coco_2017_if_not_present()
        self._fetch_coco_2017_data()

        # combine 2017 train val and index it with id
        self.data_coco_2017_trainval = self.data_train_coco_2017 + self.data_val_coco_2017

        self.dict_coco2017_id_item = {}
        for e in self.data_coco_2017_trainval:
            e_id = os.path.basename(e['file_name']).split('.')[0]
            self.dict_coco2017_id_item[e_id] = e

    def _fetch_the_copy_of_instance_if_tmp_stored(self, ):
        """takes a long time otherwise"""
        if os.path.isfile('tmp/coco_base_pipe_instance.pkl'):
            with open('tmp/coco_base_pipe_instance.pkl', 'rb') as f:
                base_train_adaptor_instance = pickle.load(f)
        else:
            os.makedirs('tmp', exist_ok=True)
            base_train_adaptor_instance = BaseTrainFSODAdaptor(
                novel_id=self.novel_id, path_config=PATH_CONFIG_BASE_TRAINING)
            with open('tmp/coco_base_pipe_instance.pkl', 'wb') as f:
                pickle.dump(base_train_adaptor_instance, f)

        return base_train_adaptor_instance

    def _register_coco_2017_if_not_present(self, ):
        if 'coco_2017_train_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2017_train_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/annotations/instances_train2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/"
            )
        if 'coco_2017_val_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2017_val_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/annotations/instances_val2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/"
            )

    def _fetch_coco_2017_data(self, ):
        self.data_train_coco_2017 = DatasetCatalog.get('coco_2017_train_copy')
        self.data_val_coco_2017 = DatasetCatalog.get('coco_2017_val_copy')

        self.meta_coco_2017 = MetadataCatalog.get('coco_2017_train_copy')

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
            "coco_base_training_query_train",
            lambda: self.get_query_set_base_training_train(self.base_train_adaptor_instance)
        )
        MetadataCatalog.get("coco_base_training_query_train").set(
            thing_classes=self.base_train_adaptor_instance.cfg.coco_classes,
            evaluator_type='coco')

        # Support: Train + Validation
        DatasetCatalog.register(
            "coco_base_training_support",
            lambda: self.get_support_set_base_training(self.base_train_adaptor_instance)
        )
        MetadataCatalog.get("coco_base_training_support").set(
            thing_classes=self.base_train_adaptor_instance.cfg.coco_classes,
            evaluator_type='coco')

        # Query: Validation
        DatasetCatalog.register(
            "coco_base_training_query_val",
            lambda: load_coco_json(
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/assets/json_base_val.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/",
                "coco_base_training_query_val"
            )
        )
        MetadataCatalog.get("coco_base_training_query_val").set(
            evaluator_type='coco',
            json_file="/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/assets/json_base_val.json"
        )

    def get_query_set_base_training_train(self, base_pipe_instance):
        """
        Query: Train
        """
        dataset_dicts = []

        for path_img in base_pipe_instance.query_dataset_train.lines:
            path_img = path_img.strip()
            img_id = os.path.basename(path_img).split('_')[-1].split('.')[0]

            record = copy.deepcopy(self.dict_coco2017_id_item[img_id])
            record['file_name'] = path_img

            # filter annotations
            annotations_filtered = []
            for ann in record['annotations']:
                if ann['category_id'] in base_pipe_instance.cfg.base_ids:
                    annotations_filtered.append(ann)

            record['annotations'] = annotations_filtered

            if len(annotations_filtered) > 0:
                dataset_dicts.append(record)

        return dataset_dicts

    def get_query_set_base_training_val(self, base_pipe_instance):
        """
        Query: Validation
        !! Similar to the Train counterpart. Might refactor later.
        """
        dataset_dicts = []

        for path_img in base_pipe_instance.query_dataset_val.lines:
            path_img = path_img.strip()
            img_id = os.path.basename(path_img).split('_')[-1].split('.')[0]

            record = copy.deepcopy(self.dict_coco2017_id_item[img_id])
            record['file_name'] = path_img

            # filter annotations
            annotations_filtered = []
            for ann in record['annotations']:
                if ann['category_id'] in base_pipe_instance.cfg.base_ids:
                    annotations_filtered.append(ann)

            record['annotations'] = annotations_filtered

            if len(annotations_filtered) > 0:
                dataset_dicts.append(record)

        return dataset_dicts

    def get_support_set_base_training(self, base_pipe_instance):
        """
        Support: Train + Validation
        All samples in the class-based bins have at least one of their respective class's bbox
        """
        dataset_dicts_label_wise = {e: [] for e in base_pipe_instance.cfg.base_ids}

        assert(len(base_pipe_instance.support_set.metalines) == len(base_pipe_instance.cfg.base_ids))

        for i, class_id in enumerate(base_pipe_instance.cfg.base_ids):

            for path_img in base_pipe_instance.support_set.metalines[i]:
                path_img = path_img.strip()

                img_id = os.path.basename(path_img).split('_')[-1].split('.')[0]
                record = copy.deepcopy(self.dict_coco2017_id_item[img_id])
                record['file_name'] = path_img

                # filter annotations
                annotations_filtered = []
                for ann in record['annotations']:
                    if ann['category_id'] in [class_id] and ann['iscrowd'] == 0:
                        annotations_filtered.append(ann)

                record['annotations'] = annotations_filtered

                if len(annotations_filtered) > 0:
                    dataset_dicts_label_wise[class_id].append(record)

        return dataset_dicts_label_wise

    def get_label2int_from_list(self, labels_list):
        lab2int = {e: i for i, e in enumerate(labels_list)}
        return lab2int