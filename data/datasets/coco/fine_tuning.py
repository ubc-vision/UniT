"""
Register COCO fine-tuning routine
"""

import os
import copy
import pickle

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from ...pipelines_adaptor.coco.fine_tuning import FineTuneFSODAdaptor
from detectron2.data.datasets.coco import load_coco_json

PATH_CONFIG = '/h/skhandel/FewshotDetection/WSASOD/data/pipelines_adaptor/coco/config_fine_tuning.yaml'
# assert(os.path.isfile(PATH_CONFIG))


class FineTuneData:
    """
    Registers data for fine-tuning phase
    """
    def __init__(self, novel_id=3, num_shots=10):
        self.novel_id = novel_id
        self.num_shots = num_shots

        self.fine_tune_adaptor_instance = self._fetch_the_copy_of_instance_if_tmp_stored()

        self.fine_tune_adaptor_instance = copy.deepcopy(self.fine_tune_adaptor_instance)

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
        if self.num_shots == 10 or self.num_shots==30:
            file_name = 'tmp/coco_fine_tune_instance_shots_{}.pkl'.format(self.num_shots)
        elif self.num_shots == 5:
            file_name = 'tmp/coco_fine_tune_instance_shots_10.pkl'
        else:
            file_name = 'tmp/coco_fine_tune_instance_shots_30.pkl'
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                fine_tune_adaptor_instance = pickle.load(f)
        else:
            os.makedirs('tmp', exist_ok=True)
            fine_tune_adaptor_instance = FineTuneFSODAdaptor(
                novel_id=self.novel_id, num_shots=self.num_shots,
                path_config=PATH_CONFIG
            )
            with open(file_name, 'wb') as f:
                pickle.dump(fine_tune_adaptor_instance, f)

        return fine_tune_adaptor_instance

    def _register_coco_2017_if_not_present(self, ):
        if 'coco_2017_train_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2017_train_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/locatron/fsod_utils/data/MSCOCO/annotations/instances_train2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/locatron/fsod_utils/data/MSCOCO/images/"
            )
        if 'coco_2017_val_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2017_val_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/locatron/fsod_utils/data/MSCOCO/annotations/instances_val2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/locatron/fsod_utils/data/MSCOCO/images/"
            )

    def _fetch_coco_2017_data(self, ):
        self.data_train_coco_2017 = DatasetCatalog.get('coco_2017_train_copy')
        self.data_val_coco_2017 = DatasetCatalog.get('coco_2017_val_copy')

        self.meta_coco_2017 = MetadataCatalog.get('coco_2017_train_copy')

    def _load_datasets(self, ):
        # Query: Train
        self.fine_tuning_query_train = self.get_query_set_fine_tuning_train(
            self.fine_tune_adaptor_instance)

        # Support: Train + Validation
        self.fine_tuning_support = self.get_support_set_fine_tuning(
                                            self.fine_tune_adaptor_instance)

        # Query: Validation
        self.fine_tuning_query_val = self.get_query_set_fine_tuning_val(
                                            self.fine_tune_adaptor_instance)

    def _register_datasets(self, ):
        # Query: Train
        DatasetCatalog.register(
            "coco_fine_tuning_query_train",
            lambda: self.get_query_set_fine_tuning_train(self.fine_tune_adaptor_instance)
        )
        MetadataCatalog.get("coco_fine_tuning_query_train").set(
            # thing_classes=self.fine_tune_adaptor_instance.cfg.coco_classes,
            thing_classes=MetadataCatalog.get("coco_2017_train").thing_classes,
            evaluator_type='coco')

        # Support: Train + Validation
        DatasetCatalog.register(
            "coco_fine_tuning_support",
            lambda: self.get_support_set_fine_tuning(self.fine_tune_adaptor_instance)
        )
        MetadataCatalog.get("coco_fine_tuning_support").set(
            # thing_classes=self.fine_tune_adaptor_instance.cfg.coco_classes,
            thing_classes=MetadataCatalog.get("coco_2017_train").thing_classes,
            evaluator_type='coco')

        # Query: Validation      
        DatasetCatalog.register(
            "coco_fine_tuning_query_val",
            lambda: load_coco_json(
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/assets/json_fine_val.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/",
                "coco_fine_tuning_query_val"
            )
        )
        MetadataCatalog.get("coco_fine_tuning_query_val").set(
            # thing_classes=self.fine_tune_adaptor_instance.cfg.coco_classes,
            evaluator_type='coco',
            json_file="/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/assets/json_fine_val.json"
        )

    def get_query_set_fine_tuning_train(self, base_pipe_instance):
        """
        Query: Train
        """
        dataset_dicts = []
        category_counter = {}
        for path_img in base_pipe_instance.query_dataset_train.lines:
            path_img = path_img.strip()
            img_id = os.path.basename(path_img).split('_')[-1].split('.')[0]

            record = copy.deepcopy(self.dict_coco2017_id_item[img_id])
            record['file_name'] = path_img
            # filter annotations
            annotations_filtered = []
            for ann in record['annotations']:
                if ann['category_id'] in base_pipe_instance.cfg.base_ids:
                    if self.num_shots == 5 or self.num_shots == 20:
                        if category_counter.get(ann['category_id'], 0) >= self.num_shots:
                            continue
                    annotations_filtered.append(ann)
                    category_counter[ann['category_id']] = category_counter.get(ann['category_id'], 0) + 1

            record['annotations'] = annotations_filtered

            if len(annotations_filtered) > 0:
                dataset_dicts.append(record)

        return dataset_dicts

    def get_query_set_fine_tuning_val(self, base_pipe_instance):
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

    def get_support_set_fine_tuning(self, base_pipe_instance):
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