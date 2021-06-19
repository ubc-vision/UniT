"""
Register COCO base-training routine
"""
import copy
import os
import pickle
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

IDS_BASE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
IDS_NOVEL_CLASSES = [e for e in range(80) if e not in IDS_BASE_CLASSES]


class BaseTrainData:
    """
    Registers data for base-training phase
    """
    def __init__(self, ):
        self._register_coco_2017_if_not_present()
        self._fetch_coco_2017_data()

    def _register_coco_2017_if_not_present(self, ):
        if 'coco_2014_train_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2014_train_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/annotations/instances_train2014.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/train2014/"
            )
        if 'coco_2014_val_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2014_val_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/annotations/instances_val2014.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/val2014/"
            )

    def _fetch_coco_2017_data(self, ):
        self.data_train_coco_2017 = DatasetCatalog.get('coco_2014_train_copy')
        self.data_val_coco_2017 = DatasetCatalog.get('coco_2014_val_copy')
        self.meta_coco_2017 = MetadataCatalog.get('coco_2014_train_copy')

    def _load_datasets(self, ):
        # Query: Train
        self.base_training_query_train = self.get_query_set_base_training_train()

        # Support
        self.base_training_support = self.get_support_set_base_training()

        # Query: Validation
        self.base_training_query_val = self.get_query_set_base_training_val()

    def _register_datasets(self, ):
        # Query: Train
        DatasetCatalog.register(
            "coco_dock_base_training_query_train",
            lambda: self.get_query_set_base_training_train()
        )
        MetadataCatalog.get("coco_dock_base_training_query_train").set(
            thing_classes=MetadataCatalog.get("coco_2014_train_copy").thing_classes,
            evaluator_type='coco')

        # Support: Train + Validation
        DatasetCatalog.register(
            "coco_dock_base_training_support",
            lambda: self.get_support_set_base_training()
        )
        MetadataCatalog.get("coco_dock_base_training_support").set(
            thing_classes=MetadataCatalog.get("coco_2014_train_copy").thing_classes,
            evaluator_type='coco')

        # Query: Validation
        DatasetCatalog.register(
            "coco_dock_base_training_query_val",
            # lambda: self.get_query_set_base_training_val(self.base_train_adaptor_instance)
            lambda: load_coco_json(
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/annotations/instances_val2014.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/images/val2014/",
                "coco_dock_base_training_query_val"
            )
        )
        MetadataCatalog.get("coco_dock_base_training_query_val").set(
            # thing_classes=self.base_train_adaptor_instance.cfg.coco_classes,
            evaluator_type='coco',
            json_file="/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO/annotations/instances_val2014.json"
        )
        
        # DatasetCatalog.register(
        #     "coco_note_base_training_query_val",
        #     lambda: self.get_query_set_base_training_val()
        # )
        # MetadataCatalog.get("coco_note_base_training_query_val").set(
        #     thing_classes=MetadataCatalog.get("coco_2017_val_copy").thing_classes,
        #     evaluator_type='coco')


    def get_query_set_base_training_train(self, ):
        """
        Query: Train
        """
        str_keywords = "query_set_base_training_train_dock"
        dataset_dicts = fetch_object_if_tmp_stored(str_keywords)

        if dataset_dicts is not None:
            return dataset_dicts
        else:
            dataset_dicts = []

            for e in self.data_train_coco_2017:
                record = copy.deepcopy(e)
                # filter annotations
                annotations_filtered = []
                for ann in record['annotations']:
                    if ann['category_id'] in IDS_BASE_CLASSES:
                        annotations_filtered.append(ann)

                record['annotations'] = annotations_filtered

                if len(annotations_filtered) > 0:
                    dataset_dicts.append(record)

            store_object_in_tmp(str_keywords, dataset_dicts)
            return dataset_dicts

    def get_query_set_base_training_val(self, ):
        """
        Query: Val
        """
        str_keywords = "query_set_base_training_val_dock"
        dataset_dicts = fetch_object_if_tmp_stored(str_keywords)

        if dataset_dicts is not None:
            return dataset_dicts
        else:
            dataset_dicts = []

            for e in self.data_val_coco_2017:
                record = copy.deepcopy(e)
                # filter annotations
                annotations_filtered = []
                for ann in record['annotations']:
                    if ann['category_id'] in IDS_BASE_CLASSES:
                        annotations_filtered.append(ann)

                record['annotations'] = annotations_filtered

                if len(annotations_filtered) > 0:
                    dataset_dicts.append(record)

            store_object_in_tmp(str_keywords, dataset_dicts)
            return dataset_dicts

    def get_support_set_base_training(self, ):
        """
        All samples in the class-based bins have at least one of their respective class's bbox
        """
        str_keywords = "support_set_base_training"
        dataset_dicts_label_wise = fetch_object_if_tmp_stored(str_keywords)

        if dataset_dicts_label_wise is not None:
            return dataset_dicts_label_wise
        else:
            dataset_dicts_label_wise = {e: [] for e in IDS_BASE_CLASSES}

            for e in self.data_train_coco_2017:
                for ann in e['annotations']:
                    if ann['iscrowd'] != 0:
                        continue
                    if ann['category_id'] not in IDS_BASE_CLASSES:
                        continue

                    record = copy.deepcopy(e)

                    # filter annotations
                    annotations_filtered = [ann]
                    record['annotations'] = annotations_filtered

                    dataset_dicts_label_wise[ann['category_id']].append(record)

            store_object_in_tmp(str_keywords, dataset_dicts_label_wise)
            return dataset_dicts_label_wise


def fetch_object_if_tmp_stored(str_keywords):
    str_file_name = "tmp/{}.pkl".format(str_keywords)

    object_loaded = None
    if os.path.isfile(str_file_name):
        with open(str_file_name, 'rb') as f:
            object_loaded = pickle.load(f)
    return object_loaded


def store_object_in_tmp(str_keywords, object_to_store):
    str_file_name = "tmp/{}.pkl".format(str_keywords)

    os.makedirs('tmp', exist_ok=True)
    with open(str_file_name, 'wb') as f:
        pickle.dump(object_to_store, f)
