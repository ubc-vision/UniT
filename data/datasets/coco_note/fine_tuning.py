"""
Register COCO fine-training routine
"""
import os
import copy
import pickle
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

IDS_BASE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
IDS_NOVEL_CLASSES = [e for e in range(80) if e not in IDS_BASE_CLASSES]


class FineTuneData:
    """
    Registers data for base-training phase
    """
    def __init__(self, num_shots):
        self.num_shots = num_shots

        self._register_coco_2017_if_not_present()
        self._fetch_coco_2017_data()

        self.label_to_annotation_dict = self.get_label_to_annotation_dict()
        self.label_to_annotation_dict_sampled = self.sample_dict_with_num_shots(
            self.label_to_annotation_dict,
            self.num_shots
        )
        self.label_to_annotation_dict_sampled_no_base = self.sample_dict_with_num_shots_no_base(
            self.label_to_annotation_dict,
            self.num_shots
        )
        self.label_to_annotation_dict_sampled_weak = self.get_label_to_annotation_dict_weak()

    def _register_coco_2017_if_not_present(self, ):
        if 'coco_2017_train_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2017_train_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_train2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/images/train2017/"
            )
        if 'coco_2017_val_copy' not in DatasetCatalog.keys():
            register_coco_instances(
                'coco_2017_val_copy',
                _get_builtin_metadata('coco'),
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_val2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/images/val2017/"
            )

    def _fetch_coco_2017_data(self, ):
        self.data_train_coco_2017 = DatasetCatalog.get('coco_2017_train_copy')
        self.data_val_coco_2017 = DatasetCatalog.get('coco_2017_val_copy')

        self.meta_coco_2017 = MetadataCatalog.get('coco_2017_train_copy')

    def _load_datasets(self, ):
        # Query: Train
        self.fine_tuning_query_train = self.get_query_set_fine_tuning_train()

        # Support
        self.fine_tuning_support = self.get_support_set_fine_tuning()

        # Query: Validation
        self.fine_tuning_query_val = self.get_query_set_fine_tuning_val()

    def _register_datasets(self, ):
        # Query: Train
        DatasetCatalog.register(
            "coco_note_fine_tuning_query_train",
            lambda: self.get_query_set_fine_tuning_train()
        )
        MetadataCatalog.get("coco_note_fine_tuning_query_train").set(
            thing_classes=MetadataCatalog.get("coco_2017_train_copy").thing_classes,
            evaluator_type='coco')

        DatasetCatalog.register(
            "coco_note_fine_tuning_query_train_no_base",
            lambda: self.get_query_set_fine_tuning_train_no_base()
        )
        MetadataCatalog.get("coco_note_fine_tuning_query_train_no_base").set(
            thing_classes=MetadataCatalog.get("coco_2017_train_copy").thing_classes,
            evaluator_type='coco')

        DatasetCatalog.register(
            "coco_note_fine_tuning_query_train_weak",
            lambda: self.get_query_set_fine_tuning_train_weak()
        )
        MetadataCatalog.get("coco_note_fine_tuning_query_train_weak").set(
            thing_classes=MetadataCatalog.get("coco_2017_train_copy").thing_classes,
            evaluator_type='coco')

        # Support: Train + Validation
        DatasetCatalog.register(
            "coco_note_fine_tuning_support",
            lambda: self.get_support_set_fine_tuning()
        )
        MetadataCatalog.get("coco_note_fine_tuning_support").set(
            thing_classes=MetadataCatalog.get("coco_2017_train_copy").thing_classes,
            evaluator_type='coco')
        """
        # Query: Validation
        DatasetCatalog.register(
            "coco_fine_tuning_query_val",
            lambda: self.get_query_set_fine_tuning_val()
        )
        MetadataCatalog.get("coco_fine_tuning_query_val").set(
            thing_classes=MetadataCatalog.get("coco_2017_val_copy").thing_classes,
            evaluator_type='coco')
        """
        # OLD
        # Query: Validation
        DatasetCatalog.register(
            "coco_note_fine_tuning_query_val",
            # lambda: self.get_query_set_fine_tuning_val(self.base_train_adaptor_instance)
            lambda: load_coco_json(
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_val2017.json",
                "/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/images/val2017/",
                "coco_note_fine_tuning_query_val"
            )
        )
        MetadataCatalog.get("coco_note_fine_tuning_query_val").set(
            # thing_classes=self.base_train_adaptor_instance.cfg.coco_classes,
            evaluator_type='coco',
            json_file="/scratch/ssd001/home/skhandel/FewshotDetection/WSASOD/data/data_utils/data/MSCOCO2017/annotations/instances_val2017.json"
        )

    # def get_label_to_annotation_dict(self, ):
    #     str_keywords = "note_label_to_annotation_dict"
    #     label_to_annotation_dict = fetch_object_if_tmp_stored(str_keywords)

    #     if label_to_annotation_dict is not None:
    #         return label_to_annotation_dict
    #     else:
    #         label_to_annotation_dict = {e: [] for e in range(80)}

    #         for e in self.data_train_coco_2017:
    #             for ann in e['annotations']:
    #                 if ann['iscrowd'] != 0:
    #                     continue

    #                 record = copy.deepcopy(e)

    #                 # filter annotations
    #                 annotations_filtered = [ann]
    #                 record['annotations'] = annotations_filtered

    #                 label_to_annotation_dict[ann['category_id']].append(record)

    #         store_object_in_tmp(str_keywords, label_to_annotation_dict)
    #         return label_to_annotation_dict
    
    def get_label_to_annotation_dict(self, ):
        str_keywords = "note_label_to_annotation_dict_per_image"
        label_to_annotation_dict = fetch_object_if_tmp_stored(str_keywords)

        if label_to_annotation_dict is not None:
            return label_to_annotation_dict
        else:
            label_to_annotation_dict = {e: [] for e in range(80)}

            for e in self.data_train_coco_2017:
                per_label_record = {}
                for ann in e['annotations']:
                    if ann['iscrowd'] != 0:
                        continue
                    if ann['category_id'] in per_label_record:
                        per_label_record[ann['category_id']]['annotations'].append(ann)
                    else:
                        record = copy.deepcopy(e)
                        # filter annotations
                        annotations_filtered = [ann]
                        record['annotations'] = annotations_filtered
                        per_label_record[ann['category_id']] = record
                for key in per_label_record.keys():
                    label_to_annotation_dict[key].append(per_label_record[key])

            store_object_in_tmp(str_keywords, label_to_annotation_dict)
            return label_to_annotation_dict

    def get_label_to_annotation_dict_weak(self, ):
        str_keywords = "note_label_to_annotation_dict_weak"
        label_to_annotation_dict = fetch_object_if_tmp_stored(str_keywords)

        if label_to_annotation_dict is not None:
            return label_to_annotation_dict
        else:
            label_to_annotation_dict = []

            for e in self.data_train_coco_2017:
                curr_annotations = []
                for ann in e['annotations']:
                    if ann['iscrowd'] != 0:
                        continue
                    if ann['category_id'] in IDS_NOVEL_CLASSES:
                        curr_annotations.append(ann)
                if len(curr_annotations) > 0:
                    record = copy.deepcopy(e)
                    record['annotations'] = curr_annotations
                    label_to_annotation_dict.append(record)
            store_object_in_tmp(str_keywords, label_to_annotation_dict)
            return label_to_annotation_dict

    def sample_dict_with_num_shots(self, label_to_annotation_dict, num_shots):
        np.random.seed(0)
        label_to_annotation_dict_sampled = {}
        for id_class, ann_list in label_to_annotation_dict.items():
            ann_list_sampled = np.random.choice(ann_list, size=num_shots, replace=False)
            label_to_annotation_dict_sampled[id_class] = ann_list_sampled
        return label_to_annotation_dict_sampled
    
    def sample_dict_with_num_shots_no_base(self, label_to_annotation_dict, num_shots):
        np.random.seed(0)
        label_to_annotation_dict_sampled = {}
        for id_class, ann_list in label_to_annotation_dict.items():
            if id_class in IDS_NOVEL_CLASSES:
                ann_list_sampled = np.random.choice(ann_list, size=num_shots, replace=False)
                label_to_annotation_dict_sampled[id_class] = ann_list_sampled
        return label_to_annotation_dict_sampled

    def sample_dict_with_num_shots_weak(self, label_to_annotation_dict, num_shots):
        np.random.seed(0)
        return label_to_annotation_dict

    def get_query_set_fine_tuning_train(self, ):
        """
        Query: Train
        """
        dataset_dicts = []
        for k, v in self.label_to_annotation_dict_sampled.items():
            dataset_dicts.extend(v)

        return dataset_dicts

    def get_query_set_fine_tuning_train_no_base(self, ):
        """
        Query: Train
        """
        dataset_dicts = []
        for k, v in self.label_to_annotation_dict_sampled_no_base.items():
            dataset_dicts.extend(v)

        return dataset_dicts

    def get_query_set_fine_tuning_train_weak(self, ):
        """
        Query: Train
        """
        return self.label_to_annotation_dict_sampled_weak

    def get_support_set_fine_tuning(self, ):
        """
        All samples in the class-based bins have at least one of their respective class's bbox
        """
        return copy.deepcopy(self.label_to_annotation_dict_sampled)


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
