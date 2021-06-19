import copy
import torch
import logging
import operator
import numpy as np
import itertools

from detectron2.data import samplers
from detectron2.utils.comm import get_world_size
from detectron2.data.build import (
    print_instances_class_histogram,
    get_detection_dataset_dicts,
    worker_init_reset_seed,
    trivial_batch_collator
)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.common import (
    DatasetFromList, MapDataset, AspectRatioGroupedDataset
)
from .common import MapSupportDataset, SupportExamplesSampler
from .dataset_mapper import DatasetMapperSupport
from detectron2.data.dataset_mapper import DatasetMapper
from tabulate import tabulate
from termcolor import colored

def get_detection_dataset_dicts_support(dataset_name, filter_empty=True,
                                        min_keypoints=0, proposal_file=None):

    dataset_dict = DatasetCatalog.get(dataset_name)
    dataset_dict_flattened = []
    for id_class, annotations_class in dataset_dict.items():
        dataset_dict_flattened.extend(annotations_class)

    # pre-extracted proposal: need to think about this later
    """
    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]
    """
    # data distribution
    class_names = MetadataCatalog.get(dataset_name).thing_classes
    check_metadata_consistency("thing_classes", [dataset_name])
    print_instances_class_histogram(dataset_dict_flattened, class_names)

    return dataset_dict


def build_detection_support_loader(cfg, dataset_name, num_shots,
        mapper=None, flag_eval=False):
    """
    flag_eval: Sampling in eval mode is deterministic o/w not
    """
    dataset_dict = get_detection_dataset_dicts_support(dataset_name)
    # duplicating along repeated categories
    for k in dataset_dict.keys():
        dataset_dict[k] = duplicate_data_acc_to_repeated_categories(dataset_dict[k])
    dataset = SupportExamplesSampler(dataset_dict, num_shots, flag_eval=flag_eval)
    if mapper is None:
        mapper = DatasetMapperSupport(cfg)
    dataset = MapSupportDataset(dataset, mapper)
    logger = logging.getLogger(__name__)
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset), shuffle=not flag_eval)
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dict, cfg.DATALOADER.REPEAT_THRESHOLD,
            shuffle=not flag_eval
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False
    )
    if flag_eval:
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader


def build_detection_query_loader(cfg, dataset_names_tuple,
                                 dataset_proposal_files_tuple,
                                 mapper=None, is_train=True):
    """
    - Modified from detectron2.data.build_detection_train_loader
    - `dataset_names_tuple`: since we need to provide dataset names using
      different variables (meta-setup) and cfg could not be modified
      (CfgNode is immutable)
    - `is_train`: will create duplicated entries according to annotations
      So if an image contains five annotations, it will appear in the dataset
      five times with different annotations
    """

    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts(
        dataset_names_tuple,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=dataset_proposal_files_tuple if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    logger = logging.getLogger(__name__)

    # Train: split annotations class-wise
    if is_train:
        print("Query dataset num instances before annotation-wise duplication: {}".
                    format(len(dataset_dicts)))
        dataset_dicts = duplicate_data_acc_to_annotation_categories(dataset_dicts)
        print("Query dataset num instances after annotation-wise duplication: {}".
                    format(len(dataset_dicts)))

    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=is_train)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader


def build_detection_test_query_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.
    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.
    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def build_classification_train_loader_1(cfg, mapper=None, multiplier=1):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.
    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.
    Returns:
        an infinite iterator of training data
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    images_per_worker = images_per_worker * multiplier
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_CLASSIFIER_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = samplers.RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = samplers.RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    
    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )  # drop_last so the batch always have the same size
        data_loader =  torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )
    return data_loader

def build_classification_train_loader(cfg, mapper=None, multiplier=1):
    """
    A data loader is created by the following steps:
    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.
    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.
    Returns:
        an infinite iterator of training data
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    sample_num = cfg.DATASETS.WEAK_CLASSIFIER_SAMPLE_NUM
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    images_per_worker = int(images_per_worker * multiplier)
    if sample_num > 0:
        np.random.seed(cfg.DATASETS.SAMPLE_SEED)
        print ("Setting sampling seed:", cfg.DATASETS.SAMPLE_SEED)
        dataset_names = cfg.DATASETS.CLASSIFIER_TRAIN
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
        dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
        label_to_annotation_dict = {e: [] for e in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
        for e in dataset_dicts:
            per_label_record = {}
            for ann in e['annotations']:
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

        label_to_annotation_dict_sampled = {}
        for id_class, ann_list in label_to_annotation_dict.items():
            if id_class in cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID:
                if not cfg.DATASETS.OVER_SAMPLE:
                    if cfg.DATASETS.BASE_MULTIPLIER > 0:
                        try:
                            ann_list_sampled = np.random.choice(ann_list, size=int(sample_num*cfg.DATASETS.BASE_MULTIPLIER), replace=False)
                        except:
                            ann_list_sampled = np.random.choice(ann_list, size=int(sample_num*cfg.DATASETS.BASE_MULTIPLIER), replace=True)
                    else:
                        ann_list_sampled = ann_list
                else:
                    print ("BASE OVER SAMPLING")
                    ann_list_sampled = ann_list
                label_to_annotation_dict_sampled[id_class] = ann_list_sampled
            else:
                if not cfg.DATASETS.OVER_SAMPLE:
                    if cfg.DATASETS.BASE_MULTIPLIER > 0:
                        try:
                            ann_list_sampled = np.random.choice(ann_list, size=sample_num, replace=False)
                        except:
                            ann_list_sampled = np.random.choice(ann_list, size=sample_num, replace=True)
                        if cfg.DATASETS.NOVEL_MULTIPLER > 0:
                            ann_list_sampled = np.repeat(ann_list_sampled, cfg.DATASETS.NOVEL_MULTIPLER)
                    else:
                        ann_list_sampled = []
                    
                else:
                    try:
                        ann_list_sampled_temp = np.random.choice(ann_list, size=sample_num, replace=False)
                        if not cfg.DATASETS.SAMPLE_WITH_REPLACEMENT:
                            print ("OVER SAMPLING")
                            ann_list_sampled = np.random.choice(ann_list_sampled_temp, size=len(ann_list), replace=True)
                        else:
                            ann_list_sampled_temp = np.random.choice(ann_list, size=sample_num, replace=False)
                            num_repeat = len(ann_list)//len(ann_list_sampled_temp)
                            num_remainder = len(ann_list)%len(ann_list_sampled_temp)
                            ann_list_sampled = np.repeat(ann_list_sampled_temp, num_repeat)
                            if num_remainder > 0:
                                ann_list_sampled = np.hstack((ann_list_sampled, np.random.choice(ann_list_sampled_temp, size=num_remainder, replace=True)))
                            print ("OVER SAMPLING FIXED NEW", len(ann_list_sampled_temp), len(ann_list_sampled))
                    except:
                        ann_list_sampled = ann_list
                label_to_annotation_dict_sampled[id_class] = ann_list_sampled
        dataset_dicts = []
        for k, v in label_to_annotation_dict_sampled.items():
            dataset_dicts.extend(v)
        
        DatasetCatalog.register(
            "classifier_train_sampled",
            lambda: dataset_dicts
        )
        MetadataCatalog.get("classifier_train_sampled").set(
            thing_classes=MetadataCatalog.get(dataset_names[0]).thing_classes,
            evaluator_type='pascal_voc')
        dataset_name = ('classifier_train_sampled',)
        # print([(x['image_id'], len(x['annotations'])) for x in dataset_dicts[:50]])
        # print_instances_class_histogram_1(dataset_dicts, MetadataCatalog.get(dataset_names[0]).thing_classes)
    else:
        dataset_name = cfg.DATASETS.CLASSIFIER_TRAIN

    dataset_dicts = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_CLASSIFIER_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
    )
        
        
    dataset = DatasetFromList(dataset_dicts, copy=False)
    # # filtering
    # dataset_filtered = []
    # for sample in dataset:
    #     e_class_ids = set([e['category_id'] for e in sample['annotations']])
    #     for e_class_ids_ in e_class_ids:
    #         if e_class_ids_ in cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID:
    #             dataset_filtered.append(sample)
    #             break
    # dataset = dataset_filtered
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader


def duplicate_data_acc_to_annotation_categories(dataset_dict):
    dataset_dict_query_duplicated_for_train = []

    for sample in dataset_dict:
        id_classes = np.array([e['category_id'] for e in sample['annotations']])

        for id_class in np.unique(id_classes):
            sample_duplicated = copy.deepcopy(sample)

            indices_id_class = np.where(np.array(id_classes) == id_class)[0]    
            annotations_id_class = [sample['annotations'][i] for i in indices_id_class]

            sample_duplicated['annotations'] = annotations_id_class
            dataset_dict_query_duplicated_for_train.append(sample_duplicated)

    return dataset_dict_query_duplicated_for_train


def duplicate_data_acc_to_repeated_categories(dataset_dict):
    dataset_dict_duplicated = []

    for sample in dataset_dict:
        id_classes = np.array([e['category_id'] for e in sample['annotations']])

        if len(np.unique(id_classes)) > 1:
            raise ValueError("Duplication w.r.t. repeated categories can only"
                "be performed when there is only one unique category present")
        for annotation in sample['annotations']:
            sample_duplicated = copy.deepcopy(sample)

            sample_duplicated['annotations'] = [annotation]
            dataset_dict_duplicated.append(sample_duplicated)

    return dataset_dict_duplicated

def print_instances_class_histogram_1(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print (table)