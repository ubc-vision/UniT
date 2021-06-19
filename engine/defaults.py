import os
import sys
import torch
import numpy as np
import logging 
import detectron2.utils.comm as comm
import time 
import datetime
import pickle
import itertools
import pycocotools.mask as mask_util
from collections import OrderedDict
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.engine import DefaultTrainer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, print_csv_format, inference_context
from imantics import Polygons, Mask

from detectron2.engine import hooks, HookBase
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator
)
from ..checkpoint import PeriodicCheckpointerWithEval
from detectron2.engine import hooks
from detectron2.data.dataset_mapper import DatasetMapper
from ..data import build_detection_support_loader, build_classification_train_loader, MetaDatasetMapper, get_evaluator
from ..evalutation import inference_on_dataset
from ..solver import build_optimizer, build_optimizer_C4, modify_optimizer_C4
from torch.nn.parallel import DistributedDataParallel
from detectron2.checkpoint import DetectionCheckpointer

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(Trainer, self).__init__(cfg)
        self.classifier_data_loader = self.build_classifier_train_loader(cfg, multiplier=cfg.DATASETS.WEAK_CLASSIFIER_MUTLIPLIER)
        self._classifier_data_loader_iter = iter(self.classifier_data_loader)
        self.meta_data_loader = self.build_meta_train_loader(cfg)
        self._meta_data_loader_iter = iter(self.meta_data_loader)
        self.inital_training = 0

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        base_data = next(self._data_loader_iter)
        classifier_data = next(self._classifier_data_loader_iter)
        if self.meta_data_loader is not None:
            meta_data = next(self._meta_data_loader_iter)
        else:
            meta_data = None
        data_time = time.perf_counter() - start
        loss_dict = self.model(base_data, weak_batched_inputs=classifier_data, meta_data=meta_data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        comm.synchronize()
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        #     ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=20))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(PeriodicCheckpointerWithEval(cfg.TEST.EVAL_PERIOD, test_and_save_results, self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=3))
        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_meta_train_loader(cls, cfg):
        return build_detection_support_loader(cfg, dataset_name=cfg.DATASETS.META_TRAIN, num_shots=cfg.DATASETS.META_SHOTS, mapper=MetaDatasetMapper(cfg, True), flag_eval=False)

    @classmethod
    def build_meta_test_loader(cls, cfg, dataset_name):
        return build_detection_support_loader(cfg, dataset_name, num_shots=cfg.DATASETS.META_VAL_SHOTS, mapper=MetaDatasetMapper(cfg, False), flag_eval=True)

    @classmethod
    def build_classifier_train_loader(cls, cfg, multiplier=1):
        return build_classification_train_loader(cfg, mapper=DatasetMapper(cfg, True), multiplier=multiplier)

    @classmethod
    def build_base_meta_loader(cls, cfg):
        return build_detection_support_loader(cfg, dataset_name=cfg.DATASETS.BASE_META, num_shots=cfg.DATASETS.BASE_META_SHOTS, mapper=MetaDatasetMapper(cfg, False), flag_eval=True)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        return evaluator
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer_C4(cfg, model)

    @classmethod    
    def get_meta_attention(cls, cfg, model):
        device = next(model.parameters()).device
        base_ids = torch.tensor(cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID).long()
        novel_ids = torch.tensor(cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID).long()
        base_ids = base_ids.to(device)
        novel_ids = novel_ids.to(device)

        base_data_loader = cls.build_base_meta_loader(cfg)
        _base_meta_data_loader_iter = iter(base_data_loader)        
        
        base_data = next(_base_meta_data_loader_iter)
        with inference_context(model), torch.no_grad():
            meta_attention = model(None, meta_data=base_data, return_attention=True)
        return meta_attention

    @classmethod
    def test(cls, cfg, model, evaluators=None, meta_attention=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        if meta_attention is None:
            meta_attention = cls.get_meta_attention(cfg, model)
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, meta_attention=meta_attention)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

class TrainerNoMeta(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(TrainerNoMeta, self).__init__(cfg)
        self.classifier_data_loader = self.build_classifier_train_loader(cfg, multiplier=cfg.DATASETS.WEAK_CLASSIFIER_MUTLIPLIER)
        self._classifier_data_loader_iter = iter(self.classifier_data_loader)
        self.inital_training = 0
        self.train_only_weak = cfg.SOLVER.TRAIN_ONLY_WEAK
        self.multi_box_head = cfg.MODEL.ROI_HEADS.MULTI_BOX_HEAD
        self.supervised_disabled = False
        self.cfg = cfg
        self.freezed_params = []
        
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        base_data = next(self._data_loader_iter)
        classifier_data = next(self._classifier_data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(base_data, weak_batched_inputs=classifier_data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        comm.synchronize()
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        #     ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=20))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(PeriodicCheckpointerWithEval(cfg.TEST.EVAL_PERIOD, test_and_save_results, self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=3))
        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_classifier_train_loader(cls, cfg, multiplier=1):
        return build_classification_train_loader(cfg, mapper=DatasetMapper(cfg, True), multiplier=multiplier)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        return evaluator
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer_C4(cfg, model)

class TrainerOnlyWeak(TrainerNoMeta):
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        base_data = next(self._data_loader_iter)
        classifier_data = next(self._classifier_data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(None, weak_batched_inputs=classifier_data, train_only_weak=True)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        comm.synchronize()
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

class TrainerOnlyWeakFineTune(TrainerNoMeta):
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        base_data = next(self._data_loader_iter)
        classifier_data = next(self._classifier_data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(classifier_data, weak_batched_inputs=None, train_only_weak=False)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        comm.synchronize()
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)


class TrainerFineTune(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(TrainerFineTune, self).__init__(cfg)
        self.inital_training = 0
        self.train_only_weak = cfg.SOLVER.TRAIN_ONLY_WEAK
        self.multi_box_head = cfg.MODEL.ROI_HEADS.MULTI_BOX_HEAD
        self.supervised_disabled = False
        self.cfg = cfg
        self.freezed_params = []
        
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        base_data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(base_data, weak_batched_inputs=None)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        comm.synchronize()
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        #     ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=20))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(PeriodicCheckpointerWithEval(cfg.TEST.EVAL_PERIOD, test_and_save_results, self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=3))
        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_classifier_train_loader(cls, cfg, multiplier=1):
        return build_classification_train_loader(cfg, mapper=DatasetMapper(cfg, True), multiplier=multiplier)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        return evaluator
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer_C4(cfg, model)

class WeakDetectorTrainer(Trainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(WeakDetectorTrainer, self).__init__(cfg)
        del self.meta_data_loader
        del self._meta_data_loader_iter 
        self.meta_data_loader = None
        self._meta_data_loader_iter = None

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        classifier_data = next(self._data_loader_iter)
        base_data = None
        meta_data = None
        data_time = time.perf_counter() - start
        loss_dict = self.model(base_data, weak_batched_inputs=classifier_data, meta_data=meta_data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        comm.synchronize()
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

    @classmethod
    def test(cls, cfg, model, evaluators=None, meta_attention=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)