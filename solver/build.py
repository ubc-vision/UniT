from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch

from detectron2.config import CfgNode

from detectron2.solver.lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
from detectron2.solver.build import maybe_add_gradient_clipping
import logging

def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if 'oicr_predictors' in module_name or 'regression_branch' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.REFINEMENT_LR_FACTOR))
                lr  = lr * cfg.SOLVER.REFINEMENT_LR_FACTOR
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(
        params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
    )
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def build_optimizer_C4(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if 'oicr_predictors' in module_name or 'regression_branch' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.REFINEMENT_LR_FACTOR))
                lr  = lr * cfg.SOLVER.REFINEMENT_LR_FACTOR
            if 'classifier_stream' in module_name or 'detection_stream' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.MIL_LR_FACTOR))
                lr  = lr * cfg.SOLVER.MIL_LR_FACTOR
            if 'cls_score_delta' in module_name or 'bbox_pred_delta' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.DELTA_LR_FACTOR))
                lr  = lr * cfg.SOLVER.DELTA_LR_FACTOR
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(
        params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
    )
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def modify_optimizer_C4(cfg, model, train_only_weak=False, freezed_params=[]):
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    multi_box_head = cfg.MODEL.ROI_HEADS.MULTI_BOX_HEAD
    for module_name, module in model.named_modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                if module_name not in freezed_params:
                    continue
                
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if 'oicr_predictors' in module_name or 'regression_branch' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.REFINEMENT_LR_FACTOR))
                lr  = lr * cfg.SOLVER.REFINEMENT_LR_FACTOR
            if 'classifier_stream' in module_name or 'detection_stream' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.MIL_LR_FACTOR))
                lr  = lr * cfg.SOLVER.MIL_LR_FACTOR
            if 'cls_score_delta' in module_name or 'bbox_pred_delta' in module_name:
                logging.getLogger('detectron2').log(logging.INFO, "Setting learning rate of {} to {}".format(module_name, lr * cfg.SOLVER.DELTA_LR_FACTOR))
                lr  = lr * cfg.SOLVER.DELTA_LR_FACTOR
            if train_only_weak:
                if 'roi_heads' in module_name:
                    if 'weak' not in module_name:
                        if 'box_head' in module_name:
                            if multi_box_head:
                                value.requires_grad = False
                                freezed_params.append(module_name)
                                continue
                        else:
                            value.requires_grad = False 
                            freezed_params.append(module_name)
                            continue
            else:
                value.requires_grad = True
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(
        params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
    )
    optimizer = maybe_add_gradient_clipping(cfg, optimizer), freezed_params
    return optimizer