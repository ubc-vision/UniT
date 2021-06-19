import copy
import inspect
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads, build_box_head, Res5ROIHeads
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from .fast_rcnn import build_fastrcnn_head
from .visual_attention_head import build_visual_attention_head
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.evaluation.evaluator import inference_context

@ROI_HEADS_REGISTRY.register()
class WeakDetectorHead(StandardROIHeads):
    @configurable
    def __init__(self,*, box_in_features, box_pooler, box_head, box_predictor, mask_in_features=None, mask_pooler=None, mask_head=None, keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None, weak_box_head=None, visual_attention_head=None, train_on_pred_boxes=False, freeze_layers=[], **kwargs):
        self._base_classes_id = kwargs['base_classes_id']
        self._novel_classes_id = kwargs['novel_classes_id']
        self.train_dataset_name = kwargs['train_dataset_name']
        self.weak_divisor = kwargs['weak_divisor']
        self.terms = kwargs['terms']

        del kwargs['base_classes_id']
        del kwargs['novel_classes_id']
        del kwargs['train_dataset_name']
        del kwargs['weak_divisor']
        del kwargs['terms']
        super(WeakDetectorHead, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor, 
                                                    mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, keypoint_in_features=keypoint_in_features, 
                                                    keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, **kwargs)
        self.compute_similarity = {'lingual' : 'lingual' in [y for _,x in self.terms.items() for y in x], 'visual' : 'visual' in [y for _,x in self.terms.items() for y in x]}
        self._freeze_layers(layers=freeze_layers)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.ROI_HEADS
        ret['base_classes_id'] = cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID
        ret['novel_classes_id'] = cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID
        ret['train_dataset_name'] = cfg.DATASETS.TRAIN[0]
        ret['weak_divisor'] = cfg.MODEL.ROI_HEADS.WEAK_CLASSIFIER_PROPOSAL_DIVISOR
        ret['terms'] = {'cls': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.CLASSIFIER, 'bbox': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.BBOX}
        if cfg.MODEL.MASK_ON:
            ret['terms']['seg']: cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.MASK
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        in_channels = [input_shape[f].channels for f in in_features]
        in_channels = in_channels[0]
    
        del ret['box_predictor']
        ret['box_predictor'] = build_fastrcnn_head(cfg, ret['box_head'].output_shape)
        return ret

    def _forward_box(self, features, proposals, weak_features=None, weak_proposals=None, weak_targets=None, meta_attention=None, tta=False):
        features = [features[f] for f in self.box_in_features]
        pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(pooled_features)
 
        predictions, weak_predictions = self.box_predictor(box_features)
        if self.training:
            losses = {}
            losses.update(self.box_predictor.losses(predictions, proposals, weak_targets))
            del box_features
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            del box_features
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, tta=tta)
            return pred_instances

    def forward(self, images, features, proposals, targets=None, weak_images=None, weak_features=None, weak_proposals=None, weak_targets=None, meta_features=None, meta_targets=None, meta_attention=None, return_attention=False, tta=False):
        # perms = []
        # for proposal in proposals:
        #     perm = torch.arange(len(proposal), device=images.device)[:self.batch_size_per_image//self.weak_divisor]
        #     perms.append(perm)
        # proposals = [proposal[perms[x]] for x,proposal in enumerate(proposals)]
        del images
            
        if self.training:
            losses = {}
            losses.update(self._forward_box(features, proposals, weak_targets=targets))

            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, meta_attention=meta_attention, tta=tta)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if not tta:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class WSROIHead(StandardROIHeads):
    @configurable
    def __init__(self,*, box_in_features, box_pooler, box_head, box_predictor, mask_in_features=None, mask_pooler=None, mask_head=None, keypoint_in_features=None, keypoint_pooler=None, keypoint_head=None, weak_box_head=None, visual_attention_head=None, train_on_pred_boxes=False, freeze_layers=[], **kwargs):
        self._base_classes_id = kwargs['base_classes_id']
        self._novel_classes_id = kwargs['novel_classes_id']
        self.train_dataset_name = kwargs['train_dataset_name']
        self.weak_divisor = kwargs['weak_divisor']
        self.terms = kwargs['terms']
        self.load_proposals = kwargs['load_proposals']
        self.visual_threshold = kwargs['visual_threshold']
        self.similarity_combination = kwargs['similarity_combination']
        self.topk = kwargs['topk']

        del kwargs['base_classes_id']
        del kwargs['novel_classes_id']
        del kwargs['train_dataset_name']
        del kwargs['weak_divisor']
        del kwargs['terms']
        del kwargs['load_proposals']
        del kwargs['visual_threshold']
        del kwargs['similarity_combination']
        del kwargs['topk']
        super(WSROIHead, self).__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head, box_predictor=box_predictor, 
                                                    mask_in_features=mask_in_features, mask_pooler=mask_pooler, mask_head=mask_head, keypoint_in_features=keypoint_in_features, 
                                                    keypoint_pooler=keypoint_pooler, keypoint_head=keypoint_head, train_on_pred_boxes=train_on_pred_boxes, **kwargs)
        self.compute_similarity = {'lingual' : 'lingual' in [y for _,x in self.terms.items() for y in x], 'visual' : 'visual' in [y for _,x in self.terms.items() for y in x]}
        self.visual_attention_head = visual_attention_head
        self.weak_box_head = weak_box_head
        self._class_mappings()
        self._freeze_layers(layers=freeze_layers)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.ROI_HEADS
        ret['base_classes_id'] = cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID
        ret['novel_classes_id'] = cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID
        ret['train_dataset_name'] = cfg.DATASETS.TRAIN[0]
        ret['weak_divisor'] = cfg.MODEL.ROI_HEADS.WEAK_CLASSIFIER_PROPOSAL_DIVISOR
        ret['terms'] = {'cls': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.CLASSIFIER, 'bbox': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.BBOX}
        if cfg.MODEL.MASK_ON:
            ret['terms']['seg'] = cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.MASK
        ret['load_proposals'] = cfg.MODEL.LOAD_PROPOSALS
        ret['visual_threshold'] = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.VISUAL_SIMILARITY_THRESHOLD
        ret['similarity_combination'] = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.SIMILARITY_COMBINATION
        ret['topk'] = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.TOPK
        return ret

    def _class_mappings(self):
        self._coco = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
        self._voc = MetadataCatalog.get(self.train_dataset_name).thing_classes
        self._coco_indexer = []
        for name in self._voc:
            if name == 'aeroplane':
                name = 'airplane'
            if name == 'diningtable':
                name = 'dining table'
            if name == 'motorbike':
                name = 'motorcycle'
            if name == 'pottedplant':
                name = 'potted plant'
            if name == 'sofa':
                name = 'couch'
            if name== 'tvmonitor':
                name = 'tv'
            self._coco_indexer.append(self._coco[name])
        self._coco_indexer = np.array(self._coco_indexer)
        self._base_classes = np.array(self._base_classes_id)
        self._novel_classes = np.array(self._novel_classes_id)

        self._remaining_ids = np.setdiff1d(np.arange(len(self._coco.keys())), self._coco_indexer)
        self._coco_indexer_tensor = torch.tensor(self._coco_indexer).long()
        self._remaining_ids_tensor = torch.tensor(self._remaining_ids).long()
        self._base_classes_tensor = torch.tensor(self._base_classes).long()
        self._novel_classes_tensor = torch.tensor(self._novel_classes).long()

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        in_channels = [input_shape[f].channels for f in in_features]
        in_channels = in_channels[0]  
    
        del ret['box_predictor']
        ret['box_predictor'] = build_fastrcnn_head(cfg, ret['box_head'].output_shape)
        ret['visual_attention_head'] = build_visual_attention_head(cfg, input_shape)
        if cfg.MODEL.ROI_HEADS.MULTI_BOX_HEAD:
            ret['weak_box_head'] = build_box_head(cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution))
        return ret

    def move_mappings_to_gpu(self):
        if not self._coco_indexer_tensor.is_cuda:
            device = next(self.box_predictor.parameters()).device
            self._coco_indexer_tensor = self._coco_indexer_tensor.to(device)
            self._remaining_ids_tensor = self._remaining_ids_tensor.to(device)
            self._base_classes_tensor = self._base_classes_tensor.to(device)
            self._novel_classes_tensor = self._novel_classes_tensor.to(device)

    def get_similarity_matrices(self, box_features, return_similarity=False):
        if self.compute_similarity['lingual']:
            lingual_similarity = self.box_predictor.get_similarity(base_classes=self._base_classes_tensor, novel_classes=self._novel_classes_tensor, indexer=self._coco_indexer_tensor)
        if self.compute_similarity['visual']:
            if len(box_features.size()) > 2:
                probs = torch.mean(torch.stack(self.box_predictor.weak_detector_head.evaluation(box_features.mean(dim=[2,3]))[0][0], 0), 0)
            else:
                probs = torch.mean(torch.stack(self.box_predictor.weak_detector_head.evaluation(box_features)[0][0], 0), 0)
            if return_similarity:
                visual_similarity_normalized_viz = probs.index_select(1, self._base_classes_tensor).clone().detach()
            visual_similarity_normalized = torch.softmax(probs, -1).index_select(1, self._base_classes_tensor)
            visual_similarity_normalized = visual_similarity_normalized/visual_similarity_normalized.sum(-1, keepdim=True).clamp(min=1e-9)
            visual_similarity_normalized[visual_similarity_normalized < self.visual_threshold] = 0
            # class_weights = torch.mean(torch.stack([self.box_predictor.weak_detector_head.oicr_predictors[oicr_iter].weight.clone().detach() for oicr_iter in range(self.box_predictor.weak_detector_head.oicr_iter)]), dim=0)
            # base_class_weights = class_weights.index_select(0, self._base_classes_tensor) 
            # base_class_weights_norm = base_class_weights / torch.norm(base_class_weights, dim=-1, keepdim=True).clamp(min=1e-9)
            # input_norm = box_features / torch.norm(box_features, dim=-1, keepdim=True).clamp(min=1e-9)
            # visual_similarity_normalized = torch.mm(input_norm, base_class_weights_norm.transpose(0,1))
            # visual_similarity_normalized = torch.softmax(visual_similarity_normalized * 20, -1)
            # visual_similarity_normalized[visual_similarity_normalized < self.visual_threshold] = 0

        similarity= {}
        for idx, head_type in enumerate(self.terms.keys()):
            similarity[head_type] =  torch.zeros(self._novel_classes_tensor.size(0), self._base_classes_tensor.size(0)).to(box_features.device)
            if self.similarity_combination == 'Sum':
                weight = 1.0/len(self.terms[head_type])
                if 'lingual' in self.terms[head_type]:
                    similarity[head_type] = similarity[head_type] + (weight * torch.softmax(lingual_similarity, dim=-1))
                if np.any(['TopK' in x for x in self.terms[head_type]]):
                    topk_value = int([x for x in self.terms[head_type] if 'TopK' in x][0].split("-")[1])
                    class_weights = torch.mean(torch.stack([self.box_predictor.weak_detector_head.oicr_predictors[oicr_iter].weight.clone().detach() for oicr_iter in range(self.box_predictor.weak_detector_head.oicr_iter)]), dim=0)
                    base_class_weights = class_weights.index_select(0, self._base_classes_tensor)
                    novel_class_weights = class_weights.index_select(0, self._novel_classes_tensor)
                    weight_similarity = torch.mm(novel_class_weights, base_class_weights.transpose(0,1))
                    topk, indices = torch.topk(weight_similarity, topk_value, dim=-1)
                    topk_similarity = torch.zeros(self._novel_classes_tensor.size(0), self._base_classes_tensor.size(0)).to(box_features.device)
                    topk_similarity = topk_similarity.scatter(1, indices, 1.)
                    topk_similarity = topk_similarity/torch.sum(topk_similarity, dim=-1, keepdim=True)
                    similarity[head_type] = similarity[head_type] + (weight * topk_similarity)
                if np.any(['WTopK' in x for x in self.terms[head_type]]):
                    topk_value = int([x for x in self.terms[head_type] if 'WTopK' in x][0].split("-")[1])
                    class_weights = torch.mean(torch.stack([self.box_predictor.weak_detector_head.oicr_predictors[oicr_iter].weight.clone().detach() for oicr_iter in range(self.box_predictor.weak_detector_head.oicr_iter)]), dim=0)
                    base_class_weights = class_weights.index_select(0, self._base_classes_tensor)
                    novel_class_weights = class_weights.index_select(0, self._novel_classes_tensor)
                    weight_similarity = torch.mm(novel_class_weights, base_class_weights.transpose(0,1))
                    topk, indices = torch.topk(weight_similarity, topk_value, dim=-1)
                    topk_similarity = torch.zeros(self._novel_classes_tensor.size(0), self._base_classes_tensor.size(0)).to(box_features.device)
                    topk_similarity = topk_similarity.scatter(1, indices, topk)
                    topk_similarity = topk_similarity/torch.sum(topk_similarity, dim=-1, keepdim=True)
                    similarity[head_type] = similarity[head_type] + (weight * topk_similarity)
                if np.any(['LSDA' in x for x in self.terms[head_type]]):
                    topk_value = int([x for x in self.terms[head_type] if 'LSDA' in x][0].split("-")[1])
                    class_weights = torch.mean(torch.stack([self.box_predictor.weak_detector_head.oicr_predictors[oicr_iter].weight.clone().detach() for oicr_iter in range(self.box_predictor.weak_detector_head.oicr_iter)]), dim=0)
                    base_class_weights = class_weights.index_select(0, self._base_classes_tensor)
                    novel_class_weights = class_weights.index_select(0, self._novel_classes_tensor)
                    weight_similarity = torch.norm((novel_class_weights.unsqueeze(1) - base_class_weights.unsqueeze(0)), dim=-1)
                    topk, indices = torch.topk(weight_similarity, topk_value, dim=-1, largest=False)
                    topk_similarity = torch.zeros(self._novel_classes_tensor.size(0), self._base_classes_tensor.size(0)).to(box_features.device)
                    topk_similarity = topk_similarity.scatter(1, indices, 1.)
                    topk_similarity = topk_similarity/torch.sum(topk_similarity, dim=-1, keepdim=True)
                    similarity[head_type] = similarity[head_type] + (weight * topk_similarity)
                if np.any(['VisualK' in x for x in self.terms[head_type]]):
                    topk_value = int([x for x in self.terms[head_type] if 'VisualK' in x][0].split("-")[1])
                    class_weights = torch.softmax(torch.mean(torch.stack(self.box_predictor.weak_detector_head.evaluation(box_features)[0][0], 0), 0).narrow(1, 0, self.num_classes), -1)
                    base_class_weights = class_weights.index_select(1, self._base_classes_tensor)
                    weight_similarity = base_class_weights / torch.sum(base_class_weights, -1, keepdim=True).clamp(min=1e-9)
                    topk, indices = torch.topk(weight_similarity, topk_value, dim=-1)
                    topk_similarity = torch.zeros(weight_similarity.size(0), self._base_classes_tensor.size(0)).to(box_features.device)
                    topk_similarity = topk_similarity.scatter(1, indices, topk)
                    topk_similarity = topk_similarity/torch.sum(topk_similarity, dim=-1, keepdim=True)
                    similarity[head_type] = similarity[head_type].unsqueeze(0) + (weight * topk_similarity.unsqueeze(1))
                if 'visual' in self.terms[head_type]:
                    similarity[head_type] = similarity[head_type].unsqueeze(0) + (weight * visual_similarity_normalized.unsqueeze(1))
                if 'Average' in self.terms[head_type]:
                    similarity[head_type] = similarity[head_type].fill_(1.)
                    similarity[head_type] = similarity[head_type]/torch.sum(similarity[head_type], dim=-1, keepdim=True)
                if len(self.terms[head_type]) > 0 and ('None' not in self.terms[head_type]):
                    similarity[head_type] = similarity[head_type]/torch.sum(similarity[head_type], dim=-1, keepdim=True).clamp(min=1e-9)
                else:
                    similarity[head_type] = 0.0 * similarity[head_type]
            else:
                if 'lingual' in self.terms[head_type]:
                    similarity[head_type] = similarity[head_type] * lingual_similarity
                    weight = 0.5 
                if 'visual' in self.terms[head_type]:
                    similarity[head_type] = similarity[head_type].unsqueeze(0) * visual_similarity_normalized.unsqueeze(1)
                if len(self.terms[head_type]) > 0:
                    similarity[head_type] = torch.softmax(similarity[head_type], -1)
        if not return_similarity:
            return similarity
        else:
            return similarity, [lingual_similarity, visual_similarity_normalized_viz]

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, selected_masks = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_box(self, features, proposals, weak_features=None, weak_proposals=None, weak_targets=None, meta_attention=None, tta=False, return_similarity=False, batch_size_per_image=100, return_proposals=False):
        features = [features[f] for f in self.box_in_features]
        pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(pooled_features)
        supervised_branch_box_features = None
        if self.weak_box_head is not None:
            with torch.no_grad():
                supervised_branch_box_features = self.weak_box_head(pooled_features)
        if weak_features is not None:
            weak_features = [weak_features[f] for f in self.box_in_features]
            weak_pooled_features = self.box_pooler(weak_features, [x.proposal_boxes for x in weak_proposals])
            if self.weak_box_head is not None:
                weak_box_features = self.weak_box_head(weak_pooled_features)
            else:
                weak_box_features = self.box_head(weak_pooled_features)
        else:
            weak_box_features = None
        
        similarity = None
        if not self.training:
            if self.compute_similarity['lingual']:
                lingual_similarity = self.box_predictor.get_similarity(base_classes=self._base_classes_tensor, novel_classes=self._novel_classes_tensor, indexer=self._coco_indexer_tensor)
            if self.compute_similarity['visual']:
                ### WORKS
                # probs = torch.relu(torch.mean(torch.stack(self.box_predictor.weak_detector_head.evaluation(box_features)[0][0], 0), 0))
                # visual_similarity = probs.index_select(1, self._base_classes_tensor)
                # visual_similarity_normalized = torch.softmax(visual_similarity, -1)
                ####

                probs = torch.relu(torch.mean(torch.stack(self.box_predictor.weak_detector_head.evaluation(box_features)[0][0], 0), 0))
                visual_similarity = probs.index_select(1, self._base_classes_tensor)
                visual_similarity_normalized = torch.softmax(visual_similarity, -1)
                # visual_weights = torch.sigmoid(probs.index_select(1, self._novel_classes_tensor))
                # visual_similarity = self.visual_attention_head.inference(pooled_features, base_classes=self._base_classes_tensor, meta_attention=meta_attention)
                # visual_similarity_normalized = visual_similarity/(torch.sum(visual_similarity, -1, keepdim=True).clamp(min=1e-7))
                # visual_similarity_normalized[visual_similarity_normalized < 0.4] = 0.0
            similarity= {}
            for idx, head_type in enumerate(self.terms.keys()):
                similarity[head_type] =  torch.ones(self._novel_classes_tensor.size(0), self._base_classes_tensor.size(0)).to(box_features.device)
                weight = 1.0
                if 'lingual' in self.terms[head_type]:
                    similarity[head_type] = similarity[head_type] *  torch.softmax(lingual_similarity, dim=-1)
                    # similarity[head_type] = similarity[head_type] *  lingual_similarity
                    weight = 0.5 
                if 'visual' in self.terms[head_type]:
                    # similarity[head_type] = ((1.0 - visual_weights).unsqueeze(2) * similarity[head_type].unsqueeze(0)) + (visual_weights.unsqueeze(2) * visual_similarity_normalized.unsqueeze(1))
                    similarity[head_type] = ((1.0 - weight) * similarity[head_type].unsqueeze(0)) + (weight * visual_similarity_normalized.unsqueeze(1))
                    # similarity[head_type] = similarity[head_type].unsqueeze(0) * visual_similarity_normalized.unsqueeze(1)
                if len(self.terms[head_type]) > 0:
                    similarity[head_type] = similarity[head_type]/torch.sum(similarity[head_type], dim=-1, keepdim=True)
                    # similarity[head_type] = torch.softmax(similarity[head_type], dim=-1)
        predictions, weak_predictions = self.box_predictor(box_features, supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        
        if self.training:
            losses = {}
            with torch.no_grad():
                meta_pooled_features = []
                meta_proposals = []
                len_perms = [len(proposal) for proposal in proposals]
                indices = np.insert(np.cumsum(len_perms), 0, 0)
                for img, proposal in enumerate(proposals):
                    pos_idx, neg_idx = subsample_labels(proposal.gt_classes, batch_size_per_image, self.positive_fraction, self.num_classes)
                    sampled_idx = torch.cat([pos_idx, neg_idx], dim=0)
                    meta_proposals.append(proposal[sampled_idx])
                    meta_pooled_features.append(pooled_features.narrow(0, start=int(indices[img]), length=int(indices[img+1] - indices[img]))[sampled_idx])
                meta_pooled_features = torch.cat(meta_pooled_features, dim=0)
            losses.update(self.visual_attention_head.rank_loss(meta_pooled_features, meta_proposals, base_classes=self._base_classes_tensor, meta_attention=meta_attention))
            losses.update(self.box_predictor.losses(predictions, proposals, weak_predictions=weak_predictions, weak_proposals=weak_proposals, weak_targets=weak_targets))
            del box_features
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            del box_features
            all_proposals = None
            if return_proposals:
                all_proposals = copy.deepcopy(proposals)
            pred_instances, filter_inds = self.box_predictor.inference(predictions, proposals, tta=tta)
            if return_similarity:
                for idx, pred_instance in enumerate(pred_instances):
                    pred_instances[idx]._lingual_similarity = lingual_similarity
                    pred_instances[idx]._visual_similarity = visual_similarity[filter_inds[idx]]
            return pred_instances, all_proposals
        
    def forward(self, images, features, proposals, targets=None, weak_images=None, weak_features=None, weak_proposals=None, weak_targets=None, meta_features=None, meta_targets=None, meta_attention=None, return_attention=False, tta=False, return_similarity=False, return_proposals=False):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        self.move_mappings_to_gpu()
        # Compute meta attention
        if meta_attention is None:
            if meta_features is not None:
                meta_attention = self.visual_attention_head(meta_features, meta_targets, self._base_classes_tensor)
                if return_attention:
                    return meta_attention

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if not self.load_proposals:
            if weak_proposals is not None:
                perms = []
                for proposal in weak_proposals:
                    perm = torch.arange(len(proposal), device=weak_images.device)[:self.batch_size_per_image//self.weak_divisor]
                    perms.append(perm)
                weak_proposals = [proposal[perms[x]] for x,proposal in enumerate(weak_proposals)]
        del weak_images
        if self.training:
            losses = {}
            losses.update(self._forward_box(features, proposals, weak_features=weak_features, weak_proposals=weak_proposals, weak_targets=weak_targets, meta_attention=meta_attention))

            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, proposals = self._forward_box(features, proposals, meta_attention=meta_attention, tta=tta, return_similarity=return_similarity, return_proposals=return_proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if not tta:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, proposals

@ROI_HEADS_REGISTRY.register()
class WSROIHeadNoMeta(WSROIHead):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['visual_attention_head']
        return ret

    def _forward_box(self, features, proposals, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False, return_proposals=False):
        if not train_only_weak:
            features = [features[f] for f in self.box_in_features]
            pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(pooled_features)
            supervised_branch_box_features = None
            if self.weak_box_head is not None:
                with torch.no_grad():
                    supervised_branch_box_features = self.weak_box_head(pooled_features)
        else:
            box_features = None
            supervised_branch_box_features = None

        if weak_features is not None:
            weak_features = [weak_features[f] for f in self.box_in_features]
            weak_pooled_features = self.box_pooler(weak_features, [x.proposal_boxes for x in weak_proposals])
            if self.weak_box_head is not None:
                weak_box_features = self.weak_box_head(weak_pooled_features)
            else:
                weak_box_features = self.box_head(weak_pooled_features)
        else:
            weak_box_features = None
        
        similarity = None
        if not self.training:
            if not return_similarity:
                similarity = self.get_similarity_matrices(box_features)
            else:
                similarity, similarity_values = self.get_similarity_matrices(box_features, return_similarity=return_similarity)

        predictions, weak_predictions = self.box_predictor(box_features, supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        
        if self.training:
            losses = {}
            losses.update(self.box_predictor.losses(predictions, proposals, weak_predictions=weak_predictions, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak))
            del box_features
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            del box_features
            all_proposals = None
            if return_proposals:
                all_proposals = [copy.deepcopy(proposals), predictions]
            pred_instances, filter_inds = self.box_predictor.inference(predictions, proposals, tta=tta)
            if return_similarity:
                for idx, pred_instance in enumerate(pred_instances):
                    pred_instances[idx]._lingual_similarity = similarity_values[0]
                    pred_instances[idx]._visual_similarity = similarity_values[1][filter_inds[idx]]
            return pred_instances, all_proposals

    def forward(self, images, features, proposals, targets=None, weak_images=None, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False, return_proposals=False):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        self.move_mappings_to_gpu()

        if not train_only_weak:
            if self.training:
                assert targets
                proposals = self.label_and_sample_proposals(proposals, targets)
            del targets

        if not self.load_proposals:
            if weak_proposals is not None:
                perms = []
                for proposal in weak_proposals:
                    perm = torch.arange(len(proposal), device=weak_images.device)[:self.batch_size_per_image//self.weak_divisor]
                    perms.append(perm)
                weak_proposals = [proposal[perms[x]] for x,proposal in enumerate(weak_proposals)]
        del weak_images

        if self.training:
            losses = {}
            losses.update(self._forward_box(features, proposals, weak_features=weak_features, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak))

            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_proposals = self._forward_box(features, proposals, tta=tta, return_similarity=return_similarity, return_proposals=return_proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if not tta:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, all_proposals

@ROI_HEADS_REGISTRY.register()
class WSROIHeadFineTune(WSROIHeadNoMeta):
    def _forward_box(self, features, proposals, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False, return_proposals=False):
        if not train_only_weak:
            features = [features[f] for f in self.box_in_features]
            pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(pooled_features)
            supervised_branch_box_features = None
            if self.weak_box_head is not None:
                with torch.no_grad():
                    supervised_branch_box_features = self.weak_box_head(pooled_features)
        else:
            box_features = None
            supervised_branch_box_features = None

        if weak_features is not None:
            weak_features = [weak_features[f] for f in self.box_in_features]
            weak_pooled_features = self.box_pooler(weak_features, [x.proposal_boxes for x in weak_proposals])
            if self.weak_box_head is not None:
                weak_box_features = self.weak_box_head(weak_pooled_features)
            else:
                weak_box_features = self.box_head(weak_pooled_features)
        else:
            weak_box_features = None
        
        similarity = self.get_similarity_matrices(box_features)
        predictions, weak_predictions = self.box_predictor(box_features, supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        
        if self.training:
            losses = {}
            losses.update(self.box_predictor.losses(predictions, proposals, weak_predictions=weak_predictions, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak))
            del box_features
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            del box_features
            all_proposals = None
            if return_proposals:
                all_proposals = [copy.deepcopy(proposals), predictions]
            pred_instances, filter_inds = self.box_predictor.inference(predictions, proposals, tta=tta)
            if return_similarity:
                for idx, pred_instance in enumerate(pred_instances):
                    pred_instances[idx]._lingual_similarity = lingual_similarity
                    pred_instances[idx]._visual_similarity = visual_similarity[filter_inds[idx]]
            return pred_instances, all_proposals

@ROI_HEADS_REGISTRY.register()
class WSROIHeadNoMetaWithMask(WSROIHead):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['visual_attention_head']
        return ret

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}

        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE 
        if pooler_type == "None":
            pooler_type=None
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = build_box_head(cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)).output_shape
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    def _forward_mask(self, features, instances, box_features=None, similarity=None):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, fg_selection_masks = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            if (box_features is not None) and self.training:
                features = box_features[torch.cat(fg_selection_masks, dim=0)]
            else:
                features = [features[f] for f in self.mask_in_features]
                pooled_features = self.box_pooler(features, [x.pred_boxes for x in instances])
                features = self.box_head(pooled_features)
        return self.mask_head(features, instances, similarity=similarity, base_classes=self._base_classes_tensor, novel_classes=self._novel_classes_tensor)

    def _forward_box(self, features, proposals, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False):
        if not train_only_weak:
            features = [features[f] for f in self.box_in_features]
            pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(pooled_features)
            supervised_branch_box_features = None
            if self.weak_box_head is not None:
                with torch.no_grad():
                    supervised_branch_box_features = self.weak_box_head(pooled_features)
                    if len(supervised_branch_box_features.size()) > 2:
                        supervised_branch_box_features = supervised_branch_box_features.mean(dim=[2,3])
        else:
            box_features = None
            supervised_branch_box_features = None

        if weak_features is not None:
            weak_features = [weak_features[f] for f in self.box_in_features]
            weak_pooled_features = self.box_pooler(weak_features, [x.proposal_boxes for x in weak_proposals])
            if self.weak_box_head is not None:
                weak_box_features = self.weak_box_head(weak_pooled_features)
            else:
                weak_box_features = self.box_head(weak_pooled_features)
            if len(weak_box_features.size()) > 2:
                weak_box_features = weak_box_features.mean(dim=[2,3])
        else:
            weak_box_features = None
        
        similarity = None
        if not self.training:
            similarity = self.get_similarity_matrices(box_features)
            
        if len(box_features.size()) > 2:
            predictions, weak_predictions = self.box_predictor(box_features.mean(dim=[2,3]), supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        else:
            predictions, weak_predictions = self.box_predictor(box_features, supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        if self.training:
            losses = {}
            losses.update(self.box_predictor.losses(predictions, proposals, weak_predictions=weak_predictions, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak))
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            if self.mask_pooler is None:
                return losses, box_features
            else:
                del box_features
                return losses, None
        else:
            pred_instances, filter_inds = self.box_predictor.inference(predictions, proposals, tta=tta)
            if return_similarity:
                for idx, pred_instance in enumerate(pred_instances):
                    pred_instances[idx]._lingual_similarity = lingual_similarity
                    pred_instances[idx]._visual_similarity = visual_similarity[filter_inds[idx]]
            if similarity is not None:
                assert len(pred_instances) == 1
                for idx, head_type in enumerate(self.terms.keys()):
                    if len(similarity[head_type].size()) > 2:
                        similarity[head_type] = similarity[head_type][filter_inds[0]]
            del box_features
            return pred_instances, similarity

    def forward_with_given_boxes(self, features, instances, similarity=None):
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances, similarity=similarity)
        return instances

    def forward(self, images, features, proposals, targets=None, weak_images=None, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        self.move_mappings_to_gpu()

        if not train_only_weak:
            if self.training:
                assert targets
                proposals = self.label_and_sample_proposals(proposals, targets)
            del targets

        if not self.load_proposals:
            if weak_proposals is not None:
                perms = []
                for proposal in weak_proposals:
                    perm = torch.arange(len(proposal), device=weak_images.device)[:self.batch_size_per_image//self.weak_divisor]
                    perms.append(perm)
                weak_proposals = [proposal[perms[x]] for x,proposal in enumerate(weak_proposals)]
        del weak_images

        if self.training:
            losses = {}
            box_losses, box_features = self._forward_box(features, proposals, weak_features=weak_features, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak)
            losses.update(box_losses)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals, box_features=box_features))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, similarity = self._forward_box(features, proposals, tta=tta, return_similarity=return_similarity)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if not tta:
                pred_instances = self.forward_with_given_boxes(features, pred_instances, similarity=similarity)
            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class WSROIHeadWithMaskFineTune(WSROIHeadNoMetaWithMask):
    def _forward_box(self, features, proposals, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False):
        if not train_only_weak:
            features = [features[f] for f in self.box_in_features]
            pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(pooled_features)
            supervised_branch_box_features = None
            if self.weak_box_head is not None:
                with torch.no_grad():
                    supervised_branch_box_features = self.weak_box_head(pooled_features)
                    if len(supervised_branch_box_features.size()) > 2:
                        supervised_branch_box_features = supervised_branch_box_features.mean(dim=[2,3])
        else:
            box_features = None
            supervised_branch_box_features = None

        if weak_features is not None:
            weak_features = [weak_features[f] for f in self.box_in_features]
            weak_pooled_features = self.box_pooler(weak_features, [x.proposal_boxes for x in weak_proposals])
            if self.weak_box_head is not None:
                weak_box_features = self.weak_box_head(weak_pooled_features)
            else:
                weak_box_features = self.box_head(weak_pooled_features)
            if len(weak_box_features.size()) > 2:
                weak_box_features = weak_box_features.mean(dim=[2,3])
        else:
            weak_box_features = None
        
        similarity = self.get_similarity_matrices(box_features)
            
        if len(box_features.size()) > 2:
            predictions, weak_predictions = self.box_predictor(box_features.mean(dim=[2,3]), supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        else:
            predictions, weak_predictions = self.box_predictor(box_features, supervised_branch_x_weak=supervised_branch_box_features, novel_classes=self._novel_classes_tensor, base_classes=self._base_classes_tensor, x_weak=weak_box_features, similarity=similarity)
        if self.training:
            losses = {}
            losses.update(self.box_predictor.losses(predictions, proposals, weak_predictions=weak_predictions, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak))
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            if self.mask_pooler is None:
                return losses, box_features, similarity
            else:
                del box_features
                return losses, None, similarity
        else:
            pred_instances, filter_inds = self.box_predictor.inference(predictions, proposals, tta=tta)
            if return_similarity:
                for idx, pred_instance in enumerate(pred_instances):
                    pred_instances[idx]._lingual_similarity = lingual_similarity
                    pred_instances[idx]._visual_similarity = visual_similarity[filter_inds[idx]]
            if similarity is not None:
                assert len(pred_instances) == 1
                for idx, head_type in enumerate(self.terms.keys()):
                    if len(similarity[head_type].size()) > 2:
                        similarity[head_type] = similarity[head_type][filter_inds[0]]
            del box_features
            return pred_instances, similarity

    def _forward_mask(self, features, instances, box_features=None, similarity=None):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, fg_selection_masks = select_foreground_proposals(instances, self.num_classes)
            if similarity is not None:
                similarity_selection_masks = torch.cat(fg_selection_masks, dim=0)
                for idx, head_type in enumerate(self.terms.keys()):
                    if len(similarity[head_type].size()) > 2:
                        similarity[head_type] = similarity[head_type][similarity_selection_masks]

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            if (box_features is not None) and self.training:
                features = box_features[torch.cat(fg_selection_masks, dim=0)]
            else:
                features = [features[f] for f in self.mask_in_features]
                pooled_features = self.box_pooler(features, [x.pred_boxes for x in instances])
                features = self.box_head(pooled_features)
        return self.mask_head(features, instances, similarity=similarity, base_classes=self._base_classes_tensor, novel_classes=self._novel_classes_tensor)

    def forward(self, images, features, proposals, targets=None, weak_images=None, weak_features=None, weak_proposals=None, weak_targets=None, tta=False, return_similarity=False, train_only_weak=False):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        self.move_mappings_to_gpu()

        if not train_only_weak:
            if self.training:
                assert targets
                proposals = self.label_and_sample_proposals(proposals, targets)
            del targets

        if not self.load_proposals:
            if weak_proposals is not None:
                perms = []
                for proposal in weak_proposals:
                    perm = torch.arange(len(proposal), device=weak_images.device)[:self.batch_size_per_image//self.weak_divisor]
                    perms.append(perm)
                weak_proposals = [proposal[perms[x]] for x,proposal in enumerate(weak_proposals)]
        del weak_images

        if self.training:
            losses = {}
            box_losses, box_features, similarity = self._forward_box(features, proposals, weak_features=weak_features, weak_proposals=weak_proposals, weak_targets=weak_targets, train_only_weak=train_only_weak)
            losses.update(box_losses)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals, box_features=box_features, similarity=similarity))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, similarity = self._forward_box(features, proposals, tta=tta, return_similarity=return_similarity)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if not tta:
                pred_instances = self.forward_with_given_boxes(features, pred_instances, similarity=similarity)
            return pred_instances, {}