import logging
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import batched_nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads import StandardROIHeads, build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet

VISUAL_ATTENTION_HEAD_REGISTRY = Registry("VISUAL_ATTENTION_HEAD_REGISTRY")

class VisualAttentionHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        in_features       = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.POOLER_TYPE
        in_channels = [input_shape[f].channels for f in in_features]
        in_channels = in_channels[0]
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.box_in_features = in_features
        self.meta_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.meta_box_head = build_box_head(cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution))
        input_shape_box = self.meta_box_head.output_shape
        if isinstance(input_shape_box, int):  # some backward compatibility
            input_shape_box = ShapeSpec(channels=input_shape_box)
        input_size = input_shape_box.channels * (input_shape_box.width or 1) * (input_shape_box.height or 1)
        self.input_size = input_size
        self.pi_normalizer = 0.5 * input_size * np.log(2 * np.pi)
        self.rank_loss_classifier = Linear(input_size, self.num_classes + 1)
        nn.init.normal_(self.rank_loss_classifier.weight, std=0.01)
        nn.init.constant_(self.rank_loss_classifier.bias, 0.0)

    def _roi_transform_meta(self, features, proposals):
        with torch.no_grad():
            features = [features[f] for f in self.box_in_features]
            box_features = self.meta_box_pooler(features, proposals)
        box_features = self.meta_box_head(box_features)
        return box_features

    def estimate_conv(self, examples, rowvar=False, inplace=False):
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()
    
    def compute_similarity(self, means, covariances, logdets, query):
        distance_from_mean = query.unsqueeze(1) - means.unsqueeze(0)
        distance_from_mean = distance_from_mean.transpose(0,1)
        mahalanobis_distance = torch.mul(torch.bmm(distance_from_mean, covariances), distance_from_mean).sum(dim=2).transpose(0,1)
        log_likelihood = 0.5 * logdets.transpose(0,1) - self.pi_normalizer - 0.5 * mahalanobis_distance
        likelihood = log_likelihood.exp()
        return likelihood

    def get_query_features(self, pooled_features):
        pooled_features = pooled_features.clone().detach()
        features = self.meta_box_head(pooled_features)
        return features

    def get_meta_features(self, meta_attention, base_classes):
        meta_mean, meta_covariance, meta_logdet = meta_attention
        meta_mean_base = meta_mean.index_select(0, base_classes)
        meta_covariance_base = meta_covariance.index_select(0, base_classes)
        meta_logdet_base = meta_logdet.index_select(0, base_classes)

        # Add dummy mean for background class
        meta_mean_base = nn.functional.pad(meta_mean_base.unsqueeze(1), (0,0,0,0,0,1), 'constant', 0.).squeeze(1)
        meta_covariance_base = torch.cat([meta_covariance_base, torch.eye(meta_covariance_base.size(1)).unsqueeze(0).to(meta_covariance_base.device)], dim=0)
        meta_logdet_base = torch.cat([meta_logdet_base, torch.zeros(1).unsqueeze(0).to(meta_logdet_base.device)], dim=0)
        return meta_mean_base, meta_covariance_base, meta_logdet_base

    def rank_loss(self, pooled_features, proposals, base_classes, meta_attention=None):
        features = self.get_query_features(pooled_features)
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        meta_mean_base, meta_covariance_base, meta_logdet_base = self.get_meta_features(meta_attention, base_classes)

        base_similarity = self.compute_similarity(meta_mean_base, meta_covariance_base, meta_logdet_base, features)
        
        similarity = torch.zeros(base_similarity.size(0), self.num_classes + 1).to(base_similarity.device)
        padded_classes = torch.cat([base_classes, (torch.zeros(1) + self.num_classes).to(base_classes.device).long()])
        similarity = similarity.index_copy(1, padded_classes, base_similarity)

        # u_ix_i > u_jx_i : Example should be closer to GT class mean than other class means
        gt_index_scores = torch.gather(similarity, 1, gt_classes.unsqueeze(1))
        distance_matrix = gt_index_scores - similarity
        labels = torch.zeros_like(distance_matrix) - 1.
        labels = labels.scatter(1, gt_classes.unsqueeze(-1), 1.)
        margin_loss = nn.functional.hinge_embedding_loss(distance_matrix, labels, margin=0.5, reduction='none')
        margin_loss = torch.mean(margin_loss.index_select(dim=1, index=padded_classes))

        # Force similarity between example and GT class mean to be 1
        labels_reg = torch.zeros_like(gt_index_scores) - 1.
        reg_loss = torch.mean(nn.functional.hinge_embedding_loss(gt_index_scores, labels_reg, margin=1.0, reduction='none'))

        # Classification loss for the class means
        classification_logits = self.rank_loss_classifier(meta_mean_base)
        classification_loss = nn.functional.cross_entropy(classification_logits, padded_classes)
        return {'loss_rank_margin' : margin_loss, 'loss_rank_reg' : reg_loss, 'loss_rank_cls' : classification_loss}

    def inference(self, pooled_features, base_classes, meta_attention):
        pooled_features = pooled_features.clone().detach()
        meta_mean, meta_covariance, meta_logdet = meta_attention
        meta_mean_base = meta_mean.index_select(0, base_classes)
        meta_covariance_base = meta_covariance.index_select(0, base_classes)
        meta_logdet_base = meta_logdet.index_select(0, base_classes)
        features = self.get_query_features(pooled_features)

        # Add dummy mean for background class
        # meta_mean_base = nn.functional.pad(meta_mean_base.unsqueeze(1), (0,0,0,0,0,1), 'constant', 0.).squeeze(1)
        # meta_covariance_base = torch.cat([meta_covariance_base, torch.eye(meta_covariance_base.size(1)).unsqueeze(0).to(meta_covariance_base.device)], dim=0)
        # meta_logdet_base = torch.cat([meta_logdet_base, torch.zeros(1).unsqueeze(0).to(meta_logdet_base.device)], dim=0)

        base_similarity = self.compute_similarity(meta_mean_base, meta_covariance_base, meta_logdet_base, features)
        return base_similarity

    def forward(self, meta_features, meta_targets, base_classes_tensor):
        raise NotImplementedError

@VISUAL_ATTENTION_HEAD_REGISTRY.register()
class MahalanobisSimilarity(VisualAttentionHead):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def forward(self, meta_features, meta_targets, base_classes_tensor):
        meta_features_res5 = {}
        meta_features_mean = {}
        meta_features_covariance = {}
        meta_features_logdet = {}
        meta_classes = np.sort(list(meta_features.keys()))
        meta_idx_to_class = {x:y for y,x in enumerate(meta_classes)}
        meta_proposal_boxes = {idx:[x.gt_boxes for x in proposals] for idx, proposals in meta_targets.items()} 
        for idx, meta_feature in meta_features.items():
            meta_features_res5[idx] = self._roi_transform_meta(meta_feature, meta_proposal_boxes[idx])
            meta_features_mean[idx] = torch.mean(meta_features_res5[idx], dim=0)
            meta_features_covariance[idx] = torch.inverse(self.estimate_conv(meta_features_res5[idx]) + torch.eye(meta_features_res5[idx].size(1), meta_features_res5[idx].size(1)).to(meta_features_res5[idx].device))
            meta_features_logdet[idx] = torch.zeros(1).to(meta_features_res5[idx].device)
            # meta_features_logdet[idx] = torch.logdet(meta_features_covariance[idx]).unsqueeze(0)

        base_meta_mean = torch.cat([meta_features_mean[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_covariance = torch.cat([meta_features_covariance[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_logdet = torch.cat([meta_features_logdet[idx].unsqueeze(0) for idx in meta_classes], dim=0)

        meta_mean = torch.zeros(self.num_classes+1, base_meta_mean.size(1)).to(base_meta_mean.get_device())
        meta_covariance = torch.zeros(self.num_classes+1, base_meta_covariance.size(1), base_meta_covariance.size(1)).to(base_meta_covariance.get_device())
        meta_logdet = torch.zeros(self.num_classes+1, 1).to(base_meta_logdet.get_device()) - np.float("inf")

        meta_mean = meta_mean.index_copy(0, base_classes_tensor, base_meta_mean)
        meta_covariance = meta_covariance.index_copy(0, base_classes_tensor, base_meta_covariance)
        meta_logdet = meta_logdet.index_copy(0, base_classes_tensor, base_meta_logdet)
        return [meta_mean, meta_covariance, meta_logdet]

@VISUAL_ATTENTION_HEAD_REGISTRY.register()
class MeanSimilarity(VisualAttentionHead):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def forward(self, meta_features, meta_targets, base_classes_tensor):
        meta_features_res5 = {}
        meta_features_mean = {}
        meta_features_covariance = {}
        meta_features_logdet = {}
        meta_classes = np.sort(list(meta_features.keys()))
        meta_idx_to_class = {x:y for y,x in enumerate(meta_classes)}
        meta_proposal_boxes = {idx:[x.gt_boxes for x in proposals] for idx, proposals in meta_targets.items()} 
        for idx, meta_feature in meta_features.items():
            meta_features_res5[idx] = self._roi_transform_meta(meta_feature, meta_proposal_boxes[idx])
            meta_features_mean[idx] = torch.mean(meta_features_res5[idx], dim=0)
            meta_features_covariance[idx] = torch.eye(meta_features_res5[idx].size(1), meta_features_res5[idx].size(1)).to(meta_features_res5[idx].device)
            meta_features_logdet[idx] = torch.zeros(1).to(meta_features_res5[idx].device)
        base_meta_mean = torch.cat([meta_features_mean[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_covariance = torch.cat([meta_features_covariance[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_logdet = torch.cat([meta_features_logdet[idx].unsqueeze(0) for idx in meta_classes], dim=0)

        meta_mean = torch.zeros(self.num_classes+1, base_meta_mean.size(1)).to(base_meta_mean.get_device())
        meta_covariance = torch.zeros(self.num_classes+1, base_meta_covariance.size(1), base_meta_covariance.size(1)).to(base_meta_covariance.get_device())
        meta_logdet = torch.zeros(self.num_classes+1, 1).to(base_meta_logdet.get_device()) - np.float("inf")

        meta_mean = meta_mean.index_copy(0, base_classes_tensor, base_meta_mean)
        meta_covariance = meta_covariance.index_copy(0, base_classes_tensor, base_meta_covariance)
        meta_logdet = meta_logdet.index_copy(0, base_classes_tensor, base_meta_logdet)
        return [meta_mean, meta_covariance, meta_logdet]

@VISUAL_ATTENTION_HEAD_REGISTRY.register()
class MeanMatrixSimilarity(VisualAttentionHead):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.sim_matrix = Linear(self.input_size, self.input_size, bias=False)
        nn.init.constant_(self.sim_matrix.weight, 0.)
        with torch.no_grad():
            self.sim_matrix.weight.fill_diagonal_(1.)
    
    def inference(self, pooled_features, base_classes, meta_attention, threshold=0.6, eps=0.2):
        similarity = super().inference(pooled_features, base_classes, meta_attention)
        # similarity[similarity < threshold] = 0.0
        # similarity_sum = torch.sum(similarity, -1)
        # similarity[similarity_sum == 0] = 1.
        similarity = torch.relu(similarity)
        # similarity[similarity <= 0] = eps
        return similarity

    def compute_similarity(self, means, covariances, logdets, query):
        similarity = torch.mm(query, means.transpose(0,1))
        return torch.relu(similarity)

    def get_query_features(self, pooled_features, eps=1e-6):
        pooled_features = pooled_features.clone().detach()
        features = self.meta_box_head(pooled_features)
        norm_features = torch.norm(features, dim=-1, keepdim=True).clamp(min=eps)
        features = torch.div(features, norm_features)
        return features

    def get_meta_features(self, meta_attention, base_classes):
        meta_mean, meta_covariance, meta_logdet = meta_attention
        meta_mean_base = meta_mean.index_select(0, base_classes)
        meta_covariance_base = meta_covariance.index_select(0, base_classes)
        meta_logdet_base = meta_logdet.index_select(0, base_classes)

        # Add dummy mean for background class
        # meta_mean_base = nn.functional.pad(meta_mean_base.unsqueeze(1), (0,0,0,0,0,1), 'constant', 1./np.sqrt(meta_mean_base.size(1))).squeeze(1)
        # meta_covariance_base = torch.cat([meta_covariance_base, torch.eye(meta_covariance_base.size(1)).unsqueeze(0).to(meta_covariance_base.device)], dim=0)
        # meta_logdet_base = torch.cat([meta_logdet_base, torch.zeros(1).unsqueeze(0).to(meta_logdet_base.device)], dim=0)
        return meta_mean_base, meta_covariance_base, meta_logdet_base

    def rank_loss(self, pooled_features, proposals, base_classes, meta_attention=None):
        pooled_features = pooled_features.clone().detach()
        features = self.get_query_features(pooled_features)
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        meta_mean_base, meta_covariance_base, meta_logdet_base = self.get_meta_features(meta_attention, base_classes)

        base_classes_mask = gt_classes.lt(self.num_classes).unsqueeze(1)
        base_similarity = self.compute_similarity(meta_mean_base, meta_covariance_base, meta_logdet_base, features)
        
        similarity = torch.zeros(base_similarity.size(0), self.num_classes + 1).to(base_similarity.device) - 2.
        
         # Add contribution of background class
        similarity = similarity.scatter(1, gt_classes.unsqueeze(-1), 0.5)
        similarity = similarity.index_copy(1, base_classes, base_similarity)

        # u_ix_i > u_jx_i : Example should be closer to GT class mean than other class means
        gt_index_scores = torch.gather(similarity, 1, gt_classes.unsqueeze(1))
        distance_matrix = gt_index_scores - similarity
        labels = torch.zeros_like(distance_matrix) - 1.
        labels = labels.scatter(1, gt_classes.unsqueeze(-1), 1.)
        margin_loss = nn.functional.hinge_embedding_loss(distance_matrix, labels, margin=0.5, reduction='none')
        margin_loss = torch.mean(margin_loss.index_select(dim=1, index=base_classes))

        # Force similarity between example and GT class mean to be 1
        labels_reg = torch.zeros_like(gt_index_scores) - 1.
        reg_loss = nn.functional.hinge_embedding_loss(gt_index_scores, labels_reg, margin=1.0, reduction='none')
        reg_loss = torch.mean(torch.masked_select(reg_loss, base_classes_mask))

        # Classification loss for the class means
        classification_logits = self.rank_loss_classifier(meta_mean_base)
        classification_loss = nn.functional.cross_entropy(classification_logits, base_classes)
        return {'loss_rank_margin' : margin_loss, 'loss_rank_reg' : reg_loss, 'loss_rank_cls' : classification_loss}

    def forward(self, meta_features, meta_targets, base_classes_tensor, eps=1e-6):
        meta_features_res5 = {}
        meta_features_mean = {}
        meta_features_covariance = {}
        meta_features_logdet = {}
        meta_classes = np.sort(list(meta_features.keys()))
        meta_idx_to_class = {x:y for y,x in enumerate(meta_classes)}
        meta_proposal_boxes = {idx:[x.gt_boxes for x in proposals] for idx, proposals in meta_targets.items()} 
        for idx, meta_feature in meta_features.items():
            meta_features_res5[idx] = self._roi_transform_meta(meta_feature, meta_proposal_boxes[idx])
            meta_features_mean[idx] = torch.mean(meta_features_res5[idx], dim=0)
            meta_features_covariance[idx] = torch.eye(meta_features_res5[idx].size(1), meta_features_res5[idx].size(1)).to(meta_features_res5[idx].device)
            meta_features_logdet[idx] = torch.zeros(1).to(meta_features_res5[idx].device)
        base_meta_mean = torch.cat([meta_features_mean[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_covariance = torch.cat([meta_features_covariance[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_logdet = torch.cat([meta_features_logdet[idx].unsqueeze(0) for idx in meta_classes], dim=0)

        base_meta_mean = self.sim_matrix(base_meta_mean)
        base_meta_mean = torch.div(base_meta_mean, torch.norm(base_meta_mean, dim=-1, keepdim=True).clamp(min=eps))

        meta_mean = torch.zeros(self.num_classes+1, base_meta_mean.size(1)).to(base_meta_mean.get_device())
        meta_covariance = torch.zeros(self.num_classes+1, base_meta_covariance.size(1), base_meta_covariance.size(1)).to(base_meta_covariance.get_device())
        meta_logdet = torch.zeros(self.num_classes+1, 1).to(base_meta_logdet.get_device()) - np.float("inf")

        meta_mean = meta_mean.index_copy(0, base_classes_tensor, base_meta_mean)
        meta_covariance = meta_covariance.index_copy(0, base_classes_tensor, base_meta_covariance)
        meta_logdet = meta_logdet.index_copy(0, base_classes_tensor, base_meta_logdet)

        return [meta_mean, meta_covariance, meta_logdet]

@VISUAL_ATTENTION_HEAD_REGISTRY.register()
class MeanMatrixSimilarityLocatron(MeanMatrixSimilarity):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        del self.rank_loss_classifier
        self.sim_matrix = Linear(self.input_size, self.input_size, bias=False)
        nn.init.constant_(self.sim_matrix.weight, 0.)
        with torch.no_grad():
            self.sim_matrix.weight.fill_diagonal_(1.)
    
    def inference(self, pooled_features, base_classes, meta_attention, threshold=0.6):
        similarity = super().inference(pooled_features, base_classes, meta_attention)
        # similarity[similarity < threshold] = 0.0
        # similarity_sum = torch.sum(similarity, -1)
        # similarity[similarity_sum == 0] = 1.
        return torch.relu(similarity)

    def compute_similarity(self, means, covariances, logdets, query):
        similarity = torch.mm(query, means.transpose(0,1))
        return similarity

    def get_query_features(self, pooled_features, eps=1e-6):
        pooled_features = pooled_features.clone().detach()
        features = self.meta_box_head(pooled_features)
        norm_features = torch.norm(features, dim=-1, keepdim=True).clamp(min=eps)
        features = torch.div(features, norm_features)
        return features

    def get_meta_features(self, meta_attention, base_classes):
        meta_mean, meta_covariance, meta_logdet = meta_attention
        meta_mean_base = meta_mean.index_select(0, base_classes)
        meta_covariance_base = meta_covariance.index_select(0, base_classes)
        meta_logdet_base = meta_logdet.index_select(0, base_classes)

        # Add dummy mean for background class
        # meta_mean_base = nn.functional.pad(meta_mean_base.unsqueeze(1), (0,0,0,0,0,1), 'constant', 1./np.sqrt(meta_mean_base.size(1))).squeeze(1)
        # meta_covariance_base = torch.cat([meta_covariance_base, torch.eye(meta_covariance_base.size(1)).unsqueeze(0).to(meta_covariance_base.device)], dim=0)
        # meta_logdet_base = torch.cat([meta_logdet_base, torch.zeros(1).unsqueeze(0).to(meta_logdet_base.device)], dim=0)
        return meta_mean_base, meta_covariance_base, meta_logdet_base

    def rank_loss(self, pooled_features, proposals, base_classes, meta_attention=None):
        features = self.get_query_features(pooled_features)
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        meta_mean_base, meta_covariance_base, meta_logdet_base = self.get_meta_features(meta_attention, base_classes)

        base_classes_mask = gt_classes.lt(self.num_classes).unsqueeze(1)
        base_similarity = self.compute_similarity(meta_mean_base, meta_covariance_base, meta_logdet_base, features)
        
        similarity = torch.zeros(base_similarity.size(0), self.num_classes + 1).to(base_similarity.device) - 2.
        
         # Add contribution of background class
        similarity = similarity.scatter(1, gt_classes.unsqueeze(-1), 0.5)
        similarity = similarity.index_copy(1, base_classes, base_similarity)

        # u_ix_i > u_jx_i : Example should be closer to GT class mean than other class means
        gt_index_scores = torch.gather(similarity, 1, gt_classes.unsqueeze(1))
        distance_matrix = gt_index_scores - similarity
        labels = torch.zeros_like(distance_matrix) - 1.
        labels = labels.scatter(1, gt_classes.unsqueeze(-1), 1.)
        margin_loss = nn.functional.hinge_embedding_loss(distance_matrix, labels, margin=0.5, reduction='none')
        margin_loss = torch.mean(margin_loss.index_select(dim=1, index=base_classes))

        # Force similarity between example and GT class mean to be 1
        labels_reg = torch.zeros_like(gt_index_scores) - 1.
        reg_loss = nn.functional.hinge_embedding_loss(gt_index_scores, labels_reg, margin=1.0, reduction='none')
        reg_loss = torch.mean(torch.masked_select(reg_loss, base_classes_mask))

        # Classification loss for the class means
        return {'loss_rank_margin' : margin_loss, 'loss_rank_reg' : reg_loss}

    def forward(self, meta_features, meta_targets, base_classes_tensor, eps=1e-6):
        meta_features_res5 = {}
        meta_features_mean = {}
        meta_features_covariance = {}
        meta_features_logdet = {}
        meta_classes = np.sort(list(meta_features.keys()))
        meta_idx_to_class = {x:y for y,x in enumerate(meta_classes)}
        meta_proposal_boxes = {idx:[x.gt_boxes for x in proposals] for idx, proposals in meta_targets.items()} 
        for idx, meta_feature in meta_features.items():
            meta_features_res5[idx] = self._roi_transform_meta(meta_feature, meta_proposal_boxes[idx])
            meta_features_mean[idx] = torch.mean(meta_features_res5[idx], dim=0)
            meta_features_covariance[idx] = torch.eye(meta_features_res5[idx].size(1), meta_features_res5[idx].size(1)).to(meta_features_res5[idx].device)
            meta_features_logdet[idx] = torch.zeros(1).to(meta_features_res5[idx].device)
        base_meta_mean = torch.cat([meta_features_mean[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_covariance = torch.cat([meta_features_covariance[idx].unsqueeze(0) for idx in meta_classes], dim=0)
        base_meta_logdet = torch.cat([meta_features_logdet[idx].unsqueeze(0) for idx in meta_classes], dim=0)

        base_meta_mean = self.sim_matrix(base_meta_mean)
        base_meta_mean = torch.div(base_meta_mean, torch.norm(base_meta_mean, dim=-1, keepdim=True).clamp(min=eps))

        meta_mean = torch.zeros(self.num_classes+1, base_meta_mean.size(1)).to(base_meta_mean.get_device())
        meta_covariance = torch.zeros(self.num_classes+1, base_meta_covariance.size(1), base_meta_covariance.size(1)).to(base_meta_covariance.get_device())
        meta_logdet = torch.zeros(self.num_classes+1, 1).to(base_meta_logdet.get_device()) - np.float("inf")

        meta_mean = meta_mean.index_copy(0, base_classes_tensor, base_meta_mean)
        meta_covariance = meta_covariance.index_copy(0, base_classes_tensor, base_meta_covariance)
        meta_logdet = meta_logdet.index_copy(0, base_classes_tensor, base_meta_logdet)

        return [meta_mean, meta_covariance, meta_logdet]

def build_visual_attention_head(cfg, input_shape):
    name = cfg.MODEL.ROI_HEADS.VISUAL_ATTENTION_HEAD.NAME
    return VISUAL_ATTENTION_HEAD_REGISTRY.get(name)(cfg, input_shape)