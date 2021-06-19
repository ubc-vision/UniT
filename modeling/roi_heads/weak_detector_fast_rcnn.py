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
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.utils.registry import Registry
from ..matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.events import get_event_storage
from .pcl_loss import PCLFunction

WEAK_DETECTOR_FAST_RCNN_REGISTRY = Registry("WEAK_DETECTOR_FAST_RCNN")

class FastRCNNOutputsRegression(FastRCNNOutputs):
    def __init__(self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, weights, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1"):
        super().__init__(box2box_transform=box2box_transform, pred_class_logits=pred_class_logits, pred_proposal_deltas=pred_proposal_deltas, proposals=proposals, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type)
        self.weights = weights

    def softmax_cross_entropy_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            loss = F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="none") * self.weights
            return loss.mean()
    
    def losses(self):
        return {"loss_regression_cls": self.softmax_cross_entropy_loss(), "loss_regression_bbox": self.box_reg_loss()}

@WEAK_DETECTOR_FAST_RCNN_REGISTRY.register()
class WeakDetectorOutputsBase(nn.Module):
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1", oicr_iter=3, fg_threshold=0.5, bg_threshold=0.1, freeze_layers=[], mil_multiplier=4.0, detector_temp=1.0, classifier_temp=1.0, regression_branch=False, proposal_matcher=None, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, weak_detector_type="OICR", num_kmeans_cluster=3, graph_iou_threshold=0.4, max_pc_num=5, oicr_regression_branch=False, only_base=False, base_classes=None, novel_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.oicr_iter = oicr_iter
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.box_dim = box_dim
        self.mil_multiplier = mil_multiplier
        self.detector_temp = detector_temp
        self.classifier_temp = classifier_temp
        self.regression_branch = regression_branch
        self.proposal_matcher = proposal_matcher
        self.num_bbox_reg_classes = num_bbox_reg_classes
        self.box2box_transform = box2box_transform
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.weak_detector_type = weak_detector_type
        self.num_kmeans_cluster = num_kmeans_cluster
        self.graph_iou_threshold = graph_iou_threshold
        self.max_pc_num = max_pc_num
        self.oicr_regression_branch = oicr_regression_branch
        self.only_base = only_base
        self.base_classes = torch.tensor(base_classes).long()
        self.novel_classes = torch.tensor(novel_classes).long()
        # Define delta predictors
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.input_size = input_size
        self.classifier_stream = Linear(input_size, self.num_classes)
        self.detection_stream = Linear(input_size, self.num_classes)
        nn.init.normal_(self.classifier_stream.weight, std=0.01)
        nn.init.normal_(self.detection_stream.weight, std=0.01)
        for l in [self.detection_stream, self.classifier_stream]:
            nn.init.constant_(l.bias, 0.)

        if self.oicr_iter > 0:
            self.oicr_predictors = nn.ModuleList([Linear(input_size, self.num_classes + 1) for _ in range(self.oicr_iter)])
            for oicr_iter in range(self.oicr_iter):
                nn.init.normal_(self.oicr_predictors[oicr_iter].weight, std=0.01)
                nn.init.constant_(self.oicr_predictors[oicr_iter].bias, 0.)
            if self.oicr_regression_branch:
                self.oicr_predictors_regressor = nn.ModuleList([Linear(self.input_size, self.num_bbox_reg_classes * self.box_dim) for _ in range(self.oicr_iter)])
                for oicr_iter in range(self.oicr_iter):
                    nn.init.normal_(self.oicr_predictors_regressor[oicr_iter].weight, std=0.001)
                    nn.init.constant_(self.oicr_predictors_regressor[oicr_iter].bias, 0.)

        if self.regression_branch:
            self.regression_branch_cls = Linear(self.input_size, self.num_classes + 1)
            self.regression_branch_bbox = Linear(self.input_size, self.num_bbox_reg_classes * self.box_dim)
            nn.init.normal_(self.regression_branch_bbox.weight, std=0.001)
            nn.init.normal_(self.regression_branch_cls.weight, std=0.01)
            for l in [self.regression_branch_cls, self.regression_branch_bbox]:
                nn.init.constant_(l.bias, 0.)
        self._freeze_layers(layers=freeze_layers)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def move_mappings_to_gpu(self):
        if not self.novel_classes.is_cuda:
            device = next(self.classifier_stream.parameters()).device
            self.novel_classes = self.novel_classes.to(device)
            self.base_classes = self.base_classes.to(device)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
        "input_shape": input_shape,
        "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
        # fmt: off
        "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
        "oicr_iter"             : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.OICR_ITER,
        "fg_threshold"          : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.FG_THRESHOLD,
        "bg_threshold"          : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.BG_THRESHOLD,
        "freeze_layers"         : cfg.MODEL.FREEZE_LAYERS.FAST_RCNN,
        "mil_multiplier"        : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.MIL_MULTIPLIER,
        "detector_temp"         : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.DETECTOR_TEMP,
        "classifier_temp"       : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.CLASSIFIER_TEMP,
        "regression_branch"     : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.REGRESSION_BRANCH,
        "proposal_matcher"      : Matcher(cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS, cfg.MODEL.ROI_HEADS.IOU_LABELS, allow_low_quality_matches=False),
        "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
        "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
        "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
        "weak_detector_type"    : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.TYPE,
        'num_kmeans_cluster'    : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.NUM_KMEANS_CLUSTER,
        'graph_iou_threshold'   : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.GRAPH_IOU_THRESHOLD,
        'max_pc_num'            : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.MAX_PC_NUM,
        "oicr_regression_branch": cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.OICR_REGRESSION_BRANCH,
        "only_base" : cfg.DATASETS.BASE_MULTIPLIER == 0,
        'base_classes': cfg.DATASETS.FEWSHOT.BASE_CLASSES_ID,
        'novel_classes': cfg.DATASETS.FEWSHOT.NOVEL_CLASSES_ID
        # fmt: on
                }

    def forward(self, x_weak):
        self.move_mappings_to_gpu()
        if self.training:
            classifier_stream = self.classifier_stream(x_weak) / self.classifier_temp
            detection_stream = self.detection_stream(x_weak) / self.detector_temp
            oicr_scores = []
            oicr_bbox = []
            for idx in range(self.oicr_iter):
                oicr_scores.append(self.oicr_predictors[idx](x_weak))
                if self.oicr_regression_branch:
                    oicr_bbox.append(self.oicr_predictors_regressor[idx](x_weak))
            regression_cls, regression_bbox = None, None
            if self.regression_branch:
                regression_cls = self.regression_branch_cls(x_weak)
                regression_bbox = self.regression_branch_bbox(x_weak)
            return [classifier_stream, detection_stream, oicr_scores, oicr_bbox, regression_cls, regression_bbox], None
        else:
            return self.evaluation(x_weak)

    def evaluation(self, x_weak):
        self.move_mappings_to_gpu()
        if self.regression_branch:
            cls_output = self.regression_branch_cls(x_weak)
            bbox_output = self.regression_branch_bbox(x_weak)
        elif self.oicr_iter > 0:
            cls_output = []
            for idx in range(self.oicr_iter):
                cls_output.append(self.oicr_predictors[idx](x_weak))
            if self.oicr_regression_branch:
                bbox_output = []
                for idx in range(self.oicr_iter):
                    bbox_output.append(self.oicr_predictors_regressor[idx](x_weak))
            else:
                bbox_output = torch.zeros(x_weak.size(0), self.num_bbox_reg_classes * self.box_dim).to(x_weak.device)

        else:
            cls_output = self.classifier_stream(x_weak) / self.classifier_temp
            bbox_output = torch.zeros(x_weak.size(0), self.num_bbox_reg_classes * self.box_dim).to(x_weak.device)
        
        return [cls_output, bbox_output], None

    def losses(self, weak_predictions, weak_proposals, weak_targets):
        final_losses = {}
        weak_classification_stream, weak_detection_stream, oicr_scores, oicr_bbox, regression_cls, regression_bbox = weak_predictions
        len_perms = [len(proposal) for proposal in weak_proposals]
        indices = np.insert(np.cumsum(len_perms), 0, 0)
        mil_scores = []
        class_vectors = []
        gt_classes = []
        # if self.only_base:
        #     weak_classification_stream = weak_classification_stream.index_fill(1, self.novel_classes, -np.float("inf"))
        #     weak_detection_stream = weak_detection_stream.index_fill(1, self.novel_classes, -np.float("inf"))
        #     for idx in range(len(oicr_scores)):
        #         oicr_scores[idx] = oicr_scores[idx].index_fill(1, self.novel_classes, -np.float("inf"))
        for img, gt_class in enumerate(weak_targets):
            unique_gt_class = torch.unique(gt_class)
            curr_img_logits = weak_classification_stream.narrow(0, start=int(indices[img]), length=int(indices[img+1] - indices[img]))
            curr_det_logits = weak_detection_stream.narrow(0, start=int(indices[img]), length=int(indices[img+1] - indices[img]))
            x_r = torch.softmax(curr_img_logits, -1) * torch.softmax(curr_det_logits, 0)
            mil_scores.append(x_r)
            class_vectors.append(torch.sum(x_r, dim=0))
            gt_classes.append(unique_gt_class)
        class_vectors = torch.stack(class_vectors)
        gt_vector = torch.zeros_like(class_vectors)
        for index, gt_class in enumerate(gt_classes):
            gt_vector[index, gt_class] = 1.
        final_losses['loss_im_cls'] = self.cross_entropy_loss(class_vectors, gt_vector) * self.mil_multiplier
        mil_scores = torch.cat(mil_scores, 0).clone().detach()
        
        # OICR Loss
        return_proposals = True if self.oicr_regression_branch else False
        for idx, oicr_score in enumerate(oicr_scores):
            with torch.no_grad():
                if idx == 0:
                    if self.weak_detector_type == "OICR":
                        oicr_loss_inputs = self.compute_loss_inputs(weak_proposals, mil_scores, gt_classes, None, indices, return_proposals=return_proposals)
                    else:
                        oicr_loss_inputs = self.compute_pcl_loss_inputs(weak_proposals, mil_scores, gt_classes, torch.softmax(oicr_scores[idx].clone().detach(), dim=-1), indices, return_proposals=return_proposals)
                else:
                    if self.weak_detector_type == "OICR":
                        oicr_loss_inputs = self.compute_loss_inputs(weak_proposals, torch.softmax(oicr_scores[idx - 1].clone().detach(), dim=-1), gt_classes, None, indices, return_proposals=return_proposals)
                    else:
                        oicr_loss_inputs = self.compute_pcl_loss_inputs(weak_proposals, torch.softmax(oicr_scores[idx - 1].clone().detach(), dim=-1), gt_classes, torch.softmax(oicr_scores[idx].clone().detach(), dim=-1), indices, return_proposals=return_proposals)
            if not self.oicr_regression_branch:
                if self.weak_detector_type == "OICR":
                    final_losses['loss_oicr_{}'.format(idx + 1)] = self.weighted_softmax_with_loss(oicr_scores[idx], oicr_loss_inputs['labels'], oicr_loss_inputs['cls_weights'])
                else:
                    final_losses['loss_oicr_{}'.format(idx + 1)] = 0.0
                    for img_idx in range(len(weak_targets)):
                        final_losses['loss_oicr_{}'.format(idx + 1)] = final_losses['loss_oicr_{}'.format(idx + 1)] + PCLFunction.apply(torch.softmax(oicr_score.narrow(0, start=int(indices[img_idx]), length=int(indices[img_idx+1] - indices[img_idx])), -1), *[oicr_loss_inputs[key][img_idx] for key in ['labels', 'cls_weights', 'gt_assignment', 'pc_labels', 'pc_probs', 'pc_count','img_cls_weights', 'im_labels']])     
                    final_losses['loss_oicr_{}'.format(idx + 1)] = final_losses['loss_oicr_{}'.format(idx + 1)] / len(weak_targets)
            else:
                assert self.weak_detector_type == "OICR"
                oicr_losses = FastRCNNOutputsRegression(self.box2box_transform, oicr_scores[idx], oicr_bbox[idx], oicr_loss_inputs['proposals'], oicr_loss_inputs['cls_weights'], self.smooth_l1_beta, self.box_reg_loss_type,).losses()
                final_losses['loss_oicr_{}'.format(idx + 1)] = oicr_losses['loss_regression_cls']
                final_losses['loss_oicr_bbox_{}'.format(idx + 1)] = oicr_losses['loss_regression_bbox']

        # Regression Branch Loss
        if self.regression_branch:
            with torch.no_grad():
                oicr_mean_scores = torch.mean(torch.stack([torch.softmax(oicr_scores[idx].clone().detach(), -1) for idx in range(len(oicr_scores))], 0), 0)
                if self.weak_detector_type == "OICR":
                    regression_branch_loss_inputs = self.compute_loss_inputs(weak_proposals, oicr_mean_scores, gt_classes, None, indices, return_proposals=True)
                else:
                    regression_branch_loss_inputs = self.compute_pcl_loss_inputs(weak_proposals, oicr_mean_scores, gt_classes, torch.softmax(regression_cls.clone().detach(), -1), indices, return_proposals=True)
            regression_losses = FastRCNNOutputsRegression(self.box2box_transform, regression_cls, regression_bbox, regression_branch_loss_inputs['proposals'], regression_branch_loss_inputs['cls_weights'], self.smooth_l1_beta, self.box_reg_loss_type,).losses()
            final_losses.update(regression_losses)
        return final_losses 

    def cross_entropy_loss(self, cls_score, labels, eps=1e-6):
        cls_score = cls_score.clamp(eps, 1 - eps)
        loss = nn.functional.binary_cross_entropy(cls_score, labels)
        return loss

    def weighted_softmax_with_loss(self, score, labels, weights, eps=1e-9):
        if score.numel() > 0:
            loss = nn.functional.cross_entropy(score, labels, reduction='none') * weights
            return loss.mean()
        else:
            loss = 0.0 * score.sum()
            return loss

    def predict_boxes(self, predictions, proposals):
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        scores, _ = predictions
        if self.oicr_iter > 0 and not self.regression_branch:
            scores = torch.sum(torch.softmax(torch.stack(scores, 0), -1), 0)
        else:
            scores = torch.softmax(scores, -1)
        num_inst_per_image = [len(p) for p in proposals]
        return scores.split(num_inst_per_image, dim=0)

    def inference(self, predictions, proposals, tta=False):
        scores = self.predict_probs(predictions, proposals)
        if tta:
            if self.oicr_regression_branch:
                regression_output = torch.mean(torch.stack(predictions[1], 0), 0)
                return [torch.cat(scores, 0), regression_output], None
            else:
                return [torch.cat(scores, 0), predictions[1]], None
        boxes = self.predict_boxes(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
    
    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.size(0)).to(gt_classes.device)
        return sampled_idxs, gt_classes[sampled_idxs]

    def label_and_sample_proposals(self, proposals, targets, return_match_vals=False):
        gt_boxes = [x.gt_boxes for x in targets]
        proposals_with_gt = []
        weight_assignments = []
        iou_values = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels, matched_vals = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)
            if return_match_vals:
                iou_values.append(matched_vals)
            weight_assignments.append(matched_idxs[sampled_idxs])
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            proposals_with_gt.append(proposals_per_image)
        if not return_match_vals:
            return proposals_with_gt, weight_assignments
        else:
            return proposals_with_gt, weight_assignments, iou_values
    
    def get_proposal_clusters(self, box, proposals, label, cls_prob):
        gt_boxes = []
        gt_classes = []
        gt_scores = []
        if cls_prob.numel() > 0:
            for idx, gt_class in enumerate(label):
                curr_cls_prob = cls_prob.index_select(1, index=gt_class).clone().detach()
                max_gt_score, max_index = curr_cls_prob.max(dim=0)
                gt_boxes.append(box[max_index])
                gt_classes.append(gt_class)
                gt_scores.append(max_gt_score)
                cls_prob[max_index, :] = 0.0
            gt_classes = torch.stack(gt_classes)
            gt_scores = torch.cat(gt_scores)
            new_instance = Instances(box.image_size)
            new_instance.gt_boxes = copy.deepcopy(Boxes.cat([x.proposal_boxes for x in gt_boxes]))
            new_instance.gt_classes = label.clone().detach()
        else:
            new_instance = Instances(box.image_size)
            new_instance.gt_boxes = Boxes(torch.zeros(0, self.box_dim)).to(cls_prob.device)
            new_instance.gt_classes = torch.zeros(0).to(cls_prob.device)
            gt_scores = torch.zeros(0).to(cls_prob.device)

        return new_instance, gt_scores

    def compute_loss_inputs(self, boxes, cls_probs, im_labels, cls_probs_next_iter, indices, return_proposals=False):
        img_instances = []
        img_gt_scores = []
        for idx, label in enumerate(im_labels):
            box = boxes[idx]
            cls_prob = cls_probs.narrow(0, start=int(indices[idx]), length=int(indices[idx+1] - indices[idx]))
            img_instance, img_gt_score = self.get_proposal_clusters(copy.deepcopy(box), None, label.clone().detach(), cls_prob.clone().detach())
            img_instances.append(img_instance)
            img_gt_scores.append(img_gt_score)
        new_proposals, weight_assignments, iou_values = self.label_and_sample_proposals(boxes, img_instances, return_match_vals=True)
        if not return_proposals:
            loss_inputs = {'labels' : [], 'cls_weights' : []}
            loss_inputs['labels'] = cat([p.gt_classes for p in new_proposals], dim=0)
            loss_inputs['cls_weights'] = [torch.index_select(img_gt_scores[x], 0, weight_assignments[x]) for x in range(len(im_labels))]
            if self.bg_threshold > 0.0:
                ig_inds = [torch.where(x < self.bg_threshold)[0] for x in iou_values]
                for idx, cls_weight in enumerate(loss_inputs['cls_weights']):
                    if cls_weight.numel() > 0:
                        loss_inputs['cls_weights'][idx][ig_inds[idx]] = 0.0 
            loss_inputs['cls_weights'] = cat(loss_inputs['cls_weights'])
        else:
            loss_inputs = {'proposals' : [], 'cls_weights' : []}
            loss_inputs['proposals'] = new_proposals
            loss_inputs['cls_weights'] = [torch.index_select(img_gt_scores[x], 0, weight_assignments[x]) for x in range(len(im_labels))]
            if self.bg_threshold > 0.0:
                ig_inds = [torch.where(x < self.bg_threshold)[0] for x in iou_values]
                for idx, cls_weight in enumerate(loss_inputs['cls_weights']):
                    if cls_weight.numel() > 0:
                        loss_inputs['cls_weights'][idx][ig_inds[idx]] = 0.0 
            loss_inputs['cls_weights'] = cat(loss_inputs['cls_weights'])
        return loss_inputs

    def build_graph(self, box):
        overlaps = pairwise_iou(box.proposal_boxes, box.proposal_boxes)
        overlaps_mask = overlaps > self.graph_iou_threshold
        return overlaps_mask.float()

    def get_graph_centers(self, box, cls_prob, label):
        gt_boxes = []
        gt_classes = []
        gt_scores = []
        for idx, gt_class in enumerate(label):
            curr_cls_prob = cls_prob.index_select(1, index=gt_class)
            non_zero_idxs = torch.where(curr_cls_prob >= 0)[0]
            top_ranking_idxs = self.get_top_ranking_proposals(curr_cls_prob[non_zero_idxs])
            non_zero_idxs = non_zero_idxs[top_ranking_idxs]
            curr_box = box[non_zero_idxs]
            curr_cls_prob = curr_cls_prob[non_zero_idxs]

            graph = self.build_graph(curr_box)
            count = curr_cls_prob.size(0)
            keep_idxs = []
            curr_gt_scores = []
            while True:
                order = torch.sum(graph, 1).argsort(descending=True)
                keep_idxs.append(order[0])

                graph_idx = torch.where(graph[order[0], :] > 0)[0]
                curr_gt_scores.append(torch.max(curr_cls_prob[graph_idx]))

                graph[:, graph_idx] = 0
                graph[graph_idx, :] = 0
                count = count - len(graph_idx)
                if count <= 5:
                    break
            keep_idxs = torch.stack(keep_idxs, 0)
            curr_gt_scores = torch.stack(curr_gt_scores, 0)
            curr_gt_boxes = curr_box[keep_idxs]
            
            keep_idxs_selected = curr_gt_scores.argsort().flip([0])[: min(len(curr_gt_scores), self.max_pc_num)].clone().detach()
            gt_boxes.append(curr_gt_boxes[keep_idxs_selected])
            gt_scores.append(curr_gt_scores[keep_idxs_selected])
            gt_classes.append((torch.zeros_like(keep_idxs_selected) + gt_class).long())

            # Delete selected proposals
            ids_to_remove = non_zero_idxs[keep_idxs][keep_idxs_selected]
            indexer = torch.ones(cls_prob.size(0)).to(cls_prob.device) 
            indexer[ids_to_remove] = 0.
            indexer_mask = indexer == 1.
            cls_prob = cls_prob.clone().detach()[indexer_mask]
            box = copy.deepcopy(box)[indexer_mask]
        new_instance = Instances(box.image_size)
        new_instance.gt_boxes = copy.deepcopy(Boxes.cat([x.proposal_boxes for x in gt_boxes]))
        new_instance.gt_classes = torch.cat(gt_classes)
        gt_scores = torch.cat(gt_scores)
        return new_instance, gt_scores

    def get_top_ranking_proposals(self, probs, rng_seed=3):
        if probs.size(0) < self.num_kmeans_cluster:
            index = torch.argmax(probs, 0)
        else:
            kmeans = KMeans(n_clusters=self.num_kmeans_cluster, random_state=rng_seed).fit(probs.data.cpu().numpy())
            high_score_label = np.argmax(kmeans.cluster_centers_)
            index = torch.from_numpy(np.where(kmeans.labels_ == high_score_label)[0]).to(probs.device)
            if len(index) == 0:
                index = torch.argmax(probs).unsqueeze(0)
        return index

    def compute_pcl_loss_inputs(self, boxes, cls_probs, im_labels, cls_probs_next_iter, indices, return_proposals=False, eps=1e-9):
        img_instances = []
        img_gt_scores = []
        cls_probs = cls_probs.clamp(eps, 1 - eps)
        cls_probs_next_iter = cls_probs_next_iter.clamp(eps, 1 - eps)
        for idx, label in enumerate(im_labels):
            box = boxes[idx]
            cls_prob = cls_probs.narrow(0, start=int(indices[idx]), length=int(indices[idx+1] - indices[idx]))
            img_instance, img_gt_score = self.get_graph_centers(copy.deepcopy(box), cls_prob.clone().detach(), label.clone().detach())
            img_instances.append(img_instance)
            img_gt_scores.append(img_gt_score)
        new_proposals, weight_assignments, iou_values = self.label_and_sample_proposals(boxes, img_instances, return_match_vals=True)
        if not return_proposals:
            loss_inputs = {'labels':[], 'cls_weights':[], 'img_cls_weights':[], 'pc_labels':[], 'pc_count':[], 'pc_probs':[], 'gt_assignment':[], 'im_labels':[]}
            loss_inputs['labels'] = [p.gt_classes for p in new_proposals]
            bg_inds = [torch.where(x < self.fg_threshold)[0] for x in iou_values]
            ig_inds = [torch.where(x < self.bg_threshold)[0] for x in iou_values]
            cls_weights = [torch.index_select(img_gt_scores[x], 0, weight_assignments[x]) for x in range(len(im_labels))]
            for idx, cls_weight in enumerate(cls_weights):
                cls_weights[idx][ig_inds[idx]] = 0.0 
            for idx, weight_assignment in enumerate(weight_assignments):
                weight_assignments[idx][bg_inds[idx]] = -1
            loss_inputs['cls_weights'] = cls_weights
            loss_inputs['gt_assignment'] = weight_assignments
            loss_inputs['im_labels'] = [torch.cat([x, torch.tensor([self.num_classes]).long().to(cls_probs.device)], 0) for x in im_labels]
            for idx, label in enumerate(im_labels):
                po_index = (weight_assignments[idx][None,:] == torch.arange(len(img_instances[idx].gt_boxes)).to(label.device)[:,None]).clone().detach()
                loss_inputs['img_cls_weights'].append((po_index * cls_weights[idx].unsqueeze(0)).sum(-1).clone().detach())
                loss_inputs['pc_labels'].append(img_instances[idx].gt_classes.clone().detach())
                loss_inputs['pc_count'].append(po_index.sum(-1).clone().detach())
                curr_cls_probs_next_iter = cls_probs_next_iter.narrow(0, start=int(indices[idx]), length=int(indices[idx+1] - indices[idx]))
                loss_inputs['pc_probs'].append(torch.sum((curr_cls_probs_next_iter.index_select(1, loss_inputs['pc_labels'][idx]) * po_index.transpose(0,1)), 0) / loss_inputs['pc_count'][idx])   
        else:
            loss_inputs = {'proposals' : [], 'cls_weights' : []}
            loss_inputs['proposals'] = new_proposals
            bg_inds = [torch.where(x < self.fg_threshold)[0] for x in iou_values]
            ig_inds = [torch.where(x < self.bg_threshold)[0] for x in iou_values]
            cls_weights = [torch.index_select(img_gt_scores[x], 0, weight_assignments[x]) for x in range(len(im_labels))]
            for idx, cls_weight in enumerate(cls_weights):
                cls_weights[idx][ig_inds[idx]] = 0.0 
            for idx, weight_assignment in enumerate(weight_assignments):
                weight_assignments[idx][bg_inds[idx]] = -1
            loss_inputs['cls_weights'] = torch.cat(cls_weights)
        return loss_inputs


@WEAK_DETECTOR_FAST_RCNN_REGISTRY.register()
class WeakDetectorOutputsFT(WeakDetectorOutputsBase):
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1", oicr_iter=3, fg_threshold=0.5, bg_threshold=0.1, freeze_layers=[], mil_multiplier=4.0, detector_temp=1.0, classifier_temp=1.0, regression_branch=False, proposal_matcher=None, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, weak_detector_type="OICR", num_kmeans_cluster=3, graph_iou_threshold=0.4, max_pc_num=5, oicr_regression_branch=False, only_base=False, base_classes=None, novel_classes=None):
        super().__init__(input_shape, box2box_transform=box2box_transform, num_classes=num_classes, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, oicr_iter=oicr_iter, fg_threshold=fg_threshold, bg_threshold=bg_threshold, freeze_layers=freeze_layers, mil_multiplier=mil_multiplier, detector_temp=detector_temp, classifier_temp=classifier_temp, regression_branch=regression_branch, proposal_matcher=proposal_matcher, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, weak_detector_type=weak_detector_type, num_kmeans_cluster=num_kmeans_cluster, graph_iou_threshold=graph_iou_threshold, max_pc_num=max_pc_num, oicr_regression_branch=oicr_regression_branch, only_base=only_base, base_classes=base_classes, novel_classes=novel_classes)
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.input_size = input_size
        self.classifier_stream_delta = Linear(input_size, self.num_classes)
        self.detection_stream_delta = Linear(input_size, self.num_classes)
        for l in [self.detection_stream, self.classifier_stream]:
            nn.init.constant_(l.bias, 0.)
            nn.init.constant_(l.weight, 0.)

        if self.oicr_iter > 0:
            self.oicr_predictors_delta = nn.ModuleList([Linear(input_size, self.num_classes + 1) for _ in range(self.oicr_iter)])
            for oicr_iter in range(self.oicr_iter):
                nn.init.constant_(self.oicr_predictors[oicr_iter].weight, 0.)
                nn.init.constant_(self.oicr_predictors[oicr_iter].bias, 0.)
    
    def move_mappings_to_gpu(self):
        if not self.novel_classes.is_cuda:
            device = next(self.classifier_stream.parameters()).device
            self.novel_classes = self.novel_classes.to(device)
            self.base_classes = self.base_classes.to(device)


    def forward(self, x_weak):
        self.move_mappings_to_gpu()
        if self.training:
            classifier_stream = self.classifier_stream(x_weak) / self.classifier_temp
            detection_stream = self.detection_stream(x_weak) / self.detector_temp
            classifier_stream_delta = self.classifier_stream_delta(x_weak) / self.classifier_temp
            detection_stream_delta = self.detection_stream_delta(x_weak) / self.detector_temp
            classifier_stream = classifier_stream_delta + classifier_stream.index_fill(1, self.novel_classes, 0.0)
            detection_stream = detection_stream_delta + detection_stream.index_fill(1, self.novel_classes, 0.0)
            oicr_scores = []
            oicr_bbox = []
            for idx in range(self.oicr_iter):
                oicr_scores.append(self.oicr_predictors[idx](x_weak).index_fill(1, self.novel_classes, 0.0) + self.oicr_predictors_delta[idx](x_weak))
                if self.oicr_regression_branch:
                    oicr_bbox.append(self.oicr_predictors_regressor[idx](x_weak))
            regression_cls, regression_bbox = None, None
            if self.regression_branch:
                regression_cls = self.regression_branch_cls(x_weak)
                regression_bbox = self.regression_branch_bbox(x_weak)

            return [classifier_stream, detection_stream, oicr_scores, oicr_bbox, regression_cls, regression_bbox], None
        else:
            return self.evaluation(x_weak)

    def evaluation(self, x_weak):
        self.move_mappings_to_gpu()
        if self.regression_branch:
            cls_output = self.regression_branch_cls(x_weak)
            bbox_output = self.regression_branch_bbox(x_weak)
        elif self.oicr_iter > 0:
            cls_output = []
            for idx in range(self.oicr_iter):
                cls_output.append(self.oicr_predictors[idx](x_weak).index_fill(1, self.novel_classes, 0.0) + self.oicr_predictors_delta[idx](x_weak))
            if self.oicr_regression_branch:
                bbox_output = []
                for idx in range(self.oicr_iter):
                    bbox_output.append(self.oicr_predictors_regressor[idx](x_weak))
            else:
                bbox_output = torch.zeros(x_weak.size(0), self.num_bbox_reg_classes * self.box_dim).to(x_weak.device)

        else:
            cls_output = self.classifier_stream(x_weak) / self.classifier_temp
            bbox_output = torch.zeros(x_weak.size(0), self.num_bbox_reg_classes * self.box_dim).to(x_weak.device)
        
        return [cls_output, bbox_output], None

        