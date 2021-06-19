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
from detectron2.modeling.matcher import  Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.events import get_event_storage
from .weak_detector_fast_rcnn import WEAK_DETECTOR_FAST_RCNN_REGISTRY, WeakDetectorOutputsBase
from fvcore.nn import giou_loss, smooth_l1_loss

FAST_RCNN_REGISTRY = Registry("FAST_RCNN_REGISTRY")

class FastRCNNOutputsReduction(FastRCNNOutputs):
    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="none")

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="none",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="none",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg  

class FastRCNNOutputsNLL(FastRCNNOutputs):
    def softmax_cross_entropy_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.nll_loss(self.pred_class_logits, self.gt_classes, reduction="mean")

    def predict_probs(self):
        """
        Deprecated
        """
        return probs.split(self.num_preds_per_image, dim=0)

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

class FastRCNNOutputsBase(FastRCNNOutputLayers):
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0,box_reg_loss_type="smooth_l1", loss_weight=1.0, oicr_iter=3, fg_threshold=0.5, bg_threshold=0.1, freeze_layers=[], embedding_path='', terms={}, mode='Pre_Softmax', mil_multiplier=4.0, detector_temp=1.0, classifier_temp=1.0):
        super(FastRCNNOutputsBase, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)
        self.num_classes = num_classes
        self.oicr_iter = oicr_iter
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.terms = terms
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.box_dim = box_dim
        self.num_bbox_reg_classes = num_bbox_reg_classes
        self.mode = mode
        self.mil_multiplier = mil_multiplier
        self.detector_temp = detector_temp
        self.classifier_temp = classifier_temp
        # Delete instances defined by super
        del self.cls_score
        del self.bbox_pred

        # Define delta predictors
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.input_size = input_size
        self.classifier_stream = Linear(input_size, self.num_classes)
        self.detection_stream = Linear(input_size, self.num_classes)
        self.oicr_predictors = nn.ModuleList([Linear(input_size, self.num_classes + 1) for _ in range(self.oicr_iter)])
        self.cls_score_delta = Linear(input_size, self.num_classes + 1)
        self.bbox_pred_delta = Linear(input_size, num_bbox_reg_classes * box_dim)

        # Init Predictors
        nn.init.normal_(self.bbox_pred_delta.weight, std=0.001)
        nn.init.normal_(self.classifier_stream.weight, std=0.01)
        nn.init.normal_(self.detection_stream.weight, std=0.01)
        for oicr_iter in range(self.oicr_iter):
            nn.init.normal_(self.oicr_predictors[oicr_iter].weight, std=0.01)
            nn.init.constant_(self.oicr_predictors[oicr_iter].bias, 0.)
        nn.init.constant_(self.cls_score_delta.weight, 0.)
        # nn.init.constant_(self.bbox_pred_delta.weight, 0.)
        for l in [self.cls_score_delta, self.bbox_pred_delta, self.detection_stream, self.classifier_stream]:
            nn.init.constant_(l.bias, 0.)
        
        pretrained_embeddings = torch.load(embedding_path)['embeddings']
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self._freeze_layers(layers=freeze_layers)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
        "input_shape": input_shape,
        "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
        # fmt: off
        "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
        "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
        "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
        "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
        "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
        "oicr_iter"             : cfg.MODEL.ROI_HEADS.OICR_ITER,
        "fg_threshold"          : cfg.MODEL.ROI_HEADS.FG_THRESHOLD,
        "bg_threshold"          : cfg.MODEL.ROI_HEADS.BG_THRESHOLD,
        "freeze_layers"         : cfg.MODEL.FREEZE_LAYERS.FAST_RCNN,
        "embedding_path"        : cfg.MODEL.ROI_HEADS.EMBEDDING_PATH,
        "terms"                 : {'cls': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.CLASSIFIER, 'bbox': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.BBOX, 'seg': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.MASK},
        "mode"                  : cfg.MODEL.ROI_HEADS.FAST_RCNN.MODE,
        "mil_multiplier"        : cfg.MODEL.ROI_HEADS.FAST_RCNN.MIL_MULTIPLIER,
        "detector_temp"         : cfg.MODEL.ROI_HEADS.FAST_RCNN.DETECTOR_TEMP,
        "classifier_temp"       : cfg.MODEL.ROI_HEADS.FAST_RCNN.CLASSIFIER_TEMP

        # fmt: on
                }

    def cross_entropy_loss(self, cls_score, labels, eps=1e-6):
        cls_score = cls_score.clamp(eps, 1 - eps)
        loss = nn.functional.binary_cross_entropy(cls_score, labels)
        return loss

    def weighted_softmax_with_loss(self, score, labels, weights, eps=1e-9):
        loss = nn.functional.cross_entropy(score, labels, reduction='none') * weights
        # valid_sum = weights.gt(eps).float().sum()
        # if valid_sum < eps:
        #     return loss.sum() / loss.numel() 
        # else:
        #     return loss.sum() / valid_sum
        return loss.mean()

    def compute_loss_inputs(self, boxes, cls_probs, im_labels, cls_probs_next_iter, indices, eps=1e-9):
        raise NotImplementedError

    def losses(self, predictions, proposals, weak_predictions=None, weak_proposals=None, weak_targets=None, similarity_matrices={}):
        raise NotImplementedError

    def get_similarity(self, base_classes, novel_classes, indexer):
        label_embeddings = self.embeddings(indexer)
        base_label_embeddings = label_embeddings.index_select(dim=0, index=base_classes)
        novel_label_embeddings = label_embeddings.index_select(dim=0, index=novel_classes)
        similarity = torch.mm(novel_label_embeddings, base_label_embeddings.transpose(0,1))

        return similarity

    def forward(self, x, supervised_branch_x_weak, novel_classes, base_classes, x_weak=None, similarity=None):
        # Supervised Detection Branch
        delta_scores = self.cls_score_delta(x)
        proposal_deltas = self.bbox_pred_delta(x)
        with torch.no_grad():
            weak_scores = torch.mean(torch.stack([self.oicr_predictors[oicr_iter](supervised_branch_x_weak) for oicr_iter in range(self.oicr_iter)]), dim=0)
        if self.training:
            weak_scores = weak_scores.index_fill(1, novel_classes, -np.float("inf"))
        else:
            # Transfer from base to novel
            if similarity is not None:
                transfered_delta_scores = torch.zeros_like(delta_scores)
                base_delta_scores = delta_scores.index_select(1, index=base_classes)
                if len(similarity['cls'].size()) > 2:
                    cls_base_to_novel_transfer = torch.bmm(similarity['cls'], base_delta_scores.unsqueeze(2)).squeeze(2)
                else:
                    cls_base_to_novel_transfer = torch.mm(base_delta_scores, similarity['cls'].transpose(0,1))
                transfered_delta_scores = transfered_delta_scores.index_copy(1, novel_classes, cls_base_to_novel_transfer)
                delta_scores = delta_scores + transfered_delta_scores
                
                proposal_deltas_reshaped = proposal_deltas.view(-1, self.num_classes, self.box_dim)
                base_proposal_deltas = proposal_deltas_reshaped.index_select(1, index=base_classes)
                transferred_proposal_deltas = torch.zeros_like(proposal_deltas_reshaped)
                if len(similarity['bbox'].size()) > 2:
                    bbox_base_to_novel_transfer = torch.bmm(similarity['bbox'], base_proposal_deltas)
                else:
                    bbox_base_to_novel_transfer = torch.matmul(base_proposal_deltas.transpose(1,2), similarity['bbox'].transpose(0,1)).transpose(1,2)
                transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, novel_classes, bbox_base_to_novel_transfer)
                transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, base_classes, base_proposal_deltas)
                proposal_deltas = transferred_proposal_deltas.view(-1, self.num_classes * self.box_dim)
                
        scores = weak_scores + delta_scores

        # Weak Detection Branch
        weak_branch_return = []
        if x_weak is not None:
            classifier_stream = self.classifier_stream(x_weak)
            detection_stream = self.detection_stream(x_weak)
            oicr_scores = []
            for idx in range(self.oicr_iter):
                oicr_scores.append(self.oicr_predictors[idx](x_weak))
            weak_branch_return = [classifier_stream, detection_stream, oicr_scores]
        return [scores, proposal_deltas], weak_branch_return

@FAST_RCNN_REGISTRY.register()
class WeakDetectorOutputsBaseWrapper(WeakDetectorOutputsBase):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

@FAST_RCNN_REGISTRY.register()
class SupervisedDetectorOutputsBase(FastRCNNOutputLayers):
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1", loss_weight=1.0, weak_detector_head=None, regression_branch=False, terms={}, freeze_layers=[], embedding_path=''):
        super(SupervisedDetectorOutputsBase, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight)
        self.num_classes = num_classes
        self.terms = terms
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.box_dim = box_dim
        self.num_bbox_reg_classes = num_bbox_reg_classes
        self.weak_detector_head = weak_detector_head
        self.regression_branch = regression_branch

        # Delete instances defined by super
        del self.cls_score
        del self.bbox_pred

        # Define delta predictors
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.input_size = input_size
        self.cls_score_delta = Linear(input_size, self.num_classes + 1)
        self.bbox_pred_delta = Linear(input_size, num_bbox_reg_classes * box_dim)

        # Init Predictors
        nn.init.constant_(self.cls_score_delta.weight, 0.)
        if not self.regression_branch:
            nn.init.normal_(self.bbox_pred_delta.weight, std=0.001)
        else:
            nn.init.constant_(self.bbox_pred_delta.weight, 0.)
        for l in [self.cls_score_delta, self.bbox_pred_delta]:
            nn.init.constant_(l.bias, 0.)
        
        pretrained_embeddings = torch.load(embedding_path)['embeddings']
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self._freeze_layers(layers=freeze_layers)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            "weak_detector_head"    : WEAK_DETECTOR_FAST_RCNN_REGISTRY.get(cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.NAME)(cfg, input_shape),
            "regression_branch"     : cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.REGRESSION_BRANCH,
            "terms"                 : {'cls': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.CLASSIFIER, 'bbox': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.BBOX, 'seg': cfg.MODEL.ROI_HEADS.FINETUNE_TERMS.MASK},
            "embedding_path"        : cfg.MODEL.ROI_HEADS.EMBEDDING_PATH,
            "freeze_layers"         : cfg.MODEL.FREEZE_LAYERS.FAST_RCNN
            # fmt: on
        }
    
    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def get_cls_logits(self, x, x_weak=None):
        if x_weak is None:
            return x
        if self.weak_detector_head.regression_branch:
            return (x + x_weak)
        elif self.weak_detector_head.oicr_iter > 0:
            return (x + torch.mean(torch.stack(x_weak, 0), 0))
        else:
            return (x + nn.functional.pad(x_weak, (0,1), 'constant', 0.0))

    def get_cls_bbox(self, x, x_weak=None):
        if x_weak is None:
            return x
        else:
            return (x + x_weak)

    def get_similarity(self, base_classes, novel_classes, indexer):
        label_embeddings = self.embeddings(indexer)
        base_label_embeddings = label_embeddings.index_select(dim=0, index=base_classes)
        novel_label_embeddings = label_embeddings.index_select(dim=0, index=novel_classes)
        similarity = torch.mm(novel_label_embeddings, base_label_embeddings.transpose(0,1))

        return similarity

    def forward(self, x, novel_classes, base_classes, supervised_branch_x_weak=None, x_weak=None, similarity=None):
        if x is not None:
            delta_scores = self.cls_score_delta(x)
            proposal_deltas = self.bbox_pred_delta(x)
            with torch.no_grad():
                if supervised_branch_x_weak is None:
                    [weak_scores, weak_proposal_deltas], _ = self.weak_detector_head.evaluation(x)
                else:
                    [weak_scores, weak_proposal_deltas], _ = self.weak_detector_head.evaluation(supervised_branch_x_weak)
        else:
            delta_scores = torch.zeros(x_weak.size(0), self.num_classes + 1).to(x_weak.device)
            proposal_deltas = torch.zeros(x_weak.size(0), self.num_classes * self.box_dim).to(x_weak.device)
            weak_scores = None
            weak_proposal_deltas = None

        # delta_scores = delta_scores * 0.0
        # proposal_deltas = proposal_deltas * 0.0
        if not self.training:
        # if False:
            # Transfer from base to novel
            if similarity is not None:
                transfered_delta_scores = torch.zeros_like(delta_scores)
                base_delta_scores = delta_scores.index_select(1, index=base_classes)
                if len(similarity['cls'].size()) > 2:
                    cls_base_to_novel_transfer = torch.bmm(similarity['cls'], base_delta_scores.unsqueeze(2)).squeeze(2)
                else:
                    cls_base_to_novel_transfer = torch.mm(base_delta_scores, similarity['cls'].transpose(0,1))
                transfered_delta_scores = transfered_delta_scores.index_copy(1, novel_classes, cls_base_to_novel_transfer)
                delta_scores = delta_scores + transfered_delta_scores
                
                proposal_deltas_reshaped = proposal_deltas.view(-1, self.num_classes, self.box_dim)
                base_proposal_deltas = proposal_deltas_reshaped.index_select(1, index=base_classes)
                transferred_proposal_deltas = torch.zeros_like(proposal_deltas_reshaped)
                if len(similarity['bbox'].size()) > 2:
                    bbox_base_to_novel_transfer = torch.bmm(similarity['bbox'], base_proposal_deltas)
                else:
                    bbox_base_to_novel_transfer = torch.matmul(base_proposal_deltas.transpose(1,2), similarity['bbox'].transpose(0,1)).transpose(1,2)
                transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, novel_classes, bbox_base_to_novel_transfer)
                transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, base_classes, base_proposal_deltas)
                proposal_deltas = transferred_proposal_deltas.view(-1, self.num_classes * self.box_dim)

        scores = self.get_cls_logits(x=delta_scores, x_weak=weak_scores)
        bbox = self.get_cls_bbox(x=proposal_deltas, x_weak=weak_proposal_deltas)
        if self.training:
            scores = scores.index_fill(1, novel_classes, -np.float("inf"))

        weak_branch_return = None
        if x_weak is not None:
            weak_branch_return, _ = self.weak_detector_head(x_weak)     
        return [scores, bbox], weak_branch_return

    def losses(self, predictions, proposals, weak_predictions=None, weak_proposals=None, weak_targets=None, train_only_weak=False):
        if not train_only_weak:
            scores, proposal_deltas = predictions
            final_losses = FastRCNNOutputs(
                self.box2box_transform,
                scores,
                proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.box_reg_loss_type,
            ).losses()
        else:
            final_losses = {}
            
        if weak_predictions is not None:
            weak_losses = self.weak_detector_head.losses(weak_predictions, weak_proposals, weak_targets)
            final_losses.update(weak_losses)
        
        return final_losses
    
    def inference(self, predictions, proposals, tta=False):
        scores = self.predict_probs(predictions, proposals)
        if tta:
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

@FAST_RCNN_REGISTRY.register()
class SupervisedDetectorOutputsFineTune(SupervisedDetectorOutputsBase):
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1", loss_weight=1.0, weak_detector_head=None, regression_branch=False, terms={}, freeze_layers=[], embedding_path=''):
        super(SupervisedDetectorOutputsFineTune, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight, weak_detector_head=weak_detector_head, regression_branch=regression_branch, terms=terms, freeze_layers=freeze_layers, embedding_path=embedding_path)

        # Define delta predictors
        self.cls_score_ft = Linear(self.input_size, self.num_classes + 1)
        self.bbox_pred_ft = Linear(self.input_size, self.num_bbox_reg_classes * self.box_dim)
        # Init Predictors
        for l in [self.cls_score_ft, self.bbox_pred_ft]:
            nn.init.constant_(l.weight, 0.)
            nn.init.constant_(l.bias, 0.)

    def forward(self, x, novel_classes, base_classes, supervised_branch_x_weak=None, x_weak=None, similarity=None):
        if x is not None:
            delta_scores = self.cls_score_delta(x)
            proposal_deltas = self.bbox_pred_delta(x)
            delta_ft = self.cls_score_ft(x)
            proposal_ft = self.bbox_pred_ft(x)
            with torch.no_grad():
                if supervised_branch_x_weak is None:
                    [weak_scores, weak_proposal_deltas], _ = self.weak_detector_head.evaluation(x)
                else:
                    [weak_scores, weak_proposal_deltas], _ = self.weak_detector_head.evaluation(supervised_branch_x_weak)
        else:
            delta_scores = torch.zeros(x_weak.size(0), self.num_classes + 1).to(x_weak.device)
            proposal_deltas = torch.zeros(x_weak.size(0), self.num_classes * self.box_dim).to(x_weak.device)
            delta_ft = torch.zeros(x_weak.size(0), self.num_classes + 1).to(x_weak.device)
            proposal_ft = torch.zeros(x_weak.size(0), self.num_classes * self.box_dim).to(x_weak.device)
            weak_scores = None
            weak_proposal_deltas = None
        
        # Transfer from base to novel
        if similarity is not None:
            transfered_delta_scores = torch.zeros_like(delta_scores)
            base_delta_scores = delta_scores.index_select(1, index=base_classes)
            if len(similarity['cls'].size()) > 2:
                cls_base_to_novel_transfer = torch.bmm(similarity['cls'], base_delta_scores.unsqueeze(2)).squeeze(2)
            else:
                cls_base_to_novel_transfer = torch.mm(base_delta_scores, similarity['cls'].transpose(0,1))
            transfered_delta_scores = transfered_delta_scores.index_copy(1, novel_classes, cls_base_to_novel_transfer)
            delta_scores = delta_scores + transfered_delta_scores
            
            proposal_deltas_reshaped = proposal_deltas.view(-1, self.num_classes, self.box_dim)
            base_proposal_deltas = proposal_deltas_reshaped.index_select(1, index=base_classes)
            transferred_proposal_deltas = torch.zeros_like(proposal_deltas_reshaped)
            if len(similarity['bbox'].size()) > 2:
                bbox_base_to_novel_transfer = torch.bmm(similarity['bbox'], base_proposal_deltas)
            else:
                bbox_base_to_novel_transfer = torch.matmul(base_proposal_deltas.transpose(1,2), similarity['bbox'].transpose(0,1)).transpose(1,2)
            transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, novel_classes, bbox_base_to_novel_transfer)
            transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, base_classes, base_proposal_deltas)
            proposal_deltas = transferred_proposal_deltas.view(-1, self.num_classes * self.box_dim)

        scores = self.get_cls_logits(x=delta_scores, x_weak=weak_scores)
        bbox = self.get_cls_bbox(x=proposal_deltas, x_weak=weak_proposal_deltas)
        scores = scores + delta_ft
        bbox = bbox + proposal_ft

        weak_branch_return = None
        if x_weak is not None:
            weak_branch_return, _ = self.weak_detector_head(x_weak)     
        return [scores, bbox], weak_branch_return

@FAST_RCNN_REGISTRY.register()
class SupervisedDetectorOutputsWeakFineTune(SupervisedDetectorOutputsBase):
    @configurable
    def __init__(self, input_shape, *, box2box_transform, num_classes, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, cls_agnostic_bbox_reg=False, smooth_l1_beta=0.0, box_reg_loss_type="smooth_l1", loss_weight=1.0, weak_detector_head=None, regression_branch=False, terms={}, freeze_layers=[], embedding_path=''):
        super(SupervisedDetectorOutputsWeakFineTune, self).__init__(input_shape=input_shape, box2box_transform=box2box_transform, num_classes=num_classes, test_score_thresh=test_score_thresh, test_nms_thresh=test_nms_thresh, test_topk_per_image=test_topk_per_image, cls_agnostic_bbox_reg=cls_agnostic_bbox_reg, smooth_l1_beta=smooth_l1_beta, box_reg_loss_type=box_reg_loss_type, loss_weight=loss_weight, weak_detector_head=weak_detector_head, regression_branch=regression_branch, terms=terms, freeze_layers=freeze_layers, embedding_path=embedding_path)


    def forward(self, x, novel_classes, base_classes, supervised_branch_x_weak=None, x_weak=None, similarity=None):
        if x is not None:
            delta_scores = self.cls_score_delta(x)
            proposal_deltas = self.bbox_pred_delta(x)
            with torch.no_grad():
                if supervised_branch_x_weak is None:
                    [weak_scores, weak_proposal_deltas], _ = self.weak_detector_head.evaluation(x)
                else:
                    [weak_scores, weak_proposal_deltas], _ = self.weak_detector_head.evaluation(supervised_branch_x_weak)
        else:
            delta_scores = torch.zeros(x_weak.size(0), self.num_classes + 1).to(x_weak.device)
            proposal_deltas = torch.zeros(x_weak.size(0), self.num_classes * self.box_dim).to(x_weak.device)
            weak_scores = None
            weak_proposal_deltas = None

        # Transfer from base to novel
        if similarity is not None:
            transfered_delta_scores = torch.zeros_like(delta_scores)
            base_delta_scores = delta_scores.index_select(1, index=base_classes)
            if len(similarity['cls'].size()) > 2:
                cls_base_to_novel_transfer = torch.bmm(similarity['cls'], base_delta_scores.unsqueeze(2)).squeeze(2)
            else:
                cls_base_to_novel_transfer = torch.mm(base_delta_scores, similarity['cls'].transpose(0,1))
            transfered_delta_scores = transfered_delta_scores.index_copy(1, novel_classes, cls_base_to_novel_transfer)
            delta_scores = delta_scores + transfered_delta_scores.detach()
            
            proposal_deltas_reshaped = proposal_deltas.view(-1, self.num_classes, self.box_dim)
            base_proposal_deltas = proposal_deltas_reshaped.index_select(1, index=base_classes)
            transferred_proposal_deltas = torch.zeros_like(proposal_deltas_reshaped)
            if len(similarity['bbox'].size()) > 2:
                bbox_base_to_novel_transfer = torch.bmm(similarity['bbox'], base_proposal_deltas)
            else:
                bbox_base_to_novel_transfer = torch.matmul(base_proposal_deltas.transpose(1,2), similarity['bbox'].transpose(0,1)).transpose(1,2)
            transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, novel_classes, bbox_base_to_novel_transfer.detach())
            transferred_proposal_deltas = transferred_proposal_deltas.index_copy(1, base_classes, base_proposal_deltas)
            proposal_deltas = transferred_proposal_deltas.view(-1, self.num_classes * self.box_dim)

        scores = self.get_cls_logits(x=delta_scores, x_weak=weak_scores)
        bbox = self.get_cls_bbox(x=proposal_deltas, x_weak=weak_proposal_deltas)

        weak_branch_return = None
        if x_weak is not None:
            weak_branch_return, _ = self.weak_detector_head(x_weak)     
        return [scores, bbox], weak_branch_return

def build_fastrcnn_head(cfg, input_shape):
    name = cfg.MODEL.ROI_HEADS.FAST_RCNN.NAME
    return FAST_RCNN_REGISTRY.get(name)(cfg, input_shape)


