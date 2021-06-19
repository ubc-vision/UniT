import sys
import os
import torch
import logging
import numpy as np
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.config import configurable
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.structures import ImageList
from detectron2.data import transforms as T
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
import cv2
from ..roi_heads.fast_rcnn import FastRCNNOutputsReduction

@META_ARCH_REGISTRY.register()
class WeakRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *, backbone, proposal_generator, roi_heads, pixel_mean, pixel_std, input_format=None, vis_period=0, freeze_layers=[], test_augmentations=None, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, regression_branch=False):
        super(WeakRCNN, self).__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)
        self._freeze_layers(layers=freeze_layers)
        self.test_augmentations = test_augmentations
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.regression_branch = regression_branch
        if self.test_augmentations.ENABLED:
            self.tta = self._init_tta_fn(test_augmentations)
        else:
            self.tta = lambda x: x
    
    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def _init_tta_fn(self, cfg):
        max_size = cfg.MAX_SIZE
        size_gens = [T.ResizeShortestEdge(sz, max_size, 'choice') for sz in cfg.MIN_SIZES]
        flip = T.RandomFlip(prob=1.0)
        
        def tta_fn(batched_input):
            image = batched_input['image']
            image = image.permute(1, 2, 0).to('cpu').numpy()
            dtype = image.dtype
            image = image.astype(np.uint8)
            
            out_inputs = []
            for tfm_gen in size_gens:
                augmentation_resize = T.AugmentationList([tfm_gen])
                resize_aug_input = T.AugInput(image)
                resize_transforms = augmentation_resize(resize_aug_input)
                resized_image = resize_aug_input.image
                resized_bbox = resize_transforms.apply_box(batched_input['proposals'].proposal_boxes.tensor.data.cpu().numpy()).clip(min=0)

                out_inputs.append( {'file_name': batched_input['file_name'],
                                    'image_id': batched_input['image_id'],
                                    'image': torch.from_numpy(resized_image.astype(dtype)).permute(2,0,1), 
                                    'proposals': Instances(image_size=resized_image.shape[:2], proposal_boxes=Boxes(torch.from_numpy(resized_bbox)), objectness_logits=batched_input['proposals'].objectness_logits),
                                    'height': batched_input['height'],
                                    'width': batched_input['width']} )
                    
                if cfg.FLIP:
                    augmentation_flipped = T.AugmentationList([flip])
                    flip_aug_input = T.AugInput(resized_image)
                    flip_transforms = augmentation_flipped(flip_aug_input)
                    flipped_image = flip_aug_input.image
                    flipped_bbox = flip_transforms.apply_box(resized_bbox).clip(min=0)
                    out_inputs.append( {'file_name': batched_input['file_name'],
                                    'image_id': batched_input['image_id'],
                                    'image': torch.from_numpy(flipped_image.astype(dtype)).permute(2,0,1), 
                                    'proposals': Instances(image_size=flipped_image.shape[:2], proposal_boxes=Boxes(torch.from_numpy(flipped_bbox)), objectness_logits=batched_input['proposals'].objectness_logits),
                                    'height': batched_input['height'],
                                    'width': batched_input['width']} )
            return out_inputs
        return tta_fn

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.META_ARCH
        ret['test_augmentations'] = cfg.TEST.AUG
        ret["test_score_thresh"] = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        ret["test_nms_thresh"] = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        ret["test_topk_per_image"] = cfg.TEST.DETECTIONS_PER_IMAGE
        ret['regression_branch'] = cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.REGRESSION_BRANCH
        return ret

    def forward(self, batched_inputs, weak_batched_inputs=None, meta_data=None, meta_attention=None, return_attention=False):
        if (not self.training) or return_attention:
            return self.inference(batched_inputs, meta_data=meta_data, meta_attention=meta_attention, return_attention=return_attention)

        if self.proposal_generator:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
        
        if weak_batched_inputs is not None:
            weak_images = self.preprocess_image(weak_batched_inputs)
            weak_features = self.backbone(weak_images.tensor)
            if "instances" in weak_batched_inputs[0]:
                weak_gt_instances = [x["instances"].gt_classes.to(self.device) for x in weak_batched_inputs]
            else:
                weak_gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            with torch.no_grad():
                weak_proposals, _ = self.proposal_generator(weak_images, weak_features, None)
        else:
            assert "proposals" in weak_batched_inputs[0]
            weak_proposals = [x["proposals"].to(self.device) for x in weak_batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(weak_images, weak_features, weak_proposals, weak_gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, meta_data=None, meta_attention=None, return_attention=False, do_postprocess=True):   
        assert len(batched_inputs) == 1
        if self.test_augmentations.ENABLED:
            batched_inputs_tta = [self.tta(batched_input) for batched_input in batched_inputs]
            all_results = []
            for idx, batched_inputs_all in enumerate(batched_inputs_tta):
                cls_pred_scores, cls_pred_boxes = [], []
                for batched_inputs_transformed in batched_inputs_all:
                    images = self.preprocess_image([batched_inputs_transformed])
                    features = self.backbone(images.tensor)    
                    if detected_instances is None:
                        if self.proposal_generator:
                            proposals, _ = self.proposal_generator(images, features, None)
                        else:
                            assert "proposals" in [batched_inputs_transformed][0]
                            proposals = [x["proposals"].to(self.device) for x in [batched_inputs_transformed]]

                        results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                        cls_pred_scores.append(results[0])
                        cls_pred_boxes.append(results[1])
                    else:
                        raise NotImplementedError
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                cls_pred_scores = (torch.stack(cls_pred_scores).sum(0),)
                # if not self.regression_branch:
                #     cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                # else:
                #     images = self.preprocess_image(batched_inputs)
                #     features = self.backbone(images.tensor)    
                #     results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                #     cls_pred_boxes = results[1]
                cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                cls_pred_boxes = self.roi_heads.box_predictor.predict_boxes([cls_pred_scores, cls_pred_boxes], proposals)
                results, _ = fast_rcnn_inference(cls_pred_boxes, cls_pred_scores, [x.image_size for x in proposals], self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image) 
                all_results = all_results + results
        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                all_results, _ = self.roi_heads(images, features, proposals, None, tta=False)
        if do_postprocess:
            return GeneralizedRCNN._postprocess(all_results, batched_inputs, images.image_sizes)
        else:
            return all_results

@META_ARCH_REGISTRY.register()
class WeaklySupervisedRCNN(GeneralizedRCNN):
    @configurable
    def __init__(self, *, backbone, proposal_generator, roi_heads, pixel_mean, pixel_std, input_format=None, vis_period=0, freeze_layers=[], test_augmentations=None, test_score_thresh=0.0, test_nms_thresh=0.5, test_topk_per_image=100, regression_branch=False, normalize_images=False, weak_rpn_score_threshold=0.95, train_using_weak=False, train_proposal_regressor=True, weak_proposal_divisor=1.0):
        super(WeaklySupervisedRCNN, self).__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period)
 
        self._freeze_layers(layers=freeze_layers)
        self.test_augmentations = test_augmentations
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.regression_branch = regression_branch
        self.normalize_images = normalize_images
        self.weak_rpn_score_threshold = weak_rpn_score_threshold
        self.train_using_weak = train_using_weak
        self.train_proposal_regressor = train_proposal_regressor
        self.weak_proposal_divisor = weak_proposal_divisor
        if self.test_augmentations.ENABLED:
            self.tta = self._init_tta_fn(test_augmentations)
        else:
            self.tta = lambda x: x

    def _init_tta_fn(self, cfg):
        max_size = cfg.MAX_SIZE
        size_gens = [T.ResizeShortestEdge(sz, max_size, 'choice') for sz in cfg.MIN_SIZES]
        flip = T.RandomFlip(prob=1.0)
        
        def tta_fn(batched_input):
            image = batched_input['image']
            image = image.permute(1, 2, 0).to('cpu').numpy()
            dtype = image.dtype
            image = image.astype(np.uint8)
            
            out_inputs = []
            for tfm_gen in size_gens:
                augmentation_resize = T.AugmentationList([tfm_gen])
                resize_aug_input = T.AugInput(image)
                resize_transforms = augmentation_resize(resize_aug_input)
                resized_image = resize_aug_input.image
                resized_bbox = resize_transforms.apply_box(batched_input['proposals'].proposal_boxes.tensor.data.cpu().numpy()).clip(min=0)

                out_inputs.append( {'file_name': batched_input['file_name'],
                                    'image_id': batched_input['image_id'],
                                    'image': torch.from_numpy(resized_image.astype(dtype)).permute(2,0,1), 
                                    'proposals': Instances(image_size=resized_image.shape[:2], proposal_boxes=Boxes(torch.from_numpy(resized_bbox)), objectness_logits=batched_input['proposals'].objectness_logits),
                                    'height': batched_input['height'],
                                    'width': batched_input['width']} )
                    
                if cfg.FLIP:
                    augmentation_flipped = T.AugmentationList([flip])
                    flip_aug_input = T.AugInput(resized_image)
                    flip_transforms = augmentation_flipped(flip_aug_input)
                    flipped_image = flip_aug_input.image
                    flipped_bbox = flip_transforms.apply_box(resized_bbox).clip(min=0)
                    out_inputs.append( {'file_name': batched_input['file_name'],
                                    'image_id': batched_input['image_id'],
                                    'image': torch.from_numpy(flipped_image.astype(dtype)).permute(2,0,1), 
                                    'proposals': Instances(image_size=flipped_image.shape[:2], proposal_boxes=Boxes(torch.from_numpy(flipped_bbox)), objectness_logits=batched_input['proposals'].objectness_logits),
                                    'height': batched_input['height'],
                                    'width': batched_input['width']} )
            return out_inputs
        return tta_fn

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.normalize_images:
            images = [x/255.0 for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.META_ARCH
        ret['test_augmentations'] = cfg.TEST.AUG
        ret["test_score_thresh"] = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        ret["test_nms_thresh"] = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        ret["test_topk_per_image"] = cfg.TEST.DETECTIONS_PER_IMAGE
        ret['regression_branch'] = cfg.MODEL.ROI_HEADS.FAST_RCNN.WEAK_DETECTOR.REGRESSION_BRANCH
        ret['normalize_images'] = cfg.INPUT.NORMALIZE_IMAGES
        ret['weak_rpn_score_threshold'] = cfg.MODEL.PROPOSAL_GENERATOR.WEAK_RPN_SCORE_TRESHOLD
        ret['train_using_weak'] = cfg.MODEL.ROI_HEADS.TRAIN_USING_WEAK
        ret['train_proposal_regressor'] = cfg.MODEL.ROI_HEADS.TRAIN_PROPOSAL_REGRESSOR
        ret['weak_proposal_divisor'] = cfg.MODEL.ROI_HEADS.WEAK_PROPOSAL_DIVISOR
        return ret

    def preprocess_meta_image(self, meta_data):
        """
        Normalize, pad and batch the input images.
        """
        batched_inputs = meta_data[0]
        images = {idx:[x['image'].to(self.device) for x in shots] for idx, shots in batched_inputs.items()}
        images = {idx:[(x - self.pixel_mean) / self.pixel_std for x in shots] for idx, shots in images.items()}
        images = {idx:ImageList.from_tensors(imgs, self.backbone.size_divisibility) for idx, imgs in images.items()}
        return images

    def process_meta_data(self, meta_data):
        classes = np.sort(list(meta_data[0].keys()))
        num_shots = len(meta_data[0][classes[0]])
        meta_images = self.preprocess_meta_image(meta_data)

        if "instances" in meta_data[0][classes[0]][0]:
            meta_gt_instances = {idx:[x["instances"].to(self.device) for x in shots] for idx,shots in meta_data[0].items()}
        else:
            meta_gt_instances = None

        meta_features = {}
        for idx, meta_image in meta_images.items():
            meta_features[idx] = self.backbone(meta_image.tensor)
        return meta_features, meta_gt_instances

    def forward(self, batched_inputs, weak_batched_inputs=None, meta_data=None, meta_attention=None, return_attention=False, return_similarity=False):
        if (not self.training) or return_attention:
            return self.inference(batched_inputs, meta_data=meta_data, meta_attention=meta_attention, return_attention=return_attention, return_similarity=return_similarity)
            
        meta_features, meta_gt_instances = None, None
        if meta_data is not None:
            with torch.no_grad():
                meta_features, meta_gt_instances = self.process_meta_data(meta_data)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        if weak_batched_inputs is not None:
            with torch.no_grad():
                weak_images = self.preprocess_image(weak_batched_inputs)
            weak_features = self.backbone(weak_images.tensor)
            if "instances" in weak_batched_inputs[0]:
                weak_gt_instances = [x["instances"].gt_classes.to(self.device) for x in weak_batched_inputs]
            else:
                weak_gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            with torch.no_grad():
                weak_proposals, _ = self.proposal_generator(weak_images, weak_features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            weak_proposals = [x["proposals"].to(self.device) for x in weak_batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, weak_images=weak_images, weak_features=weak_features, weak_proposals=weak_proposals, weak_targets=weak_gt_instances, meta_features=meta_features, meta_targets=meta_gt_instances, meta_attention=meta_attention, return_attention=return_attention)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, meta_data=None, meta_attention=None, return_attention=False, do_postprocess=True, return_similarity=False):   
        meta_features, meta_gt_instances = None, None
        if return_attention or (meta_attention is None):
            if meta_data is not None:
                meta_features, meta_gt_instances = self.process_meta_data(meta_data)
                return self.roi_heads(None, None, None, meta_features=meta_features, meta_targets=meta_gt_instances, return_attention=return_attention)
        assert len(batched_inputs) == 1
        if self.test_augmentations.ENABLED:
            batched_inputs_tta = [self.tta(batched_input) for batched_input in batched_inputs]
            all_results = []
            for idx, batched_inputs_all in enumerate(batched_inputs_tta):
                cls_pred_scores, cls_pred_boxes = [], []
                for batched_inputs_transformed in batched_inputs_all:
                    images = self.preprocess_image([batched_inputs_transformed])
                    features = self.backbone(images.tensor)    
                    if detected_instances is None:
                        if self.proposal_generator:
                            proposals, _ = self.proposal_generator(images, features, None)
                        else:
                            assert "proposals" in [batched_inputs_transformed][0]
                            proposals = [x["proposals"].to(self.device) for x in [batched_inputs_transformed]]

                        results, _ = self.roi_heads(images, features, proposals, None, meta_features=meta_features, meta_targets=meta_gt_instances, meta_attention=meta_attention, return_attention=return_attention, tta=True)
                        cls_pred_scores.append(results[0])
                        cls_pred_boxes.append(results[1])
                    else:
                        raise NotImplementedError
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                cls_pred_scores = (torch.stack(cls_pred_scores).sum(0),)
                # if not self.regression_branch:
                #     cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                # else:
                #     images = self.preprocess_image(batched_inputs)
                #     features = self.backbone(images.tensor)    
                #     results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                #     cls_pred_boxes = results[1]
                cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                cls_pred_boxes = self.roi_heads.box_predictor.predict_boxes([cls_pred_scores, cls_pred_boxes], proposals)
                results, _ = fast_rcnn_inference(cls_pred_boxes, cls_pred_scores, [x.image_size for x in proposals], self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image) 
                all_results = all_results + results
        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                all_results, _ = self.roi_heads(images, features, proposals, None, meta_features=meta_features, meta_targets=meta_gt_instances, meta_attention=meta_attention, return_attention=return_attention, tta=False, return_similarity=return_similarity)
        if do_postprocess:
            return WeaklySupervisedRCNN._postprocess(all_results, batched_inputs, images.image_sizes)
        else:
            return all_results

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            if hasattr(results_per_image, '_visual_similarity'):
                r._visual_similarity = results_per_image._visual_similarity
            if hasattr(results_per_image, '_lingual_similarity'):
                r._lingual_similarity = results_per_image._lingual_similarity
            processed_results.append({"instances": r})
        return processed_results

@META_ARCH_REGISTRY.register()
class WeaklySupervisedRCNNNoMeta(WeaklySupervisedRCNN):
    def forward(self, batched_inputs, weak_batched_inputs=None, return_similarity=False, train_only_weak=False):
        if not self.training:
            return self.inference(batched_inputs, return_similarity=return_similarity)

        if batched_inputs is not None:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
        else:
            images = None
            features = None
            gt_instances = None
        
        if weak_batched_inputs is not None:
            with torch.no_grad():
                weak_images = self.preprocess_image(weak_batched_inputs)
            weak_features = self.backbone(weak_images.tensor)
            if "instances" in weak_batched_inputs[0]:
                weak_gt_instances = [x["instances"].gt_classes.to(self.device) for x in weak_batched_inputs]
            else:
                weak_gt_instances = None
        else:
            weak_features = None
            weak_images = None
            weak_gt_instances = None

        if self.proposal_generator:
            if batched_inputs is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                proposals = None
                proposal_losses = {}
            if weak_batched_inputs is not None:
                with torch.no_grad():
                    weak_proposals, _ = self.proposal_generator(weak_images, weak_features, None)
            else:
                weak_proposals = None
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            if weak_batched_inputs is not None:
                weak_proposals = [x["proposals"].to(self.device) for x in weak_batched_inputs]
            else:
                weak_proposals = None
            proposal_losses = {}
   
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, weak_images=weak_images, weak_features=weak_features, weak_proposals=weak_proposals, weak_targets=weak_gt_instances, train_only_weak=train_only_weak)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, return_similarity=False):   
        assert len(batched_inputs) == 1
        if self.test_augmentations.ENABLED:
            batched_inputs_tta = [self.tta(batched_input) for batched_input in batched_inputs]
            all_results = []
            for idx, batched_inputs_all in enumerate(batched_inputs_tta):
                cls_pred_scores, cls_pred_boxes = [], []
                for batched_inputs_transformed in batched_inputs_all:
                    images = self.preprocess_image([batched_inputs_transformed])
                    features = self.backbone(images.tensor)    
                    if detected_instances is None:
                        if self.proposal_generator:
                            proposals, _ = self.proposal_generator(images, features, None)
                        else:
                            assert "proposals" in [batched_inputs_transformed][0]
                            proposals = [x["proposals"].to(self.device) for x in [batched_inputs_transformed]]

                        results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                        cls_pred_scores.append(results[0])
                        cls_pred_boxes.append(results[1])
                    else:
                        raise NotImplementedError
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                cls_pred_scores = (torch.stack(cls_pred_scores).sum(0),)
                # if not self.regression_branch:
                #     cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                # else:
                #     images = self.preprocess_image(batched_inputs)
                #     features = self.backbone(images.tensor)    
                #     results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                #     cls_pred_boxes = results[1]
                cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                cls_pred_boxes = self.roi_heads.box_predictor.predict_boxes([cls_pred_scores, cls_pred_boxes], proposals)
                results, _ = fast_rcnn_inference(cls_pred_boxes, cls_pred_scores, [x.image_size for x in proposals], self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image) 
                all_results = all_results + results
        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                all_results, _ = self.roi_heads(images, features, proposals, None, tta=False, return_similarity=return_similarity)
        if do_postprocess:
            return WeaklySupervisedRCNNNoMeta._postprocess(all_results, batched_inputs, images.image_sizes)
        else:
            return all_results

@META_ARCH_REGISTRY.register()
class WeaklySupervisedRCNNRPN(WeaklySupervisedRCNN):
    def forward(self, batched_inputs, weak_batched_inputs=None, return_similarity=False, train_only_weak=False):
        if not self.training:
            return self.inference(batched_inputs, return_similarity=return_similarity)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        if weak_batched_inputs is not None:
            with torch.no_grad():
                weak_images = self.preprocess_image(weak_batched_inputs)
                weak_features = self.backbone(weak_images.tensor)
                if "instances" in weak_batched_inputs[0]:
                    weak_gt_instances = [x["instances"].gt_classes.to(self.device) for x in weak_batched_inputs]
                else:
                    weak_gt_instances = None
        else:
            weak_features = None
            weak_images = None
            weak_gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # Use weak classes as pseudo labels for RPN
            if weak_batched_inputs is not None:
                if not self.train_using_weak:
                    with torch.no_grad():
                        weak_proposals, _ = self.proposal_generator(weak_images, weak_features, None)
                        self.roi_heads.eval()
                        weak_pred_instances, _ = self.roi_heads(weak_images, weak_features, weak_proposals, None, tta=False, return_similarity=False)
                        self.roi_heads.train()
                else:
                    with torch.no_grad():
                        weak_proposals, _ = self.proposal_generator(weak_images, weak_features, None)
                        for idx in range(len(weak_proposals)):
                            weak_proposals[idx] = weak_proposals[idx][:self.proposal_generator.batch_size_per_image]
                    self.roi_heads.eval()
                    weak_pred_instances, weak_pred_proposals = self.roi_heads(weak_images, weak_features, weak_proposals, None, tta=False, return_similarity=False, return_proposals=True)
                    self.roi_heads.train()
                weak_proposal_losses = {'weak_loss_rpn_cls': [], 'weak_loss_rpn_loc': []}
                if self.train_using_weak:
                    weak_proposal_losses['weak_loss_cls'] = []
                    weak_proposal_losses['weak_loss_bbox'] = []
                weak_features_start = 0
                for idx, weak_instance in enumerate(weak_pred_instances):
                    labels_mask = (weak_pred_instances[idx].pred_classes == weak_gt_instances[idx].unsqueeze(1)).sum(0) > 0
                    weak_pred_instances[idx] = weak_pred_instances[idx][labels_mask]
                    score_mask = weak_pred_instances[idx].scores > self.weak_rpn_score_threshold
                    weak_pred_instances[idx] = weak_pred_instances[idx][score_mask]
                    weak_pred_instances[idx].gt_boxes = weak_pred_instances[idx].pred_boxes
                    weak_pred_instances[idx].gt_classes = weak_pred_instances[idx].pred_classes
                    if len(weak_pred_instances[idx]) > 0:
                        _, weak_proposal_loss = self.proposal_generator(None, {x:y.narrow(0, idx, 1) for x,y in weak_features.items()}, [weak_pred_instances[idx]])
                        weak_proposal_losses['weak_loss_rpn_cls'].append(weak_proposal_loss['loss_rpn_cls'] * self.weak_rpn_score_threshold * self.weak_proposal_divisor)
                        if self.train_proposal_regressor:
                            weak_proposal_losses['weak_loss_rpn_loc'].append(weak_proposal_loss['loss_rpn_loc'] * self.weak_rpn_score_threshold * self.weak_proposal_divisor)
                        else:
                            weak_proposal_losses['weak_loss_rpn_loc'].append(weak_proposal_loss['loss_rpn_loc'] * 0.0)
                    else:
                        weak_proposal_losses['weak_loss_rpn_cls'].append(0.0 * weak_pred_instances[idx].pred_classes.sum())
                        weak_proposal_losses['weak_loss_rpn_loc'].append(0.0 * weak_pred_instances[idx].pred_classes.sum())
                    
                    if self.train_using_weak and len(weak_pred_instances[idx]) > 0:
                        labelled_weak_proposals, _ = self.roi_heads.box_predictor.weak_detector_head.label_and_sample_proposals([weak_proposals[idx]], [weak_pred_instances[idx]])
                        weak_scores, weak_proposal_deltas = weak_pred_proposals[1][0].narrow(0, weak_features_start, len(weak_proposals[idx])), weak_pred_proposals[1][1].narrow(0, weak_features_start, len(weak_proposals[idx]))
                        weak_features_start = weak_features_start + len(weak_proposals[idx])
                        weak_losses = FastRCNNOutputsReduction(self.roi_heads.box_predictor.box2box_transform, weak_scores, weak_proposal_deltas, labelled_weak_proposals, self.roi_heads.box_predictor.smooth_l1_beta, self.roi_heads.box_predictor.box_reg_loss_type).losses()   
                        weak_proposal_losses['weak_loss_cls'].append((weak_losses['loss_cls'][labelled_weak_proposals[0].gt_classes != self.roi_heads.num_classes]).mean())
                        weak_proposal_losses['weak_loss_bbox'].append((weak_losses['loss_box_reg']).mean())
                    else:
                        weak_proposal_losses['weak_loss_cls'].append(0.0 * weak_pred_proposals[1][0].sum())
                        weak_proposal_losses['weak_loss_bbox'].append(0.0 * weak_pred_proposals[1][1].sum())

                if len(weak_proposal_losses['weak_loss_rpn_cls']) > 0:
                    weak_proposal_losses['weak_loss_rpn_cls'] = torch.mean(torch.stack(weak_proposal_losses['weak_loss_rpn_cls']))
                    weak_proposal_losses['weak_loss_rpn_loc'] = torch.mean(torch.stack(weak_proposal_losses['weak_loss_rpn_loc']))
                    if self.train_using_weak:
                        weak_proposal_losses['weak_loss_cls'] = torch.mean(torch.stack(weak_proposal_losses['weak_loss_cls']))
                        weak_proposal_losses['weak_loss_bbox'] = torch.mean(torch.stack(weak_proposal_losses['weak_loss_bbox']))
                else:
                    weak_proposal_losses = {}
            else:
                weak_proposals = None
                weak_proposal_losses = {}
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            if weak_batched_inputs is not None:
                weak_proposals = [x["proposals"].to(self.device) for x in weak_batched_inputs]
            else:
                weak_proposals = None
            if self.train_using_weak:
                import ipdb; ipdb.set_trace()
            proposal_losses = {}
   
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, weak_images=None, weak_features=None, weak_proposals=None, weak_targets=None, train_only_weak=train_only_weak)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(weak_proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True, return_similarity=False):   
        assert len(batched_inputs) == 1
        if self.test_augmentations.ENABLED:
            batched_inputs_tta = [self.tta(batched_input) for batched_input in batched_inputs]
            all_results = []
            for idx, batched_inputs_all in enumerate(batched_inputs_tta):
                cls_pred_scores, cls_pred_boxes = [], []
                for batched_inputs_transformed in batched_inputs_all:
                    images = self.preprocess_image([batched_inputs_transformed])
                    features = self.backbone(images.tensor)    
                    if detected_instances is None:
                        if self.proposal_generator:
                            proposals, _ = self.proposal_generator(images, features, None)
                        else:
                            assert "proposals" in [batched_inputs_transformed][0]
                            proposals = [x["proposals"].to(self.device) for x in [batched_inputs_transformed]]

                        results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                        cls_pred_scores.append(results[0])
                        cls_pred_boxes.append(results[1])
                    else:
                        raise NotImplementedError
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                cls_pred_scores = (torch.stack(cls_pred_scores).sum(0),)
                # if not self.regression_branch:
                #     cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                # else:
                #     images = self.preprocess_image(batched_inputs)
                #     features = self.backbone(images.tensor)    
                #     results, _ = self.roi_heads(images, features, proposals, None, tta=True)
                #     cls_pred_boxes = results[1]
                cls_pred_boxes = torch.stack(cls_pred_boxes).mean(0)
                cls_pred_boxes = self.roi_heads.box_predictor.predict_boxes([cls_pred_scores, cls_pred_boxes], proposals)
                results, _ = fast_rcnn_inference(cls_pred_boxes, cls_pred_scores, [x.image_size for x in proposals], self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image) 
                all_results = all_results + results
        else:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                all_results, _ = self.roi_heads(images, features, proposals, None, tta=False, return_similarity=return_similarity)
        if do_postprocess:
            return WeaklySupervisedRCNNNoMeta._postprocess(all_results, batched_inputs, images.image_sizes)
        else:
            return all_results