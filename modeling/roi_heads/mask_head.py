from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, MaskRCNNConvUpsampleHead, mask_rcnn_inference, mask_rcnn_loss
import logging 

@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHeadWithSimilarity(MaskRCNNConvUpsampleHead):
    def forward(self, x, instances, similarity=None, base_classes=None, novel_classes=None):
        x = self.layers(x)
        if similarity is not None:
            if x.numel() > 0:
                similarity_mask = similarity['seg']
                mask_base = x.index_select(1, index=base_classes)
                mask_base_reshaped = mask_base.view(*mask_base.size()[:2], -1)
                if len(similarity_mask.size()) > 2:
                    mask_combination = torch.bmm(similarity_mask, mask_base_reshaped)
                else:
                    mask_combination = torch.matmul(mask_base_reshaped.transpose(1,2), similarity_mask.transpose(0,1)).transpose(1,2)
                mask_novel = mask_combination.view(mask_base.size(0), -1, *mask_base.size()[2:])
                mask_final = torch.zeros_like(x)
                mask_final = mask_final.index_copy(1, novel_classes, mask_novel)
                mask_final = mask_final.index_copy(1, base_classes, mask_base)
                x = mask_final
        if self.training:
            assert not torch.jit.is_scripting()
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHeadWithFineTune(MaskRCNNConvUpsampleHead):
    @configurable
    def __init__(self, input_shape, *, num_classes, conv_dims, conv_norm="", **kwargs):
        freeze_layers = kwargs['freeze_layers']
        del kwargs['freeze_layers']
        super().__init__(input_shape, num_classes=num_classes, conv_dims=conv_dims, conv_norm=conv_norm, **kwargs)
        self.predictor_delta = Conv2d(self.predictor.in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.predictor_delta.weight, 0.)
        if self.predictor_delta.bias is not None:
            nn.init.constant_(self.predictor_delta.bias, 0.)
        self._freeze_layers(freeze_layers)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['freeze_layers'] = cfg.MODEL.FREEZE_LAYERS.MASK_HEAD
        return ret

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False
    
    def layers(self, x):
        x = self.deconv(x)
        x = self.deconv_relu(x)
        x_fixed = self.predictor(x)
        x_delta = self.predictor_delta(x)
        return x_fixed, x_delta

    def forward(self, x, instances, similarity=None, base_classes=None, novel_classes=None):
        x, x_delta = self.layers(x)
        if similarity is not None:
            if x.numel() > 0:
                similarity_mask = similarity['seg']
                mask_base = x.index_select(1, index=base_classes)
                mask_base_reshaped = mask_base.view(*mask_base.size()[:2], -1)
                if len(similarity_mask.size()) > 2:
                    mask_combination = torch.bmm(similarity_mask, mask_base_reshaped)
                else:
                    mask_combination = torch.matmul(mask_base_reshaped.transpose(1,2), similarity_mask.transpose(0,1)).transpose(1,2)
                mask_novel = mask_combination.view(mask_base.size(0), -1, *mask_base.size()[2:])
                mask_final = torch.zeros_like(x)
                mask_final = mask_final.index_copy(1, novel_classes, mask_novel)
                mask_final = mask_final.index_copy(1, base_classes, mask_base)
                x = mask_final
        x = x + x_delta
        if self.training:
            assert not torch.jit.is_scripting()
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances