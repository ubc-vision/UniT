import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import logging
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY

from ..backbone import extract_components
from torchvision import models as M, ops
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from ..backbone.inception_resnet_v2 import inceptionresnetv2
from ..backbone.vgg_cnn_f import extract_vgg_cnn_f_components

@ROI_BOX_HEAD_REGISTRY.register()
class VGGConvFCHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        _, fc = extract_components(M.vgg16 ,pretrained=True)
        _output_size = input_shape.channels
        for c in fc:
            _output_size = getattr(c, 'out_features') if isinstance(c, nn.Linear) else _output_size
        self.fc = fc
        self._output_size = _output_size

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

@ROI_BOX_HEAD_REGISTRY.register()
class Res5BoxHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.res5, self.out_channels = self._build_res5_block(cfg)

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def forward(self, x):
        x = self.res5(x)
        return x.mean(dim=[2,3])

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        return ShapeSpec(channels=self.out_channels, height=1, width=1)

@ROI_BOX_HEAD_REGISTRY.register()
class Res5BoxHeadNOTE(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.res5, self.out_channels = self._build_res5_block(cfg)
        self.out_channels = 1536

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=1088,
            bottleneck_channels=bottleneck_channels,
            out_channels=1536,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def forward(self, x):
        x = self.res5(x)
        return x.mean(dim=[2,3])

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        return ShapeSpec(channels=self.out_channels, height=1, width=1)

@ROI_BOX_HEAD_REGISTRY.register()
class Res5BoxHeadWithMask(Res5BoxHead):
    def forward(self, x):
        x = self.res5(x)
        return x

@ROI_BOX_HEAD_REGISTRY.register()
class VGGCNNFBoxHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        _, features = extract_vgg_cnn_f_components(pretrained=True)
        self.fc = features
        self._output_size = 4096

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

@ROI_BOX_HEAD_REGISTRY.register()
class InceptionResNetHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        layers = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.mixed_7a = layers.mixed_7a
        self.repeat_2 = layers.repeat_2
        self.block8 = layers.block8
        self.conv2d_7b = layers.conv2d_7b
        self.avgpool_1a = layers.avgpool_1a
        self._output_size = 1536
        self._freeze_layers(cfg.MODEL.FREEZE_LAYERS.BOX_HEAD)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer == name.split(".")[0] for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def forward(self, x):
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x = self.avgpool_1a(x)
        return x.flatten(1)

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
