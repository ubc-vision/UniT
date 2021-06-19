import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from torchvision import models as M, ops
from .inception_resnet_v2 import inceptionresnetv2
from .vgg_cnn_f import extract_vgg_cnn_f_components
from collections import OrderedDict
import logging


def extract_components(model_fn, pretrained=False):
    model = model_fn(pretrained)
    convs = model.features[:-1]
    fc    = model.classifier[:-1]
    return convs, fc

def dilate_convs(convs):
    i = -1
    while not isinstance(convs[i], nn.MaxPool2d):
        if isinstance(convs[i], nn.Conv2d):
            convs[i].dilation = (2, 2)
            convs[i].padding = (2, 2)
        i -= 1
    del convs[i]
    return convs

def freeze_convs(convs, k):
    """
    Freezes `k` conv layers
    """
    i = 0
    while k > 0:
        if isinstance(convs[i], nn.Conv2d):
            k -= 1
            for p in convs[i].parameters():
                p.requires_grad = False
        i += 1

def get_conv_scale(convs):
    """
    Determines the downscaling performed by a sequence of convolutional and pooling layers
    """
    scale = 1.
    channels = 3
    for c in convs:
        stride = getattr(c, 'stride', 1.)
        scale *= stride if isinstance(stride, (int, float)) else stride[0]
        channels = getattr(c, 'out_channels') if isinstance(c, nn.Conv2d) else channels
    return scale, channels

@BACKBONE_REGISTRY.register()
class VGG(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        dilated = cfg.MODEL.BACKBONE.DILATED
        convs, _ = extract_components(M.vgg16 ,pretrained=True)
        if dilated:
            convs = dilate_convs(convs)
        freeze_convs(convs, cfg.MODEL.BACKBONE.FREEZE_CONVS)
        self.convs = convs
        self.scale, self.channels = get_conv_scale(convs)
        self._out_features = ['vgg_conv']

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.channels, stride=self.scale
            )
            for name in self._out_features
        }
    def forward(self, x):
        output = self.convs(x)
        return {self._out_features[0]: output}

@BACKBONE_REGISTRY.register()
class VGG_CNN_F(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        convs, _ = extract_vgg_cnn_f_components(pretrained=True)
        self.convs = convs
        self.scale, self.channels = 16, 256
        self._out_features = ['vgg_conv']

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.channels, stride=self.scale
            )
            for name in self._out_features
        }

    def forward(self, x):
        output = self.convs(x)
        return {self._out_features[0]: output}

@BACKBONE_REGISTRY.register()
class InceptionResNetV2(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        layers = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.conv2d_1a = layers.conv2d_1a
        self.conv2d_2a = layers.conv2d_2a
        self.conv2d_2b = layers.conv2d_2b
        self.maxpool_3a = layers.maxpool_3a
        self.conv2d_3b = layers.conv2d_3b
        self.conv2d_4a = layers.conv2d_4a
        self.maxpool_5a =layers.maxpool_5a
        self.mixed_5b = layers.mixed_5b
        self.repeat = layers.repeat
        self.mixed_6a = layers.mixed_6a
        self.repeat_1 = layers.repeat_1
        self.scale, self.channels =  16, 1088
        self._out_features = ['block17']
        if cfg.MODEL.BACKBONE.FREEZE_CONVS > 0:
            logging.getLogger("detectron2").warn("FREEZING BACKBONE LAYERS")
            for layer in [self.conv2d_1a, self.conv2d_2a, self.conv2d_2b, self.conv2d_3b, self.conv2d_4a]:
                for name, param in layer.named_parameters():
                    param.requires_grad = False


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.channels, stride=self.scale
            )
            for name in self._out_features
        }

    def features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        return x

    def forward(self, x):
        output = self.features(x)
        return {self._out_features[0]: output}