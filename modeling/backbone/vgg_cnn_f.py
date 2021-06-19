import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
from torch.nn.modules.normalization import CrossMapLRN2d
import os

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def extract_vgg_cnn_f_components(pretrained=False):
    VGG_CNN_F_torch = nn.Sequential( # Sequential,
        nn.Conv2d(3,64,(11, 11),(4, 4)),
        nn.ReLU(),
        # Lambda(lambda x,lrn=torch.legacy.nn.SpatialCrossMapLRN(*(5, 0.0005, 0.75, 2)): Variable(lrn.forward(x.data))),
        Lambda(lambda x,lrn=CrossMapLRN2d(*(5, 0.0005, 0.75, 2)): lrn.forward(x)),
        nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
        nn.Conv2d(64,256,(5, 5),(1, 1),(2, 2)),
        nn.ReLU(),
        # Lambda(lambda x,lrn=torch.legacy.nn.SpatialCrossMapLRN(*(5, 0.0005, 0.75, 2)): Variable(lrn.forward(x.data))),
        Lambda(lambda x,lrn=CrossMapLRN2d(*(5, 0.0005, 0.75, 2)): lrn.forward(x)),
        nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
        Lambda(lambda x: x.view(x.size(0),-1)), # View,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(9216,4096)), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,1000)), # Linear,
        nn.Softmax(),
    )
    if pretrained:
        VGG_CNN_F_torch.load_state_dict(torch.load(os.path.abspath(__file__ + '/../../../models/VGG_CNN_F_torch.pth')))
    layers = list(VGG_CNN_F_torch.children())
    convs = nn.Sequential(*layers[:14])
    features = nn.Sequential(*layers[15:22])
    return convs, features


    
