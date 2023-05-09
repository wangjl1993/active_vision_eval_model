import torch 
from torchvision import models
import torch.nn as nn
from typing import Callable

class EvalModel(nn.Module):
    def __init__(self, build_backbone_fun: Callable, **params) -> None:
        super(EvalModel, self).__init__()
        self.backbone = build_backbone_fun(**params)
    
    def forward(self, x):
        return self.backbone(x)

def build_MobileNetV2(input_channels: int=4, num_classes: int=2, **params):
    model = models.MobileNetV2(num_classes=num_classes, **params)
    if input_channels != 3:
        model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # replace the first layer
    return model

def build_ResNet18(input_channels: int=4, num_classes: int=2, **params):
    model = models.resnet18(num_classes=num_classes, **params)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model