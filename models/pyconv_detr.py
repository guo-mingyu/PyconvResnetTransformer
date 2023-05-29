import torch
from torch import nn
from torchvision.models import resnet

from models.pyconvresnet import PyConvBottleneck, PyConv


class PyConvResNet(nn.Module):
    def __init__(self, original_resnet):
        super(PyConvResNet, self).__init__()

        # Remove the last layer of original ResNet
        layers = list(original_resnet.children())[:-1]
        self.original_resnet = nn.Sequential(*layers)

        # Replace the last layer with PyConvBottleneck
        self.original_resnet[-1] = PyConvBottleneck(1024, 2048, stride=1, downsample=self.original_resnet[-1].downsample)

    def forward(self, x):
        x = self.original_resnet(x)
        return x


def build_backbone():
    original_resnet = resnet.__dict__['resnet50'](pretrained=True)
    backbone = PyConvResNet(original_resnet)
    return backbone