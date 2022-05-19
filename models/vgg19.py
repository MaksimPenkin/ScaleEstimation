"""
 @author   Maksim Penkin
"""


import torch
import torch.nn as nn
from torchvision import models


class VGG19(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        vgg19_pretrained = models.vgg19(pretrained=True)
        backbone = vgg19_pretrained.features

        self.avgpool = vgg19_pretrained.avgpool
        self.classifier = vgg19_pretrained.classifier

        self.add_module(f"block1", backbone[:3])
        self.add_module(f"pool1", backbone[4])

        self.add_module(f"block2", backbone[5:8])
        self.add_module(f"pool2", backbone[9])

        self.add_module(f"block3", backbone[10:17])
        self.add_module(f"pool3", backbone[18])

        self.add_module(f"block4", backbone[19:26])
        self.add_module(f"pool4", backbone[27])

        self.add_module(f"block5", backbone[27:35])
        self.add_module(f"pool5", backbone[36])

    def forward(self, x):
        x = self.__getattr__(f"block1")(x)
        x = nn.ReLU()(x)
        x = self.__getattr__(f"pool1")(x)

        x = self.__getattr__(f"block2")(x)
        x = nn.ReLU()(x)
        x = self.__getattr__(f"pool2")(x)

        x = self.__getattr__(f"block3")(x)
        x = nn.ReLU()(x)
        x = self.__getattr__(f"pool3")(x)

        x = self.__getattr__(f"block4")(x)
        x = nn.ReLU()(x)
        x = self.__getattr__(f"pool4")(x)

        x = self.__getattr__(f"block5")(x)
        x = nn.ReLU()(x)
        x = self.__getattr__(f"pool5")(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
