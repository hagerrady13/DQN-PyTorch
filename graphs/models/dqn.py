"""
Deep Q Network model
based on the paper: https://www.nature.com/articles/nature14236
date: 1st of June 2018
"""
import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init


class DQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.config.input_channels, out_channels=self.config.conv_filters[0], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.config.conv_filters[0])

        self.conv1 = nn.Conv2d(in_channels=self.config.conv_filters[0], out_channels=self.config.conv_filters[1], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.config.conv_filters[1])

        self.conv1 = nn.Conv2d(in_channels=self.config.conv_filters[1], out_channels=self.config.conv_filters[2], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.config.conv_filters[2])

        self.linear = nn.Linear(448, self.config.num_classes)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        out = self.linear(x.view(x.size(0), -1))

        return out


"""
#########################
Architecture:
#########################
Input: (N, 3, 64, 64)

conv1: (N, 64, 32, 32)   ==> H/2, W/2
conv2: (N, 128, 16, 16)  ==> H/4, W/4
conv3: (N, 256, 8, 8)    ==> H/8, W/8

----
torch.Size([4, 3, 64, 64])
torch.Size([4, 64, 32, 32])
torch.Size([4, 128, 16, 16])
torch.Size([4, 256, 8, 8])
torch.Size([4, 512, 4, 4])
torch.Size([4, 1, 1, 1])
"""