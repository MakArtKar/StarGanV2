from abc import abstractmethod
from math import sqrt

import torch
from torch import nn as nn


class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.mean_linear = nn.Linear(style_dim, channels)
        self.std_linear = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        mean, std = self.mean_linear(style), self.std_linear(style)
        mean, std = mean.view(x.size(0), -1, 1, 1), std.view(x.size(0), -1, 1, 1)
        return self.norm(x) * (std + 1) + mean


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward_shortcut(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def forward_block(self, *args):
        raise NotImplementedError()

    def forward(self, *args):
        return (self.forward_shortcut(*args) + self.forward_block(*args)) / sqrt(2)


class DownsampleResNetBlock(ResNetBlock):
    def __init__(self, in_channels, out_channels, norm=True, downsample=True):
        super().__init__(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True) if norm else nn.Identity()

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True) if norm else nn.Identity()

        self.activ = nn.LeakyReLU(0.2)

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.downsample = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward_shortcut(self, x):
        return self.downsample(self.conv(x))

    def forward_block(self, x):
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv1(x)
        x = self.downsample(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        return x


class UpsampleResNetBlock(ResNetBlock):
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = AdaIN(in_channels, style_dim)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = AdaIN(out_channels, style_dim)

        self.activ = nn.LeakyReLU(0.2)

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2) if upsample else nn.Identity()

    def forward_shortcut(self, x, style):
        x = self.upsample(x)
        x = self.conv(x)
        return x

    def forward_block(self, x, style):
        x = self.norm1(x, style)
        x = self.activ(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, style)
        x = self.activ(x)
        x = self.conv2(x)
        return x
