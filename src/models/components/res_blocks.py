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
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)

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


def test_adain(batch_size, in_channels, out_channels, img_size, style_dim, x, style):
    adain = AdaIN(in_channels, style_dim)
    out = adain(x, style)
    assert out.shape == (batch_size, in_channels, img_size, img_size), out.shape
    print('OK adain')


def test_downsample_resnet_block(batch_size, in_channels, out_channels, img_size, style_dim, x, style):
    block = DownsampleResNetBlock(in_channels, out_channels, downsample=True)
    out = block(x)
    assert out.shape == (batch_size, out_channels, img_size // 2, img_size // 2), out.shape

    block = DownsampleResNetBlock(in_channels, out_channels, downsample=False)
    out = block(x)
    assert out.shape == (batch_size, out_channels, img_size, img_size), out.shape
    print('OK downsample blocks')


def test_upsample_resnet_block(batch_size, in_channels, out_channels, img_size, style_dim, x, style):
    block = UpsampleResNetBlock(in_channels, out_channels, style_dim, upsample=True)
    out = block(x, style)
    assert out.shape == (batch_size, out_channels, img_size * 2, img_size * 2), out.shape

    block = UpsampleResNetBlock(in_channels, out_channels, style_dim, upsample=False)
    out = block(x, style)
    assert out.shape == (batch_size, out_channels, img_size, img_size), out.shape

    print('OK upsample blocks')


def test():
    batch_size, in_channels, out_channels, img_size, style_dim = 8, 128, 64, 4, 32
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    style = torch.randn(batch_size, style_dim)

    test_adain(batch_size, in_channels, out_channels, img_size, style_dim, x, style)
    test_downsample_resnet_block(batch_size, in_channels, out_channels, img_size, style_dim, x, style)
    test_upsample_resnet_block(batch_size, in_channels, out_channels, img_size, style_dim, x, style)


if __name__ == '__main__':
    test()
