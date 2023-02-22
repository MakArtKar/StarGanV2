import torch
from omegaconf import DictConfig

from src.models.components.res_blocks import AdaIN, DownsampleResNetBlock, UpsampleResNetBlock


def test_adain(cfg_train: DictConfig):
    channels = 64
    x = torch.randn(cfg_train.data.batch_size, channels, cfg_train.image_size, cfg_train.image_size)
    style = torch.randn(cfg_train.data.batch_size, cfg_train.style_dim)

    adain = AdaIN(channels, cfg_train.style_dim)
    out = adain(x, style)
    assert out.shape == (cfg_train.data.batch_size, channels, cfg_train.image_size, cfg_train.image_size), out.shape


def test_downsample_resnet_block(cfg_train: DictConfig):
    in_channels, out_channels = 64, 128
    x = torch.randn(cfg_train.data.batch_size, in_channels, cfg_train.image_size, cfg_train.image_size)

    block = DownsampleResNetBlock(in_channels, out_channels, downsample=True)
    out = block(x)
    out_size = (cfg_train.data.batch_size, out_channels, cfg_train.image_size // 2, cfg_train.image_size // 2)
    assert out.shape == out_size, out.shape

    block = DownsampleResNetBlock(in_channels, out_channels, downsample=False)
    out = block(x)
    out_size = (cfg_train.data.batch_size, out_channels, cfg_train.image_size, cfg_train.image_size)
    assert out.shape == out_size, out.shape


def test_upsample_resnet_block(cfg_train: DictConfig):
    in_channels, out_channels = 128, 64
    x = torch.randn(cfg_train.data.batch_size, in_channels, cfg_train.image_size, cfg_train.image_size)
    style = torch.randn(cfg_train.data.batch_size, cfg_train.style_dim)

    block = UpsampleResNetBlock(in_channels, out_channels, cfg_train.style_dim, upsample=True)
    out = block(x, style)
    out_size = (cfg_train.data.batch_size, out_channels, cfg_train.image_size * 2, cfg_train.image_size * 2)
    assert out.shape == out_size, out.shape

    block = UpsampleResNetBlock(in_channels, out_channels, cfg_train.style_dim, upsample=False)
    out = block(x, style)
    out_size = (cfg_train.data.batch_size, out_channels, cfg_train.image_size, cfg_train.image_size)
    assert out.shape == out_size, out.shape
