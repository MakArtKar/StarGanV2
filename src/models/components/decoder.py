import numpy as np
import torch.nn as nn

from src.models.components.res_blocks import DownsampleResNetBlock


class Decoder(nn.Module):
    def __init__(self, image_size: int, hid_channels: int = 64, max_channels_scale: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_channels, 1)

        channels = hid_channels
        decoder = []
        depth = int(np.log2(image_size)) - 2
        for _ in range(depth):
            new_channels = min(max_channels_scale * hid_channels, channels * 2)
            decoder.append(DownsampleResNetBlock(channels, new_channels, norm=False))
            channels = new_channels
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = channels

    def forward(self, x):
        x = self.conv1(x)
        return self.decoder(x)
