import torch
import torch.nn as nn

from src.models.components.decoder import Decoder


class Discriminator(nn.Module):
    def __init__(self, n_domains: int, image_size: int = 256, hid_channels: int = 64, max_channels_scale: int = 8):
        super().__init__()
        self.decoder = Decoder(image_size, hid_channels, max_channels_scale)
        channels = self.decoder.out_channels
        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, n_domains, 1)
        )

    def forward(self, x, y):
        result = self.decoder(x)
        result = result.view(result.size(0), -1)
        ids = torch.arange(y.size(0)).to(y)
        return result[ids, y]
