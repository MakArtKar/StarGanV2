import torch
import torch.nn as nn

from src.models.components.decoder import Decoder


class StyleEncoder(nn.Module):
    def __init__(self, style_dim: int, n_domains: int,
                 image_size: int = 256, hid_channels: int = 64, max_channels_scale: int = 8):
        super().__init__()
        self.decoder = Decoder(image_size, hid_channels, max_channels_scale)
        channels = self.decoder.out_channels
        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 4),
            nn.LeakyReLU(0.2),
        )
        self.unshared = nn.ModuleList([nn.Linear(channels, style_dim) for _ in range(n_domains)])

    def forward(self, x, y):
        x = self.tail(self.decoder(x)).view(x.size(0), -1)
        result = torch.stack([m(x) for m in self.unshared], dim=1)
        ids = torch.arange(y.size(0)).to(y)
        return result[ids, y]
