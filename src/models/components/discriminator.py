import numpy as np
import torch
import torch.nn as nn

from src.models.components.res_blocks import DownsampleResNetBlock


class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, y):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, num_domains: int, image_size: int = 256, hid_channels: int = 64, max_channels_scale=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_channels, 1)
        channels = hid_channels
        decoder = []
        depth = int(np.log2(image_size)) - 2
        for _ in range(depth):
            new_channels = min(max_channels_scale * hid_channels, channels * 2)
            decoder.append(DownsampleResNetBlock(channels, new_channels))
            channels = new_channels
        self.decoder = nn.Sequential(*decoder)
        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, num_domains, 1)
        )

    def forward(self, x, y):
        x = self.conv1(x)
        result = self.decoder(x)
        result = result.view(result.size(0), -1)
        ids = torch.arange(y.size(0)).to(y)
        return result[ids, y]


def test():
    batch_size, image_size, n_domains = 8, 64, 2
    discriminator = Discriminator(n_domains, image_size=image_size)
    image = torch.randn(batch_size, 3, image_size, image_size)
    y = torch.randint(low=0, high=n_domains, size=(batch_size,))
    out = discriminator(image, y)
    assert out.shape == (batch_size,), out.shape

    print('OK discriminator')


if __name__ == '__main__':
    test()
