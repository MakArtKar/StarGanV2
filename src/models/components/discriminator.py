import torch
import torch.nn as nn

from src.models.components.decoder import Decoder


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
    def __init__(self, num_domains: int, image_size: int = 256, hid_channels: int = 64, max_channels_scale: int = 8):
        super().__init__()
        self.decoder = Decoder(image_size, hid_channels, max_channels_scale)
        channels = self.decoder.out_channels
        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, num_domains, 1)
        )

    def forward(self, x, y):
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
