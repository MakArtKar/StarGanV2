import torch
import torch.nn as nn

from src.models.components.res_blocks import DownsampleResNetBlock, UpsampleResNetBlock


class DummyGenerator(nn.Identity):
    ...


class Generator(nn.Module):
    def __init__(self, style_dim, hid_channels=64, depth=4, bottleneck_num: int = 4):
        super().__init__()
        self.channels = hid_channels
        self.bottleneck_num = bottleneck_num

        self.conv1 = nn.Conv2d(3, hid_channels, 1)
        self.encoder_blocks = self.build_encoder(depth)
        self.bottleneck = self.build_bottleneck(bottleneck_num, style_dim)
        self.decoder_blocks = self.build_decoder(depth, style_dim)
        self.conv2 = nn.Conv2d(self.channels, 3, 1)

    def build_encoder(self, depth):
        encoder_blocks = []
        for _ in range(depth):
            encoder_blocks.append(DownsampleResNetBlock(self.channels, self.channels * 2))
            self.channels *= 2
        return nn.ModuleList(encoder_blocks)

    def build_bottleneck(self, bottleneck_num, style_dim):
        bottleneck_blocks = []
        for _ in range(bottleneck_num // 2):
            bottleneck_blocks.append(DownsampleResNetBlock(self.channels, self.channels, downsample=False))
        for _ in range(bottleneck_num // 2):
            bottleneck_blocks.append(UpsampleResNetBlock(self.channels, self.channels, style_dim, upsample=False))
        return nn.ModuleList(bottleneck_blocks)

    def build_decoder(self, depth, style_dim):
        decoder_blocks = []
        for _ in range(depth):
            decoder_blocks.append(UpsampleResNetBlock(self.channels, self.channels // 2, style_dim))
            self.channels //= 2
        return nn.ModuleList(decoder_blocks)

    def forward(self, x, style):
        x = self.conv1(x)
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)

        for i in range(self.bottleneck_num // 2):
            x = self.bottleneck[i](x)
        for i in range(self.bottleneck_num // 2, self.bottleneck_num):
            x = self.bottleneck[i](x, style)

        for i, decoder_block in enumerate(self.decoder_blocks):
            x = x + encoder_outputs[-i - 1]
            x = decoder_block(x, style)
        x = self.conv2(x)
        return x


def test_generator(batch_size, in_channels, out_channels, img_size, style_dim, x, style):
    img_size = 256
    image = torch.rand(batch_size, 3, img_size, img_size)
    generator = Generator(style_dim)
    out = generator(image, style)
    assert out.shape == (batch_size, 3, img_size, img_size), out.shape

    print('OK generator')


def test():
    batch_size, in_channels, out_channels, img_size, style_dim = 8, 128, 64, 4, 32
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    style = torch.randn(batch_size, style_dim)

    test_generator(batch_size, in_channels, out_channels, img_size, style_dim, x, style)


if __name__ == '__main__':
    test()
