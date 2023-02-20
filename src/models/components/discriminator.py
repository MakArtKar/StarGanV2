import torch.nn as nn


class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, y):
        return self.model(x)
