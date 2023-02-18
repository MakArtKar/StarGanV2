import torch.nn as nn


class DummyStyleEncoder(nn.Module):
    def __init__(self, style_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, style_dim, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, y):
        return self.model(x)
