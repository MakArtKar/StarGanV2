import torch.nn as nn


class DummyMappingNetwork(nn.Module):
    def __init__(self, n_latent=16, n_style=64):
        super().__init__()
        self.linear = nn.Linear(n_latent, n_style)

    def forward(self, z, y):
        return self.linear(z)
