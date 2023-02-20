import torch.nn as nn


class DummyMappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64):
        super().__init__()
        self.linear = nn.Linear(latent_dim, style_dim)

    def forward(self, z, y):
        return self.linear(z)
