import torch
import torch.nn as nn


class DummyMappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64):
        super().__init__()
        self.linear = nn.Linear(latent_dim, style_dim)

    def forward(self, z, y):
        return self.linear(z)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim: int, style_dim: int, n_domains: int):
        super().__init__()
        shared = []
        for i in range(4):
            shared.extend([
                nn.Linear(latent_dim if i == 0 else 512, 512),
                nn.ReLU()
            ])
        self.shared = nn.Sequential(*shared)

        unshared = []
        for _ in range(n_domains):
            modules = []
            for i in range(4):
                modules.extend([
                    nn.Linear(512, style_dim if i == 3 else 512),
                    nn.ReLU()
                ])
            unshared.append(nn.Sequential(*modules))
        self.unshared = nn.ModuleList(unshared)

    def forward(self, z, y):
        x = self.shared(z)
        result = torch.stack([m(x) for m in self.unshared], dim=1)
        ids = torch.arange(y.size(0)).to(y)
        return result[ids, y]
