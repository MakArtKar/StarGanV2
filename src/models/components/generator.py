import torch.nn as nn


class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x, s):
        return x
