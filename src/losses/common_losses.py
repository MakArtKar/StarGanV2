from abc import abstractmethod
from typing import List, Tuple

import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError()


class ConcatLoss(BaseLoss):
    def __init__(self, losses: List[Tuple[BaseLoss, float]]):
        super().__init__()
        self.losses = losses

    def forward(self, **kwargs):
        result = 0.
        for loss, scalar in self.losses:
            result += scalar * loss(**kwargs)
        return result
