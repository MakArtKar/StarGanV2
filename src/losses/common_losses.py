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
    def __init__(self, losses: List[BaseLoss], weights: List[float]):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, **kwargs):
        result = 0.
        for loss, weight in zip(self.losses, self.weights):
            result += weight * loss(**kwargs)
        return result
