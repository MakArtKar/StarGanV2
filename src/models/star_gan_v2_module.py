import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.losses import BaseLoss


class StarGanV2LitModule(LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            mapping_network: nn.Module,
            style_encoder: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            criterion: BaseLoss,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.generator = generator
        self.discriminator = discriminator
        self.mapping_network = mapping_network
        self.style_encoder = style_encoder

        self.criterion = criterion

    def training_step(self, batch, batch_idx: int):
        ...
