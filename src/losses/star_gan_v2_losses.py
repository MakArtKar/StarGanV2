import torch
import torch.nn.functional as F

from src.losses.common_losses import BaseLoss


class AdversarialLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, real_disc_output: torch.Tensor, fake_disc_output: torch.Tensor, **kwargs):
        return self.coef * (torch.log(real_disc_output).mean() + torch.log(1 - fake_disc_output).mean())


class StyleReconstructionLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, style_code: torch.Tensor, encoded_style_code: torch.Tensor, **kwargs):
        return F.l1_loss(style_code, encoded_style_code)


class StyleDiversificationLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, first_gen_output: torch.Tensor, second_gen_output: torch.Tensor, **kwargs):
        return F.l1_loss(first_gen_output, second_gen_output)


class CycleLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, cycled_gen_output: torch.Tensor, image: torch.Tensor, **kwargs):
        return F.l1_loss(cycled_gen_output, image)
