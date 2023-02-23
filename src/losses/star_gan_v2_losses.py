import torch
import torch.nn.functional as F

from src.losses.common_losses import BaseLoss


class AdversarialLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, real_disc_output: torch.Tensor, fake_disc_output: torch.Tensor, **kwargs):
        return F.binary_cross_entropy_with_logits(real_disc_output, torch.ones_like(real_disc_output)) + \
            F.binary_cross_entropy_with_logits(fake_disc_output, torch.zeros_like(fake_disc_output))


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


class R1RegularizationLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor, real_disc_output: torch.Tensor, **kwargs):
        grad = torch.autograd.grad(
            outputs=real_disc_output.sum(), inputs=image, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return (grad ** 2).view(image.size(0), -1).sum(1).mean(0) / 2
