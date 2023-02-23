import torch
from pytorch_lightning import LightningModule
from munch import Munch
from lpips_pytorch import lpips

from src.losses import BaseLoss


class StarGanV2LitModule(LightningModule):
    def __init__(
            self,
            latent_dim,
            n_domains,
            models: Munch,
            optimizers: Munch,
            gen_criterion: BaseLoss,
            disc_criterion: BaseLoss,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['models', 'optimizers', 'gen_criterion', 'disc_criterion'])

        self.models = models
        self.optimizers = optimizers
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.hparams.latent_dim).to(self.device)

    def sample_y(self, batch_size):
        return torch.randint(low=0, high=self.hparams.n_domains, size=(batch_size,)).to(self.device)

    def adversarial_step(self, image, y, style_code, y_ref, **kwargs):
        real_disc_output = self.models.discriminator(image, y)
        gen_output = self.models.generator(image, style_code)
        fake_disc_output = self.models.discriminator(gen_output, y_ref)
        return {
            'real_disc_output': real_disc_output,
            'gen_output': gen_output,
            'fake_disc_output': fake_disc_output,
        }

    def style_reconstruction_step(self, gen_output, y_ref, **kwargs):
        encoded_style_code = self.models.style_encoder(gen_output, y_ref)
        return {
            'encoded_style_code': encoded_style_code
        }

    def style_diversification_step(self, image, y_ref, **kwargs):
        first_z = self.sample_z(image.size(0))
        second_z = self.sample_z(image.size(0))
        first_s = self.models.mapping_network(first_z, y_ref)
        second_s = self.models.mapping_network(second_z, y_ref)
        first_gen_output = self.models.generator(image, first_s)
        second_gen_output = self.models.generator(image, second_s)
        return {
            'first_gen_output': first_gen_output,
            'second_gen_output': second_gen_output,
        }

    def cycle_step(self, image, y, gen_output, **kwargs):
        s = self.models.style_encoder(image, y)
        cycled_gen_output = self.models.generator(gen_output, s)
        return {
            'cycled_gen_output': cycled_gen_output,
        }

    def init_step(self, batch):
        image, y = batch
        image.requires_grad_()
        z = self.sample_z(image.size(0))
        y_ref = self.sample_y(image.size(0))
        style_code = self.models.mapping_network(z, y_ref)
        result = {'image': image, 'y': y, 'z': z, 'y_ref': y_ref, 'style_code': style_code}
        return result

    def generator_loss(self, batch):
        batch.update(self.adversarial_step(**batch))
        batch.update(self.style_reconstruction_step(**batch))
        batch.update(self.style_diversification_step(**batch))
        batch.update(self.cycle_step(**batch))
        losses = self.gen_criterion(**batch)
        return losses

    def discriminator_loss(self, batch):
        batch.update(self.adversarial_step(**batch))
        loss = self.disc_criterion(**batch)
        return loss

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        batch = self.init_step(batch)
        if optimizer_idx == 0:
            losses = self.generator_loss(batch)
            for name, value in losses.items():
                self.log(f'generator_train/{name}', value, prog_bar=True)
            return losses
        elif optimizer_idx == 1:
            losses = self.discriminator_loss(batch)
            for name, value in losses.items():
                self.log(f'discriminator_train/{name}', value, prog_bar=True)
            return losses
        elif optimizer_idx == 2:
            losses = self.generator_loss(batch)
            return losses
        else:
            losses = self.generator_loss(batch)
            return losses

    def val_step(self, mode: str, batch, batch_idx: int):
        batch = self.init_step(batch)
        fake_images = self.models.generator(batch['image'], batch['style_code'])
        metrics = {
            'lpips': lpips(fake_images.detach().cpu(), batch['image'].cpu()).squeeze().item()
        }
        self.log(f'{mode}/lpips', metrics['lpips'], on_step=True, on_epoch=True, prog_bar=True)
        return {
            'metrics': metrics,
            'real_images': batch['image'],
            'fake_images': fake_images,
        }

    def validation_step(self, batch, batch_idx: int):
        return self.val_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx: int):
        return self.val_step('test', batch, batch_idx)

    def configure_optimizers(self):
        return [self.optimizers[key](self.models[key].parameters()) for key in self.optimizers]


if __name__ == '__main__':
    StarGanV2LitModule(1, 1, Munch(), Munch(), BaseLoss(), BaseLoss())
