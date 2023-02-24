import copy
from math import ceil

import torch
from pytorch_lightning import LightningModule
from munch import Munch
from lpips_pytorch import lpips

from src.losses import BaseLoss
from src.models.utils import he_init, moving_average


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
        self.save_hyperparameters(logger=False, ignore=['models', 'optims', 'gen_criterion', 'disc_criterion'])

        self.models = models
        for model in self.models.values():
            model.apply(he_init)
        self.models_ema = copy.deepcopy(self.models)
        self.optims = optimizers
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion

        self.automatic_optimization = False
        self._init_diversity_loss_decay()

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.hparams.latent_dim).to(self.device)

    def sample_y(self, batch_size):
        return torch.randint(low=0, high=self.hparams.n_domains, size=(batch_size,)).to(self.device)

    def fake_image_step(self, image, style_code, **kwargs):
        gen_output = self.models.generator(image, style_code)
        return {'gen_output': gen_output}

    def fake_adversarial_step(self, gen_output, y_ref, **kwargs):
        fake_disc_output = self.models.discriminator(gen_output, y_ref)
        return {'fake_disc_output': fake_disc_output}

    def real_adversarial_step(self, image, y, **kwargs):
        real_disc_output = self.models.discriminator(image, y)
        return {'real_disc_output': real_disc_output}

    def style_reconstruction_step(self, gen_output, y_ref, **kwargs):
        encoded_style_code = self.models.style_encoder(gen_output, y_ref)
        return {
            'encoded_style_code': encoded_style_code
        }

    def style_diversification_step(self, image, gen_output, second_z, y_ref, image_ref2, use_latents=True, **kwargs):
        if use_latents:
            second_style_code = self.models.mapping_network(second_z, y_ref)
        else:
            second_style_code = self.models.style_encoder(image_ref2, y_ref)
        second_gen_output = self.models.generator(image, second_style_code)
        return {
            'first_gen_output': gen_output,
            'second_gen_output': second_gen_output,
        }

    def cycle_step(self, image, y, gen_output, **kwargs):
        s = self.models.style_encoder(image, y)
        cycled_gen_output = self.models.generator(gen_output, s)
        return {
            'cycled_gen_output': cycled_gen_output,
        }

    def get_style_code(self, y_ref, z, image_ref1=None, use_latents=True, **kwargs):
        if use_latents:
            style_code = self.models.mapping_network(z, y_ref)
        else:
            style_code = self.models.style_encoder(image_ref1, y_ref)
        return {'style_code': style_code}

    def init_step(self, image, **kwargs):
        y_ref = self.sample_y(image.size(0))
        z = self.sample_z(image.size(0))
        second_z = self.sample_z(image.size(0))
        return {'y_ref': y_ref, 'z': z, 'second_z': second_z}

    def discriminator_loss(self, batch, use_latents=True):
        batch['image'].requires_grad_()
        batch.update(self.real_adversarial_step(**batch))
        with torch.no_grad():
            batch.update(self.get_style_code(**batch))
            batch.update(self.fake_image_step(**batch))
        batch.update(self.fake_adversarial_step(**batch))
        losses = self.disc_criterion(**batch)
        suffix = '_latents' if use_latents else '_refs'
        losses = {key + suffix: value for key, value in losses.items()}
        return losses

    def generator_loss(self, batch, use_latents=True):
        batch.update(self.get_style_code(**batch))
        batch.update(self.fake_image_step(**batch))
        batch.update(self.fake_adversarial_step(**batch))

        batch.update(self.style_reconstruction_step(**batch))
        batch.update(self.style_diversification_step(**batch, use_latents=use_latents))
        batch.update(self.cycle_step(**batch))
        losses = self.gen_criterion(**batch)
        suffix = '_latents' if use_latents else '_refs'
        losses = {key + suffix: value for key, value in losses.items()}
        return losses

    def training_step(self, batch, batch_idx: int):
        batch.update(self.init_step(**batch))
        g_opt, d_opt, m_opt, s_opt = self.optimizers()

        # Discriminator
        latent_losses = self.discriminator_loss(batch, use_latents=True)
        self.log_dict({f'disc_train/{key}': value for key, value in latent_losses.items()}, prog_bar=True)

        d_opt.zero_grad()
        self.manual_backward(latent_losses['loss_latents'])
        d_opt.step()

        ref_losses = self.discriminator_loss(batch, use_latents=False)
        self.log_dict({f'disc_train/{key}': value for key, value in ref_losses.items()}, prog_bar=True)

        d_opt.zero_grad()
        self.manual_backward(ref_losses['loss_refs'])
        d_opt.step()

        # Generator
        latent_losses = self.generator_loss(batch, use_latents=True)
        self.log_dict({f'gen_train/{key}': value for key, value in latent_losses.items()}, prog_bar=True)

        g_opt.zero_grad()
        m_opt.zero_grad()
        s_opt.zero_grad()
        self.manual_backward(latent_losses['loss_latents'])
        g_opt.step()
        m_opt.step()
        s_opt.step()

        ref_losses = self.generator_loss(batch, use_latents=False)
        self.log_dict({f'gen_train/{key}': value for key, value in ref_losses.items()}, prog_bar=True)

        g_opt.zero_grad()
        self.manual_backward(ref_losses['loss_refs'])
        g_opt.step()

        for key in self.models:
            if key != 'discriminator':
                moving_average(self.models[key], self.models_ema[key])
        self._diversity_loss_weight_decay(batch['image'].size(0))

    def _init_diversity_loss_decay(self):
        self.ds_loss_idx = \
            [loss.__class__.__name__ for loss in self.gen_criterion.losses].index('StyleDiversificationLoss')
        self.init_ds_weight = self.gen_criterion.weights[self.ds_loss_idx]

    def _diversity_loss_weight_decay(self, batch_size):
        num_steps = self.trainer.max_epochs * ceil(len(self.trainer.train_dataloader.dataset) // batch_size)
        self.gen_criterion.weights[self.ds_loss_idx] -= self.init_ds_weight / num_steps
        self.log('style_diversity_weight', self.gen_criterion.weights[self.ds_loss_idx], on_step=True)

    def val_step(self, mode: str, batch, batch_idx: int):
        batch.update(self.init_step(**batch))
        batch.update(self.get_style_code(**batch))
        fake_images = self.models_ema.generator(batch['image'], batch['style_code'])
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
        return [self.optims[key](self.models[key].parameters()) for key in self.optims]


if __name__ == '__main__':
    StarGanV2LitModule(1, 1, Munch(), Munch(), BaseLoss(), BaseLoss())
