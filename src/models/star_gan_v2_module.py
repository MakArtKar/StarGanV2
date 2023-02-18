import torch
from pytorch_lightning import LightningModule
from munch import Munch

from src.losses import BaseLoss


class StarGanV2LitModule(LightningModule):
    def __init__(
            self,
            latent_dim,
            n_domains,
            models: Munch,
            optimizers: Munch,
            criterion: BaseLoss,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.models = models
        self.optimizers = optimizers
        self.criterion = criterion

    def sample_z(self, batch_size: int):
        return torch.randn(batch_size, self.hparams.latent_dim)

    def adversarial_step(self, x, y, s_ref, y_ref, **kwargs):
        real_disc_output = self.models.discriminator(x, y)
        gen_output = self.models.generator(x, s_ref)
        fake_disc_output = self.models.discriminator(gen_output, y_ref)
        return {
            'real_disc_output': real_disc_output,
            'gen_output': gen_output,
            'fake_disc_output': fake_disc_output,
        }

    def style_reconstruction_step(self, gen_output, y_ref, **kwargs):
        encoded_s_ref = self.models.style_encoder(gen_output, y_ref)
        return {
            'encoded_s_ref': encoded_s_ref
        }

    def style_diversification_step(self, x, y_ref, **kwargs):
        first_z = self.sample_z(x.size(0))
        second_z = self.sample_z(x.size(0))
        first_s = self.models.mapping_network(first_z, y_ref)
        second_s = self.models.mapping_network(second_z, y_ref)
        first_gen_output = self.models.generator(x, first_s)
        second_gen_output = self.models.generator(x, second_s)
        return {
            'first_gen_output': first_gen_output,
            'second_gen_output': second_gen_output,
        }

    def cycle_step(self, x, y, gen_output, **kwargs):
        s = self.style_encoder(x, y)
        cycled_gen_output = self.models.generator(gen_output, s)
        return {
            'cycled_gen_output': cycled_gen_output,
        }

    def init_step(self, batch):
        x, y = batch
        z = self.sample_z(x.size(0))
        y_ref = torch.randint(low=0, high=self.hparams.n_domains, size=(x.size(0),))
        s_ref = self.models.mapping_network(z, y_ref)
        result = {'x': x, 'y': y, 'z': z, 'y_ref': y_ref, 's_ref': s_ref}
        return result

    def step(self, batch):
        result = self.init_step(batch)
        result.update(
            self.adversarial_step(**result)
        )
        result.update(
            self.style_reconstruction_step(**result)
        )
        result.update(
            self.style_diversification_step(**result)
        )
        result.update(
            self.cycle_step(**result)
        )
        loss = self.criterion(**result)
        return loss, result

    def training_step(self, batch, batch_idx: int):
        loss, result = self.step(batch)
        return {'loss': loss, 'result': result}

    def configure_optimizers(self):
        return [self.optimizers[key](self.models[key].parameters()) for key in self.optimizers]


if __name__ == '__main__':
    StarGanV2LitModule(1, 1, Munch(), Munch(), BaseLoss())
