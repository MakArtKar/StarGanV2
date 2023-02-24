from pytorch_lightning import Callback
from torchvision.utils import make_grid


class WandbImageLogging(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:
            fake_images = outputs['gen_output'][:8]
            fake_images = make_grid(fake_images)
            trainer.logger.log_image(key='generated_images', images=[(fake_images + 1) / 2])

            real_images = batch['image'][:8]
            real_images = make_grid(real_images)
            trainer.logger.log_image(key='original_images', images=[(real_images + 1) / 2])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            fake_images = outputs['fake_images'][:8]
            fake_images = make_grid(fake_images)
            trainer.logger.log_image(key='generated_images', images=[(fake_images + 1) / 2])

            real_images = outputs['real_images'][:8]
            real_images = make_grid(real_images)
            trainer.logger.log_image(key='original_images', images=[(real_images + 1) / 2])
