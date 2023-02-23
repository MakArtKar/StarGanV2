from pytorch_lightning import Callback
from torchvision.utils import make_grid


class WandbImageLogging(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            fake_images = outputs['fake_images']
            fake_images = make_grid(fake_images)
            trainer.logger.log_image(key='generated_images', images=[fake_images])

            real_images = outputs['real_images']
            real_images = make_grid(real_images)
            trainer.logger.log_image(key='original_images', images=[real_images])
