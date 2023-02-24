from typing import Optional

import requests
from albumentations import ImageOnlyTransform
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from src.data.components.celeba_wrappers import WrappedCelebADatasetWithRefs


class CelebADataModule(LightningDataModule):
    LISTATTR_URL = """https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt"""

    def __init__(
            self,
            data_dir: str = 'data/celeba',
            batch_size: int = 8,
            num_workers: int = 8,
            pin_memory: int = True,
            train_transform: Optional[ImageOnlyTransform] = None,
            val_transform: Optional[ImageOnlyTransform] = None,
            test_iters: int = 100,
    ):
        super().__init__()

        self.data_train = self.data_val = self.data_test = None

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        open(f'list_attr_celeba.txt', 'wb').write(requests.get(self.LISTATTR_URL).content)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = \
                WrappedCelebADatasetWithRefs(self.hparams.data_dir, num_refs=2, transform=self.train_transform)
            self.data_val = \
                WrappedCelebADatasetWithRefs(self.hparams.data_dir, num_refs=0, transform=self.val_transform)
            truncated_size = min(self.hparams.test_iters * self.hparams.batch_size, len(self.data_val))
            self.data_val = self.data_test = Subset(self.data_val, range(truncated_size))

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )


if __name__ == '__main__':
    CelebADataModule()
