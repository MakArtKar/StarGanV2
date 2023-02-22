from typing import Optional

import requests
from albumentations import ImageOnlyTransform
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.celeba_wraper import WrappedCelebADataset


class CelebADataModule(LightningDataModule):
    LISTATTR_URL = """https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt"""

    def __init__(
            self,
            batch_size: int = 8,
            num_workers: int = 8,
            pin_memory: int = True,
            train_transform: Optional[ImageOnlyTransform] = None,
            val_transform: Optional[ImageOnlyTransform] = None,
    ):
        super().__init__()

        self.data_train = self.data_val = self.data_test = None

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        open('list_attr_celeba.txt', 'wb').write(requests.get(self.LISTATTR_URL).content)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = WrappedCelebADataset(transform=self.train_transform)
            self.data_val = WrappedCelebADataset(transform=self.val_transform)
            self.data_test = WrappedCelebADataset(transform=self.val_transform)

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
