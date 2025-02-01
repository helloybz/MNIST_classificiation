from enum import StrEnum, auto
import os
from pathlib import Path

import click
import lightning.pytorch as pl
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam


class LIGHTNINGSTAGE(StrEnum):
    FIT = auto()
    TEST = auto()

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path = Path("./data_root"),
        global_mean: float = 0.3081,
        global_std: tuple[float] = (0.1037,),
        train_val_ratio: tuple[float] = (0.9, 0.1),
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.train_val_ratio = train_val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=global_mean,
                std=global_std,
            )
        ])
        self.target_transform = torch.nn.functional.one_hot

    def prepare_data(self) -> None:
        MNIST(
            root=self.data_root,
            train=True,
            download=True,
        )
        MNIST(
            root=self.data_root,
            train=False,
            download=True,
        )

    def setup(self, stage: LIGHTNINGSTAGE | None = None) -> None:
        if stage == LIGHTNINGSTAGE.FIT:
            train_dataset = MNIST(
                root=self.data_root,
                train=True,
                download=False,
                transform=self.transforms,
                target_transform=lambda x: self.target_transform(torch.tensor(x), 10).float(),
            )

            train_set_size = len(train_dataset) * self.train_val_ratio[0] / sum(self.train_val_ratio)
            train_set_size = int(train_set_size)
            valid_set_size = len(train_dataset) - train_set_size

            self.train_split, self.valid_split = random_split(
                dataset=train_dataset,
                lengths=[train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(int(os.environ.get("PL_GLOBAL_SEED", 42))),
            )

        if stage == LIGHTNINGSTAGE.TEST:
            self.test_split = MNIST(
                root=self.data_dir,
                train=False,
                download=False,
                transform=self.transforms,
                target_transform=lambda x: self.target_transform(torch.tensor(x), 10).float(),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    # @classmethod
    # def to_image(cls, tensors):
    #     tensors = tensors * cls.MNIST_STD + cls.MNIST_MEAN 
    #     tensors = tensors.view(-1, 28, 28)
    #     images = [transforms.ToPILImage()(tensor) for tensor in tensors]

    #     return images
class MNISTClassifier(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(128, 10)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        x, y = batch
        x = self.encoder(x)
        x = self.classifier(x)
        logit = x.softmax(-1)

        loss = torch.nn.functional.cross_entropy(logit, y)
         
        return loss
    
    def configure_optimizers(self):
        return Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
    
@click.command("train")
def train() -> None:
    data_module = MNISTDataModule()
    model = MNISTClassifier()

    trainer = pl.Trainer()
    trainer.fit(model, data_module)
    
if __name__ == "__main__":
    train()
