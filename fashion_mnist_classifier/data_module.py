from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataParams:
    dataset_dir: Path
    batch_size: int = 64
    num_workers: int = 2


class FashionMnistDataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams) -> None:
        super().__init__()
        self.params = params
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self) -> None:
        datasets.FashionMNIST(
            root=str(self.params.dataset_dir), train=True, download=True
        )
        datasets.FashionMNIST(
            root=str(self.params.dataset_dir), train=False, download=True
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = datasets.FashionMNIST(
                root=str(self.params.dataset_dir),
                train=True,
                download=False,
                transform=self.transform,
            )
            self.val_ds = datasets.FashionMNIST(
                root=str(self.params.dataset_dir),
                train=False,
                download=False,
                transform=self.transform,
            )
        if stage in (None, "test"):
            self.test_ds = datasets.FashionMNIST(
                root=str(self.params.dataset_dir),
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=int(self.params.batch_size),
            shuffle=True,
            num_workers=int(self.params.num_workers),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=int(self.params.batch_size),
            shuffle=False,
            num_workers=int(self.params.num_workers),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=int(self.params.batch_size),
            shuffle=False,
            num_workers=int(self.params.num_workers),
            pin_memory=True,
        )
