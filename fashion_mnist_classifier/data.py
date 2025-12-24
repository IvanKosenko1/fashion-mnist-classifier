from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    batch_size: int
    num_workers: int
    dataset_dir: Path


def get_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=str(config.dataset_dir),
        train=True,
        download=True,
        transform=transform_pipeline,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(config.dataset_dir),
        train=False,
        download=True,
        transform=transform_pipeline,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
