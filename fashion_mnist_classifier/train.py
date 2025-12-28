from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from fashion_mnist_classifier.data import DataConfig, get_dataloaders
from fashion_mnist_classifier.lit_module import LitFashionMNIST


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    model_name: str
    dataset_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    batch_size: int
    num_workers: int


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    model.eval()

    all_true_labels: list[int] = []
    all_pred_labels: list[int] = []

    with torch.no_grad():
        for images_batch, labels_batch in data_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(images_batch)
            predicted = torch.argmax(logits, dim=1)

            all_true_labels.extend(labels_batch.cpu().tolist())
            all_pred_labels.extend(predicted.cpu().tolist())

    accuracy = float(accuracy_score(all_true_labels, all_pred_labels))
    macro_f1 = float(f1_score(all_true_labels, all_pred_labels, average="macro"))
    matrix = confusion_matrix(all_true_labels, all_pred_labels)

    return accuracy, macro_f1, matrix


def train_model(config: TrainConfig, mlflow_cfg, trainer_cfg) -> None:
    seed_everything(config.seed, workers=True)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_dir=config.dataset_dir,
    )
    train_loader, test_loader = get_dataloaders(data_cfg)

    mlf_logger = MLFlowLogger(
        tracking_uri=str(mlflow_cfg.tracking_uri),
        experiment_name=str(mlflow_cfg.experiment_name),
        run_name=(
            None if mlflow_cfg.run_name in (None, "null") else str(mlflow_cfg.run_name)
        ),
    )

    ckpt = ModelCheckpoint(
        dirpath=str(config.checkpoint_dir),
        monitor="val/macro_f1",
        mode="max",
        save_top_k=1,
        filename="best",
    )

    lit = LitFashionMNIST(
        model_name=config.model_name,
        dropout=config.dropout,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator=str(trainer_cfg.accelerator),
        devices=trainer_cfg.devices,
        precision=trainer_cfg.precision,
        log_every_n_steps=int(trainer_cfg.log_every_n_steps),
        logger=mlf_logger,
        callbacks=[ckpt],
        enable_checkpointing=True,
    )

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=test_loader)
