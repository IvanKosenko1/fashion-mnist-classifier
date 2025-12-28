from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn.functional as functional
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


@dataclass(frozen=True)
class ModelParams:
    num_classes: int = 10
    dropout: float = 0.1


class SimpleCnn(nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=params.dropout),
            nn.Linear(128, params.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        return self.classifier(x)


class FashionMnistLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        model_params: ModelParams | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_params"])

        self.model_params = model_params or ModelParams()
        self.model = SimpleCnn(self.model_params)

        self.train_acc = MulticlassAccuracy(num_classes=self.model_params.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.model_params.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.model_params.num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = functional.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        logits = self(images)
        loss = functional.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        logits = self(images)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
