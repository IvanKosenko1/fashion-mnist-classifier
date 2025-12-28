from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from fashion_mnist_classifier.model import build_model


class LitFashionMNIST(L.LightningModule):
    def __init__(self, model_name: str, dropout: float, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_name=model_name, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_acc = MulticlassAccuracy(num_classes=10)
        self.val_f1 = MulticlassF1Score(num_classes=10, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/macro_f1", f1, prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
