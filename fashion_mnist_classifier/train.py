from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from fashion_mnist_classifier.data import DataConfig, get_dataloaders
from fashion_mnist_classifier.model import build_model


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


def train_model(config: TrainConfig) -> Dict[str, object]:
    set_global_seed(config.seed)

    device = torch.device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_dir=config.dataset_dir,
    )
    train_loader, test_loader = get_dataloaders(data_config)

    model = build_model(model_name=config.model_name, dropout=config.dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_macro_f1 = -1.0
    best_checkpoint_path = config.checkpoint_dir / "best.pt"

    for epoch_index in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []

        progress = tqdm(train_loader, desc=f"epoch {epoch_index + 1}/{config.epochs}", leave=False)
        for images_batch, labels_batch in progress:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images_batch)
            loss_value = loss_function(logits, labels_batch)
            loss_value.backward()
            optimizer.step()

            epoch_losses.append(float(loss_value.item()))
            progress.set_postfix(loss=float(np.mean(epoch_losses)))

        accuracy, macro_f1, matrix = evaluate_model(model, test_loader, device=device)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "model_name": config.model_name,
                    "dropout": config.dropout,
                    "state_dict": model.state_dict(),
                },
                str(best_checkpoint_path),
            )

        print(
            f"Epoch {epoch_index + 1}/{config.epochs}: "
            f"acc={accuracy:.4f} macro_f1={macro_f1:.4f} best_macro_f1={best_macro_f1:.4f}"
        )

    final_accuracy, final_macro_f1, final_matrix = evaluate_model(model, test_loader, device=device)

    metrics_payload: Dict[str, object] = {
        "final_accuracy": final_accuracy,
        "final_macro_f1": final_macro_f1,
        "best_macro_f1": best_macro_f1,
        "confusion_matrix": final_matrix.tolist(),
    }

    metrics_path = config.metrics_dir / "metrics.json"
    # файл сохранится локально, но в git не попадёт (из-за .gitignore)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "checkpoint_path": str(best_checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_payload,
    }
