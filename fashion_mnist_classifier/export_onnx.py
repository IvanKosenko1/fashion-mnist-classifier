from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from fashion_mnist_classifier.model import build_model


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: Path
    onnx_path: Path
    model_name: str
    dropout: float
    opset_version: int = 17


def export_to_onnx(config: ExportConfig) -> str:
    config.onnx_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(config.checkpoint_path), map_location="cpu")
    model: nn.Module = build_model(model_name=config.model_name, dropout=config.dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(config.onnx_path),
        opset_version=int(config.opset_version),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )

    return str(config.onnx_path)
