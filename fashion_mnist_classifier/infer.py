from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms


FASHION_MNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass(frozen=True)
class InferConfig:
    onnx_path: Path
    dataset_dir: Path


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _preprocess_pil_image(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L").resize((28, 28))
    array_u8 = np.array(grayscale, dtype=np.float32)  # 0..255
    array_01 = array_u8 / 255.0
    normalized = (array_01 - 0.5) / 0.5  # как в train: Normalize((0.5,), (0.5,))
    batch = normalized[None, None, :, :]  # [1,1,28,28]
    return batch.astype(np.float32)


def infer_from_image_path(config: InferConfig, image_path: Path) -> Tuple[int, float, str]:
    session = ort.InferenceSession(str(config.onnx_path), providers=["CPUExecutionProvider"])

    image = Image.open(str(image_path))
    input_tensor = _preprocess_pil_image(image)

    outputs = session.run(None, {"images": input_tensor})
    logits = outputs[0]
    probabilities = _softmax(logits)

    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0, predicted_class])
    label_name = FASHION_MNIST_LABELS[predicted_class]
    return predicted_class, confidence, label_name


def infer_from_test_sample(config: InferConfig, test_index: int) -> Tuple[int, int, float, str]:
    session = ort.InferenceSession(str(config.onnx_path), providers=["CPUExecutionProvider"])

    transform_pipeline = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_dataset = datasets.FashionMNIST(
        root=str(config.dataset_dir),
        train=False,
        download=True,
        transform=transform_pipeline,
    )

    image_tensor, true_label = test_dataset[int(test_index)]
    input_tensor = image_tensor.unsqueeze(0).numpy().astype(np.float32)  # [1,1,28,28]

    outputs = session.run(None, {"images": input_tensor})
    logits = outputs[0]
    probabilities = _softmax(logits)

    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0, predicted_class])
    label_name = FASHION_MNIST_LABELS[predicted_class]
    return int(true_label), predicted_class, confidence, label_name
