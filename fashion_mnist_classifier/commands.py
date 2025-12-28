from __future__ import annotations

from pathlib import Path

import fire

from fashion_mnist_classifier.data import DataConfig, get_dataloaders
from fashion_mnist_classifier.export_onnx import ExportConfig, export_to_onnx
from fashion_mnist_classifier.infer import (
    InferConfig,
    infer_from_image_path,
    infer_from_test_sample,
)
from fashion_mnist_classifier.train import TrainConfig, train_model
from fashion_mnist_classifier.utils import load_config, parse_overrides


class Commands:
    def smoke(self, config_name: str = "base", overrides: str | None = None) -> None:
        override_list = parse_overrides(overrides)
        config = load_config(config_name=config_name, overrides=override_list)

        dataset_dir = Path(str(config.paths.dataset_dir))
        data_config = DataConfig(
            batch_size=int(config.data.batch_size),
            num_workers=int(config.data.num_workers),
            dataset_dir=dataset_dir,
        )

        train_loader, _ = get_dataloaders(data_config)
        images_batch, labels_batch = next(iter(train_loader))

        print(f"images: {tuple(images_batch.shape)}")
        print(f"labels: {tuple(labels_batch.shape)}")
        print(f"labels_sample: {labels_batch[:10].tolist()}")

    def train(self, config_name: str = "base", overrides: str | None = None) -> None:
        override_list = parse_overrides(overrides)
        cfg = load_config(config_name=config_name, overrides=override_list)

        training_config = TrainConfig(
            seed=int(cfg.seed),
            device=str(cfg.device),
            epochs=int(cfg.train.epochs),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
            model_name=str(cfg.model.name),
            dropout=float(cfg.model.dropout),
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            dataset_dir=Path(cfg.paths.dataset_dir),
            checkpoint_dir=Path(cfg.paths.checkpoint_dir),
            metrics_dir=Path(cfg.paths.metrics_dir),
            onnx_dir=Path(cfg.paths.onnx_dir),
        )
        train_model(training_config, mlflow_cfg=cfg.mlflow, trainer_cfg=cfg.trainer)

    def export_onnx(
        self, config_name: str = "base", overrides: str | None = None
    ) -> None:
        override_list = parse_overrides(overrides)
        config = load_config(config_name=config_name, overrides=override_list)

        export_config = ExportConfig(
            checkpoint_path=Path(str(config.paths.checkpoint_dir)) / "best.pt",
            onnx_path=Path(str(config.paths.onnx_dir)) / "model.onnx",
            model_name=str(config.model.name),
            dropout=float(config.model.dropout),
            opset_version=17,
        )

        onnx_path = export_to_onnx(export_config)
        print(f"Saved ONNX model: {onnx_path}")

    def infer_onnx(
        self,
        config_name: str = "base",
        overrides: str | None = None,
        image_path: str | None = None,
        test_index: int = 0,
    ) -> None:
        override_list = parse_overrides(overrides)
        config = load_config(config_name=config_name, overrides=override_list)

        infer_config = InferConfig(
            onnx_path=Path(str(config.paths.onnx_dir)) / "model.onnx",
            dataset_dir=Path(str(config.paths.dataset_dir)),
        )

        if image_path is not None and image_path.strip():
            predicted_class, confidence, label_name = infer_from_image_path(
                infer_config, Path(image_path)
            )
            print(
                f"predicted_class={predicted_class} label={label_name} confidence={confidence:.4f}"
            )
            return

        true_label, predicted_class, confidence, label_name = infer_from_test_sample(
            infer_config, test_index=test_index
        )
        print(
            f"true_label={true_label} predicted_class={predicted_class} "
            f"label={label_name} confidence={confidence:.4f}"
        )
        return


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
