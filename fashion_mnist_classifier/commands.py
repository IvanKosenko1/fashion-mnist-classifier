from __future__ import annotations

from pathlib import Path

import fire

from fashion_mnist_classifier.data import DataConfig, get_dataloaders
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


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
