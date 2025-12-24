from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_name: str, overrides: Optional[Iterable[str]] = None) -> DictConfig:
    overrides_list: List[str] = list(overrides) if overrides is not None else []
    config_dir = get_project_root() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed_config = compose(config_name=config_name, overrides=overrides_list)
    return composed_config


def parse_overrides(overrides: Optional[str]) -> List[str]:
    if overrides is None:
        return []
    cleaned = overrides.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]
