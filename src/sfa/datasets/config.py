import json
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

from .constants import ROOT_PATH

__all__ = ["DatasetConfig", "get_anns_path", "get_data_path"]

LOGGER = getLogger(__name__)


def get_data_path(dataset_name: str) -> Path:
    data_config_path = ROOT_PATH / "data" / "data_paths.json"
    if not data_config_path.exists():
        raise FileNotFoundError(
            f"data_paths.json file not found in {data_config_path}. "
            "Configure dataset paths by running scripts/configure_data_paths.py"
        )
    with open(data_config_path, "r") as f:
        data_paths = json.load(f)
        try:
            return Path(data_paths[dataset_name])
        except KeyError:
            raise KeyError(
                f"{dataset_name} not found in data_paths.json: "
                f"{data_paths.keys()}\n"
            )


def get_anns_path(dataset_name: str) -> Path:
    return ROOT_PATH / "data" / dataset_name / "split_info.json"


@dataclass(slots=True)
class DatasetConfig:
    dataset_name: str = "faceforensics"
    split: str = "test"
    batch_size: int = 1
    image_size: int = 224
    num_frames: int = 16
    shuffle: bool = False
    num_workers: int = 4
    drop_last: bool = False
    pin_memory: bool = True
    transforms: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> "DatasetConfig":
        config = cls()
        for key, val in state_dict.items():
            if hasattr(config, key):
                setattr(config, key, val)
            else:
                LOGGER.warning(f"Received an invalid param: {key}")
        return config
