import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .augmentation import VideoCompose, load_transforms_from_dict
from .config import DatasetConfig, get_anns_path, get_data_path
from .constants import (
    FAKE_CLASS,
    FRAMES,
    LABEL,
    REAL_CLASS,
    VIDEO,
)

__all__ = ["get_dataloader"]


class DeepfakeDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        if config.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid data split {config.split}")

        self.image_size = config.image_size
        self.num_frames = config.num_frames
        self.transforms = None
        if config.transforms is not None:
            self.transforms = VideoCompose(
                load_transforms_from_dict(config.transforms),
                num_frames=self.num_frames,
            )
        self.logger = logging.getLogger(__name__)

        self.data = self._load_data(config.dataset_name)
        if config.dataset_name == "deeperforensics":
            self.data += self._load_data("faceshifter", filter_label="real")

    def _load_data(
        self,
        dataset_name: str,
        filter_label: str | None = None,
    ) -> list[dict[str, Any]]:
        split_path = get_anns_path(dataset_name)
        with open(split_path, "r") as f:
            split_data = json.load(f)

        data_path = get_data_path(dataset_name)
        data = []
        for video, label in split_data.items():
            if filter_label is not None and label != filter_label:
                continue
            frames_dir = Path(data_path, "frames", video)
            label = REAL_CLASS if label == "real" else FAKE_CLASS
            data += self._process_anns(frames_dir, video, label)
        return data

    def _process_anns(
        self,
        frames_dir: Path,
        video: str,
        label: float,
    ) -> list[dict[str, Any]]:
        frames_files = sorted([file for file in frames_dir.glob("*.png")])
        if len(frames_files) < self.num_frames:
            self.logger.warning(
                f"Not enough frames, skipping {frames_dir} ({len(frames_files)} < {self.num_frames})"
            )
            return []

        data = []
        frames = []
        for frame in frames_files:
            if len(frames) < self.num_frames:
                frames.append(frame)
            else:
                data.append(
                    {
                        "video": video,
                        "label": label,
                        "frames_list": frames,
                    }
                )
                frames = [frame]
        return data

    def _read_frames(self, frames_list: list[Path]) -> np.ndarray:
        frames = []
        dsize = (self.image_size, self.image_size)
        for frame_path in frames_list:
            image = cv2.imread(str(frame_path))
            if image is None:
                raise FileNotFoundError(
                    f"Image file does not exist: {frame_path}"
                )
            height, width = image.shape[:2]
            scale = self.image_size / max(height, width)
            interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            image = cv2.resize(image, dsize, interpolation=interpolation)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_sample = self.data[index]
        video = data_sample["video"]
        label = data_sample["label"]
        frames_list = data_sample["frames_list"]

        frames = self._read_frames(frames_list)
        if self.transforms is not None:
            frames = np.array(self.transforms(frames=frames)[0])
        frames = frames.transpose(1, 0, 2, 3).copy()  # t c h w -> c t h w

        return {
            VIDEO: video,
            LABEL: label,
            FRAMES: frames,
        }

    def __len__(self) -> int:
        return len(self.data)


def get_dataloader(config: DatasetConfig) -> DataLoader:
    dataset = DeepfakeDataset(config=config)
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
