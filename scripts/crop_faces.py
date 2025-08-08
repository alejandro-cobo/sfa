#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sfa.datasets import DATASET_LIST
from sfa.datasets.config import get_anns_path, get_data_path
from sfa.utils.video import VideoReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=DATASET_LIST,
        help="Dataset name",
    )
    parser.add_argument(
        "--crop-size",
        "-c",
        type=int,
        help="Crop size [default: do not resize]",
    )
    parser.add_argument(
        "--bbox-scale",
        "-b",
        type=float,
        default=1.3,
        help="Bounding box scale factor [default: 1.3]",
    )
    args = parser.parse_args()
    return args


def _expand_bbox(bbox, scale) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    new_x = int(center_x - side / 2)
    new_y = int(center_y - side / 2)
    side = int(side)
    return (new_x, new_y, new_x + side, new_y + side)


def process_video(
    data_path: Path,
    video: str,
    crop_size: int | None,
    bbox_scale: float,
) -> None:
    out_dir = Path(data_path, "frames", video)
    if out_dir.exists():
        return

    video_path = Path(data_path, "videos", video + ".mp4")
    video_reader = VideoReader(video_path).start()

    bbox_path = Path(data_path, "annotations", video + "_bbox.json")
    with open(bbox_path, "r") as json_file:
        bbox_anns = json.load(json_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, bbox in bbox_anns.items():
        frame = video_reader.get(frame_id.split("_")[1])
        bbox = np.array(bbox)
        bbox = _expand_bbox(bbox, bbox_scale)
        frame_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_crop = Image.fromarray(frame_crop)
        frame_crop = frame_crop.crop(bbox)
        frame_crop = cv2.cvtColor(np.array(frame_crop), cv2.COLOR_RGB2BGR)
        crop_path = Path(out_dir, f"{frame_id}.png")
        cv2.imwrite(str(crop_path), frame_crop)

    video_reader.stop()


def main():
    args = parse_args()

    split_path = get_anns_path(args.dataset)
    with open(split_path, "r") as json_file:
        split_info = json.load(json_file)

    data_path = get_data_path(args.dataset)
    for idx, video in enumerate(split_info):
        print(f"[{idx + 1}/{len(split_info)}] Processing {video}")
        process_video(
            data_path=data_path,
            video=video,
            crop_size=args.crop_size,
            bbox_scale=args.bbox_scale,
        )


if __name__ == "__main__":
    main()
