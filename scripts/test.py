#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch

from sfa.datasets import DATASET_LIST, DatasetConfig, get_dataloader
from sfa.models import SFA
from sfa.test.tester import DFDTester
from sfa.utils.random import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to config.json file",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        type=str,
        nargs="+",
        choices=DATASET_LIST,
        default=DATASET_LIST,
        help="Testing datasets [default: all]",
    )
    parser.add_argument(
        "--weights-path",
        "-w",
        type=str,
        help="Path to pre-trained weights checkpoint [default: auto-detect]",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size [default: 1]",
    )
    parser.add_argument(
        "--num-clips",
        "-c",
        type=int,
        help="Number of clips to load for each video file [default: all]",
    )
    parser.add_argument(
        "--sample-level",
        action="store_true",
        help="Report sample-level metrics instead of video-level metrics",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold [default: 0.5]",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save test results to files",
    )
    parser.add_argument(
        "--random-seed",
        "-r",
        type=int,
        default=0,
        help="Random seed [default: 0]",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_random_seed(args.random_seed)

    with open(args.config_path, "r") as f:
        exp_config = json.load(f)
    exp_path = Path(args.config_path).parent

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"- Device: {device}")

    model = SFA(**exp_config["model"])
    weights_path = args.weights_path
    if weights_path is None:
        try:
            ckpt_path = exp_path / "checkpoints"
            weights_path = next(ckpt_path.glob("*.pt"))
        except StopIteration:
            raise FileNotFoundError(
                f"Could not find any model checkpoint in {ckpt_path}"
            )
    weights_path = Path(weights_path)
    state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"- Loaded weights from path {weights_path}")

    val_config = DatasetConfig.from_dict(exp_config["val_dataset"])
    for dataset in args.datasets:
        data_config = DatasetConfig(
            dataset_name=dataset,
            batch_size=args.batch_size,
            image_size=val_config.image_size,
            num_frames=val_config.num_frames,
            transforms=val_config.transforms,
        )
        test_data = get_dataloader(data_config)
        print(f"- Dataset {dataset}: {len(test_data)} samples")

        if args.save:
            save_dir = Path(
                exp_path,
                "benchmark",
                dataset,
                "test",
            )
        else:
            save_dir = None
        tester = DFDTester(
            model=model,
            test_data=test_data,
            out_dir=save_dir,
            sample_level=args.sample_level,
            threshold=args.threshold,
            device=device,
        )
        tester.test()


if __name__ == "__main__":
    main()
