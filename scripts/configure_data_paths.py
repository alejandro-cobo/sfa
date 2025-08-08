#!/usr/bin/env python

import argparse
import hashlib
import json
from pathlib import Path

from sfa.datasets.constants import DATASET_LIST, ROOT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        help="Search recursively from this directory",
    )
    args = parser.parse_args()
    return args


def get_hash_dict() -> dict[str, str]:
    hash_dict = {}
    for dataset in DATASET_LIST:
        hash = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
        hash_dict[hash] = dataset
    return hash_dict


def main():
    args = parse_args()

    root_path = Path(args.path).expanduser().resolve()
    hash_dict = get_hash_dict()
    paths = {}
    for id_file in root_path.rglob(".dfd-db-id"):
        with open(id_file, "r") as f:
            hash = f.readline().strip()
        if hash in hash_dict:
            db_path = str(id_file.parent)
            dataset = hash_dict[hash]
            paths[dataset] = db_path
            print(f"Found location of dataset {dataset} in {db_path}")

    if len(paths) == 0:
        print(f"Could not find any dataset in {root_path}")
        return

    save_path = ROOT_PATH / "data" / "data_paths.json"
    with open(save_path, "w") as out_file:
        json.dump(paths, out_file, indent=4)
    print(f"Saved dataset paths to {save_path}")


if __name__ == "__main__":
    main()
