import json
from datetime import datetime
from os import PathLike
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets import FRAMES, LABEL, VIDEO
from ..utils.metrics import Metrics

__all__ = ["DFDTester"]


class DFDTester:
    def __init__(
        self,
        model: nn.Module,
        test_data: DataLoader,
        sample_level: bool = False,
        threshold: float = 0.5,
        out_dir: PathLike | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.test_data = test_data
        self.out_dir = out_dir
        self.sample_level = sample_level
        self.device = device
        self._metrics = Metrics(
            sample_level=sample_level,
            threshold=threshold,
        )

    @torch.inference_mode()
    def test(self) -> None:
        self.model.eval()
        start = datetime.now()
        for batch_idx, batch in enumerate(self.test_data):
            frames = batch[FRAMES].to(self.device, non_blocking=True)
            scores = self.model(frames)["classifier"].sigmoid()
            for video, label, score in zip(batch[VIDEO], batch[LABEL], scores):
                self._metrics.add_score(video, score.item())
                self._metrics.add_target(video, label.item())

            if batch_idx % 10 == 0:
                elapsed = datetime.now() - start
                remaining = (
                    elapsed
                    * (len(self.test_data) - batch_idx - 1)
                    / (batch_idx + 1)
                )
                print(
                    f"\33[2K\r[{batch_idx + 1}/{len(self.test_data)}] elapsed time: {elapsed}, remaining time: {remaining}",
                    end="",
                )

        metrics = self._metrics.compute_metrics()
        print("\nResults:")
        for k, v in metrics.items():
            print(f"\t{k}: {v:.4g}")

        if self.out_dir is None:
            return

        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "sample" if self.sample_level else "video"
        scores_file = out_dir / f"scores_{suffix}.json"
        metrics_file = out_dir / f"metrics_{suffix}.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics in {metrics_file}")

        target_names = self._metrics.target_names
        scores = self._metrics.scores.tolist()
        targets = self._metrics.targets.tolist()
        scores = {
            target_names[idx]: {"target": targets[idx], "score": scores[idx]}
            for idx in range(len(target_names))
        }
        with open(scores_file, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"Saved scores in {scores_file}")
