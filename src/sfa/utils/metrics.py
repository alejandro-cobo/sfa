from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from ..datasets import FAKE_CLASS, REAL_CLASS
from .meter import AverageMeter

__all__ = ["Metrics"]


def cross_entropy_loss(targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    targets = targets.astype(np.float64)
    scores = scores.astype(np.float64)
    eps = np.finfo(scores.dtype).eps
    scores = np.clip(scores, eps, 1 - eps)
    return -(targets * np.log(scores) + (1 - targets) * np.log(1 - scores))


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    indices_min_diff = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[indices_min_diff] + fnr[indices_min_diff]) / 2
    return eer


class Metrics:
    def __init__(self, sample_level: bool = False, threshold: float = 0.5):
        self.sample_level = sample_level
        self.threshold = threshold
        self._scores = defaultdict(AverageMeter)
        self._targets = dict()

    def _validate_scores_and_targets(self) -> None:
        scores_keys = sorted(self._scores.keys())
        targets_keys = sorted(self._targets.keys())
        if scores_keys != targets_keys:
            raise RuntimeError("Mismatch between scores and targets")

    def add_score(self, video_id: str, value: float) -> None:
        key = video_id
        if self.sample_level:
            sample_id = 0
            key = video_id + f"_{sample_id}"
            while key in self._scores:
                sample_id += 1
                key = video_id + f"_{sample_id}"
        self._scores[key].add(value)

    def add_target(self, video_id: str, value: int) -> None:
        key = video_id
        if self.sample_level:
            sample_id = 0
            key = video_id + f"_{sample_id}"
            while key in self._targets:
                sample_id += 1
                key = video_id + f"_{sample_id}"
            self._targets[key] = value
            return

        if key not in self._targets:
            self._targets[key] = value
        elif value != self._targets[key]:
            raise RuntimeError(
                "Tried to add different target values for the same video "
                f"{key}: {self._targets[key]} and {value}."
            )

    @property
    def scores(self) -> np.ndarray:
        ordered_scores = dict(sorted(self._scores.items()))
        ordered_scores = [meter.compute() for meter in ordered_scores.values()]
        return np.array(ordered_scores).astype(float)

    @property
    def targets(self) -> np.ndarray:
        ordered_targets = dict(sorted(self._targets.items()))
        ordered_targets = list(ordered_targets.values())
        return np.array(ordered_targets).astype(int)

    @property
    def target_names(self) -> list[str]:
        return sorted(list(self._targets.keys()))

    def compute_metrics(self) -> dict[str, float]:
        self._validate_scores_and_targets()
        if len(self._targets) == 0:
            return {}

        targets = self.targets
        scores = self.scores
        predictions = (scores > self.threshold).astype(int)
        log_loss = cross_entropy_loss(targets, scores)
        return {
            "accuracy": float(accuracy_score(targets, predictions)),
            "ap": float(average_precision_score(targets, scores)),
            "auc": float(roc_auc_score(targets, scores)),
            "eer": float(equal_error_rate(targets, scores)),
            "log_loss": float(log_loss.mean()),
            "log_loss_real": float(log_loss[targets == REAL_CLASS].mean()),
            "log_loss_fake": float(log_loss[targets == FAKE_CLASS].mean()),
        }

    def reset(self) -> None:
        self._scores.clear()
        self._targets.clear()
