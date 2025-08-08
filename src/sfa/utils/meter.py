import math
from logging import getLogger

__all__ = ["AverageMeter"]


class AverageMeter:
    def __init__(self) -> None:
        self._value = 0
        self._count = 0
        self._logger = getLogger(__name__)

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            self._logger.warning(
                "Invalid value encountered in AverageMeter.add. Skipping..."
            )
            return
        self._value += x
        self._count += 1

    def compute(self) -> float:
        if self.zero_count:
            self._logger.warning(
                "Cannot compute average value when count is 0."
            )
            return float("inf")
        return self._value / self._count

    def reset(self) -> None:
        self._value = 0
        self._count = 0

    @property
    def zero_count(self) -> bool:
        return self._count == 0
