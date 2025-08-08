import time
from os import PathLike
from threading import Thread

import cv2
import numpy as np

__all__ = ["VideoReader"]


class VideoReader:
    def __init__(self, path: PathLike):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = {}
        self.thread = Thread(target=self._run)
        self.thread.daemon = True

    def __enter__(self) -> "VideoReader":
        self.start()
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.stop()

    def _run(self) -> None:
        for frame_id in range(self.num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames[frame_id] = frame

    def start(self) -> "VideoReader":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.thread.join()
        self.cap.release()

    def get(self, frame_id: int | str, wait_max_sec: float = 5) -> np.ndarray:
        frame_id = int(frame_id)
        start = time.time()
        while time.time() - start < wait_max_sec:
            if frame_id in self.frames:
                return self.frames[frame_id]
            time.sleep(0.1)
        raise RuntimeError(
            f"Could not read frame {self.path}:{frame_id} after waiting {wait_max_sec:.2f}s"
        )
