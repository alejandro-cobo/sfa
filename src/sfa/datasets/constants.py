from pathlib import Path

__all__ = [
    "DATASET_LIST",
    "FAKE_CLASS",
    "FRAMES",
    "LABEL",
    "REAL_CLASS",
    "ROOT_PATH",
    "VIDEO",
]

ROOT_PATH: Path = Path(*Path(__file__).parts[:-4])
DATASET_LIST: tuple[str, ...] = (
    "celebdf",
    "deeperforensics",
    "dfdcp",
    "faceshifter",
)
REAL_CLASS: float = 0.0
FAKE_CLASS: float = 1.0
VIDEO: str = "video"
LABEL: str = "label"
FRAMES: str = "frames"
