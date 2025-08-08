from typing import Any

import albumentations as alb
import numpy as np
from albumentations import pytorch as alb_pytorch
from albumentations.core.composition import TransformsSeqType

__all__ = ["VideoCompose", "load_transforms_from_dict"]


class VideoCompose:
    def __init__(
        self,
        transform: TransformsSeqType,
        num_frames: int,
        keypoints: bool = False,
        masks: bool = False,
    ):
        self.image_targets = ["image"] + [
            f"image{idx}" for idx in range(num_frames - 1)
        ]

        self.keypoint_targets = []
        if keypoints:
            self.keypoint_targets = ["keypoints"] + [
                f"keypoints{idx}" for idx in range(num_frames - 1)
            ]

        self.mask_targets = []
        if masks:
            self.mask_targets = ["mask"] + [
                f"mask{idx}" for idx in range(num_frames - 1)
            ]

        targets = {target: "image" for target in self.image_targets}
        targets.update(
            {target: "keypoints" for target in self.keypoint_targets}
        )
        targets.update({target: "mask" for target in self.mask_targets})
        keypoint_params = (
            alb.KeypointParams("xy", remove_invisible=False)
            if keypoints
            else None
        )
        self.transform = alb.Compose(
            transform,
            additional_targets=targets,
            keypoint_params=keypoint_params,
        )

    def __call__(
        self,
        frames: np.ndarray,
        keypoints: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> tuple[list, ...]:
        params = {
            target: frame for target, frame in zip(self.image_targets, frames)
        }
        if keypoints is not None:
            params.update(
                {
                    target: kp
                    for target, kp in zip(self.keypoint_targets, keypoints)
                }
            )
        if masks is not None:
            params.update(
                {target: mask for target, mask in zip(self.mask_targets, masks)}
            )
        transformed_data = self.transform(**params)

        transformed_frames = [
            transformed_data[target] for target in self.image_targets
        ]
        res = [transformed_frames]
        if keypoints is not None:
            transformed_landmarks = [
                transformed_data[target] for target in self.keypoint_targets
            ]
            res.append(transformed_landmarks)
        if masks is not None:
            transformed_masks = [
                transformed_data[target] for target in self.mask_targets
            ]
            res.append(transformed_masks)

        return tuple(res)


def load_transforms_from_dict(
    transforms_dict: dict[str, Any],
) -> list[alb.BasicTransform] | None:
    if transforms_dict is None or len(transforms_dict) == 0:
        return None
    transforms = []
    for name, params in transforms_dict.items():
        aug = _get_transform(name, **params)
        transforms.append(aug)
    return transforms


def _get_transform(name: str, **kwargs) -> alb.BasicTransform:
    for module in (alb, alb_pytorch):
        if hasattr(module, name):
            return getattr(module, name)(**kwargs)
    raise ValueError(f"Invalid augmentation name {name}")
