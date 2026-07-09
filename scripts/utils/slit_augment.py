import math
from typing import Tuple

import numpy as np


class SlitAugment:
    CROP_HEIGHT = 288
    CROP_WIDTH = 288
    INPUT_HEIGHT = 288
    INPUT_WIDTH = 512
    CROP_RATIO = CROP_WIDTH / INPUT_WIDTH

    def __init__(self) -> None:
        max_left = self.INPUT_WIDTH - self.CROP_WIDTH
        self.crop_specs = (
            ("left", 0 / max_left, -0.4),
            ("center_left", 66 / max_left, -0.2),
            ("center", 112 / max_left, 0.0),
            ("center_right", 178 / max_left, 0.2),
            ("right", 224 / max_left, 0.4),
        )

    def get_augmented(self, image: np.ndarray, action: np.ndarray, augment_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        _, left_ratio, offset = self.crop_specs[augment_idx]
        width = image.shape[1]
        crop_width = round(width * self.CROP_RATIO)
        left = round(left_ratio * (width - crop_width))
        right = left + crop_width
        return image[:, left:right, :], rotate_path(action, offset)

    def __len__(self) -> int:
        return len(self.crop_specs)


def rotate_path(path: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = cos_a * path[:, 0] - sin_a * path[:, 1]
    y = sin_a * path[:, 0] + cos_a * path[:, 1]
    return np.stack([x, y], axis=1).astype(np.float32)
