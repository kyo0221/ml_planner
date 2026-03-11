from typing import Iterator, Tuple

import numpy as np


class SlitAugment:
    CROP_HEIGHT = 288
    CROP_WIDTH = 288
    INPUT_HEIGHT = 288
    INPUT_WIDTH = 512

    def __init__(self) -> None:
        self.crop_specs = (
            ("left", 0, 0.4),
            ("center_left", 66, 0.2),
            ("center", 112, 0.0),
            ("center_right", 178, -0.2),
            ("right", 224, -0.4),
        )

    def get_augmented(self, image: np.ndarray, angular_z: float, augment_idx: int) -> Tuple[np.ndarray, float]:
        _, left, offset = self.crop_specs[augment_idx]
        right = left + self.CROP_WIDTH
        return image[:, left:right, :], angular_z + offset

    def __len__(self) -> int:
        return len(self.crop_specs)
