from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ml_planner.image_utils import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class RecoveryAugmentConfig:
    lateral_pixel_offsets: tuple[float, ...]
    angular_correction_gain: float
    horizon_decay: float


class RecoveryAugment:
    def __init__(self, config: RecoveryAugmentConfig, image_size: int):
        self.config = config
        self.image_size = image_size
        self.offsets = tuple(max(-1.0, min(1.0, float(offset))) for offset in config.lateral_pixel_offsets)
        if not self.offsets:
            self.offsets = (0.0,)

    def __len__(self) -> int:
        return len(self.offsets)

    def _crop_with_offset(self, image: np.ndarray, offset_ratio: float) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        center_left = (width - crop_size) / 2.0
        max_shift = width - crop_size
        target_left = center_left + (offset_ratio * center_left)
        left = int(np.clip(round(target_left), 0, max_shift))
        cropped = rgb[top:top + crop_size, left:left + crop_size]
        resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        applied_offset_px = float(left - center_left)
        return normalized.transpose(2, 0, 1), applied_offset_px

    def _action_correction(self, horizon: int, offset_px: float) -> torch.Tensor:
        if horizon == 1:
            weights = torch.ones(1, dtype=torch.float32)
        else:
            weights = torch.linspace(1.0, self.config.horizon_decay, steps=horizon, dtype=torch.float32)
        correction = self.config.angular_correction_gain * float(offset_px)
        return weights * correction

    def apply(self, images: list[np.ndarray], actions: torch.Tensor, augment_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset_ratio = self.offsets[augment_idx]
        augmented_images = []
        applied_offset_px = 0.0
        for image in images:
            augmented_image, applied_offset_px = self._crop_with_offset(image, offset_ratio)
            augmented_images.append(torch.from_numpy(augmented_image))

        augmented_actions = actions.clone()
        correction = self._action_correction(actions.shape[0], applied_offset_px).unsqueeze(-1)
        augmented_actions[:, 0:1] = augmented_actions[:, 0:1] + correction

        return torch.stack(augmented_images, dim=0), augmented_actions
