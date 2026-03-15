import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top:top + crop_size, left:left + crop_size]


def preprocess_policy_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
    cropped = center_crop_square(rgb)
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalized.transpose(2, 0, 1))
