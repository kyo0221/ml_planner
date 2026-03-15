import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_b_16(weights=None)
        self.feature_dim = self.backbone.hidden_dim
        self.backbone.heads = nn.Identity()

    def forward(self, obs_images: torch.Tensor) -> torch.Tensor:
        batch_size, obs_steps, channels, height, width = obs_images.shape
        features = self.backbone(obs_images.reshape(batch_size * obs_steps, channels, height, width))
        return features.view(batch_size, obs_steps, self.feature_dim)
