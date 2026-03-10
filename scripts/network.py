import torch
import torch.nn as nn
import timm


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model("efficientnetv2_s", pretrained=True, num_classes=0)

        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        return x