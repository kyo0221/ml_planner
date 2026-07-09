import torch
import torch.nn as nn
import timm


class Network(nn.Module):
    def __init__(self, num_branches=4, chunk_size=1):
        super().__init__()

        self.backbone = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=0)
        self.num_branches = num_branches
        self.chunk_size = chunk_size
        feature_dim = self.backbone.num_features

        self.fc = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU(inplace=True)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, chunk_size * 2)
            )
            for _ in range(num_branches)
        ])

    def forward(self, x, cmd):
        x = self.backbone(x)
        x = self.relu(self.fc(x))

        batch_size = x.size(0)
        action_indices = torch.argmax(cmd, dim=1)
        output = torch.zeros(batch_size, self.chunk_size * 2, device=x.device, dtype=x.dtype)

        for idx, branch in enumerate(self.branches):
            mask = (action_indices == idx)
            if mask.any():
                output[mask] = branch(x[mask])

        return output.view(batch_size, self.chunk_size, 2)
