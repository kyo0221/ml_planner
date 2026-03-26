import torch
import torch.nn as nn
import timm


class Network(nn.Module):
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2

    def __init__(self, num_branches=4):
        super().__init__()

        self.backbone = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=0)
        self.num_branches = num_branches
        feature_dim = self.backbone.num_features

        self.fc = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            dropout=self.LSTM_DROPOUT,
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.HIDDEN_SIZE, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            )
            for _ in range(num_branches)
        ])

    def forward(self, images, cmd):
        batch_size, seq_len, channels, height, width = images.shape
        frames = images.reshape(batch_size * seq_len, channels, height, width)

        features = self.backbone(frames)
        features = self.relu(self.fc(features))
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        x = lstm_out[:, -1, :]

        batch_size = x.size(0)
        action_indices = torch.argmax(cmd, dim=1)
        output = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        for idx, branch in enumerate(self.branches):
            mask = (action_indices == idx)
            if mask.any():
                output[mask] = branch(x[mask])

        return output
