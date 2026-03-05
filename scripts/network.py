import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_waypoints: int = 10):
        super(Network, self).__init__()
        self.num_waypoints = num_waypoints

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 23, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_waypoints * 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.mlp(x)
        return x.view(x.shape[0], self.num_waypoints, 2)


class ADELoss(nn.Module):
    def forward(self, pred, target):
        l2 = torch.linalg.norm(pred - target, dim=-1)
        return l2.mean()
