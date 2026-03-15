import torch


class ActionNormalizer:
    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, actions: torch.Tensor) -> None:
        dims = tuple(range(actions.ndim - 1))
        self.mean = actions.mean(dim=dims)
        self.std = actions.std(dim=dims).clamp_min(self.eps)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.mean.to(actions.device)) / self.std.to(actions.device)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.std.to(actions.device) + self.mean.to(actions.device)

    def state_dict(self) -> dict:
        return {
            'mean': self.mean,
            'std': self.std,
            'eps': self.eps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.eps = state_dict['eps']
