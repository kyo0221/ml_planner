#!/usr/bin/env python3

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ml_planner.image_utils import preprocess_policy_image
from ml_planner.models import ActionNormalizer, DiffusionPolicy


def load_scalar_csv(path: Path) -> float:
    with path.open('r', newline='') as f:
        return float(next(csv.reader(f))[0])


def load_action_csv(path: Path) -> float:
    with path.open('r', newline='') as f:
        return float(next(csv.reader(f))[1])


@dataclass
class Config:
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    weight_file: str
    n_obs_steps: int
    pred_horizon: int
    action_dim: int
    image_size: int
    diffusion_step_embed_dim: int
    global_cond_dim: int
    down_dims: tuple[int, ...]
    kernel_size: int
    n_groups: int
    num_train_timesteps: int
    num_inference_steps: int
    beta_schedule: str
    prediction_type: str
    clip_sample: bool
    weights_dir: Path
    logs_dir: Path
    device: torch.device

    @classmethod
    def load(cls, config_path: Path, package_root: Path):
        with config_path.open('r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        dataset_cfg = config_dict['dataset']
        vision_cfg = config_dict['vision']
        model_cfg = config_dict['model']
        diffusion_cfg = config_dict['diffusion']

        weights_dir = package_root / 'weights'
        logs_dir = package_root / 'runs'
        weights_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        return cls(
            epochs=int(config_dict['epochs']),
            batch_size=int(config_dict['batch_size']),
            learning_rate=float(config_dict['learning_rate']),
            num_workers=int(config_dict['num_workers']),
            weight_file=str(config_dict['weight_file']),
            n_obs_steps=int(dataset_cfg['n_obs_steps']),
            pred_horizon=int(dataset_cfg['pred_horizon']),
            action_dim=int(dataset_cfg['action_dim']),
            image_size=int(vision_cfg['image_size']),
            diffusion_step_embed_dim=int(model_cfg['diffusion_step_embed_dim']),
            global_cond_dim=int(model_cfg['global_cond_dim']),
            down_dims=tuple(model_cfg['down_dims']),
            kernel_size=int(model_cfg['kernel_size']),
            n_groups=int(model_cfg['n_groups']),
            num_train_timesteps=int(diffusion_cfg['num_train_timesteps']),
            num_inference_steps=int(diffusion_cfg['num_inference_steps']),
            beta_schedule=str(diffusion_cfg['beta_schedule']),
            prediction_type=str(diffusion_cfg['prediction_type']),
            clip_sample=bool(diffusion_cfg['clip_sample']),
            weights_dir=weights_dir,
            logs_dir=logs_dir,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )


class EpisodeSequenceDataset(Dataset):
    NUM_COMMANDS = 4

    def __init__(self, dataset_path: Path, n_obs_steps: int, pred_horizon: int, image_size: int):
        self.dataset_path = dataset_path
        self.n_obs_steps = n_obs_steps
        self.pred_horizon = pred_horizon
        self.image_size = image_size
        self.samples = []
        self._build_index()

    def _build_index(self) -> None:
        episode_dirs = sorted(path for path in self.dataset_path.glob('episode_*') if path.is_dir())
        if not episode_dirs:
            raise ValueError(f'No episode directories found in {self.dataset_path}')

        for episode_dir in episode_dirs:
            image_paths = sorted((episode_dir / 'images').glob('*.png'))
            if not image_paths:
                continue

            num_frames = len(image_paths)
            max_start = num_frames - self.pred_horizon + 1
            for current_idx in range(self.n_obs_steps - 1, max_start):
                self.samples.append((episode_dir, current_idx))

        if not self.samples:
            raise ValueError(f'No valid training windows found in {self.dataset_path}')

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, episode_dir: Path, frame_idx: int) -> torch.Tensor:
        image_path = episode_dir / 'images' / f'{frame_idx + 1:05d}.png'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Failed to read image: {image_path}')
        return preprocess_policy_image(image, self.image_size)

    def _load_command(self, episode_dir: Path, frame_idx: int) -> torch.Tensor:
        command_path = episode_dir / 'commands' / f'{frame_idx + 1:05d}.csv'
        command_idx = int(load_scalar_csv(command_path))
        command = torch.zeros(self.NUM_COMMANDS, dtype=torch.float32)
        command[command_idx] = 1.0
        return command

    def _load_actions(self, episode_dir: Path, frame_idx: int) -> torch.Tensor:
        actions = []
        for offset in range(self.pred_horizon):
            action_path = episode_dir / 'actions' / f'{frame_idx + offset + 1:05d}.csv'
            actions.append([load_action_csv(action_path)])
        return torch.tensor(actions, dtype=torch.float32)

    def __getitem__(self, idx: int):
        episode_dir, current_idx = self.samples[idx]
        obs_images = []
        for frame_idx in range(current_idx - self.n_obs_steps + 1, current_idx + 1):
            obs_images.append(self._load_image(episode_dir, frame_idx))

        return (
            torch.stack(obs_images, dim=0),
            self._load_command(episode_dir, current_idx),
            self._load_actions(episode_dir, current_idx),
        )


def build_noise_scheduler(config: Config) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type=config.prediction_type,
        clip_sample=config.clip_sample,
    )


def build_model(config: Config) -> DiffusionPolicy:
    return DiffusionPolicy(
        action_dim=config.action_dim,
        pred_horizon=config.pred_horizon,
        n_obs_steps=config.n_obs_steps,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        global_cond_dim=config.global_cond_dim,
        down_dims=config.down_dims,
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
    )


def fit_normalizer(dataloader: DataLoader) -> ActionNormalizer:
    actions = []
    for _, _, batch_actions in dataloader:
        actions.append(batch_actions)
    stacked_actions = torch.cat(actions, dim=0)
    normalizer = ActionNormalizer()
    normalizer.fit(stacked_actions)
    return normalizer


def save_checkpoint(
    config: Config,
    model: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    normalizer: ActionNormalizer,
    scheduler: DDPMScheduler,
    epoch: int,
    loss: float,
) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'normalizer': normalizer.state_dict(),
        'model_config': {
            'action_dim': config.action_dim,
            'pred_horizon': config.pred_horizon,
            'n_obs_steps': config.n_obs_steps,
            'diffusion_step_embed_dim': config.diffusion_step_embed_dim,
            'global_cond_dim': config.global_cond_dim,
            'down_dims': list(config.down_dims),
            'kernel_size': config.kernel_size,
            'n_groups': config.n_groups,
        },
        'scheduler_config': dict(scheduler.config),
        'train_config': {
            'num_inference_steps': config.num_inference_steps,
            'image_size': config.image_size,
        },
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, config.weights_dir / config.weight_file)


def train(dataset_path: Path) -> None:
    script_dir = Path(__file__).parent
    package_root = script_dir.parent
    config = Config.load(package_root / 'config' / 'train.yaml', package_root)

    dataset = EpisodeSequenceDataset(
        dataset_path=dataset_path,
        n_obs_steps=config.n_obs_steps,
        pred_horizon=config.pred_horizon,
        image_size=config.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    normalizer = fit_normalizer(DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers))
    noise_scheduler = build_noise_scheduler(config)
    model = build_model(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(config.logs_dir)

    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}')
        for obs_images, command, actions in progress:
            obs_images = obs_images.to(config.device, dtype=torch.float32)
            command = command.to(config.device, dtype=torch.float32)
            actions = normalizer.normalize(actions.to(config.device, dtype=torch.float32))

            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(obs_images, command, actions, noise_scheduler)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            progress.set_postfix(loss=f'{loss_value:.4f}')

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('loss/train', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(config, model, optimizer, normalizer, noise_scheduler, epoch, avg_loss)

    writer.close()


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 train.py <dataset_path>')
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f'Dataset path does not exist: {dataset_path}')
        sys.exit(1)

    train(dataset_path)


if __name__ == '__main__':
    main()
