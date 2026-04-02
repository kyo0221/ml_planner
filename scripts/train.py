#!/usr/bin/env python3

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import schedulefree
import torch
import torch.multiprocessing as mp
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ml_planner.models import ActionNormalizer, DiffusionPolicy
from scripts.utils.random_shadows_highlights import RandomShadows
from scripts.utils.slit_augment import SlitAugment


def load_scalar_csv(path):
    with path.open('r', newline='') as f:
        return float(next(csv.reader(f))[0])


def load_action_csv(path):
    with path.open('r', newline='') as f:
        return float(next(csv.reader(f))[1])


@dataclass
class Config:
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    weight_file: str
    pred_horizon: int
    action_dim: int
    image_size: int
    augment: dict
    randomshadow: dict
    weights_dir: Path
    logs_dir: Path
    device: torch.device

    @classmethod
    def load(cls, config_path, package_root):
        with config_path.open('r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        dataset_cfg = config_dict['dataset']
        vision_cfg = config_dict['vision']
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
            pred_horizon=int(dataset_cfg['pred_horizon']),
            action_dim=int(dataset_cfg['action_dim']),
            image_size=int(vision_cfg['image_size']),
            augment=config_dict.get('augment', {}),
            randomshadow=cls._load_randomshadow_config(config_dict.get('randomshadow', {})),
            weights_dir=weights_dir,
            logs_dir=logs_dir,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    @staticmethod
    def _load_randomshadow_config(config):
        required_keys = (
            'p',
            'high_ratio',
            'low_ratio',
            'left_low_ratio',
            'left_high_ratio',
            'right_low_ratio',
            'right_high_ratio',
        )

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f'Missing randomshadow config keys in train.yaml: {missing_keys}')

        parsed = dict(config)
        for key in required_keys[1:]:
            parsed[key] = tuple(parsed[key])
        return parsed


class MLDataset(Dataset):
    NUM_COMMANDS = 4
    OFFSET_DECAY_STEPS = 5

    def __init__(self, dataset_path, pred_horizon, image_size, augment_config, randomshadow_config):
        dataset_root = Path(dataset_path)
        self.episode_dirs = sorted([path for path in dataset_root.glob('episode*') if path.is_dir()])
        self.chunk_size = pred_horizon
        self.image_size = image_size
        self.use_slit_augment = bool(augment_config.get('crop', True))
        self.use_randomshadow = bool(augment_config.get('randomshadow', False))
        self.augmentor = SlitAugment()
        self.randomshadow = RandomShadows(**randomshadow_config)
        self.samples = []
        self.episode_actions = []
        self.episode_commands = []
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f'No training samples found in dataset: {dataset_root}')

    def __len__(self):
        return len(self.samples) * self._num_augmentations()

    def num_augmentations(self):
        return self._num_augmentations()

    def _num_augmentations(self):
        if self.use_slit_augment:
            return len(self.augmentor)
        return 1

    def _build_action_chunk(self, episode_index, frame_index):
        actions = self.episode_actions[episode_index]
        return [
            actions[frame_index + step]
            for step in range(self.chunk_size)
        ]

    def _apply_decayed_action_offset(self, action_chunk, action_offset):
        adjusted_actions = []
        for step, action in enumerate(action_chunk):
            decay_ratio = max(0.0, 1.0 - (step / self.OFFSET_DECAY_STEPS))
            adjusted_actions.append(action + (action_offset * decay_ratio))
        return adjusted_actions

    def _apply_slit_augment(self, image, angular_z, augment_idx):
        if not self.use_slit_augment:
            return image, 0.0

        augmented_image, augmented_angular_z = self.augmentor.get_augmented(image, angular_z, augment_idx)
        return augmented_image, augmented_angular_z - angular_z

    def _apply_randomshadow(self, image):
        if not self.use_randomshadow or self.randomshadow is None:
            return image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        augmented_pil = self.randomshadow(pil_image)
        augmented_rgb = np.array(augmented_pil)
        return cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

    def _preprocess_image(self, image):
        resized = cv2.resize(image[..., :3], (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return torch.from_numpy(normalized.transpose(2, 0, 1))

    def __getitem__(self, idx):
        sample_idx = idx // self._num_augmentations()
        augment_idx = idx % self._num_augmentations()
        return self.get_item(sample_idx, augment_idx)

    def get_item(self, sample_idx, augment_idx, apply_randomshadow=True):
        episode_index, current_idx, image_path = self.samples[sample_idx]
        action_chunk = self._build_action_chunk(episode_index, current_idx)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Failed to read image: {image_path}')

        image, action_offset = self._apply_slit_augment(image, action_chunk[0], augment_idx)
        if apply_randomshadow:
            image = self._apply_randomshadow(image)

        action_chunk = self._apply_decayed_action_offset(action_chunk, action_offset)

        actions = torch.tensor([[value] for value in action_chunk], dtype=torch.float32)
        image_tensor = self._preprocess_image(image)
        command = torch.zeros(self.NUM_COMMANDS, dtype=torch.float32)
        command_idx = self.episode_commands[episode_index][current_idx]
        command[command_idx] = 1.0
        return image_tensor, command, actions


    def _build_samples(self):
        if not self.episode_dirs:
            raise ValueError('No episode directories found in dataset path')

        samples = []
        for episode_dir in self.episode_dirs:
            image_paths = sorted((episode_dir / 'images').glob('*.png'))
            if not image_paths:
                continue

            valid_image_paths = []
            actions = []
            commands = []
            for image_path in image_paths:
                action_path = episode_dir / 'actions' / f'{image_path.stem}.csv'
                command_path = episode_dir / 'commands' / f'{image_path.stem}.csv'
                if not action_path.exists() or not command_path.exists():
                    continue

                valid_image_paths.append(image_path)
                actions.append(load_action_csv(action_path))
                commands.append(int(load_scalar_csv(command_path)))

            max_start = len(valid_image_paths) - self.chunk_size + 1
            if max_start <= 0:
                continue

            episode_index = len(self.episode_actions)
            self.episode_actions.append(actions)
            self.episode_commands.append(commands)

            for current_idx in range(max_start):
                samples.append((episode_index, current_idx, valid_image_paths[current_idx]))

        return samples


class SampleSplitDataset(Dataset):
    def __init__(self, dataset, sample_indices, apply_randomshadow):
        self.dataset = dataset
        self.sample_indices = sample_indices
        self.apply_randomshadow = apply_randomshadow

    def __len__(self):
        return len(self.sample_indices) * self.dataset.num_augmentations()

    def __getitem__(self, idx):
        local_sample_idx = idx // self.dataset.num_augmentations()
        augment_idx = idx % self.dataset.num_augmentations()
        sample_idx = self.sample_indices[local_sample_idx]
        return self.dataset.get_item(
            sample_idx,
            augment_idx,
            apply_randomshadow=self.apply_randomshadow,
        )


def build_fixed_range_normalizer(action_dim, action_min=-1.0, action_max=1.0):
    if action_max <= action_min:
        raise ValueError(f'Invalid action range: min={action_min}, max={action_max}')

    midpoint = (action_max + action_min) / 2.0
    half_range = (action_max - action_min) / 2.0
    mean = torch.full((action_dim,), midpoint, dtype=torch.float32)
    std = torch.full((action_dim,), half_range, dtype=torch.float32)
    return ActionNormalizer(mean=mean, std=std)


def split_sample_indices(total_samples, train_ratio=0.7):
    if total_samples < 2:
        raise ValueError('At least 2 samples are required to split into train_data and test_data')

    train_count = int(total_samples * train_ratio)
    train_count = max(1, min(total_samples - 1, train_count))
    shuffled_indices = torch.randperm(total_samples).tolist()
    return shuffled_indices[:train_count], shuffled_indices[train_count:]


def save_checkpoint(
    config,
    model,
    optimizer,
    normalizer,
    scheduler,
    epoch,
    loss,
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'normalizer': normalizer.state_dict(),
        'model_config': {
            'action_dim': config.action_dim,
            'pred_horizon': config.pred_horizon,
        },
        'scheduler_config': dict(scheduler.config),
        'train_config': {
            'image_size': config.image_size,
            'augment': config.augment,
            'randomshadow': config.randomshadow,
        },
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, config.weights_dir / config.weight_file)


def train(dataset_path):
    mp.set_sharing_strategy('file_system')

    script_dir = Path(__file__).parent
    package_root = script_dir.parent
    config = Config.load(package_root / 'config' / 'train.yaml', package_root)

    dataset = MLDataset(
        dataset_path=dataset_path,
        pred_horizon=config.pred_horizon,
        image_size=config.image_size,
        augment_config=config.augment,
        randomshadow_config=config.randomshadow,
    )
    train_indices, test_indices = split_sample_indices(len(dataset.samples), train_ratio=0.7)
    train_dataset = SampleSplitDataset(dataset, train_indices, apply_randomshadow=True)
    test_dataset = SampleSplitDataset(dataset, test_indices, apply_randomshadow=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f'Split dataset into train_data={len(train_indices)} samples and '
        f'test_data={len(test_indices)} samples (ratio 7:3)'
    )

    normalizer = build_fixed_range_normalizer(config.action_dim, action_min=-1.0, action_max=1.0)
    noise_scheduler = DDPMScheduler()
    model = DiffusionPolicy(
        action_dim=config.action_dim,
        pred_horizon=config.pred_horizon,
    ).to(config.device)
    optimizer = schedulefree.RAdamScheduleFree(model.parameters(), lr=config.learning_rate)
    optimizer.train()
    writer = SummaryWriter(config.logs_dir)

    best_test_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs} [train]')
        for obs_images, command, actions in progress:
            obs_images = obs_images.to(config.device, dtype=torch.float32).unsqueeze(1)
            command = command.to(config.device, dtype=torch.float32)
            actions = normalizer.normalize(actions.to(config.device, dtype=torch.float32))

            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(obs_images, command, actions, noise_scheduler)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_train_loss += loss_value
            progress.set_postfix(loss=f'{loss_value:.4f}')

        model.eval()
        optimizer.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            eval_progress = tqdm(test_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs} [test]')
            for obs_images, command, actions in eval_progress:
                obs_images = obs_images.to(config.device, dtype=torch.float32).unsqueeze(1)
                command = command.to(config.device, dtype=torch.float32)
                actions = normalizer.normalize(actions.to(config.device, dtype=torch.float32))

                loss = model.compute_loss(obs_images, command, actions, noise_scheduler)
                total_test_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_test_loss = total_test_loss / len(test_dataloader)
        writer.add_scalar('loss/train', avg_train_loss, epoch)
        writer.add_scalar('loss/test', avg_test_loss, epoch)
        print(
            f'Epoch {epoch + 1}/{config.epochs}, '
            f'train_loss={avg_train_loss:.6f}, test_loss={avg_test_loss:.6f}'
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_checkpoint(config, model, optimizer, normalizer, noise_scheduler, epoch, avg_test_loss)

        optimizer.train()

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
