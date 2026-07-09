#!/usr/bin/env python3

import yaml
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from navvla_dataset import load_navvla_episode
from navvla_split import ensure_split
from network import Network
from utils.random_shadows_highlights import RandomShadows
from utils.slit_augment import SlitAugment


class MLDataset(Dataset):
    NUM_BRANCHES = 4

    def __init__(self, dataset_specs: List[Dict], chunk_size: int, augment_config: Dict, randomshadow_config: Dict,
                 split_ratio: float = 0.8, split_seed: Optional[int] = None):
        self.chunk_size = chunk_size
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.use_slit_augment = bool(augment_config.get('crop', True))
        self.use_randomshadow = bool(augment_config.get('randomshadow', False))
        self.augmentor = SlitAugment()
        self.randomshadow = RandomShadows(**randomshadow_config)
        self.episode_images: List[List[Path]] = []
        self.episode_actions: List[List[np.ndarray]] = []
        self.episode_commands: List[List[int]] = []
        self.samples = self._build_samples(dataset_specs)

        if not self.samples:
            raise ValueError(f'No training samples found in dataset_specs: {dataset_specs}')

    def __len__(self):
        return len(self.samples) * self._num_augmentations()

    def __getitem__(self, idx):
        sample_idx = idx // self._num_augmentations()
        augment_idx = idx % self._num_augmentations()
        return self.get_item(sample_idx, augment_idx)

    def get_item(self, sample_idx: int, augment_idx: int, apply_randomshadow: bool = True):
        episode_index, frame_index = self.samples[sample_idx]
        image_path = self.episode_images[episode_index][frame_index]
        action = self.episode_actions[episode_index][frame_index]
        command = self.episode_commands[episode_index][frame_index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.use_slit_augment:
            image, action = self.augmentor.get_augmented(image, action, augment_idx)
        if apply_randomshadow and self.use_randomshadow:
            image = self._apply_randomshadow(image)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        action_tensor = torch.tensor(action, dtype=torch.float32)

        command_tensor = torch.zeros(self.NUM_BRANCHES, dtype=torch.float32)
        command_tensor[int(command)] = 1.0

        return image_tensor, action_tensor, command_tensor

    def _num_augmentations(self) -> int:
        return len(self.augmentor) if self.use_slit_augment else 1

    def _apply_randomshadow(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        augmented_pil = self.randomshadow(pil_image)
        augmented_rgb = np.array(augmented_pil)
        return cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

    def _resolve_episode_names(self, dataset_root: Path, episodes: Optional[List[str]]) -> List[str]:
        if episodes is not None:
            return list(episodes)
        train_names, _ = ensure_split(dataset_root, split=self.split_ratio, seed=self.split_seed)
        return train_names

    def _build_samples(self, dataset_specs: List[Dict]):
        samples = []
        for spec in dataset_specs:
            dataset_root = Path(spec['path'])
            default_command = spec.get('default_command')
            episode_names = self._resolve_episode_names(dataset_root, spec.get('episodes'))

            for episode_name in episode_names:
                episode_dir = dataset_root / episode_name
                image_paths, actions, commands = load_navvla_episode(episode_dir, default_command, self.chunk_size)

                episode_index = len(self.episode_actions)
                self.episode_images.append(image_paths)
                self.episode_actions.append(actions)
                self.episode_commands.append(commands)

                for frame_index in range(len(actions)):
                    samples.append((episode_index, frame_index))

        return samples


class Config:
    def __init__(self, config_path, package_root):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.epochs = config_dict['epochs']
        self.batch_size = config_dict['batch_size']
        self.learning_rate = config_dict['learning_rate']
        self.num_workers = config_dict['num_workers']
        self.weight_file = config_dict['weight_file']
        self.chunk_size = int(config_dict['chunk_size'])
        self.augment = config_dict.get('augment', {})
        self.randomshadow = self._load_randomshadow_config(config_dict.get('randomshadow', {}))

        with open(package_root / 'config' / config_dict['dataset_config'], 'r') as f:
            dataset_dict = yaml.safe_load(f)
        self.split_ratio = dataset_dict.get('split_ratio', 0.8)
        self.split_seed = dataset_dict.get('split_seed')
        self.dataset_entries = dataset_dict['datasets']

        self.weights_dir = package_root / 'weights'
        self.logs_dir = package_root / 'runs'
        self.weights_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda')

    def _load_randomshadow_config(self, config: Dict) -> Dict:
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


def build_dataset_specs(dataset_entries: List[Dict], split_ratio: float, split_seed: Optional[int]):
    train_specs = []
    test_specs = []
    for entry in dataset_entries:
        dataset_root = Path(entry['path'])
        train_names, test_names = ensure_split(dataset_root, split=split_ratio, seed=split_seed)
        default_command = entry.get('default_command')
        train_specs.append({'path': entry['path'], 'default_command': default_command, 'episodes': train_names})
        test_specs.append({'path': entry['path'], 'default_command': default_command, 'episodes': test_names})
    return train_specs, test_specs


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = Network(chunk_size=config.chunk_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss = nn.MSELoss()
        self.writer = SummaryWriter(config.logs_dir)

    def train(self, train_dataloader, test_dataloader):
        self.model.to(self.config.device)
        best_test_loss = float('inf')

        for epoch in range(self.config.epochs):
            self.model.train()
            total_train_loss = 0.0

            for image, action, command in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs} [train]'):
                image = image.to(self.config.device)
                action = action.to(self.config.device)
                command = command.to(self.config.device)

                self.optimizer.zero_grad()
                outputs = self.model(image, command)
                loss = self.loss(outputs, action)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            self.model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for image, action, command in tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs} [test]'):
                    image = image.to(self.config.device)
                    action = action.to(self.config.device)
                    command = command.to(self.config.device)

                    outputs = self.model(image, command)
                    loss = self.loss(outputs, action)
                    total_test_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_test_loss = total_test_loss / len(test_dataloader)
            self.writer.add_scalar('train_loss', avg_train_loss, epoch)
            self.writer.add_scalar('test_loss', avg_test_loss, epoch)
            print(
                f'Epoch [{epoch+1}/{self.config.epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}'
            )

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.jit.script(self.model).save(self.config.weights_dir / self.config.weight_file)

        self.writer.close()


def main():
    script_dir = Path(__file__).parent
    package_root = script_dir.parent
    config_path = package_root / 'config' / 'train.yaml'
    config = Config(config_path, package_root)

    train_specs, test_specs = build_dataset_specs(config.dataset_entries, config.split_ratio, config.split_seed)

    test_augment = {**config.augment, 'randomshadow': False}
    train_dataset = MLDataset(train_specs, config.chunk_size, config.augment, config.randomshadow)
    test_dataset = MLDataset(test_specs, config.chunk_size, test_augment, config.randomshadow)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print(
        f'Split dataset into train_data={len(train_dataset.samples)} samples and '
        f'test_data={len(test_dataset.samples)} samples'
    )

    trainer = Trainer(config)
    trainer.train(train_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
