#!/usr/bin/env python3

import csv
import yaml
import sys
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

from network import Network
from utils.random_shadows_highlights import RandomShadows
from utils.slit_augment import SlitAugment


class MLDataset(Dataset):
    NUM_BRANCHES = 4
    OFFSET_DECAY_STEPS = 5

    def __init__(self, dataset_path: str, chunk_size: int, augment_config: Dict, randomshadow_config: Dict):
        dataset_root = Path(dataset_path)
        self.episode_dirs = sorted([path for path in dataset_root.glob('episode*') if path.is_dir()])
        self.chunk_size = chunk_size
        self.use_slit_augment = bool(augment_config.get('crop', True))
        self.use_randomshadow = bool(augment_config.get('randomshadow', False))
        self.augmentor = SlitAugment()
        self.randomshadow: Optional[RandomShadows] = None
        self.randomshadow = RandomShadows(**randomshadow_config)
        self.episode_actions: List[List[float]] = []
        self.episode_commands: List[List[int]] = []
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f'No training samples found in dataset: {dataset_root}')

    def __len__(self):
        return len(self.samples) * self._num_augmentations()

    def __getitem__(self, idx):
        sample_idx = idx // self._num_augmentations()
        augment_idx = idx % self._num_augmentations()
        return self.get_item(sample_idx, augment_idx)

    def get_item(self, sample_idx: int, augment_idx: int, apply_randomshadow: bool = True):
        episode_index, frame_index, img_file = self.samples[sample_idx]

        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        action_chunk = self._build_action_chunk(episode_index, frame_index)
        command = self.episode_commands[episode_index][frame_index]

        image, action_offset = self._apply_slit_augment(image, action_chunk[0], augment_idx)
        action_chunk = self._apply_decayed_action_offset(action_chunk, action_offset)
        if apply_randomshadow:
            image = self._apply_randomshadow(image)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        action_tensor = torch.tensor(action_chunk, dtype=torch.float32)

        command_tensor = torch.zeros(self.NUM_BRANCHES, dtype=torch.float32)
        command_tensor[int(command)] = 1.0
        
        return image_tensor, action_tensor, command_tensor

    def num_augmentations(self) -> int:
        return self._num_augmentations()

    def _num_augmentations(self) -> int:
        if self.use_slit_augment:
            return len(self.augmentor)
        return 1

    def _apply_slit_augment(self, image: np.ndarray, angular_z: float, augment_idx: int):
        if not self.use_slit_augment:
            return image, 0.0
        image, augmented_angular_z = self.augmentor.get_augmented(image, angular_z, augment_idx)
        return image, augmented_angular_z - angular_z

    def _apply_randomshadow(self, image: np.ndarray) -> np.ndarray:
        if not self.use_randomshadow or self.randomshadow is None:
            return image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        augmented_pil = self.randomshadow(pil_image)
        augmented_rgb = np.array(augmented_pil)
        return cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

    def _build_action_chunk(self, episode_index: int, frame_index: int) -> List[float]:
        actions = self.episode_actions[episode_index]
        return [
            actions[frame_index + step]
            for step in range(self.chunk_size)
        ]

    def _apply_decayed_action_offset(self, action_chunk: List[float], action_offset: float) -> List[float]:
        adjusted_actions: List[float] = []
        for step, action in enumerate(action_chunk):
            decay_ratio = max(0.0, 1.0 - (step / self.OFFSET_DECAY_STEPS))
            adjusted_actions.append(action + (action_offset * decay_ratio))
        return adjusted_actions

    def _build_samples(self):
        samples = []
        for episode_dir in self.episode_dirs:
            image_dir = episode_dir / 'images'
            action_dir = episode_dir / 'actions'
            command_dir = episode_dir / 'commands'
            image_paths = sorted(image_dir.glob('*.png'))
            if not image_paths:
                continue

            episode_actions: List[float] = []
            episode_commands: List[int] = []

            for image_path in image_paths:
                action_file = action_dir / f'{image_path.stem}.csv'
                command_file = command_dir / f'{image_path.stem}.csv'

                with open(action_file, 'r', newline='') as f:
                    episode_actions.append(float(next(csv.reader(f))[1]))

                with open(command_file, 'r', newline='') as f:
                    episode_commands.append(int(float(next(csv.reader(f))[0])))

            valid_start_count = len(image_paths) - self.chunk_size + 1
            if valid_start_count <= 0:
                continue

            episode_index = len(self.episode_actions)
            self.episode_actions.append(episode_actions)
            self.episode_commands.append(episode_commands)

            for frame_index in range(valid_start_count):
                samples.append((episode_index, frame_index, image_paths[frame_index]))

        return samples


class SampleSplitDataset(Dataset):
    def __init__(self, dataset: MLDataset, sample_indices: List[int], apply_randomshadow: bool):
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
    
    
class Config:
    def __init__(self, config_path, package_root):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.epochs = config_dict['epochs']
        self.batch_size = config_dict['batch_size']
        self.learning_rate = config_dict['learning_rate']
        self.num_workers = config_dict['num_workers']
        self.weight_file = config_dict['weight_file']
        self.action_chunk_size = int(config_dict.get('action_chunk_size', 1))
        self.augment = config_dict.get('augment', {})
        self.randomshadow = self._load_randomshadow_config(config_dict.get('randomshadow', {}))

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


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = Network(chunk_size=config.action_chunk_size)
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
    if len(sys.argv) != 2:
        print('Usage: python3 train.py <dataset_path>')
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f'Dataset path does not exist: {dataset_path}')
        sys.exit(1)

    script_dir = Path(__file__).parent
    package_root = script_dir.parent
    config_path = package_root / 'config' /'train.yaml'
    config = Config(config_path, package_root)

    dataset = MLDataset(
        str(dataset_path),
        chunk_size=config.action_chunk_size,
        augment_config=config.augment,
        randomshadow_config=config.randomshadow,
    )
    num_samples = len(dataset.samples)
    if num_samples < 2:
        raise ValueError('At least 2 samples are required to split into train_data and test_data')

    train_count = int(num_samples * 0.7)
    train_count = max(1, min(num_samples - 1, train_count))
    sample_indices = torch.randperm(num_samples).tolist()
    train_indices = sample_indices[:train_count]
    test_indices = sample_indices[train_count:]

    train_data = SampleSplitDataset(dataset, train_indices, apply_randomshadow=True)
    test_data = SampleSplitDataset(dataset, test_indices, apply_randomshadow=False)

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    print(
        f'Split dataset into train_data={len(train_indices)} samples and '
        f'test_data={len(test_indices)} samples (ratio 7:3)'
    )

    trainer = Trainer(config)
    trainer.train(train_dataloader, test_dataloader)

if __name__ == '__main__':
    main()
