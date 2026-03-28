#!/usr/bin/env python3

import csv
import yaml
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network import Network
from utils.slit_augment import SlitAugment


class MLDataset(Dataset):
    NUM_BRANCHES = 4

    def __init__(self, dataset_path: str, sequence_length: int = 10):
        dataset_root = Path(dataset_path)
        self.image_dir = dataset_root / 'images'
        self.action_dir = dataset_root / 'actions'
        self.command_dir = dataset_root / 'commands'
        self.sequence_length = int(sequence_length)
        self.image_paths = sorted(self.image_dir.glob('*.png'))
        self.stem_to_path = {int(path.stem): path for path in self.image_paths}
        self.action_indices = {int(path.stem) for path in self.action_dir.glob('*.csv')}
        self.command_indices = {int(path.stem) for path in self.command_dir.glob('*.csv')}
        self.sequence_end_indices = self._build_sequence_end_indices()
        self.augmentor = SlitAugment()

    def _build_sequence_end_indices(self):
        if not self.image_paths:
            return []

        sorted_indices = sorted(self.stem_to_path.keys())
        valid_ends = []
        window_span = self.sequence_length - 1

        for end_idx in sorted_indices:
            start_idx = end_idx - window_span
            if start_idx < 0:
                continue

            frame_indices = [start_idx + i for i in range(self.sequence_length)]
            if not all(index in self.stem_to_path for index in frame_indices):
                continue

            if not all(index in self.action_indices and index in self.command_indices for index in frame_indices):
                continue

            if end_idx not in self.action_indices or end_idx not in self.command_indices:
                continue

            valid_ends.append(end_idx)

        return valid_ends

    def __len__(self):
        return len(self.sequence_end_indices) * len(self.augmentor)

    def __getitem__(self, idx):
        sequence_idx = idx // len(self.augmentor)
        augment_idx = idx % len(self.augmentor)
        end_idx = self.sequence_end_indices[sequence_idx]
        start_idx = end_idx - (self.sequence_length - 1)
        frame_indices = [start_idx + i for i in range(self.sequence_length)]

        sequence_images = []
        augment_offset = 0.0
        for frame_idx in frame_indices:
            img_file = self.stem_to_path[frame_idx]
            image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            image, offset = self.augmentor.get_augmented(image, 0.0, augment_idx)
            augment_offset = offset
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            sequence_images.append(image)

        action_file = self.action_dir / f'{end_idx:05d}.csv'
        command_file = self.command_dir / f'{end_idx:05d}.csv'
        with open(action_file, 'r', newline='') as f:
            angular_z = float(next(csv.reader(f))[1])
        angular_z += augment_offset

        with open(command_file, 'r', newline='') as f:
            command = float(next(csv.reader(f))[0])

        image_tensor = torch.from_numpy(np.stack(sequence_images, axis=0))

        action_tensor = torch.tensor([angular_z], dtype=torch.float32)

        command_tensor = torch.zeros(self.NUM_BRANCHES, dtype=torch.float32)
        command_tensor[int(command)] = 1.0
        
        return image_tensor, action_tensor, command_tensor
    
    
class Config:
    def __init__(self, config_path, package_root):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.epochs = config_dict['epochs']
        self.batch_size = config_dict['batch_size']
        self.learning_rate = config_dict['learning_rate']
        self.num_workers = config_dict['num_workers']
        self.weight_file = config_dict['weight_file']
        self.sequence_length = int(config_dict.get('sequence_length', 10))

        self.weights_dir = package_root / 'weights'
        self.logs_dir = package_root / 'runs'
        self.weights_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = Network()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss = nn.MSELoss()
        self.writer = SummaryWriter(config.logs_dir)
        
    def train(self, dataloader):
        self.model.to(self.config.device)
        best_loss = float('inf')

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0

            for images, action, command in tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}'):
                images = images.to(self.config.device)
                action = action.to(self.config.device)
                command = command.to(self.config.device)

                self.optimizer.zero_grad()
                outputs = self.model(images, command)
                loss = self.loss(outputs, action)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.writer.add_scalar("loss", avg_loss, epoch)
            print(f'Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
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
        sequence_length=config.sequence_length,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    trainer = Trainer(config)
    trainer.train(dataloader)

if __name__ == '__main__':
    main()
