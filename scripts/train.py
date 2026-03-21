#!/usr/bin/env python3

import csv
import yaml
import sys
from pathlib import Path
from typing import List

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
    OFFSET_DECAY_STEPS = 5

    def __init__(self, dataset_path: str, chunk_size: int):
        dataset_root = Path(dataset_path)
        self.image_dir = dataset_root / 'images'
        self.action_dir = dataset_root / 'actions'
        self.command_dir = dataset_root / 'commands'
        self.image_paths = sorted(self.image_dir.glob('*.png'))
        self.chunk_size = chunk_size
        self.augmentor = SlitAugment()
        self.actions = self._load_actions()
        self.commands = self._load_commands()

    def __len__(self):
        return len(self.image_paths) * len(self.augmentor)

    def __getitem__(self, idx):
        image_idx = idx // len(self.augmentor)
        augment_idx = idx % len(self.augmentor)
        img_file = self.image_paths[image_idx]

        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        action_chunk = self._build_action_chunk(image_idx)
        command = self.commands[image_idx]

        image, augmented_angular_z = self.augmentor.get_augmented(image, action_chunk[0], augment_idx)
        action_offset = augmented_angular_z - action_chunk[0]
        action_chunk = self._apply_decayed_action_offset(action_chunk, action_offset)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

        action_tensor = torch.tensor(action_chunk, dtype=torch.float32)

        command_tensor = torch.zeros(self.NUM_BRANCHES, dtype=torch.float32)
        command_tensor[int(command)] = 1.0
        
        return image_tensor, action_tensor, command_tensor

    def _build_action_chunk(self, image_idx: int) -> List[float]:
        last_index = len(self.actions) - 1
        return [
            self.actions[min(image_idx + step, last_index)]
            for step in range(self.chunk_size)
        ]

    def _apply_decayed_action_offset(self, action_chunk: List[float], action_offset: float) -> List[float]:
        adjusted_actions: List[float] = []
        for step, action in enumerate(action_chunk):
            decay_ratio = max(0.0, 1.0 - (step / self.OFFSET_DECAY_STEPS))
            adjusted_actions.append(action + (action_offset * decay_ratio))
        return adjusted_actions

    def _load_actions(self) -> List[float]:
        actions: List[float] = []
        for image_path in self.image_paths:
            action_file = self.action_dir / f'{image_path.stem}.csv'
            with open(action_file, 'r', newline='') as f:
                actions.append(float(next(csv.reader(f))[1]))
        return actions

    def _load_commands(self) -> List[int]:
        commands: List[int] = []
        for image_path in self.image_paths:
            command_file = self.command_dir / f'{image_path.stem}.csv'
            with open(command_file, 'r', newline='') as f:
                commands.append(int(float(next(csv.reader(f))[0])))
        return commands
    
    
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

        self.weights_dir = package_root / 'weights'
        self.logs_dir = package_root / 'runs'
        self.weights_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = Network(chunk_size=config.action_chunk_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss = nn.MSELoss()
        self.writer = SummaryWriter(config.logs_dir)
        
    def train(self, dataloader):
        self.model.to(self.config.device)
        best_loss = float('inf')

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0

            for image, action, command in tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}'):
                image = image.to(self.config.device)
                action = action.to(self.config.device)
                command = command.to(self.config.device)

                self.optimizer.zero_grad()
                outputs = self.model(image, command)
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

    dataset = MLDataset(str(dataset_path), chunk_size=config.action_chunk_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    trainer = Trainer(config)
    trainer.train(dataloader)

if __name__ == '__main__':
    main()
