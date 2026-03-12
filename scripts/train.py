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

    def __init__(self, dataset_path: str):
        dataset_root = Path(dataset_path)
        self.image_dir = dataset_root / 'images'
        self.action_dir = dataset_root / 'actions'
        self.command_dir = dataset_root / 'commands'
        self.image_paths = sorted(self.image_dir.glob('*.png'))
        self.augmentor = SlitAugment()

    def __len__(self):
        return len(self.image_paths) * len(self.augmentor)

    def __getitem__(self, idx):
        image_idx = idx // len(self.augmentor)
        augment_idx = idx % len(self.augmentor)
        img_file = self.image_paths[image_idx]
        action_file = self.action_dir / f'{img_file.stem}.csv'
        command_file = self.command_dir / f'{img_file.stem}.csv'

        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        with open(action_file, 'r', newline='') as f:
            angular_z = float(next(csv.reader(f))[1])

        with open(command_file, 'r', newline='') as f:
            command = float(next(csv.reader(f))[0])

        image, angular_z = self.augmentor.get_augmented(image, angular_z, augment_idx)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)

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

        self.weights_dir = package_root / 'weights'
        self.logs_dir = package_root / 'runs'
        self.weights_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = Network()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss = nn.MSELoss()
        self.writer = SummaryWriter(config.logs_dir)
        
    def train(self, dataloader):
        self.model.to(self.config.device)

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

    dataset = MLDataset(str(dataset_path))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    trainer = Trainer(config)
    trainer.train(dataloader)

if __name__ == '__main__':
    main()
