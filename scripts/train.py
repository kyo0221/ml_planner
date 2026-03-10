#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import numpy as np

import yaml
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

from network import Network


class MLDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.image_dir = dataset_path + '/images'
        self.action_dir = dataset_path + '/actions'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_file = self.image_dir[idx]
        csv_file = self.action_dir / f'{img_file.stem}.csv'

        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        image_norm = image.astype(np.float32)
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0)

        with open(csv_file, 'r', newline='') as f:
            row = next(csv.reader(f))

        action_tensor = torch.tensor(float(row[1]), dtype=torch.float32)
        return image_tensor, action_tensor
    
    
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
        
    def train(self, dataloader):
        self.model.to(self.config.device)

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0

            for image, action in dataloader:
                image = image.to(self.config.device)
                action = action.to(self.config.device)

                self.optimizer.zero_grad()
                outputs = self.model(image)
                loss = self.loss(outputs, action)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}')

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
