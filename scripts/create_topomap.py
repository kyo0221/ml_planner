#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import cv2
import torch
import yaml
from torchvision import transforms


class TopomapGenerator:
    COMMAND_TO_ACTION = {
        0: 'roadside',
        1: 'straight',
        2: 'left',
        3: 'right',
    }
    CROP_SIZE = 288
    OUTPUT_SIZE = 85
    SAVED_STEP = 10

    def __init__(self, dataset_path):
        self.dataset_root = Path(dataset_path)
        self.episode_dirs = sorted([path for path in self.dataset_root.glob('episode*') if path.is_dir()])

        script_dir = Path(__file__).parent
        self.package_root = script_dir.parent
        self.topomap_dir = self.package_root / 'config' / 'topomap'
        self.topomap_images_dir = self.topomap_dir / 'images'
        self.topomap_yaml_path = self.topomap_dir / 'topomap.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.placenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.OUTPUT_SIZE, self.OUTPUT_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        weight_path = self.package_root / 'weights' / 'placenet.pt'
        model = torch.jit.load(weight_path, map_location=self.device)
        model.eval()
        return model

    def _prepare_directories(self):
        self.topomap_images_dir.mkdir(parents=True, exist_ok=True)

    def _load_command(self, command_dir, image_path):
        command_path = command_dir / f'{image_path.stem}.csv'
        with command_path.open('r', newline='') as f:
            return int(float(next(csv.reader(f))[0]))

    def _center_crop(self, image):
        height, width = image.shape[:2]
        if height < self.CROP_SIZE or width < self.CROP_SIZE:
            raise ValueError(f'Image is smaller than {self.CROP_SIZE}x{self.CROP_SIZE}: {height}x{width}')

        top = (height - self.CROP_SIZE) // 2
        left = (width - self.CROP_SIZE) // 2
        return image[top:top + self.CROP_SIZE, left:left + self.CROP_SIZE]

    def _preprocess_image(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Failed to read image: {image_path}')

        cropped_image = self._center_crop(image)
        resized_image = cv2.resize(
            cropped_image,
            (self.OUTPUT_SIZE, self.OUTPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return resized_image

    def extract_feature(self, image):
        image_tensor = self.placenet_transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(image_tensor)

        return output.squeeze(0).flatten().tolist()

    def build_nodes(self):
        nodes = []
        node_id = 0

        for episode_dir in self.episode_dirs:
            image_dir = episode_dir / 'images'
            command_dir = episode_dir / 'commands'
            episode_image_paths = sorted(image_dir.glob('*.png'))[::self.SAVED_STEP]
            episode_nodes = []

            for image_path in episode_image_paths:
                command = self._load_command(command_dir, image_path)
                if command not in self.COMMAND_TO_ACTION:
                    raise ValueError(f'Unsupported command value: {command}')

                processed_image = self._preprocess_image(image_path)
                output_image_name = f'img{node_id + 1:05d}.png'
                output_image_path = self.topomap_images_dir / output_image_name
                cv2.imwrite(str(output_image_path), processed_image)

                episode_nodes.append({
                    'id': node_id,
                    'image': output_image_name,
                    'feature': self.extract_feature(processed_image),
                    'action': self.COMMAND_TO_ACTION[command],
                })
                node_id += 1

            for idx, node in enumerate(episode_nodes):
                target = episode_nodes[idx + 1]['id'] if idx + 1 < len(episode_nodes) else node['id']
                node['edges'] = [{'target': target, 'action': node.pop('action')}]

            nodes.extend(episode_nodes)

        if not nodes:
            raise ValueError(f'No images found in dataset: {self.dataset_root}')

        return nodes

    def generate(self):
        self._prepare_directories()
        topomap = {'nodes': self.build_nodes()}

        with self.topomap_yaml_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(topomap, f, sort_keys=False, allow_unicode=False)


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 create_topomap.py <dataset_path>')
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f'Dataset path does not exist: {dataset_path}')
        sys.exit(1)

    topomap_generator = TopomapGenerator(dataset_path)
    topomap_generator.generate()


if __name__ == '__main__':
    main()
