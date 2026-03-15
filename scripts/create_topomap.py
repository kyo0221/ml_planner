#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import cv2
import torch
import yaml


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
        self.image_dir = self.dataset_root / 'images'
        self.command_dir = self.dataset_root / 'commands'

        script_dir = Path(__file__).parent
        self.package_root = script_dir.parent
        self.topomap_dir = self.package_root / 'config' / 'topomap'
        self.topomap_images_dir = self.topomap_dir / 'images'
        self.topomap_yaml_path = self.topomap_dir / 'topomap.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def load_model(self):
        weight_path = self.package_root / 'weights' / 'placenet.pt'
        model = torch.jit.load(weight_path, map_location=self.device)
        model.eval()
        return model

    def _prepare_directories(self):
        self.topomap_images_dir.mkdir(parents=True, exist_ok=True)

    def _load_command(self, image_path):
        command_path = self.command_dir / f'{image_path.stem}.csv'
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
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(image_tensor)

        return output.squeeze(0).flatten().tolist()

    def build_nodes(self):
        image_paths = sorted(self.image_dir.glob('*.png'))
        nodes = []

        for idx, image_path in enumerate(image_paths[::self.SAVED_STEP]):
            command = self._load_command(image_path)
            if command not in self.COMMAND_TO_ACTION:
                raise ValueError(f'Unsupported command value: {command}')

            processed_image = self._preprocess_image(image_path)
            output_image_name = f'img{idx + 1:05d}.png'
            output_image_path = self.topomap_images_dir / output_image_name
            cv2.imwrite(str(output_image_path), processed_image)

            nodes.append({
                'id': idx,
                'image': output_image_name,
                'feature': self.extract_feature(processed_image),
                'action': self.COMMAND_TO_ACTION[command],
            })

        if not nodes:
            raise ValueError(f'No images found in dataset: {self.image_dir}')

        for idx, node in enumerate(nodes):
            target = idx + 1 if idx + 1 < len(nodes) else idx
            node['edges'] = [{'target': target, 'action': node.pop('action')}]

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
