#!/usr/bin/env python3

import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# order/meaning must match ml_planner/ml_planner/planner.py's COMMAND_LABELS
COMMAND_LABELS = ['roadside', 'straight', 'left', 'right']

SAMPLE_INTERVAL = 0.2


def derive_command_index(prompt: str, default_command: str) -> int:
    lower = prompt.lower()
    if 'left' in lower:
        label = 'left'
    elif 'right' in lower:
        label = 'right'
    else:
        label = default_command
    return COMMAND_LABELS.index(label)


def local_path_from_positions(positions: np.ndarray, yaw: np.ndarray, start_idx: int, chunk_size: int) -> np.ndarray:
    origin = positions[start_idx]
    cos_yaw = math.cos(float(yaw[start_idx]))
    sin_yaw = math.sin(float(yaw[start_idx]))
    future = positions[start_idx + 1: start_idx + chunk_size + 1] - origin
    local_x = cos_yaw * future[:, 0] + sin_yaw * future[:, 1]
    local_y = -sin_yaw * future[:, 0] + cos_yaw * future[:, 1]
    return np.stack([local_x, local_y], axis=1).astype(np.float32)


def load_navvla_episode(episode_dir, default_command: Optional[str], chunk_size: int) -> Tuple[List[Path], List[np.ndarray], List[int]]:
    episode_dir = Path(episode_dir)
    with open(episode_dir / 'traj_data.pkl', 'rb') as f:
        traj_data = pickle.load(f)

    positions = np.asarray(traj_data['position'], dtype=np.float32)
    yaw = np.asarray(traj_data['yaw'], dtype=np.float32)
    prompts = (episode_dir / 'traj_prompt.txt').read_text(encoding='utf-8').splitlines()

    num_actions = len(yaw) - chunk_size
    actions = [
        local_path_from_positions(positions, yaw, i, chunk_size)
        for i in range(num_actions)
    ]
    commands = [
        derive_command_index(prompts[i], default_command)
        for i in range(num_actions)
    ]
    image_paths = [episode_dir / f'{i}.jpg' for i in range(num_actions)]

    return image_paths, actions, commands
