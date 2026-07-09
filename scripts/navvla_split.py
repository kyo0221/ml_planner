#!/usr/bin/env python3

import random
from pathlib import Path
from typing import List, Optional, Tuple

TRAJ_DATA_FILE = 'traj_data.pkl'
TRAJ_NAMES_FILE = 'traj_names.txt'


def list_trajectories(data_dir) -> List[str]:
    data_dir = Path(data_dir)
    return sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / TRAJ_DATA_FILE).exists()
    )


def split_trajectories(
    traj_names: List[str],
    split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    shuffled = list(traj_names)
    random.Random(seed).shuffle(shuffled)

    split_index = int(split * len(shuffled))
    return shuffled[:split_index], shuffled[split_index:]


def write_split(
    data_dir,
    train_names: List[str],
    test_names: List[str],
) -> Tuple[Path, Path]:
    base = Path(data_dir)
    train_dir = base / 'train'
    test_dir = base / 'test'

    for dir_path, names in ((train_dir, train_names), (test_dir, test_names)):
        dir_path.mkdir(parents=True, exist_ok=True)
        with (dir_path / TRAJ_NAMES_FILE).open('w', encoding='utf-8') as f:
            for name in names:
                f.write(name + '\n')

    return train_dir, test_dir


def read_traj_names(split_folder) -> List[str]:
    traj_names_file = Path(split_folder) / TRAJ_NAMES_FILE
    lines = traj_names_file.read_text(encoding='utf-8').split('\n')
    return [line for line in lines if line.strip()]


def ensure_split(
    data_dir,
    split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    base = Path(data_dir)
    train_file = base / 'train' / TRAJ_NAMES_FILE
    test_file = base / 'test' / TRAJ_NAMES_FILE

    if not (train_file.exists() and test_file.exists()):
        traj_names = list_trajectories(base)
        train_names, test_names = split_trajectories(traj_names, split=split, seed=seed)
        write_split(base, train_names, test_names)

    return read_traj_names(base / 'train'), read_traj_names(base / 'test')
