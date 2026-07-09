import pickle

import cv2
import numpy as np
import pytest

from navvla_dataset import (
    COMMAND_LABELS,
    derive_command_index,
    load_navvla_episode,
    local_path_from_positions,
)


def test_derive_command_index_left():
    assert derive_command_index('turn left here', 'roadside') == COMMAND_LABELS.index('left')


def test_derive_command_index_right_case_insensitive():
    assert derive_command_index('please turn RIGHT', 'roadside') == COMMAND_LABELS.index('right')


def test_derive_command_index_default():
    assert derive_command_index('go straight ahead', 'roadside') == COMMAND_LABELS.index('roadside')


def test_derive_command_index_left_priority_over_right():
    assert derive_command_index('not right, go left', 'roadside') == COMMAND_LABELS.index('left')


def test_local_path_from_positions_straight_heading():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    yaw = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    path = local_path_from_positions(positions, yaw, 0, chunk_size=2)

    assert path == pytest.approx(np.array([[1.0, 0.0], [2.0, 0.0]]))


def test_local_path_from_positions_rotates_into_robot_frame():
    positions = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    yaw = np.array([np.pi / 2, np.pi / 2], dtype=np.float32)

    path = local_path_from_positions(positions, yaw, 0, chunk_size=1)

    assert path == pytest.approx(np.array([[1.0, 0.0]]), abs=1e-6)


def _write_navvla_episode(base, name, positions, yaw, prompts):
    episode_dir = base / name
    episode_dir.mkdir()
    num_images = len(yaw)
    fake_image = np.zeros((288, 512, 3), dtype=np.uint8)
    for i in range(num_images):
        cv2.imwrite(str(episode_dir / f'{i}.jpg'), fake_image)

    traj_data = {
        'position': np.array(positions, dtype=np.float32),
        'yaw': np.array(yaw, dtype=np.float32),
    }
    with open(episode_dir / 'traj_data.pkl', 'wb') as f:
        pickle.dump(traj_data, f)

    (episode_dir / 'traj_prompt.txt').write_text('\n'.join(prompts) + '\n', encoding='utf-8')
    return episode_dir


def test_load_navvla_episode_basic(tmp_path):
    positions = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    yaw = [0.0, 0.0, 0.0, 0.0]
    prompts = ['go straight', 'turn left here', 'turn RIGHT now']
    episode_dir = _write_navvla_episode(tmp_path, 'nav_episode01', positions, yaw, prompts)

    image_paths, actions, commands = load_navvla_episode(episode_dir, default_command='roadside', chunk_size=1)

    assert [p.name for p in image_paths] == ['0.jpg', '1.jpg', '2.jpg']
    assert len(actions) == 3
    assert len(commands) == 3
    assert actions[0] == pytest.approx(np.array([[1.0, 0.0]]))

    assert commands[0] == COMMAND_LABELS.index('roadside')
    assert commands[1] == COMMAND_LABELS.index('left')
    assert commands[2] == COMMAND_LABELS.index('right')


def test_load_navvla_episode_chunk_size_shortens_valid_range(tmp_path):
    positions = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    yaw = [0.0, 0.0, 0.0, 0.0]
    prompts = ['go straight', 'turn left here', 'turn RIGHT now']
    episode_dir = _write_navvla_episode(tmp_path, 'nav_episode01', positions, yaw, prompts)

    image_paths, actions, commands = load_navvla_episode(episode_dir, default_command='roadside', chunk_size=2)

    assert [p.name for p in image_paths] == ['0.jpg', '1.jpg']
    assert len(actions) == 2
    assert actions[0] == pytest.approx(np.array([[1.0, 0.0], [2.0, 0.0]]))


def _randomshadow_config():
    return {
        'p': 0.5,
        'high_ratio': [1.0, 2.0],
        'low_ratio': [0.01, 0.5],
        'left_low_ratio': [0.4, 0.6],
        'left_high_ratio': [0.0, 0.2],
        'right_low_ratio': [0.4, 0.6],
        'right_high_ratio': [0.0, 0.2],
    }


def test_mldataset_build_samples_from_navvla_datasets(tmp_path):
    from train import MLDataset

    navvla_root = tmp_path / 'navvla_root'
    navvla_root.mkdir()
    _write_navvla_episode(
        navvla_root, 'nav_episode01',
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [0.0, 0.0, 0.0, 0.0],
        ['go straight', 'turn left', 'turn right'],
    )
    (navvla_root / 'train').mkdir()
    (navvla_root / 'train' / 'traj_names.txt').write_text('nav_episode01\n')
    (navvla_root / 'test').mkdir()
    (navvla_root / 'test' / 'traj_names.txt').write_text('')

    dataset_specs = [
        {'path': str(navvla_root), 'default_command': 'roadside', 'episodes': None},
    ]

    dataset = MLDataset(
        dataset_specs,
        chunk_size=1,
        augment_config={'crop': False, 'randomshadow': False},
        randomshadow_config=_randomshadow_config(),
    )

    assert len(dataset.episode_actions) == 1
    assert len(dataset.samples) == 3

    image_tensor, action_tensor, command_tensor = dataset[0]
    assert image_tensor.shape[0] == 3
    assert action_tensor.shape == (1, 2)
    assert command_tensor.shape == (MLDataset.NUM_BRANCHES,)


def test_mldataset_build_samples_respects_episodes_filter(tmp_path):
    from train import MLDataset

    navvla_root = tmp_path / 'navvla_root'
    navvla_root.mkdir()
    _write_navvla_episode(
        navvla_root, 'nav_episode01',
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [0.0, 0.0, 0.0, 0.0],
        ['go straight', 'turn left', 'turn right'],
    )
    _write_navvla_episode(
        navvla_root, 'nav_episode02',
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [0.0, 0.0, 0.0],
        ['go straight', 'turn left'],
    )

    dataset_specs = [
        {'path': str(navvla_root), 'default_command': 'roadside', 'episodes': ['nav_episode01']},
    ]

    dataset = MLDataset(
        dataset_specs,
        chunk_size=1,
        augment_config={'crop': False, 'randomshadow': False},
        randomshadow_config=_randomshadow_config(),
    )

    assert len(dataset.episode_actions) == 1
    assert len(dataset.samples) == 3


def test_mldataset_slit_augment_rotates_path(tmp_path):
    from train import MLDataset

    navvla_root = tmp_path / 'navvla_root'
    navvla_root.mkdir()
    _write_navvla_episode(
        navvla_root, 'nav_episode01',
        [[0.0, 0.0], [1.0, 0.0]],
        [0.0, 0.0],
        ['go straight'],
    )

    dataset_specs = [
        {'path': str(navvla_root), 'default_command': 'roadside', 'episodes': ['nav_episode01']},
    ]

    dataset = MLDataset(
        dataset_specs,
        chunk_size=1,
        augment_config={'crop': True, 'randomshadow': False},
        randomshadow_config=_randomshadow_config(),
    )

    assert len(dataset) == 1 * len(dataset.augmentor)

    _, center_action, _ = dataset.get_item(0, 2)
    assert center_action.numpy() == pytest.approx(np.array([[1.0, 0.0]]), abs=1e-5)

    _, left_action, _ = dataset.get_item(0, 0)
    assert not np.allclose(left_action.numpy(), center_action.numpy())
