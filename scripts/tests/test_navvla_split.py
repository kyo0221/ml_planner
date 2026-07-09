from navvla_split import (
    ensure_split,
    list_trajectories,
    read_traj_names,
    split_trajectories,
    write_split,
)


def _make_traj_dir(base, name):
    d = base / name
    d.mkdir()
    (d / 'traj_data.pkl').write_bytes(b'dummy')
    return d


def test_list_trajectories(tmp_path):
    _make_traj_dir(tmp_path, 'episode02')
    _make_traj_dir(tmp_path, 'episode01')
    (tmp_path / 'not_a_traj').mkdir()

    assert list_trajectories(tmp_path) == ['episode01', 'episode02']


def test_split_trajectories_ratio_and_determinism():
    names = [f'ep{i:02d}' for i in range(10)]

    train_names, test_names = split_trajectories(names, split=0.8, seed=42)

    assert len(train_names) == 8
    assert len(test_names) == 2
    assert set(train_names) | set(test_names) == set(names)
    assert set(train_names) & set(test_names) == set()

    train_names2, test_names2 = split_trajectories(names, split=0.8, seed=42)
    assert train_names2 == train_names
    assert test_names2 == test_names


def test_write_split_and_read_traj_names(tmp_path):
    train_names = ['a', 'b']
    test_names = ['c']

    write_split(tmp_path, train_names, test_names)

    assert read_traj_names(tmp_path / 'train') == train_names
    assert read_traj_names(tmp_path / 'test') == test_names


def test_ensure_split_generates_and_persists(tmp_path):
    for i in range(5):
        _make_traj_dir(tmp_path, f'episode{i:02d}')

    train_names, test_names = ensure_split(tmp_path, split=0.8, seed=1)

    assert len(train_names) + len(test_names) == 5
    assert set(train_names) & set(test_names) == set()
    assert (tmp_path / 'train' / 'traj_names.txt').exists()
    assert (tmp_path / 'test' / 'traj_names.txt').exists()

    train_names2, test_names2 = ensure_split(tmp_path, split=0.8, seed=1)
    assert train_names2 == train_names
    assert test_names2 == test_names


def test_ensure_split_does_not_overwrite_existing(tmp_path):
    for i in range(5):
        _make_traj_dir(tmp_path, f'episode{i:02d}')

    (tmp_path / 'train').mkdir()
    (tmp_path / 'test').mkdir()
    (tmp_path / 'train' / 'traj_names.txt').write_text('preexisting_train\n')
    (tmp_path / 'test' / 'traj_names.txt').write_text('preexisting_test\n')

    train_names, test_names = ensure_split(tmp_path, split=0.8, seed=1)

    assert train_names == ['preexisting_train']
    assert test_names == ['preexisting_test']
