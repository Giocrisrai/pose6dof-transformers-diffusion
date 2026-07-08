"""Smoke tests para SimPickDataset."""

import torch


def _make_dummy_pt(path, n=10):
    """Crea un .pt con n elementos dummy."""
    torch.save({
        "conds": torch.randn(n, 64),
        "trajs": torch.randn(n, 16, 7),
        "split": "test",
    }, path)


def test_sim_pick_dataset_len(tmp_path):
    from src.planning.diffusion_dataset import SimPickDataset
    pt = tmp_path / "test.pt"
    _make_dummy_pt(pt, n=23)
    ds = SimPickDataset(pt)
    assert len(ds) == 23


def test_sim_pick_dataset_getitem_shapes(tmp_path):
    from src.planning.diffusion_dataset import SimPickDataset
    pt = tmp_path / "test.pt"
    _make_dummy_pt(pt, n=5)
    ds = SimPickDataset(pt)
    cond, traj = ds[0]
    assert cond.shape == (64,)
    assert traj.shape == (16, 7)
    assert cond.dtype == torch.float32
    assert traj.dtype == torch.float32


def test_sim_pick_dataset_dataloader(tmp_path):
    from torch.utils.data import DataLoader

    from src.planning.diffusion_dataset import SimPickDataset
    pt = tmp_path / "test.pt"
    _make_dummy_pt(pt, n=16)
    ds = SimPickDataset(pt)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    assert batch[0].shape == (4, 64)
    assert batch[1].shape == (4, 16, 7)
