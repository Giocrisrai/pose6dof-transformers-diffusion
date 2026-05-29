"""Tests para SimpleDDPMScheduler.add_noise_batch."""
import numpy as np
import torch

from src.planning.diffusion_policy import SimpleDDPMScheduler


def test_add_noise_batch_shape():
    sch = SimpleDDPMScheduler(num_timesteps=100)
    # Batch de 4 trayectorias de horizon=16, action_dim=7
    x_0 = torch.randn(4, 16, 7)
    t = torch.tensor([10, 20, 30, 50], dtype=torch.long)
    x_t, eps = sch.add_noise_batch(x_0, t)
    assert x_t.shape == x_0.shape
    assert eps.shape == x_0.shape
    assert x_t.dtype == torch.float32


def test_add_noise_batch_t_zero_returns_almost_clean():
    sch = SimpleDDPMScheduler(num_timesteps=100)
    x_0 = torch.randn(2, 16, 7)
    t = torch.tensor([0, 0], dtype=torch.long)
    x_t, eps = sch.add_noise_batch(x_0, t)
    # En t=0, alpha_bar ≈ 0.9999; x_t debe estar cerca de x_0
    diff = (x_t - x_0).abs().mean().item()
    assert diff < 0.5
