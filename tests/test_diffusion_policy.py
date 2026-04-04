"""Tests for Diffusion Policy module."""

import numpy as np
import torch
import pytest
from src.planning.diffusion_policy import (
    SimpleDDPMScheduler,
    ConditionalUNet1D,
    SinusoidalPosEmb,
    DiffusionGraspPlanner,
)
from src.utils.lie_groups import so3_exp, pose_from_Rt


class TestDDPMScheduler:
    """Tests for the DDPM noise scheduler."""

    def setup_method(self):
        self.scheduler = SimpleDDPMScheduler(num_timesteps=100)

    def test_alpha_bar_decreasing(self):
        """alpha_bar should monotonically decrease."""
        diffs = np.diff(self.scheduler.alpha_bar)
        assert np.all(diffs < 0)

    def test_alpha_bar_bounds(self):
        """alpha_bar should be in (0, 1)."""
        assert np.all(self.scheduler.alpha_bar > 0)
        assert np.all(self.scheduler.alpha_bar < 1)

    def test_add_noise_shape(self):
        x_0 = np.random.randn(16, 7)
        x_t, eps = self.scheduler.add_noise(x_0, t=50)
        assert x_t.shape == x_0.shape
        assert eps.shape == x_0.shape

    def test_add_noise_at_t0_close_to_original(self):
        """At t=0, noised data should be close to original."""
        x_0 = np.random.randn(16, 7)
        x_t, _ = self.scheduler.add_noise(x_0, t=0)
        # alpha_bar[0] is close to 1, so x_t ≈ x_0
        assert np.allclose(x_t, x_0, atol=0.5)

    def test_remove_noise_shape(self):
        x_t = np.random.randn(16, 7)
        eps_pred = np.random.randn(16, 7)
        x_prev = self.scheduler.remove_noise(x_t, eps_pred, t=50)
        assert x_prev.shape == x_t.shape

    def test_remove_noise_at_t0_deterministic(self):
        """At t=0, remove_noise should return predicted x_0 directly."""
        x_t = np.random.randn(16, 7)
        eps_pred = np.zeros_like(x_t)  # No noise prediction
        x_0 = self.scheduler.remove_noise(x_t, eps_pred, t=0)
        # At t=0, x_0 = x_t / sqrt(alpha_bar[0])
        expected = x_t / np.sqrt(self.scheduler.alpha_bar[0])
        assert np.allclose(x_0, expected, atol=1e-5)


class TestConditionalUNet1D:
    """Tests for the noise prediction network."""

    def setup_method(self):
        self.model = ConditionalUNet1D(
            action_dim=7, horizon=16, cond_dim=64
        )

    def test_output_shape(self):
        B = 4
        x = torch.randn(B, 16, 7)
        t = torch.randint(0, 100, (B,))
        c = torch.randn(B, 64)
        out = self.model(x, t, c)
        assert out.shape == (B, 16, 7)

    def test_batch_size_1(self):
        x = torch.randn(1, 16, 7)
        t = torch.zeros(1, dtype=torch.long)
        c = torch.randn(1, 64)
        out = self.model(x, t, c)
        assert out.shape == (1, 16, 7)

    def test_gradient_flows(self):
        """Verify gradients flow through the network."""
        x = torch.randn(2, 16, 7, requires_grad=True)
        t = torch.randint(0, 100, (2,))
        c = torch.randn(2, 64)
        out = self.model(x, t, c)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_timesteps_different_output(self):
        """Different timesteps should produce different outputs."""
        x = torch.randn(1, 16, 7)
        c = torch.randn(1, 64)
        with torch.no_grad():
            out_t0 = self.model(x, torch.tensor([0]), c)
            out_t99 = self.model(x, torch.tensor([99]), c)
        assert not torch.allclose(out_t0, out_t99, atol=1e-4)


class TestSinusoidalPosEmb:
    """Tests for timestep embedding."""

    def test_output_shape(self):
        emb = SinusoidalPosEmb(dim=32)
        t = torch.tensor([0, 10, 50, 99])
        out = emb(t)
        assert out.shape == (4, 32)

    def test_different_timesteps(self):
        emb = SinusoidalPosEmb(dim=32)
        t1 = emb(torch.tensor([0]))
        t2 = emb(torch.tensor([50]))
        assert not torch.allclose(t1, t2)


class TestDiffusionGraspPlanner:
    """Tests for the full grasp planner."""

    def setup_method(self):
        self.planner = DiffusionGraspPlanner(
            action_dim=7, horizon=16, device="cpu"
        )
        R = so3_exp(np.array([0.1, -0.2, 0.3]))
        t = np.array([0.15, -0.08, 0.45])
        self.pose = pose_from_Rt(R, t)

    def test_heuristic_shape(self):
        traj = self.planner.plan_grasp_heuristic(self.pose)
        assert traj.shape == (1, 16, 7)

    def test_heuristic_gripper_open_to_closed(self):
        traj = self.planner.plan_grasp_heuristic(self.pose)
        assert traj[0, 0, 6] == 1.0   # gripper open at start
        assert traj[0, -1, 6] == 0.0  # gripper closed at end

    def test_heuristic_approach_from_above(self):
        """Start position should be above the object."""
        traj = self.planner.plan_grasp_heuristic(self.pose)
        obj_z = self.pose[2, 3]
        start_z = traj[0, 0, 2]
        assert start_z > obj_z  # approaching from above

    def test_heuristic_ends_above_object(self):
        """End position should be above object (lifted)."""
        traj = self.planner.plan_grasp_heuristic(self.pose)
        obj_z = self.pose[2, 3]
        end_z = traj[0, -1, 2]
        assert end_z > obj_z

    def test_diffusion_plan_shape(self):
        """Test diffusion-based planning (untrained, just shape check)."""
        traj = self.planner.plan_grasp(self.pose, n_samples=2)
        assert traj.shape == (2, 16, 7)

    def test_encode_observation(self):
        cond = self.planner.encode_observation(self.pose)
        assert cond.shape == (1, 64)
        # First 12 elements should be the flattened 3x4 pose
        pose_flat = self.pose[:3, :].flatten()
        assert np.allclose(cond[0, :12].numpy(), pose_flat, atol=1e-6)
