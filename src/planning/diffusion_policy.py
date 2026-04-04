"""
Diffusion Policy Wrapper — Visuomotor Policy Learning via Action Diffusion.

Generates grasp trajectories using a denoising diffusion process.
Given a 6-DoF object pose (from FoundationPose), produces a sequence
of gripper waypoints for bin picking.

Reference:
    Chi et al. (2023) "Diffusion Policy: Visuomotor Policy Learning
    via Action Diffusion", RSS.

GitHub: https://github.com/real-stanford/diffusion_policy
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleDDPMScheduler:
    """Minimal DDPM noise scheduler for demonstration.

    Implements the forward and reverse diffusion process:
        Forward:  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        Reverse:  x_{t-1} = mu_theta(x_t, t) + sigma_t * z

    For production, use diffusers.DDPMScheduler.
    """

    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)

    def add_noise(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forward process: add noise to clean data.

        Args:
            x_0: clean data (trajectory)
            t: timestep

        Returns:
            x_t: noisy data
            eps: noise that was added
        """
        alpha_bar_t = self.alpha_bar[t]
        eps = np.random.randn(*x_0.shape)
        x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * eps
        return x_t, eps

    def remove_noise(self, x_t: np.ndarray, eps_pred: np.ndarray, t: int) -> np.ndarray:
        """Reverse process: one denoising step.

        Args:
            x_t: noisy data at timestep t
            eps_pred: predicted noise
            t: current timestep

        Returns:
            x_{t-1}: less noisy data
        """
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        # Predicted x_0
        x_0_pred = (x_t - np.sqrt(1 - alpha_bar_t) * eps_pred) / np.sqrt(alpha_bar_t)

        # Mean of posterior
        if t > 0:
            alpha_bar_prev = self.alpha_bar[t - 1]
            beta_t = self.betas[t]
            mu = (np.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)) * x_0_pred + \
                 (np.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * x_t
            sigma = np.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
            return mu + sigma * np.random.randn(*x_t.shape)
        else:
            return x_0_pred


class ConditionalUNet1D(nn.Module):
    """Simplified 1D U-Net for noise prediction in action space.

    Architecture mirrors Diffusion Policy's temporal U-Net:
        Input: (batch, horizon, action_dim) + timestep embedding + condition
        Output: (batch, horizon, action_dim) predicted noise

    This is a simplified version for demonstration. The full version
    uses residual blocks, group norm, and multi-scale features.
    """

    def __init__(
        self,
        action_dim: int = 7,        # 6-DoF pose + gripper
        horizon: int = 16,           # prediction horizon
        cond_dim: int = 64,          # observation condition dimension
        time_emb_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition encoder (observation → features)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )

        # Middle (with time + condition injection)
        self.mid = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )

        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )

        # Output projection
        self.out = nn.Conv1d(hidden_dim, action_dim, 1)

    def forward(
        self,
        x: torch.Tensor,       # (B, horizon, action_dim)
        timestep: torch.Tensor,  # (B,) or scalar
        cond: torch.Tensor,     # (B, cond_dim)
    ) -> torch.Tensor:
        """Predict noise given noisy action sequence and condition.

        Returns:
            (B, horizon, action_dim) predicted noise
        """
        B = x.shape[0]

        # Time embedding: (B, hidden)
        t_emb = self.time_mlp(timestep)

        # Condition embedding: (B, hidden)
        c_emb = self.cond_mlp(cond)

        # Combined context: (B, hidden, 1) for broadcasting
        ctx = (t_emb + c_emb).unsqueeze(-1)

        # (B, horizon, action_dim) → (B, action_dim, horizon)
        x = x.permute(0, 2, 1)

        # Encoder
        h1 = self.enc1(x)          # (B, hidden, horizon)
        h2 = self.enc2(h1)

        # Middle with context injection
        h = self.mid(h2) + ctx      # Broadcast context

        # Decoder with skip connections
        h = self.dec2(torch.cat([h, h2], dim=1))
        h = self.dec1(torch.cat([h, h1], dim=1))

        # Output
        out = self.out(h)  # (B, action_dim, horizon)
        return out.permute(0, 2, 1)  # (B, horizon, action_dim)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DiffusionGraspPlanner:
    """Diffusion Policy for generating grasp trajectories.

    Given the 6-DoF pose of a target object, generates a sequence
    of gripper poses (waypoints) for executing a pick-and-place.

    Usage:
        planner = DiffusionGraspPlanner()
        trajectory = planner.plan_grasp(object_pose, observation)
    """

    def __init__(
        self,
        action_dim: int = 7,      # [x, y, z, rx, ry, rz, gripper]
        horizon: int = 16,         # number of waypoints
        n_diffusion_steps: int = 100,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device

        # Noise scheduler
        self.scheduler = SimpleDDPMScheduler(num_timesteps=n_diffusion_steps)

        # Noise prediction network
        self.model = ConditionalUNet1D(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=64,
        ).to(device)

        self._trained = False

    def encode_observation(self, object_pose: np.ndarray) -> torch.Tensor:
        """Encode object pose + context into condition vector.

        In the full pipeline, this would include:
        - Object 6-DoF pose (from FoundationPose)
        - RGB-D features
        - Robot state

        Args:
            object_pose: (4, 4) SE(3) transformation matrix

        Returns:
            (1, 64) condition tensor
        """
        # Flatten pose and pad to 64 dims
        pose_flat = object_pose[:3, :].flatten()  # (12,)
        cond = np.zeros(64)
        cond[:12] = pose_flat
        return torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(self.device)

    def plan_grasp(
        self,
        object_pose: np.ndarray,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate grasp trajectory using reverse diffusion.

        Args:
            object_pose: (4, 4) SE(3) pose of target object
            n_samples: number of trajectory samples (multimodal)

        Returns:
            (n_samples, horizon, action_dim) trajectories
        """
        self.model.eval()
        cond = self.encode_observation(object_pose)
        if n_samples > 1:
            cond = cond.repeat(n_samples, 1)

        # Start from pure noise
        x_t = torch.randn(
            n_samples, self.horizon, self.action_dim,
            device=self.device
        )

        # Reverse diffusion
        with torch.no_grad():
            for t in reversed(range(self.scheduler.num_timesteps)):
                timestep = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                eps_pred = self.model(x_t, timestep, cond)

                # Convert to numpy for scheduler
                x_np = x_t.cpu().numpy()
                eps_np = eps_pred.cpu().numpy()

                # Denoise
                trajectories = []
                for i in range(n_samples):
                    x_denoised = self.scheduler.remove_noise(x_np[i], eps_np[i], t)
                    trajectories.append(x_denoised)

                x_t = torch.tensor(np.array(trajectories), dtype=torch.float32, device=self.device)

        return x_t.cpu().numpy()

    def plan_grasp_heuristic(
        self,
        object_pose: np.ndarray,
        approach_distance: float = 0.15,
        lift_height: float = 0.10,
    ) -> np.ndarray:
        """Generate a simple heuristic grasp trajectory (baseline).

        Creates a top-down approach trajectory:
        1. Pre-grasp: approach from above
        2. Grasp: close gripper at object pose
        3. Lift: move up

        Args:
            object_pose: (4, 4) SE(3) pose of target object
            approach_distance: approach distance in meters
            lift_height: lift height in meters

        Returns:
            (1, horizon, 7) trajectory [x, y, z, rx, ry, rz, gripper]
        """
        from src.utils.lie_groups import pose_to_Rt, so3_log

        R, t = pose_to_Rt(object_pose)
        rot_vec = so3_log(R)

        trajectory = np.zeros((1, self.horizon, self.action_dim))

        for i in range(self.horizon):
            progress = i / (self.horizon - 1)

            if progress < 0.3:
                # Phase 1: Approach from above
                frac = progress / 0.3
                z_offset = approach_distance * (1 - frac)
                pos = t + np.array([0, 0, z_offset])
                gripper = 1.0  # open

            elif progress < 0.5:
                # Phase 2: At grasp position
                pos = t.copy()
                gripper = 1.0  # open

            elif progress < 0.6:
                # Phase 3: Close gripper
                pos = t.copy()
                frac = (progress - 0.5) / 0.1
                gripper = 1.0 - frac  # closing

            else:
                # Phase 4: Lift
                frac = (progress - 0.6) / 0.4
                pos = t + np.array([0, 0, lift_height * frac])
                gripper = 0.0  # closed

            trajectory[0, i, :3] = pos
            trajectory[0, i, 3:6] = rot_vec
            trajectory[0, i, 6] = gripper

        return trajectory
