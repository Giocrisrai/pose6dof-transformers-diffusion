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

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

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

    def add_noise_batch(self, x_0, t):
        """Versión batch de add_noise. Acepta torch tensors.

        Args:
            x_0: (B, horizon, action_dim) tensor torch (clean data).
            t:   (B,) tensor torch long (timestep por batch element).

        Returns:
            x_t: (B, horizon, action_dim) tensor noisy data.
            eps: (B, horizon, action_dim) tensor noise applied.
        """
        import torch
        device = x_0.device
        # alpha_bar es numpy de shape (T,); indexamos por t.
        alpha_bar_np = self.alpha_bar[t.cpu().numpy()]  # (B,)
        alpha_bar = torch.tensor(alpha_bar_np, dtype=torch.float32, device=device)
        # broadcast a (B, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * eps
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
        x.shape[0]

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
        hidden_dim: int = 128,
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
            hidden_dim=hidden_dim,
        ).to(device)

        self._trained = False

    def encode_observation(
        self,
        object_pose: np.ndarray,
        visual_emb: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Encode object pose + optional visual embedding into 64d cond vector.

        Layouts:
            v3 (visual_emb provided): cond[:52]=visual_emb, cond[52:64]=pose[:3,:].flatten()
            v1/v2 (visual_emb=None):  cond[:12]=pose[:3,:].flatten(), cond[12:]=0
        """
        cond = np.zeros(64, dtype=np.float32)
        pose_flat = object_pose[:3, :].flatten().astype(np.float32)
        if visual_emb is not None:
            v = np.asarray(visual_emb, dtype=np.float32).reshape(-1)
            cond[: min(52, v.size)] = v[:52]
            cond[52:64] = pose_flat[:12]
        else:
            cond[:12] = pose_flat[:12]
        return torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(self.device)

    def plan_grasp(
        self,
        object_pose: np.ndarray,
        n_samples: int = 1,
        cond: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Generate grasp trajectory using reverse diffusion.

        Args:
            object_pose: (4, 4) SE(3) pose of target object
            n_samples: number of trajectory samples (multimodal)
            cond: optional precomputed (1, 64) cond tensor. If None, the
                  default `encode_observation(object_pose)` is used.

        Returns:
            (n_samples, horizon, action_dim) trajectories
        """
        self.model.eval()
        if cond is None:
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
        with_deposit: bool = False,
        deposit_target: tuple = (-0.30, -0.30, 0.30),
    ) -> np.ndarray:
        """Generate a simple heuristic grasp trajectory (baseline).

        Si `with_deposit=False` (default, compat con Iter 1-4): aproach → descend →
        grasp → lift. Última pose: target + lift_height en Z, gripper cerrado.

        Si `with_deposit=True` (Iter 5+): aproach (25 %) → grasp (15 %) → lift (10 %)
        → move-to-deposit (30 %) → release (20 %). Última pose: deposit_target con
        gripper abierto. El cubo cae por gravedad cuando el gripper abre.

        Args:
            object_pose: (4, 4) SE(3) pose of target object
            approach_distance: approach offset en Z al inicio (m)
            lift_height: altura de lift sobre el target (m)
            with_deposit: si True, agrega fase de deposit + release (Iter 5)
            deposit_target: (x, y, z) destino del cubo cuando with_deposit=True

        Returns:
            (1, horizon, 7) trajectory [x, y, z, rx, ry, rz, gripper]
        """
        from src.utils.lie_groups import pose_to_Rt, so3_log

        R, t = pose_to_Rt(object_pose)
        rot_vec = so3_log(R)

        trajectory = np.zeros((1, self.horizon, self.action_dim))

        if not with_deposit:
            for i in range(self.horizon):
                progress = i / (self.horizon - 1)
                if progress < 0.3:
                    frac = progress / 0.3
                    z_offset = approach_distance * (1 - frac)
                    pos = t + np.array([0, 0, z_offset])
                    gripper = 1.0
                elif progress < 0.5:
                    pos = t.copy()
                    gripper = 1.0
                elif progress < 0.6:
                    pos = t.copy()
                    frac = (progress - 0.5) / 0.1
                    gripper = 1.0 - frac
                else:
                    frac = (progress - 0.6) / 0.4
                    pos = t + np.array([0, 0, lift_height * frac])
                    gripper = 0.0
                trajectory[0, i, :3] = pos
                trajectory[0, i, 3:6] = rot_vec
                trajectory[0, i, 6] = gripper
            return trajectory

        # with_deposit=True: phases 5 con deposit
        deposit_t = np.asarray(deposit_target, dtype=np.float32)
        lift_t = t + np.array([0, 0, lift_height], dtype=np.float32)
        for i in range(self.horizon):
            progress = i / (self.horizon - 1)
            if progress < 0.25:
                # Approach desde approach_distance arriba hasta target
                frac = progress / 0.25
                pos = t + np.array([0, 0, approach_distance * (1 - frac)])
                gripper = 1.0
            elif progress < 0.4:
                # Descend + close gripper
                frac = (progress - 0.25) / 0.15
                pos = t.copy()
                gripper = 1.0 - frac
            elif progress < 0.5:
                # Lift up
                frac = (progress - 0.4) / 0.1
                pos = t + np.array([0, 0, lift_height * frac])
                gripper = 0.0
            elif progress < 0.8:
                # Move horizontally + up to deposit_target
                frac = (progress - 0.5) / 0.3
                pos = lift_t + frac * (deposit_t - lift_t)
                gripper = 0.0
            else:
                # Release (gripper opens, cubo cae)
                frac = (progress - 0.8) / 0.2
                pos = deposit_t.copy()
                gripper = frac
            trajectory[0, i, :3] = pos
            trajectory[0, i, 3:6] = rot_vec
            trajectory[0, i, 6] = gripper

        return trajectory
