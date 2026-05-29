"""Losses para training de Diffusion Policy."""
from __future__ import annotations

import torch


def make_grasp_weights(
    horizon: int = 16,
    action_dim: int = 7,
    grasp_phase_start: int = 6,
    grasp_phase_end: int = 11,
    weight_grasp_phase: float = 3.0,
    weight_xyz: float = 2.0,
    weight_rot_gripper: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Construye matriz de pesos (horizon, action_dim) para weighted loss.

    weights = weights_k * weights_dim (outer product broadcast):
      - weights_k[grasp_phase_start:grasp_phase_end] = weight_grasp_phase
      - weights_k[else] = 1.0
      - weights_dim[0:3] = weight_xyz (XYZ)
      - weights_dim[3:] = weight_rot_gripper (rotación + gripper)

    Default: peso máximo en (k∈[6,11), dim XYZ) = 3 * 2 = 6.
    Peso mínimo en (k fuera grasp, dim rot/gripper) = 1 * 1 = 1.

    Returns: (horizon, action_dim) tensor.
    """
    weights_k = torch.ones(horizon, device=device)
    weights_k[grasp_phase_start:grasp_phase_end] = weight_grasp_phase
    weights_dim = torch.full((action_dim,), weight_rot_gripper, device=device)
    weights_dim[:3] = weight_xyz
    return weights_k.view(-1, 1) * weights_dim.view(1, -1)


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error ponderado por (horizon, action_dim) matrix.

    Args:
        pred:   (B, horizon, action_dim) tensor.
        target: (B, horizon, action_dim) tensor.
        weights: (horizon, action_dim) tensor — broadcasts en batch dim.

    Returns:
        Scalar tensor: mean of (pred - target)² * weights.
    """
    return ((pred - target) ** 2 * weights).mean()
