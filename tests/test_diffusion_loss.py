"""Tests para src/planning/diffusion_loss.py."""
import torch

from src.planning.diffusion_loss import make_grasp_weights, weighted_mse_loss


def test_make_grasp_weights_shape_and_values():
    w = make_grasp_weights(horizon=16, action_dim=7)
    assert w.shape == (16, 7)
    # Verificar peso en grasp phase (k=8 ∈ [6,11)) para XYZ (dim 0) = 3 * 2 = 6
    assert w[8, 0].item() == 6.0
    # Verificar peso fuera del grasp phase (k=0) para XYZ = 1 * 2 = 2
    assert w[0, 0].item() == 2.0
    # Verificar peso en grasp phase para gripper (dim 6) = 3 * 1 = 3
    assert w[8, 6].item() == 3.0
    # Verificar peso fuera del grasp phase para gripper = 1 * 1 = 1
    assert w[0, 6].item() == 1.0


def test_weighted_mse_zero_when_pred_equals_target():
    pred = torch.randn(4, 16, 7)
    target = pred.clone()
    weights = make_grasp_weights()
    loss = weighted_mse_loss(pred, target, weights)
    assert loss.item() < 1e-6


def test_weighted_mse_is_larger_than_unweighted_when_error_in_grasp_phase():
    # Error exclusivo en k=8 (grasp phase) y dim X
    pred = torch.zeros(1, 16, 7)
    target = torch.zeros(1, 16, 7)
    target[0, 8, 0] = 1.0  # error de 1 en k=8, dim X
    weights = make_grasp_weights()
    weighted = weighted_mse_loss(pred, target, weights).item()
    # weight[8, 0] = 6.0 vs uniform 1.0 → weighted should be 6× the unweighted mean
    unweighted = ((pred - target) ** 2).mean().item()
    assert abs(weighted - 6.0 * unweighted) < 1e-5
