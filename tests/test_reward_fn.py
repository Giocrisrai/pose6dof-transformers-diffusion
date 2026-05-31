import math

import numpy as np

from src.rl.reward_fn import compute_shaping_reward, compute_terminal_reward


def test_terminal_success():
    r = compute_terminal_reward(
        grasp_plausible=True, deposit_plausible=True,
        ik_converged=True, distractor_collision=False,
    )
    assert r == 10.0


def test_terminal_grasp_only():
    r = compute_terminal_reward(
        grasp_plausible=True, deposit_plausible=False,
        ik_converged=True, distractor_collision=False,
    )
    assert r == 5.0


def test_terminal_ik_fail():
    r = compute_terminal_reward(
        grasp_plausible=False, deposit_plausible=False,
        ik_converged=False, distractor_collision=False,
    )
    assert r == -5.0


def test_terminal_collision_offsets_success():
    r = compute_terminal_reward(
        grasp_plausible=True, deposit_plausible=True,
        ik_converged=True, distractor_collision=True,
    )
    assert r == 0.0  # +10 - 10


def test_shaping_grasp_phase():
    cube_pos = np.array([0.45, -0.10, 0.033])
    wp = np.array([0.45, -0.10, 0.083])  # 5 cm sobre el cubo
    r = compute_shaping_reward(
        wp, cube_pos, deposit_target=np.zeros(3), step=3, total_steps=16
    )
    assert math.isclose(r, -0.1 * 0.05, abs_tol=1e-4)


def test_shaping_deposit_phase():
    deposit = np.array([-0.30, -0.30, 0.30])
    wp = np.array([-0.30, -0.30, 0.40])  # 10 cm sobre el deposit
    r = compute_shaping_reward(
        wp, cube_pos=np.zeros(3), deposit_target=deposit, step=12, total_steps=16
    )
    assert math.isclose(r, -0.1 * 0.10, abs_tol=1e-4)


def test_shaping_intermediate_phase_zero():
    r = compute_shaping_reward(
        wp=np.array([0.5, 0.0, 0.3]),
        cube_pos=np.array([0.0, 0.0, 0.0]),
        deposit_target=np.array([-1.0, 0.0, 0.0]),
        step=7,
        total_steps=16,
    )
    assert r == 0.0
