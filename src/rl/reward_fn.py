"""Reward functions for DPPO fine-tuning (Iter 6).

Shaped per-step + terminal binary. Bounded en [-20, +12] para estabilidad PPO.
"""
from __future__ import annotations

import numpy as np


GRASP_PHASE_END = 6        # k=0..5: aproximación + grasp
DEPOSIT_PHASE_START = 9    # k=9..15: lift + deposit


def compute_terminal_reward(
    grasp_plausible: bool,
    deposit_plausible: bool,
    ik_converged: bool,
    distractor_collision: bool,
) -> float:
    """Terminal reward al final del trajectory.

    +10 si grasp + deposit ambos plausibles.
    +5 si grasp ok pero deposit no.
    -5 si IK falla.
    -10 si distractor collision (Phase B+).
    """
    r = 0.0
    if grasp_plausible and deposit_plausible:
        r += 10.0
    elif grasp_plausible:
        r += 5.0
    if not ik_converged:
        r -= 5.0
    if distractor_collision:
        r -= 10.0
    return r


def compute_shaping_reward(
    wp: np.ndarray,
    cube_pos: np.ndarray,
    deposit_target: np.ndarray,
    step: int,
    total_steps: int,
) -> float:
    """Shaping per-step.

    Durante grasp phase (k < 6): -0.1 * dist_to_cube.
    Durante deposit phase (k >= 9): -0.1 * dist_to_deposit.
    Phase intermedio: 0.
    """
    if step < GRASP_PHASE_END:
        d = float(np.linalg.norm(wp[:3] - cube_pos[:3]))
        return -0.1 * d
    if step >= DEPOSIT_PHASE_START:
        d = float(np.linalg.norm(wp[:3] - deposit_target[:3]))
        return -0.1 * d
    return 0.0
