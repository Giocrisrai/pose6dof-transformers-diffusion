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
    grasp_proximity_m: float = 0.0,
    deposit_error_m: float = 0.0,
    curriculum_phase: int = 0,  # 0 = full reward, 1 = grasp-only, 2 = balanced both
) -> float:
    """Terminal reward al final del trajectory.

    Bonuses binarios:
    - +10 si grasp + deposit ambos plausibles.
    - +5 si grasp ok pero deposit no.
    - -5 si IK falla.
    - -10 si distractor collision.

    Penalty continuo (Iter 6c, balanceado):
    - -3 * max(0, grasp_proximity - 5cm): castiga grasp impreciso
    - -1 * min(deposit_error, 0.5m): castiga deposit lejano
    Da gradiente denso en ambas phases — evita el bias hacia solo deposit.
    """
    if curriculum_phase == 1:
        # Phase 1 (Iter 7a): solo signal de grasp. Deposit ignorado.
        r = 10.0 if grasp_plausible else 0.0
        if not ik_converged:
            r -= 5.0
        return r
    if curriculum_phase == 2:
        # Phase 2 (Iter 7a): balanceado +5 grasp + +5 deposit. KL anchor a Phase 1
        # protege el grasp (manejado en el agent, no aqui).
        r = 0.0
        if grasp_plausible:
            r += 5.0
        if deposit_plausible:
            r += 5.0
        if not ik_converged:
            r -= 5.0
        return r
    # Default (curriculum_phase=0): reward full Iter 6
    r = 0.0
    if grasp_plausible and deposit_plausible:
        r += 10.0
    elif grasp_plausible:
        r += 5.0
    if not ik_converged:
        r -= 5.0
    if distractor_collision:
        r -= 10.0
    # Penalty continuo (Iter 6c)
    r -= 3.0 * max(0.0, grasp_proximity_m - 0.05)
    r -= 1.0 * min(deposit_error_m, 0.5)
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
