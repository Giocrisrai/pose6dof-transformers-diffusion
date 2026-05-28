#!/usr/bin/env python3
"""Genera dataset para re-entrenamiento de la Diffusion Policy.

Phase A.1: 200 trayectorias heurísticas (rápido, ~1s c/u).
Phase A.2: 30 trayectorias ejecutadas en sim (lento, ~50s c/u).
Phase A.3: combina + split 80/20 → train.pt + val.pt.

Uso:
    python experiments/collect_diffusion_dataset.py --phase heuristic
    python experiments/collect_diffusion_dataset.py --phase executed
    python experiments/collect_diffusion_dataset.py --phase split
    python experiments/collect_diffusion_dataset.py --phase all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collect_dp")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v1"
N_HEURISTIC = 200
N_EXECUTED = 30
SEED = 42

# Workspace bounds (matching bin_base.ttt geometry)
X_RANGE = (0.40, 0.55)
Y_RANGE = (-0.15, -0.05)
Z_FIXED = 0.033


def sample_pose(rng: np.random.Generator) -> np.ndarray:
    """Sample una pose SE(3) random dentro del workspace.

    Returns: (4, 4) matriz SE(3).
    """
    x = rng.uniform(*X_RANGE)
    y = rng.uniform(*Y_RANGE)
    z = Z_FIXED
    # Rotación: una de [identity, 45° Z, 90° Z]
    rot_choices = [0.0, np.pi / 4, np.pi / 2]
    theta = rng.choice(rot_choices)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    return pose


def phase_heuristic(n: int = N_HEURISTIC) -> None:
    """Genera n trayectorias heurísticas y las guarda."""
    logger.info(f"Phase A.1: generando {n} trayectorias heurísticas")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100,
        device="cpu",  # heurística no usa la red
    )

    rng = np.random.default_rng(SEED)
    conds = np.zeros((n, 64), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)

    for i in range(n):
        pose = sample_pose(rng)
        traj = planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)
        # plan_grasp_heuristic devuelve (1, 16, 7); extraer
        trajs[i] = traj[0]
        cond = planner.encode_observation(pose)  # (1, 64) torch
        conds[i] = cond.cpu().numpy()[0]

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{n}")

    out = DATASET_DIR / "heuristic.pt"
    torch.save({
        "conds": torch.from_numpy(conds),
        "trajs": torch.from_numpy(trajs),
        "source": "heuristic",
        "seed": SEED,
    }, out)
    logger.info(f"escrito: {out} ({n} trayectorias)")


def phase_executed(n: int = N_EXECUTED) -> None:
    """Genera n trayectorias ejecutadas en CoppeliaSim. STUB — implementado en Task 5."""
    raise NotImplementedError("Phase A.2 se implementa en Task 5 del plan")


def phase_split() -> None:
    """Combina heurístic + executed y hace train/val split. STUB — Task 6."""
    raise NotImplementedError("Phase A.3 se implementa en Task 6 del plan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["heuristic", "executed", "split", "all"],
                        default="all")
    args = parser.parse_args()

    if args.phase in ("heuristic", "all"):
        phase_heuristic()
    if args.phase in ("executed", "all"):
        phase_executed()
    if args.phase in ("split", "all"):
        phase_split()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
