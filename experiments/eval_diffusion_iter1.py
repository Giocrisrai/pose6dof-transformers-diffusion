#!/usr/bin/env python3
"""Evalua la DP entrenada (Iter 1) sobre 20 poses no vistas.

Metricas:
- mse_dp_vs_heuristic: distancia promedio entre trayectoria DP y heuristica
- dp_avg_proximity_at_grasp_phase: distancia del waypoint medio (k=8) al cubo
  (proxy de plausibilidad sin ejecutar en sim).

NO ejecuta en sim (eso seria caro por 20 picks). Solo compara trayectorias.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
POLICY = REPO / "data" / "models" / "diffusion_policy_sim_v1.pth"
N_EVAL = 20
SEED = 999  # distinto del training (42) y del collector A.2 (43)


def sample_pose_eval(rng: np.random.Generator) -> np.ndarray:
    """Mismo workspace que training, distinta seed."""
    x = rng.uniform(0.40, 0.55)
    y = rng.uniform(-0.15, -0.05)
    z = 0.033
    theta = rng.choice([0.0, np.pi / 4, np.pi / 2])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    return pose


def main() -> int:
    if not POLICY.exists():
        print(f"policy no encontrada: {POLICY}")
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
    )
    ckpt = torch.load(POLICY, map_location=device, weights_only=True)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()

    rng = np.random.default_rng(SEED)
    mses = []
    proximities = []
    for i in range(N_EVAL):
        pose = sample_pose_eval(rng)
        traj_dp = planner.plan_grasp(pose, n_samples=1)[0]  # (16, 7)
        traj_heur = planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)[0]  # (16, 7)

        # MSE entre trayectorias (componentes XYZ; ignoramos rot y gripper)
        mse = float(np.mean((traj_dp[:, :3] - traj_heur[:, :3]) ** 2))
        mses.append(mse)

        # Proximity en waypoint k=8 (mid-trajectory)
        target_xyz = pose[:3, 3]
        prox = float(np.linalg.norm(traj_dp[8, :3] - target_xyz))
        proximities.append(prox)

    summary = {
        "n_eval": N_EVAL,
        "policy": str(POLICY.relative_to(REPO)),
        "mse_dp_vs_heuristic_mean": float(np.mean(mses)),
        "mse_dp_vs_heuristic_max": float(np.max(mses)),
        "dp_grasp_proximity_mean_m": float(np.mean(proximities)),
        "dp_grasp_proximity_max_m": float(np.max(proximities)),
        "dp_grasp_plausible_pct": 100.0 * sum(p < 0.05 for p in proximities) / N_EVAL,
        "thresholds_passed": {
            "mse_dp_vs_heuristic_mean < 0.10": float(np.mean(mses)) < 0.10,
            "dp_grasp_plausible_pct >= 70": (100.0 * sum(p < 0.05 for p in proximities) / N_EVAL) >= 70,
        },
    }

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    out = REPO_OUT / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
