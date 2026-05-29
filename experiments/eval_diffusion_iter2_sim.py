#!/usr/bin/env python3
"""Eval Iter 2 EJECUTADO EN SIM — 50 picks reales.

Mide dp_grasp_plausible_pct y dp_deposit_plausible_pct sobre poses
ejecutadas en CoppeliaSim (no solo geometría). Es el acid test del Iter 2.

Uso (CoppeliaSim running on :23000):
    python experiments/eval_diffusion_iter2_sim.py
    python experiments/eval_diffusion_iter2_sim.py --n 50 --policy-version v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from experiments.run_pick_with_diffusion import pick_with_dp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_iter2_sim")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

EVAL_SEED = 2026  # distinto del training (42) y de Iter 1 eval (999)


def sample_pose_eval(rng: np.random.Generator) -> np.ndarray:
    """Mismo workspace que training; seed distinto."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50,
                        help="Número de picks a ejecutar.")
    parser.add_argument("--policy-version", default="v2",
                        help="v1 | v2. Default v2 (Iter 2).")
    args = parser.parse_args()

    policy_path = REPO / "data" / "models" / f"diffusion_policy_sim_{args.policy_version}.pth"
    if not policy_path.exists():
        logger.error(f"policy no encontrada: {policy_path}")
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(policy_path, map_location=device, weights_only=True)
    hidden_dim = ckpt.get("config", {}).get("hidden_dim", 128)
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
        hidden_dim=hidden_dim,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    logger.info(f"policy: {policy_path.name} (hidden_dim={hidden_dim})")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                result = pick_with_dp(planner, pose, bridge, frames_dir=None)
            results.append({
                "i": i,
                "target_pose_t": result["target_pose_t"],
                "grasp_proximity_m": result["grasp_proximity_m"],
                "deposit_error_m": result["deposit_error_m"],
                "ik_converged": result["ik_converged"],
                "grasp_plausible": result["grasp_plausible"],
                "deposit_plausible": result["deposit_plausible"],
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{args.n} done (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    n_valid = len(results)
    if n_valid == 0:
        logger.error("0 picks válidos — abortando")
        return 1

    grasp_plaus_count = sum(r["grasp_plausible"] for r in results)
    deposit_plaus_count = sum(r["deposit_plausible"] for r in results)
    ik_conv_count = sum(r["ik_converged"] for r in results)

    summary = {
        "n_requested": args.n,
        "n_valid": n_valid,
        "n_skipped": skipped,
        "policy": str(policy_path.relative_to(REPO)),
        "seed": EVAL_SEED,
        "dp_grasp_plausible_pct_sim": 100.0 * grasp_plaus_count / n_valid,
        "dp_deposit_plausible_pct_sim": 100.0 * deposit_plaus_count / n_valid,
        "dp_ik_converged_pct": 100.0 * ik_conv_count / n_valid,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])),
        "mean_deposit_error_m": float(np.mean([r["deposit_error_m"] for r in results])),
        "thresholds_passed": {
            "dp_grasp_plausible_pct_sim >= 50": (100.0 * grasp_plaus_count / n_valid) >= 50,
            "dp_ik_converged_pct >= 90": (100.0 * ik_conv_count / n_valid) >= 90,
        },
        "per_pick": results,
    }

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    out = REPO_OUT / "eval_v2_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    # Print solo el resumen, no `per_pick`
    print()
    print("=== RESUMEN EVAL ITER 2 EN SIM ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    print(f"\nDetalles por pick: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
