#!/usr/bin/env python3
"""Eval Iter 3 EJECUTADO EN SIM — 50 picks reales con conditioning visual.

Usa ResNet-18 sobre RGB-D del sim como cond para la Diffusion Policy v3.

Uso (CoppeliaSim running on :23000):
    python experiments/eval_diffusion_iter3_sim.py
    python experiments/eval_diffusion_iter3_sim.py --n 50
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

from experiments.eval_diffusion_iter2_sim import EVAL_SEED, sample_pose_eval
from experiments.run_pick_with_diffusion import pick_with_dp
from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_iter3_sim")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy_path = REPO / "data" / "models" / "diffusion_policy_sim_v3.pth"
    if not policy_path.exists():
        logger.error(f"policy no encontrada: {policy_path}")
        return 1

    ckpt = torch.load(policy_path, map_location=device, weights_only=True)
    hidden_dim = ckpt.get("config", {}).get("hidden_dim", 256)
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
        hidden_dim=hidden_dim,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    encoder_ckpt = REPO / "data" / "models" / "visual_encoder_iter3.pth"
    if not encoder_ckpt.exists():
        logger.error(f"encoder ckpt no encontrado: {encoder_ckpt}. Corré precompute_visual_cond.py.")
        return 1
    enc_state = torch.load(encoder_ckpt, map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state.get("out_dim", 52)).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])
    logger.info(f"policy: v3 (hidden_dim={hidden_dim}) + ResNet-18 RGB-D ({encoder_ckpt.name})")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                r = pick_with_dp(
                    planner, pose, bridge, frames_dir=None,
                    visual_encoder=encoder,
                )
            results.append({
                "i": i,
                "target_pose_t": r["target_pose_t"],
                "grasp_proximity_m": r["grasp_proximity_m"],
                "deposit_error_m": r["deposit_error_m"],
                "ik_converged": r["ik_converged"],
                "grasp_plausible": r["grasp_plausible"],
                "deposit_plausible": r["deposit_plausible"],
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{args.n} done (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    n_valid = len(results)
    if n_valid == 0:
        logger.error("0 picks válidos")
        return 1

    gp = sum(r["grasp_plausible"] for r in results)
    dp = sum(r["deposit_plausible"] for r in results)
    ik = sum(r["ik_converged"] for r in results)

    summary = {
        "n_requested": args.n, "n_valid": n_valid, "n_skipped": skipped,
        "policy": str(policy_path.relative_to(REPO)), "seed": EVAL_SEED,
        "dp_grasp_plausible_pct_sim": 100.0 * gp / n_valid,
        "dp_deposit_plausible_pct_sim": 100.0 * dp / n_valid,
        "dp_ik_converged_pct": 100.0 * ik / n_valid,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])),
        "mean_deposit_error_m": float(np.mean([r["deposit_error_m"] for r in results])),
        "thresholds_passed": {
            "dp_grasp_plausible_pct_sim >= 55": 100.0 * gp / n_valid >= 55,
            "dp_ik_converged_pct >= 90": 100.0 * ik / n_valid >= 90,
        },
        "per_pick": results,
    }
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    out = REPO_OUT / "eval_v3_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== RESUMEN EVAL ITER 3 EN SIM ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    print(f"\nDetalles por pick: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
