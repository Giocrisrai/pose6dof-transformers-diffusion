#!/usr/bin/env python3
"""Eval Iter 7b — best-of-N selection sobre DP v7a en CoppeliaSim. 50 picks seed=2026.

Sin reentrenar: muestrea N trayectorias de la policy v7a_phase2 y ejecuta la de
menor proximidad de grasp al cubo (pose conocida vía percepción). Ataca el cuello
de botella de Iter 7a (grasp 58%, 12/21 fallos borderline <8cm) por selección,
no por gradiente. Coste de sim idéntico — el sampling de difusión es barato.

Comparable directamente con Iter 5/6/7a (mismo protocolo, seed 2026).
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
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from experiments.run_pick_with_diffusion import pick_with_dp
from experiments.eval_diffusion_iter2_sim import EVAL_SEED, sample_pose_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_iter7b")

SCENE = REPO / "data/scenes/bin_base.ttt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--best-of-n", type=int, default=8,
                        help="Nº de trayectorias candidatas; se ejecuta la de mejor grasp.")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "data/models/diffusion_policy_v7a_phase2.pth")
    parser.add_argument("--policy-name", default="diffusion_policy_v7a_phase2+bon")
    parser.add_argument("--out", type=Path,
                        default=REPO / "experiments/results/pick_with_diffusion/eval_v7b_sim.json")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    hd = ckpt["config"]["hidden_dim"]
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device, hidden_dim=hd,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    enc_state = torch.load(
        REPO / "data/models/visual_encoder_iter5.pth", map_location=device, weights_only=True
    )
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])
    logger.info(f"policy: {args.policy_name} (hidden_dim={hd}, best_of_n={args.best_of_n})")

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
                    visual_encoder=encoder, best_of_n=args.best_of_n,
                )
            results.append({
                "i": i,
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
    gp = sum(r["grasp_plausible"] for r in results)
    dp = sum(r["deposit_plausible"] for r in results)
    ik = sum(r["ik_converged"] for r in results)
    pp = sum(1 for r in results if r["grasp_plausible"] and r["deposit_plausible"])

    summary = {
        "n_requested": args.n, "n_valid": n_valid, "n_skipped": skipped,
        "policy": args.policy_name,
        "best_of_n": args.best_of_n,
        "seed": EVAL_SEED,
        "dp_grasp_plausible_pct_sim": 100.0 * gp / n_valid if n_valid else 0,
        "dp_deposit_plausible_pct_sim": 100.0 * dp / n_valid if n_valid else 0,
        "dp_ik_converged_pct": 100.0 * ik / n_valid if n_valid else 0,
        "pick_and_place_success_pct": 100.0 * pp / n_valid if n_valid else 0,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])) if results else 0,
        "per_pick": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== RESUMEN EVAL ITER 7b (best-of-{args.best_of_n}) ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
