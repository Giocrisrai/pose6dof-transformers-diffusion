#!/usr/bin/env python3
"""Genera el clip del demo reel para Iter 7c (curriculum + best-of-N + fix IK).

Reproduce la mejor corrida del eval Iter 7c (por defecto la pose i=30, la de
mejor grasp+deposit+IK) con captura de frames, y compila demo_v7c.mp4 + metadata.
Mismo setup que eval_diffusion_iter7b_sim.py: policy v7a_phase2, visual_encoder_iter5,
best-of-8, y el fix de IK (re-seed desde home) ya en pick_with_dp.

Uso (CoppeliaSim en :23000):
    python experiments/make_reel_clip_iter7c.py --pose-index 30 --torch-seed 2026
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
from experiments.run_pick_with_diffusion import compile_mp4, pick_with_dp
from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("reel_clip_7c")

SCENE = REPO / "data/scenes/bin_base.ttt"
OUT = REPO / "experiments/results/pick_with_diffusion"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-index", type=int, default=30,
                        help="Índice de la pose del eval (seed 2026) a reproducir.")
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--torch-seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(REPO / "data/models/diffusion_policy_v7a_phase2.pth",
                      map_location=device, weights_only=True)
    hd = ckpt["config"]["hidden_dim"]
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device, hidden_dim=hd,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    enc_state = torch.load(REPO / "data/models/visual_encoder_iter5.pth",
                           map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])

    # Recomputar la pose del índice pedido (extracción i+1 del rng seed EVAL_SEED).
    rng = np.random.default_rng(EVAL_SEED)
    pose = None
    for _ in range(args.pose_index + 1):
        pose = sample_pose_eval(rng)
    logger.info(f"pose i={args.pose_index}: XYZ={[round(float(v),4) for v in pose[:3,3]]}")

    frames_dir = OUT / "frames_v7c"
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        result = pick_with_dp(
            planner, pose, bridge, frames_dir=frames_dir,
            visual_encoder=encoder, best_of_n=args.best_of_n,
        )

    mp4 = OUT / "demo_v7c.mp4"
    compiled = compile_mp4(frames_dir, mp4, fps=25)
    metadata = {
        "policy": "diffusion_policy_v7a_phase2 (curriculum) + best-of-8 + fix IK re-seed",
        "pose_index": args.pose_index,
        "torch_seed": args.torch_seed,
        "grasp_proximity_m": result["grasp_proximity_m"],
        "deposit_error_m": result["deposit_error_m"],
        "ik_converged": result["ik_converged"],
        "grasp_plausible": result["grasp_plausible"],
        "deposit_plausible": result["deposit_plausible"],
        "mp4": str(compiled.relative_to(REPO)) if compiled else None,
    }
    (OUT / "metadata_v7c.json").write_text(json.dumps(metadata, indent=2))
    print("\n=== CLIP ITER 7c ===")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
