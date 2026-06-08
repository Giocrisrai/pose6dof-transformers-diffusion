# experiments/make_showcase_reel.py
"""Genera reel_showcase.mp4: hero pick cinematográfico (Iter 7c) + valor."""
from __future__ import annotations

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
from src.simulation.cine_camera import CineCamera
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

SCENE = REPO / "data/scenes/bin_base.ttt"
OUT = REPO / "experiments/results/demo_reel"
WORKSPACE_CENTER = (0.0, -0.30, 0.10)  # centro aproximado bin/deposit
POSE_INDEX = 49
TORCH_SEED = 3
# 16 waypoints × (n_substeps=8 × steps_per_substep=2 + settle=30 por waypoint) = 736
# (verificado: el pick captura 736 frames; progress = i/TOTAL recorre todo el clip)
TOTAL_FRAMES_ESTIM = 16 * (8 * 2 + 30)


def run_hero_pick(frames_dir: Path) -> dict:
    torch.manual_seed(TORCH_SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(REPO / "data/models/diffusion_policy_v7a_phase2.pth",
                      map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    es = torch.load(REPO / "data/models/visual_encoder_iter5.pth",
                    map_location=device, weights_only=True)
    enc = ResNet18RGBDEncoder(out_dim=es["out_dim"]).to(device).eval()
    enc.load_state_dict(es["state_dict"])

    rng = np.random.default_rng(EVAL_SEED)
    pose = None
    for _ in range(POSE_INDEX + 1):
        pose = sample_pose_eval(rng)

    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        cam = CineCamera(bridge)
        cam.create()
        tip = bridge.sim.getObject("/tip")
        state = {"i": 0}

        def hook():
            tcp = tuple(bridge.sim.getObjectPosition(tip, -1))
            progress = min(1.0, state["i"] / TOTAL_FRAMES_ESTIM)
            cam.aim(progress, tcp, WORKSPACE_CENTER)
            cam.capture(frames_dir, state["i"])
            state["i"] += 1

        result = pick_with_dp(planner, pose, bridge, frames_dir=None,
                              visual_encoder=enc, best_of_n=8, frame_hook=hook)
        cam.remove()
    result["frames_captured"] = state["i"]
    return result
