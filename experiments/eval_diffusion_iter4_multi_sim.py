#!/usr/bin/env python3
"""Eval Iter 4 — DP v4 + ResNet-18 sobre escenas multi-object (3-8 cubos).

Mismas seed que iter3 (2026) para comparabilidad de poses target. Cada escena
agrega n_distractors ∈ [2,7] random en posiciones non-overlapping.

Uso (CoppeliaSim running on :23000):
    python experiments/eval_diffusion_iter4_multi_sim.py --n 50
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.multi_object_scene import (
    measure_collision, setup_multi_object_scene,
)
from src.simulation.pick_sequence import (
    _move_tcp_via_ik, _setup_ik, set_gripper, setup_robot_control,
)
from experiments.eval_diffusion_iter2_sim import EVAL_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_iter4_multi")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

GRASP_THRESHOLD_M = 0.05
DEPOSIT_TARGET = [-0.30, -0.30, 0.30]
DEPOSIT_THRESHOLD_M = 0.30
N_CUBES_RANGE = (3, 8)


def _pose_from_position(position: np.ndarray, theta: float = 0.0) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position[:3]
    return pose


def _capture_rgbd_only(bridge) -> np.ndarray:
    import torch.nn.functional as F
    for _ in range(3):
        bridge.step()
    rgb, depth = bridge.capture_rgbd()
    rgb_f = rgb.astype(np.float32) / 255.0
    depth_clip = np.clip(depth, 0.05, 2.0)
    depth_norm = (depth_clip - 0.05) / (2.0 - 0.05)
    depth_norm = depth_norm[..., None]
    rgbd = np.concatenate([rgb_f, depth_norm], axis=-1)
    rgbd_t = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)
    rgbd_resized = F.interpolate(rgbd_t, size=(224, 224), mode="bilinear", align_corners=False)
    return rgbd_resized.squeeze(0).numpy().astype(np.float32)


def pick_with_dp_multi(planner, encoder, n_cubes: int, rng, bridge) -> dict:
    setup_robot_control(bridge)
    env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
    bridge.set_stepping(True)
    bridge.start_simulation()
    sim = bridge.sim

    handles, positions = setup_multi_object_scene(sim, n_cubes, rng)
    target_h = handles[0]
    distractor_handles = handles[1:]
    distractor_pos0 = positions[1:].copy()
    theta = float(rng.choice([0.0, np.pi / 4, np.pi / 2]))
    target_pose = _pose_from_position(positions[0], theta)

    rgbd = _capture_rgbd_only(bridge)
    rgbd_t = torch.from_numpy(rgbd).unsqueeze(0).to(planner.device)
    with torch.no_grad():
        visual_emb = encoder(rgbd_t).cpu().numpy()[0]

    cond = planner.encode_observation(target_pose, visual_emb=visual_emb)
    traj = planner.plan_grasp(target_pose, n_samples=1, cond=cond)
    waypoints = traj[0]

    cube_pos = sim.getObjectPosition(target_h, -1)
    grasp_wp = waypoints[8]
    grasp_proximity_m = math.sqrt(
        sum((cube_pos[i] - float(grasp_wp[i])) ** 2 for i in range(3))
    )
    grasp_plausible = grasp_proximity_m < GRASP_THRESHOLD_M

    counter = [0]
    ik_convergence: list[bool] = []
    prev_gripper = 1.0
    for i, wp in enumerate(waypoints):
        x, y, z, _, _, _, gripper = wp.tolist()
        if (gripper > 0.5) != (prev_gripper > 0.5):
            set_gripper(bridge, gripper > 0.5)
            prev_gripper = gripper
        _move_tcp_via_ik(
            bridge, env, ik_group, target_dummy, ik_joints, simIK,
            [x, y, z], None, counter,
            n_substeps=8, steps_per_substep=2,
            convergence_tracker=ik_convergence,
        )

    cube_end = sim.getObjectPosition(target_h, -1)
    ik_converged = len(ik_convergence) > 0 and all(ik_convergence)
    deposit_error_m = math.sqrt(
        (cube_end[0] - DEPOSIT_TARGET[0]) ** 2 +
        (cube_end[1] - DEPOSIT_TARGET[1]) ** 2
    )
    deposit_plausible = deposit_error_m < DEPOSIT_THRESHOLD_M
    collided, max_disp = measure_collision(sim, distractor_handles, distractor_pos0)

    bridge.stop_simulation()
    try:
        simIK.eraseEnvironment(env)
    except Exception:
        pass

    return {
        "n_cubes": n_cubes,
        "n_distractors": n_cubes - 1,
        "target_pose_t": positions[0].tolist(),
        "cube_end": cube_end,
        "ik_converged": ik_converged,
        "grasp_proximity_m": grasp_proximity_m,
        "deposit_error_m": deposit_error_m,
        "grasp_plausible": grasp_plausible,
        "deposit_plausible": deposit_plausible,
        "distractor_collided": collided,
        "max_distractor_displacement_m": max_disp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy_path = REPO / "data" / "models" / "diffusion_policy_sim_v4.pth"
    encoder_ckpt = REPO / "data" / "models" / "visual_encoder_iter4.pth"
    if not policy_path.exists() or not encoder_ckpt.exists():
        logger.error("policy o encoder no encontrados; corre train + precompute primero.")
        return 1

    ckpt = torch.load(policy_path, map_location=device, weights_only=True)
    hd = ckpt.get("config", {}).get("hidden_dim", 256)
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device, hidden_dim=hd
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    enc_state = torch.load(encoder_ckpt, map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state.get("out_dim", 52)).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])
    logger.info(f"policy: v4 (hidden_dim={hd}) + ResNet-18 ({encoder_ckpt.name})")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        try:
            n_cubes = int(rng.integers(N_CUBES_RANGE[0], N_CUBES_RANGE[1] + 1))
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                r = pick_with_dp_multi(planner, encoder, n_cubes, rng, bridge)
            results.append({"i": i, **r})
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{args.n} done (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    n_valid = len(results)
    if n_valid == 0:
        return 1

    gp = sum(r["grasp_plausible"] for r in results)
    coll = sum(r["distractor_collided"] for r in results)
    ik = sum(r["ik_converged"] for r in results)
    success_no_coll = sum(
        1 for r in results if r["grasp_plausible"] and not r["distractor_collided"]
    )

    summary = {
        "n_requested": args.n, "n_valid": n_valid, "n_skipped": skipped,
        "planner": "dp_v4_multi", "seed": EVAL_SEED,
        "grasp_plausible_pct_sim": 100.0 * gp / n_valid,
        "distractor_collision_pct": 100.0 * coll / n_valid,
        "ik_converged_pct": 100.0 * ik / n_valid,
        "grasp_success_with_no_collision_pct": 100.0 * success_no_coll / n_valid,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])),
        "mean_max_distractor_displacement_m": float(
            np.mean([r["max_distractor_displacement_m"] for r in results])
        ),
        "per_pick": results,
    }
    out = REPO_OUT / "eval_v4_multi_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== RESUMEN EVAL DP v4 MULTI (sim) ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
