#!/usr/bin/env python3
"""Eval BASELINE — heurística geométrica sobre las mismas 50 poses (seed 2026).

Mismo loop que eval_diffusion_iter3_sim.py pero generando la trayectoria con
plan_grasp_heuristic en vez de la Diffusion Policy. Sirve como upper bound /
floor para responder "¿la DP entrenada vale la pena sobre la heurística?".

Uso (CoppeliaSim running on :23000):
    python experiments/eval_heuristic_baseline_sim.py
    python experiments/eval_heuristic_baseline_sim.py --n 50
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

from experiments.eval_diffusion_iter2_sim import EVAL_SEED, sample_pose_eval
from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import (
    _move_tcp_via_ik,
    _setup_ik,
    set_gripper,
    setup_robot_control,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_heuristic")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

GRASP_THRESHOLD_M = 0.05
DEPOSIT_TARGET = [-0.30, -0.30, 0.30]
DEPOSIT_THRESHOLD_M = 0.30


def pick_with_heuristic(planner, pose: np.ndarray, bridge) -> dict:
    """Mismo flujo que pick_with_dp pero waypoints vienen del heurístico."""
    setup_robot_control(bridge)
    env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
    bridge.set_stepping(True)
    bridge.start_simulation()
    sim = bridge.sim
    obj1 = sim.getObject("/object_1")
    tip_h = sim.getObject("/tip")

    sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

    traj = planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)
    waypoints = traj[0]  # (16, 7)

    cube_pos = sim.getObjectPosition(obj1, -1)
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

    cube_end = sim.getObjectPosition(obj1, -1)
    tip_end = sim.getObjectPosition(tip_h, -1)
    ik_converged = len(ik_convergence) > 0 and all(ik_convergence)
    deposit_error_m = math.sqrt(
        (cube_end[0] - DEPOSIT_TARGET[0]) ** 2 +
        (cube_end[1] - DEPOSIT_TARGET[1]) ** 2
    )
    deposit_plausible = deposit_error_m < DEPOSIT_THRESHOLD_M

    bridge.stop_simulation()
    try:
        simIK.eraseEnvironment(env)
    except Exception:
        pass

    return {
        "target_pose_t": pose[:3, 3].tolist(),
        "cube_end": cube_end,
        "tip_end": tip_end,
        "ik_converged": ik_converged,
        "grasp_proximity_m": grasp_proximity_m,
        "deposit_error_m": deposit_error_m,
        "grasp_plausible": grasp_plausible,
        "deposit_plausible": deposit_plausible,
        "n_waypoints": len(waypoints),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    device = "cpu"  # heurística no necesita GPU
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
    )
    logger.info("planner: heurística geométrica (sin red)")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                r = pick_with_heuristic(planner, pose, bridge)
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
        return 1

    gp = sum(r["grasp_plausible"] for r in results)
    dp = sum(r["deposit_plausible"] for r in results)
    ik = sum(r["ik_converged"] for r in results)

    summary = {
        "n_requested": args.n, "n_valid": n_valid, "n_skipped": skipped,
        "planner": "heuristic_geometric",
        "seed": EVAL_SEED,
        "grasp_plausible_pct_sim": 100.0 * gp / n_valid,
        "deposit_plausible_pct_sim": 100.0 * dp / n_valid,
        "ik_converged_pct": 100.0 * ik / n_valid,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])),
        "mean_deposit_error_m": float(np.mean([r["deposit_error_m"] for r in results])),
        "per_pick": results,
    }
    out = REPO_OUT / "eval_heuristic_baseline_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== RESUMEN EVAL HEURÍSTICO BASELINE (sim) ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
