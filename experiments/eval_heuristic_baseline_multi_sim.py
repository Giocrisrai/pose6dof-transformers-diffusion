#!/usr/bin/env python3
"""Eval baseline — heurístico geométrico sobre escenas multi-object (Iter 4).

Mismo escenario que eval_diffusion_iter4_multi_sim.py pero con plan_grasp_heuristic.
Mide grasp_plausible + distractor_collision_pct.

Uso (CoppeliaSim running on :23000):
    python experiments/eval_heuristic_baseline_multi_sim.py --n 50
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.eval_diffusion_iter2_sim import EVAL_SEED
from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.multi_object_scene import (
    measure_collision,
    setup_multi_object_scene,
)
from src.simulation.pick_sequence import (
    _move_tcp_via_ik,
    _setup_ik,
    set_gripper,
    setup_robot_control,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_heur_multi")

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


def pick_with_heuristic_multi(planner, n_cubes: int, rng, bridge) -> dict:
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

    traj = planner.plan_grasp_heuristic(target_pose, approach_distance=0.15, lift_height=0.10)
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

    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu")
    logger.info("planner: heurístico geométrico (multi-object)")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        try:
            n_cubes = int(rng.integers(N_CUBES_RANGE[0], N_CUBES_RANGE[1] + 1))
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                r = pick_with_heuristic_multi(planner, n_cubes, rng, bridge)
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
        "planner": "heuristic_geometric_multi", "seed": EVAL_SEED,
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
    out = REPO_OUT / "eval_heuristic_baseline_multi_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== RESUMEN EVAL HEURÍSTICO BASELINE MULTI (sim) ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
