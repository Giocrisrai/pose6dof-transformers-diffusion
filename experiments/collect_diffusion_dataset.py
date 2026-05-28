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
    """Genera n trayectorias ejecutadas en CoppeliaSim.

    Por cada pose:
      1. Mueve /object_1 a la pose target.
      2. Corre approach → descend → grasp+close → lift, capturando TCP
         pose por step.
      3. Submuestrea N steps → 16 waypoints uniformes.
      4. Acción 7-D por waypoint: [x, y, z, rx, ry, rz, gripper] donde
         (rx, ry, rz) es so3_log de la rotación del tip.

    Persiste: data/datasets/sim_pick_v1/executed.pt
    """
    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
    from src.simulation.pick_sequence import (
        _move_tcp_via_ik, _setup_ik, set_gripper, setup_robot_control,
    )
    from src.utils.lie_groups import so3_log

    logger.info(f"Phase A.2: generando {n} trayectorias ejecutadas en sim")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    planner_aux = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu",
    )
    rng = np.random.default_rng(SEED + 1)
    conds = np.zeros((n, 64), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)
    skipped = 0

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

    for i in range(n):
        pose = sample_pose(rng)
        conds[i] = planner_aux.encode_observation(pose).cpu().numpy()[0]

        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                sim = bridge.sim
                obj1 = sim.getObject("/object_1")
                sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

                setup_robot_control(bridge)
                env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
                bridge.set_stepping(True)
                bridge.start_simulation()
                tip_h = sim.getObject("/tip")

                tip_log = []  # list of (xyz, R_3x3, gripper_open)

                def log_tip(gripper_open):
                    p = sim.getObjectPosition(tip_h, -1)
                    M = sim.getObjectMatrix(tip_h, -1)  # 12 elementos (3x4)
                    R = np.array([M[0:3], M[4:7], M[8:11]])
                    tip_log.append((p, R, gripper_open))

                set_gripper(bridge, True)
                for _ in range(20):
                    bridge.step()
                    log_tip(1.0)

                # Approach (30cm above target)
                _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                                 [pose[0, 3], pose[1, 3], 0.30], frames_dir=None, counter=[0])
                for _ in range(5): log_tip(1.0)

                # Descend (cube non-respondable to avoid push)
                sim.setObjectInt32Param(obj1, sim.shapeintparam_respondable, 0)
                _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                                 [pose[0, 3], pose[1, 3], pose[2, 3]], frames_dir=None, counter=[0])
                for _ in range(5): log_tip(1.0)

                # Grasp close
                set_gripper(bridge, False)
                for _ in range(20):
                    bridge.step()
                    log_tip(0.0)

                # Lift
                _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                                 [pose[0, 3], pose[1, 3], 0.40], frames_dir=None, counter=[0])
                for _ in range(5): log_tip(0.0)

                bridge.stop_simulation()
                try: simIK.eraseEnvironment(env)
                except Exception: pass

            # Submuestrear a 16 waypoints
            n_logged = len(tip_log)
            if n_logged < 16:
                logger.warning(f"  [{i}] solo {n_logged} steps logged, skipping")
                skipped += 1
                continue

            indices = np.linspace(0, n_logged - 1, 16).astype(int)
            for k, idx in enumerate(indices):
                p, R, g = tip_log[idx]
                rot_vec = so3_log(R)
                trajs[i, k] = [p[0], p[1], p[2], rot_vec[0], rot_vec[1], rot_vec[2], g]

            if (i + 1) % 5 == 0:
                logger.info(f"  {i+1}/{n} (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    if skipped > 0:
        logger.warning(f"Phase A.2: {skipped}/{n} trayectorias saltadas")

    n_valid = n - skipped
    out = DATASET_DIR / "executed.pt"
    torch.save({
        "conds": torch.from_numpy(conds[:n_valid]),
        "trajs": torch.from_numpy(trajs[:n_valid]),
        "source": "executed",
        "seed": SEED + 1,
    }, out)
    logger.info(f"escrito: {out} ({n_valid} trayectorias)")


def phase_split() -> None:
    """Combina heuristic.pt + executed.pt y hace 80/20 split."""
    logger.info("Phase A.3: combinando + split 80/20")
    heur_path = DATASET_DIR / "heuristic.pt"
    exec_path = DATASET_DIR / "executed.pt"
    if not heur_path.exists() or not exec_path.exists():
        raise FileNotFoundError(
            "Faltan datasets parciales. Corré --phase heuristic y --phase executed primero."
        )

    h = torch.load(heur_path, weights_only=True)
    e = torch.load(exec_path, weights_only=True)
    all_conds = torch.cat([h["conds"], e["conds"]], dim=0)
    all_trajs = torch.cat([h["trajs"], e["trajs"]], dim=0)
    n = len(all_conds)

    rng = np.random.default_rng(SEED + 2)
    indices = rng.permutation(n)
    train_n = int(0.8 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    torch.save({
        "conds": all_conds[train_idx],
        "trajs": all_trajs[train_idx],
        "split": "train",
    }, DATASET_DIR / "train.pt")
    torch.save({
        "conds": all_conds[val_idx],
        "trajs": all_trajs[val_idx],
        "split": "val",
    }, DATASET_DIR / "val.pt")

    logger.info(f"escrito: train.pt ({train_n}), val.pt ({n - train_n})")


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
