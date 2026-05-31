#!/usr/bin/env python3
"""Collector Iter 5b — single-object con deposit phase.

Genera 1000 trayectorias heurísticas con `plan_grasp_heuristic(with_deposit=True)`.
Single object (sin multi-cluttered), foco en aprender el deposit move + release.

Uso:
    python experiments/collect_diffusion_dataset_v5.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from experiments.collect_diffusion_dataset import _capture_rgbd_for_pose, sample_pose

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collect_v5")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v5"
N_HEURISTIC = 1000
SEED = 42
CHUNK_SIZE = 200


def main() -> int:
    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / "heuristic.pt"

    # Resume si existe
    if out.exists():
        d = torch.load(out, weights_only=True)
        if d.get("seed") == SEED:
            done = int(d.get("n_valid", 0))
            if done >= N_HEURISTIC:
                logger.info(f"Ya hay {done} en {out}; nada que hacer")
                return 0
            logger.info(f"Resume desde {done}/{N_HEURISTIC}")
            poses = d["poses"].numpy()
            rgbds = d["rgbds"].numpy()
            trajs = d["trajs"].numpy()
            start = done
        else:
            start = 0
            poses = rgbds = trajs = None
    else:
        start = 0
        poses = rgbds = trajs = None

    if poses is None:
        poses = np.zeros((N_HEURISTIC, 16), dtype=np.float32)
        rgbds = np.zeros((N_HEURISTIC, 4, 224, 224), dtype=np.float32)
        trajs = np.zeros((N_HEURISTIC, 16, 7), dtype=np.float32)

    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu",
    )
    rng = np.random.default_rng(SEED)
    for _ in range(start):
        sample_pose(rng)  # consume RNG to match

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
    i = start
    logger.info(f"Phase A (v5): {N_HEURISTIC - start} trayectorias single-obj + deposit (chunk_size={CHUNK_SIZE})")

    while i < N_HEURISTIC:
        chunk_end = min(i + CHUNK_SIZE, N_HEURISTIC)
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(SCENE)
            bridge.set_stepping(True)
            bridge.start_simulation()
            try:
                while i < chunk_end:
                    pose = sample_pose(rng)
                    rgbd = _capture_rgbd_for_pose(bridge, pose)
                    traj = planner.plan_grasp_heuristic(
                        pose, approach_distance=0.15, lift_height=0.10,
                        with_deposit=True,
                    )
                    trajs[i] = traj[0]
                    rgbds[i] = rgbd
                    poses[i] = pose.flatten().astype(np.float32)
                    i += 1
                    if i % 50 == 0:
                        logger.info(f"  {i}/{N_HEURISTIC}")
            finally:
                bridge.stop_simulation()

        torch.save({
            "poses": torch.from_numpy(poses),
            "rgbds": torch.from_numpy(rgbds),
            "trajs": torch.from_numpy(trajs),
            "source": "heuristic_with_deposit",
            "seed": SEED,
            "n_valid": i,
        }, out)
        logger.info(f"  checkpoint: {i}/{N_HEURISTIC} guardado ({out.stat().st_size / 1e6:.0f} MB)")

    # phase split + emparejar nombre estándar
    train_path = DATASET_DIR / "train.pt"
    val_path = DATASET_DIR / "val.pt"
    n = N_HEURISTIC
    rng_split = np.random.default_rng(SEED + 2)
    indices = rng_split.permutation(n)
    train_n = int(0.9 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]
    poses_t = torch.from_numpy(poses)
    rgbds_t = torch.from_numpy(rgbds)
    trajs_t = torch.from_numpy(trajs)
    torch.save({
        "poses": poses_t[train_idx], "rgbds": rgbds_t[train_idx], "trajs": trajs_t[train_idx],
        "split": "train",
    }, train_path)
    torch.save({
        "poses": poses_t[val_idx], "rgbds": rgbds_t[val_idx], "trajs": trajs_t[val_idx],
        "split": "val",
    }, val_path)
    logger.info(f"escrito: train.pt ({train_n}), val.pt ({n - train_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
