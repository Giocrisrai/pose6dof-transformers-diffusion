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

DATASET_VERSION = "v4"
DATASET_DIR = REPO / "data" / "datasets" / f"sim_pick_{DATASET_VERSION}"
N_HEURISTIC = 2000
N_EXECUTED = 200
SEED = 42

# Workspace bounds (matching bin_base.ttt geometry)
X_RANGE = (0.40, 0.55)
Y_RANGE = (-0.15, -0.05)
Z_FIXED = 0.033

# Iter 4: clutter multi-objeto (3-8 cubos por escena)
N_CUBES_RANGE = (3, 8)
MAX_DISTRACTORS = N_CUBES_RANGE[1] - 1  # 7


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


def _capture_rgbd_only(bridge) -> np.ndarray:
    """Captura RGB-D + resize a (4, 224, 224) float32. NO mueve objetos."""
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


def _capture_rgbd_for_pose(bridge, pose: np.ndarray) -> np.ndarray:
    """Coloca el cubo en la pose, captura RGB-D, devuelve tensor (4, 224, 224) float32."""
    sim = bridge.sim
    obj1 = sim.getObject("/object_1")
    sim.setObjectPosition(obj1, -1, [float(p) for p in pose[:3, 3]])
    return _capture_rgbd_only(bridge)


def _pose_from_position(position: np.ndarray, theta: float = 0.0) -> np.ndarray:
    """Construye pose 4x4 desde (x,y,z) + ángulo Z."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position[:3]
    return pose


def phase_heuristic(n: int = N_HEURISTIC, chunk_size: int = 200) -> None:
    """Genera n trayectorias heurísticas con multi-object scene (Iter 4).

    Chunked: cada chunk_size iters cierra y reabre el bridge (resiliente a
    hangs de CoppeliaSim cuando el Mac duerme). Persiste tras cada chunk en
    heuristic.pt como dict con n_valid_so_far. Si se interrumpe, retomar
    corriendo el script de nuevo: carga el dict existente y continúa desde
    n_valid_so_far.
    """
    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
    from src.simulation.multi_object_scene import setup_multi_object_scene

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / "heuristic.pt"

    if out.exists():
        d = torch.load(out, weights_only=True)
        if d.get("source") == "heuristic" and d.get("seed") == SEED:
            done = int(d.get("n_valid", 0))
            if done >= n:
                logger.info(f"Phase A.1 (v4): ya hay {done} trayectorias en {out}; nada que hacer")
                return
            logger.info(f"Phase A.1 (v4): resume desde {done}/{n}")
            poses = d["poses"].numpy()
            rgbds = d["rgbds"].numpy()
            trajs = d["trajs"].numpy()
            n_distractors_arr = d["n_distractors"].numpy()
            distractor_pos = d["distractor_positions"].numpy()
            start = done
        else:
            logger.info("Phase A.1 (v4): heuristic.pt existe pero no es resumible — sobrescribiendo")
            start = 0
            poses = rgbds = trajs = n_distractors_arr = distractor_pos = None
    else:
        start = 0
        poses = rgbds = trajs = n_distractors_arr = distractor_pos = None

    if poses is None:
        poses = np.zeros((n, 16), dtype=np.float32)
        rgbds = np.zeros((n, 4, 224, 224), dtype=np.float32)
        trajs = np.zeros((n, 16, 7), dtype=np.float32)
        n_distractors_arr = np.zeros((n,), dtype=np.int32)
        distractor_pos = np.full((n, MAX_DISTRACTORS, 3), np.nan, dtype=np.float32)

    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu",
    )
    rng = np.random.default_rng(SEED)
    # Advance RNG to where we left off
    for _ in range(start):
        rng.integers(N_CUBES_RANGE[0], N_CUBES_RANGE[1] + 1)
        rng.uniform(0, 1, size=2 * 10)  # consume sample_non_overlapping_positions usage approx
        rng.integers(0, 2, size=8)  # paint distractor colors
        rng.choice([0.0, np.pi / 4, np.pi / 2])
    # NOTE: RNG resume es aproximado (no exacto). Las trayectorias post-resume
    # serán deterministas pero distintas de las que habría dado un run sin interrupciones.

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
    skipped = 0
    i = start
    logger.info(f"Phase A.1 (v4): generando {n - start} trayectorias multi-object (chunk_size={chunk_size})")

    while i < n:
        chunk_end = min(i + chunk_size, n)
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(SCENE)
            bridge.set_stepping(True)
            bridge.start_simulation()
            try:
                while i < chunk_end:
                    try:
                        n_cubes = int(rng.integers(N_CUBES_RANGE[0], N_CUBES_RANGE[1] + 1))
                        handles, positions = setup_multi_object_scene(bridge.sim, n_cubes, rng)
                        theta = float(rng.choice([0.0, np.pi / 4, np.pi / 2]))
                        target_pose = _pose_from_position(positions[0], theta)
                        rgbd = _capture_rgbd_only(bridge)
                        traj = planner.plan_grasp_heuristic(
                            target_pose, approach_distance=0.15, lift_height=0.10
                        )
                        trajs[i] = traj[0]
                        rgbds[i] = rgbd
                        poses[i] = target_pose.flatten().astype(np.float32)
                        n_distractors_arr[i] = n_cubes - 1
                        distractor_pos[i, : n_cubes - 1] = positions[1:n_cubes]
                    except RuntimeError as e:
                        logger.warning(f"  [{i}] skip: {e}")
                        skipped += 1
                    i += 1
                    if i % 50 == 0:
                        logger.info(f"  {i}/{n} (skipped {skipped})")
            finally:
                bridge.stop_simulation()

        # Checkpoint tras cada chunk
        torch.save({
            "poses": torch.from_numpy(poses),
            "rgbds": torch.from_numpy(rgbds),
            "trajs": torch.from_numpy(trajs),
            "n_distractors": torch.from_numpy(n_distractors_arr),
            "distractor_positions": torch.from_numpy(distractor_pos),
            "source": "heuristic",
            "seed": SEED,
            "n_valid": i,
        }, out)
        logger.info(f"  checkpoint: {i}/{n} guardado ({out.stat().st_size / 1e6:.0f} MB)")

    logger.info(f"escrito: {out} ({n - skipped} válidos)")


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

    from src.simulation.multi_object_scene import setup_multi_object_scene

    rng = np.random.default_rng(SEED + 1)
    poses = np.zeros((n, 16), dtype=np.float32)
    rgbds = np.zeros((n, 4, 224, 224), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)
    n_distractors_arr = np.zeros((n,), dtype=np.int32)
    distractor_pos = np.full((n, MAX_DISTRACTORS, 3), np.nan, dtype=np.float32)
    skipped = 0

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

    for i in range(n):
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                sim = bridge.sim

                setup_robot_control(bridge)
                env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
                bridge.set_stepping(True)
                bridge.start_simulation()
                tip_h = sim.getObject("/tip")

                # Multi-object scene setup
                n_cubes = int(rng.integers(N_CUBES_RANGE[0], N_CUBES_RANGE[1] + 1))
                handles, positions = setup_multi_object_scene(sim, n_cubes, rng)
                obj1 = handles[0]  # target rojo
                theta = float(rng.choice([0.0, np.pi / 4, np.pi / 2]))
                pose = _pose_from_position(positions[0], theta)

                rgbd_obs = _capture_rgbd_only(bridge)
                poses[i] = pose.flatten().astype(np.float32)
                rgbds[i] = rgbd_obs
                n_distractors_arr[i] = n_cubes - 1
                distractor_pos[i, : n_cubes - 1] = positions[1:n_cubes]

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
            # Checkpoint cada 20 picks
            if (i + 1) % 20 == 0:
                n_valid_so_far = (i + 1) - skipped
                out = DATASET_DIR / "executed.pt"
                torch.save({
                    "poses": torch.from_numpy(poses[:n_valid_so_far]),
                    "rgbds": torch.from_numpy(rgbds[:n_valid_so_far]),
                    "trajs": torch.from_numpy(trajs[:n_valid_so_far]),
                    "n_distractors": torch.from_numpy(n_distractors_arr[:n_valid_so_far]),
                    "distractor_positions": torch.from_numpy(distractor_pos[:n_valid_so_far]),
                    "source": "executed",
                    "seed": SEED + 1,
                    "n_valid": n_valid_so_far,
                }, out)
                logger.info(f"  checkpoint: {n_valid_so_far} válidos guardados")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    if skipped > 0:
        logger.warning(f"Phase A.2: {skipped}/{n} trayectorias saltadas")

    n_valid = n - skipped
    out = DATASET_DIR / "executed.pt"
    torch.save({
        "poses": torch.from_numpy(poses[:n_valid]),
        "rgbds": torch.from_numpy(rgbds[:n_valid]),
        "trajs": torch.from_numpy(trajs[:n_valid]),
        "n_distractors": torch.from_numpy(n_distractors_arr[:n_valid]),
        "distractor_positions": torch.from_numpy(distractor_pos[:n_valid]),
        "source": "executed",
        "seed": SEED + 1,
        "n_valid": n_valid,
    }, out)
    logger.info(f"escrito: {out} ({n_valid} trayectorias)")


def phase_split() -> None:
    """Combina heuristic.pt + executed.pt y hace 90/10 split."""
    logger.info("Phase A.3 (v3): combinando + split 90/10")
    heur_path = DATASET_DIR / "heuristic.pt"
    exec_path = DATASET_DIR / "executed.pt"
    if not heur_path.exists() or not exec_path.exists():
        raise FileNotFoundError("Faltan datasets parciales.")

    h = torch.load(heur_path, weights_only=True)
    e = torch.load(exec_path, weights_only=True)
    all_poses = torch.cat([h["poses"], e["poses"]], dim=0)
    all_rgbds = torch.cat([h["rgbds"], e["rgbds"]], dim=0)
    all_trajs = torch.cat([h["trajs"], e["trajs"]], dim=0)
    all_n_distractors = torch.cat([h["n_distractors"], e["n_distractors"]], dim=0)
    all_distractor_pos = torch.cat([h["distractor_positions"], e["distractor_positions"]], dim=0)
    n = len(all_poses)

    rng = np.random.default_rng(SEED + 2)
    indices = rng.permutation(n)
    train_n = int(0.9 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    torch.save({
        "poses": all_poses[train_idx],
        "rgbds": all_rgbds[train_idx],
        "trajs": all_trajs[train_idx],
        "n_distractors": all_n_distractors[train_idx],
        "distractor_positions": all_distractor_pos[train_idx],
        "split": "train",
    }, DATASET_DIR / "train.pt")
    torch.save({
        "poses": all_poses[val_idx],
        "rgbds": all_rgbds[val_idx],
        "trajs": all_trajs[val_idx],
        "n_distractors": all_n_distractors[val_idx],
        "distractor_positions": all_distractor_pos[val_idx],
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
