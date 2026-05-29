#!/usr/bin/env python3
"""Pick-and-place usando la Diffusion Policy entrenada (Iter 1).

Cierra la Brecha B del pipeline: la DP entrenada en Phase B genera
una trayectoria de 16 waypoints, los pasamos a _move_tcp_via_ik para
ejecutarlos en CoppeliaSim.

Uso:
    python experiments/run_pick_with_diffusion.py
    python experiments/run_pick_with_diffusion.py --pose-source fp_ckpt --fp-index 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import (
    _capture_frame,
    _move_tcp_via_ik,
    _setup_ik,
    compile_mp4,
    set_gripper,
    setup_robot_control,
)
from src.simulation.utils import map_fp_pose_to_sim_workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pick_with_dp")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
POLICY_VERSION = os.environ.get("DP_VERSION", "v1")  # default backwards-compat
POLICY = REPO / "data" / "models" / f"diffusion_policy_sim_{POLICY_VERSION}.pth"
FP_CKPT = REPO / "experiments" / "checkpoints" / "fp_ycbv_checkpoint.json"


def get_target_pose(args) -> tuple[np.ndarray, str]:
    """Obtiene la pose target segun --pose-source. Devuelve (pose 4x4, source_label)."""
    if args.pose_source == "groundtruth":
        pose = np.eye(4)
        pose[:3, 3] = [0.46, -0.10, 0.033]
        return pose, "scene_groundtruth"
    elif args.pose_source == "fp_ckpt":
        ckpt = json.loads(FP_CKPT.read_text())
        entry = ckpt["results"][args.fp_index]
        t_mapped = map_fp_pose_to_sim_workspace(entry["t_pred"])
        pose = np.eye(4)
        pose[:3, 3] = t_mapped
        return pose, f"foundation_pose_ckpt[{args.fp_index}]"
    raise ValueError(f"pose_source {args.pose_source} no soportado")


def pick_with_dp(
    planner,
    pose: np.ndarray,
    bridge,
    frames_dir=None,
    n_substeps: int = 8,
    steps_per_substep: int = 2,
):
    """Ejecuta un pick usando la DP entrenada.

    Args:
        planner: DiffusionGraspPlanner con policy cargada.
        pose: (4,4) matriz SE(3) target.
        bridge: CoppeliaSimBridge con escena ya cargada.
        frames_dir: None para skip frame capture (eval rápido).
        n_substeps / steps_per_substep: pasados a _move_tcp_via_ik.

    Returns:
        dict con métricas: {
            'target_pose_t': [x,y,z],
            'cube_end': [x,y,z],
            'tip_end': [x,y,z],
            'ik_converged': bool,
            'grasp_proximity_m': float,
            'deposit_error_m': float,
            'grasp_plausible': bool,
            'deposit_plausible': bool,
            'n_waypoints': int,
            'waypoints': list of 16 lists of 7 floats (debug),
        }
    """
    import math
    from src.simulation.pick_sequence import (
        _move_tcp_via_ik, _setup_ik, set_gripper, setup_robot_control,
    )

    GRASP_THRESHOLD_M = 0.05
    DEPOSIT_TARGET = [-0.30, -0.30, 0.30]
    DEPOSIT_THRESHOLD_M = 0.30

    setup_robot_control(bridge)
    env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
    bridge.set_stepping(True)
    bridge.start_simulation()
    sim = bridge.sim
    obj1 = sim.getObject("/object_1")
    tip_h = sim.getObject("/tip")

    # Mover el cubo al target
    sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

    # Generar trayectoria con la policy
    traj = planner.plan_grasp(pose, n_samples=1)  # (1, 16, 7)
    waypoints = traj[0]

    # PROXIMITY pre-snap: distance entre el waypoint k=8 (donde la heur tiene grasp)
    # y la pose del cubo. Mide si la DP "apunta" al cubo correctamente.
    cube_pos = sim.getObjectPosition(obj1, -1)
    grasp_wp = waypoints[8]
    grasp_proximity_m = math.sqrt(
        sum((cube_pos[i] - float(grasp_wp[i])) ** 2 for i in range(3))
    )
    grasp_plausible = grasp_proximity_m < GRASP_THRESHOLD_M

    if frames_dir is not None:
        for f in frames_dir.glob("*.png"): f.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar waypoints
    counter = [0]
    ik_convergence = []
    prev_gripper = 1.0
    for i, wp in enumerate(waypoints):
        x, y, z, _, _, _, gripper = wp.tolist()
        if (gripper > 0.5) != (prev_gripper > 0.5):
            set_gripper(bridge, gripper > 0.5)
            prev_gripper = gripper
        _move_tcp_via_ik(
            bridge, env, ik_group, target_dummy, ik_joints, simIK,
            [x, y, z], frames_dir, counter,
            n_substeps=n_substeps, steps_per_substep=steps_per_substep,
            convergence_tracker=ik_convergence,
        )

    cube_end = sim.getObjectPosition(obj1, -1)
    tip_end = sim.getObjectPosition(tip_h, -1)
    ik_converged = len(ik_convergence) > 0 and all(ik_convergence)

    # Deposit error (XY only, Z lo ignoramos porque cae por gravedad)
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
        "waypoints": waypoints.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-source", choices=["groundtruth", "fp_ckpt"], default="groundtruth")
    parser.add_argument("--fp-index", type=int, default=0)
    args = parser.parse_args()

    if not POLICY.exists():
        logger.error(f"policy no encontrada: {POLICY}. Corre train_diffusion_on_sim.py primero.")
        return 1

    # 1. Cargar policy
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
    )
    ckpt = torch.load(POLICY, map_location=device, weights_only=True)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    logger.info(f"policy cargada: {POLICY.name}")

    # 2. Obtener target pose
    pose, source_label = get_target_pose(args)
    logger.info(f"target: t={pose[:3,3].tolist()}, source={source_label}")

    # 3. Ejecutar pick + capture frames
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = REPO_OUT / "frames"
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        result = pick_with_dp(planner, pose, bridge, frames_dir=frames_dir)

    # 4. Compilar MP4
    mp4_path = REPO_OUT / "demo.mp4"
    compiled = compile_mp4(frames_dir, mp4_path, fps=25)

    # 5. Reporte
    metadata = {
        "policy": str(POLICY.relative_to(REPO)),
        "pose_source": source_label,
        **result,
        "mp4": str(compiled.relative_to(REPO)) if compiled else None,
    }
    (REPO_OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print("=== RESULTADOS pick con Diffusion Policy ===")
    for k, v in metadata.items():
        if k != "waypoints":  # demasiado verboso
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
