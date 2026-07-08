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
from typing import Callable, Optional

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


def _min_clearance_m(cand, obstacles) -> float:
    """Mínima distancia entre el camino del efector (segmentos entre waypoints
    consecutivos) y los centros de las piezas que NO son el objetivo.

    Verificación geométrica ANTES de actuar: el robot imagina la trayectoria
    y comprueba si pasa demasiado cerca de otra pieza, sin mover el brazo.
    Aproximación: se chequea el camino del efector; la pieza transportada
    cuelga ~9.5 cm por debajo, pero el tránsito post-lift va alto (z≈0.30).
    """
    mind = float("inf")
    pts = np.asarray(cand)[:, :3]
    obs = np.asarray(obstacles, dtype=float)
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        ab = b - a
        denom = float(ab @ ab)
        for o in obs:
            t = 0.0 if denom < 1e-12 else float(np.clip((o - a) @ ab / denom, 0.0, 1.0))
            d = float(np.linalg.norm(a + t * ab - o))
            if d < mind:
                mind = d
    return mind


def pick_with_dp(
    planner,
    pose: np.ndarray,
    bridge,
    frames_dir=None,
    n_substeps: int = 8,
    steps_per_substep: int = 2,
    visual_encoder=None,
    best_of_n: int = 1,
    frame_hook: Optional[Callable[[], None]] = None,
    obstacles=None,
    clearance_m: float = 0.07,
    track_handles=None,
):
    """Ejecuta un pick usando la DP entrenada.

    Args:
        planner: DiffusionGraspPlanner con policy cargada.
        pose: (4,4) matriz SE(3) target.
        bridge: CoppeliaSimBridge con escena ya cargada.
        frames_dir: None para skip frame capture (eval rápido).
        n_substeps / steps_per_substep: pasados a _move_tcp_via_ik.
        visual_encoder: opcional ResNet18RGBDEncoder. Si está presente,
            captura RGB-D y construye cond v3.
        obstacles: opcional, lista de posiciones [x,y,z] de piezas que NO son
            el objetivo (de la percepción). Con best_of_n > 1, los candidatos
            cuyo camino pasa a < clearance_m de un obstáculo se descartan
            ANTES de ejecutar (verify-then-act); si ninguno es seguro, se
            ejecuta el de mayor holgura.
        clearance_m: holgura mínima exigida al camino del efector (default
            7 cm: media pieza de 5 cm + margen de pinza).

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

    # Conditioning: visual (Iter 3) o solo pose (v1/v2)
    visual_emb = None
    if visual_encoder is not None:
        from experiments.collect_diffusion_dataset import _capture_rgbd_for_pose
        rgbd = _capture_rgbd_for_pose(bridge, pose)
        rgbd_t = torch.from_numpy(rgbd).unsqueeze(0).to(planner.device)
        with torch.no_grad():
            visual_emb = visual_encoder(rgbd_t).cpu().numpy()[0]
    cond = planner.encode_observation(pose, visual_emb=visual_emb)

    # Generar trayectoria con la policy.
    # best_of_n > 1 (Iter 7b): muestrea N trayectorias estocásticas y selecciona
    # la de menor proximidad de grasp al cubo (pose conocida vía percepción).
    # Coste de sim idéntico — solo se EJECUTA la mejor; el sampling de difusión es barato.
    n = max(1, best_of_n)
    traj = planner.plan_grasp(pose, n_samples=n, cond=cond)
    n_unsafe = 0
    clearance_selected = None
    if n > 1:
        cube_xyz = [float(v) for v in pose[:3, 3]]
        proxs = []
        for cand_i in range(n):
            cand = traj[cand_i]
            g_idx = next(
                (k for k in range(len(cand)) if float(cand[k, 6]) < 0.5), 8,
            )
            proxs.append(math.sqrt(
                sum((cube_xyz[j] - float(cand[g_idx, j])) ** 2 for j in range(3))
            ))
        if obstacles:
            # verify-then-act: descartar candidatos que pasan demasiado cerca
            # de otra pieza; elegir el que mejor apunta ENTRE LOS SEGUROS
            clears = [_min_clearance_m(traj[k], obstacles) for k in range(n)]
            seguros = [k for k in range(n) if clears[k] >= clearance_m]
            n_unsafe = n - len(seguros)
            pool = seguros if seguros else [max(range(n), key=lambda k: clears[k])]
            best_i = min(pool, key=lambda k: proxs[k])
            clearance_selected = clears[best_i]
        else:
            best_i = min(range(n), key=lambda k: proxs[k])
        waypoints = traj[best_i]
    else:
        waypoints = traj[0]
        if obstacles:
            clearance_selected = _min_clearance_m(waypoints, obstacles)

    # PROXIMITY pre-snap: distance entre el waypoint del grasp (primera vez que
    # gripper cruza < 0.5 = "cerrándose") y la pose del cubo. Mide si la DP
    # "apunta" al cubo en el momento del grasp.
    # Compat: v1-v4 (sin deposit) tienen grasp en k=8; v5 (con deposit) en k=5.
    cube_pos = sim.getObjectPosition(obj1, -1)
    grasp_idx = next(
        (k for k in range(len(waypoints)) if float(waypoints[k, 6]) < 0.5),
        8,  # fallback al k=8 si nunca cierra (no debería pasar)
    )
    grasp_wp = waypoints[grasp_idx]
    grasp_proximity_m = math.sqrt(
        sum((cube_pos[i] - float(grasp_wp[i])) ** 2 for i in range(3))
    )
    grasp_plausible = grasp_proximity_m < GRASP_THRESHOLD_M

    if frames_dir is not None:
        for f in frames_dir.glob("*.png"): f.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar waypoints (con attach/release automático en transición de gripper)
    counter = [0]
    ik_convergence = []
    prev_gripper = 1.0
    attached = False
    for i, wp in enumerate(waypoints):
        x, y, z, _, _, _, gripper = wp.tolist()
        gripper_open = gripper > 0.5
        prev_open = prev_gripper > 0.5
        if gripper_open != prev_open:
            set_gripper(bridge, gripper_open)
            prev_gripper = gripper
            if not gripper_open and not attached:
                # Cierre del gripper → snap+attach (técnica estándar sim)
                tip_pos = sim.getObjectPosition(tip_h, -1)
                sim.setObjectInt32Param(obj1, sim.shapeintparam_respondable, 0)
                sim.setObjectInt32Param(obj1, sim.shapeintparam_static, 1)
                sim.setObjectPosition(obj1, -1, tip_pos)
                sim.setObjectParent(obj1, tip_h, True)
                attached = True
            elif gripper_open and attached:
                # Apertura del gripper → release (cubo cae)
                sim.setObjectParent(obj1, -1, True)
                sim.setObjectInt32Param(obj1, sim.shapeintparam_respondable, 1)
                sim.setObjectInt32Param(obj1, sim.shapeintparam_static, 0)
                try:
                    sim.resetDynamicObject(obj1)
                except Exception:
                    pass
                attached = False
        _move_tcp_via_ik(
            bridge, env, ik_group, target_dummy, ik_joints, simIK,
            [x, y, z], frames_dir, counter,
            n_substeps=n_substeps, steps_per_substep=steps_per_substep,
            convergence_tracker=ik_convergence,
            frame_hook=frame_hook,
        )

    # Settle post-release para que el cubo asiente
    if not attached:
        for _ in range(30):
            bridge.step()

    cube_end = sim.getObjectPosition(obj1, -1)
    tip_end = sim.getObjectPosition(tip_h, -1)
    # posiciones de objetos rastreados (p.ej. distractores) ANTES del stop:
    # al detener la simulación CoppeliaSim restaura las posiciones iniciales
    tracked_end = ([sim.getObjectPosition(h, -1) for h in track_handles]
                   if track_handles else [])
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
        "n_candidates_unsafe": n_unsafe,
        "clearance_m_selected": clearance_selected,
        "tracked_end": tracked_end,
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
    ckpt = torch.load(POLICY, map_location=device, weights_only=True)
    hidden_dim = ckpt.get("config", {}).get("hidden_dim", 128)
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
        hidden_dim=hidden_dim,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    logger.info(f"policy cargada: {POLICY.name} (hidden_dim={hidden_dim})")

    # 1.b Auto-cargar visual encoder si DP_VERSION=v3
    visual_encoder = None
    if POLICY_VERSION == "v3":
        from src.planning.visual_encoder import ResNet18RGBDEncoder
        enc_path = REPO / "data" / "models" / "visual_encoder_iter3.pth"
        if not enc_path.exists():
            logger.error(f"DP_VERSION=v3 requiere {enc_path}. Corré precompute_visual_cond.py.")
            return 1
        enc_state = torch.load(enc_path, map_location=device, weights_only=True)
        visual_encoder = ResNet18RGBDEncoder(out_dim=enc_state.get("out_dim", 52)).to(device).eval()
        visual_encoder.load_state_dict(enc_state["state_dict"])
        logger.info(f"visual encoder: {enc_path.name}")

    # 2. Obtener target pose
    pose, source_label = get_target_pose(args)
    logger.info(f"target: t={pose[:3,3].tolist()}, source={source_label}")

    # 3. Ejecutar pick + capture frames
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = REPO_OUT / "frames"
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        result = pick_with_dp(
            planner, pose, bridge, frames_dir=frames_dir,
            visual_encoder=visual_encoder,
        )

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
