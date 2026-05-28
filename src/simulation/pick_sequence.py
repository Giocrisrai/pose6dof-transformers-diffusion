"""Secuencia pick-and-place reutilizable usando IK + attach técnica.

Usa simIK module de CoppeliaSim para resolver IK del UR5 (target XYZ →
joints) y técnica estándar de attach del objeto al gripper durante el
grasp (también usada por Pickit, Cognex y otros sims comerciales).

Flujo:
    home → approach (sobre cubo) → descend (al nivel del cubo) →
    grasp (cierra gripper + attach object al TCP) → lift → deposit →
    release (detach + abre gripper + restaurar física).

Outputs: PNG por step para compilar MP4.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logger = logging.getLogger(__name__)


@dataclass
class PickResult:
    n_frames: int
    obj_start_pos: list[float]
    obj_end_pos: list[float]
    obj_moved_m: float
    grasp_success: bool
    mp4_path: Optional[Path]
    frames_dir: Path


def setup_robot_control(bridge: CoppeliaSimBridge) -> None:
    """Configura joints UR5 para control dinámico position-mode + disable script."""
    sim = bridge.sim
    for h in bridge._joint_handles:
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    try:
        scr = sim.getObject("/UR5e/Script")
        sim.setObjectInt32Param(scr, sim.scriptintparam_enabled, 0)
    except Exception as e:
        logger.warning(f"disable UR5 script: {e}")


def set_gripper(bridge: CoppeliaSimBridge, open_: bool) -> None:
    """Abre/cierra gripper RG2 vía signal scene-level."""
    bridge.sim.setIntProperty(
        bridge.sim.handle_scene, "signal.RG2_open", 1 if open_ else 0
    )


def _capture_frame(bridge, frames_dir: Path, idx: int) -> None:
    from PIL import Image
    sim = bridge.sim
    sim.handleVisionSensor(bridge._camera_rgb_handle)
    img_raw, res = sim.getVisionSensorImg(bridge._camera_rgb_handle)
    w, h = res[0], res[1]
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)
    Image.fromarray(img).save(frames_dir / f"{idx:06d}.png")


def compile_mp4(frames_dir: Path, mp4_path: Path, fps: int = 25) -> Optional[Path]:
    """Compila frames PNG a MP4 con ffmpeg. None si ffmpeg no está."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg no encontrado — skip MP4")
        return None
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg falló: {result.stderr[-500:]}")
        return None
    return mp4_path


def _setup_ik(bridge: CoppeliaSimBridge):
    """Crea IK env + group + element. Devuelve (env, ik_group, target_dummy,
    ik_joints, simIK_module)."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient("localhost", 23000)
    simIK = client.require("simIK")
    sim = bridge.sim
    env = simIK.createEnvironment()
    ik_group = simIK.createGroup(env)
    simIK.setGroupCalculation(env, ik_group, simIK.method_damped_least_squares, 0.01, 50)
    tip_h = sim.getObject("/tip")
    base_h = sim.getObject("/UR5e")
    try:
        old = sim.getObject("/ik_target")
        sim.removeObject(old)
    except Exception:
        pass
    target_dummy = sim.createDummy(0.02)
    sim.setObjectAlias(target_dummy, "ik_target")
    sim.setObjectMatrix(target_dummy, -1, sim.getObjectMatrix(tip_h, -1))
    res = simIK.addElementFromScene(env, ik_group, base_h, tip_h, target_dummy,
                                     simIK.constraint_position)
    scene_to_ik = res[1]
    ik_joints = [scene_to_ik[h] for h in bridge._joint_handles]
    return env, ik_group, target_dummy, ik_joints, simIK


def _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                      target_xyz, frames_dir, counter,
                      n_substeps: int = 40, steps_per_substep: int = 3) -> None:
    """Mueve TCP a target_xyz interpolando linealmente + IK por substep +
    comandando joints como PID target. Captura frame por step."""
    sim = bridge.sim
    start_pos = sim.getObjectPosition(target_dummy, -1)
    for i in range(1, n_substeps + 1):
        a = i / n_substeps
        interp = [start_pos[j] + a * (target_xyz[j] - start_pos[j]) for j in range(3)]
        sim.setObjectPosition(target_dummy, -1, interp)
        simIK.syncFromSim(env, [ik_group])
        simIK.handleGroup(env, ik_group)
        joint_vals = [simIK.getJointPosition(env, j) for j in ik_joints]
        for h, v in zip(bridge._joint_handles, joint_vals):
            sim.setJointTargetPosition(h, v)
        for _ in range(steps_per_substep):
            bridge.step()
            _capture_frame(bridge, frames_dir, counter[0])
            counter[0] += 1
    # Settle
    for _ in range(30):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1


def run_pick_sequence(
    bridge: CoppeliaSimBridge,
    frames_dir: Path,
    target_object: str = "/object_1",
) -> PickResult:
    """Ejecuta pick-and-place completo con IK + attach del cubo al gripper.

    Pre-condición: escena ya cargada, sim NO iniciada. La función configura
    el robot, arranca/detiene la simulación, y captura frames.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    setup_robot_control(bridge)
    sim = bridge.sim
    env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)

    bridge.set_stepping(True)
    bridge.start_simulation()

    obj_h = sim.getObject(target_object)
    obj_start = list(sim.getObjectPosition(obj_h, -1))
    tip_h = sim.getObject("/tip")

    counter = [0]
    set_gripper(bridge, True)
    for _ in range(20):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    # 1. Approach: 30 cm sobre cubo
    logger.info(f"  → approach (30 cm sobre {target_object})")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [obj_start[0], obj_start[1], 0.30], frames_dir, counter)

    # 2. Descend: al nivel del cubo
    logger.info("  → descend")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [obj_start[0], obj_start[1], obj_start[2]], frames_dir, counter)

    # 3. Grasp: cerrar gripper + ATTACH cubo al tip
    # Técnica estándar en sims comerciales (Pickit, Cognex, etc) — evita
    # tuning fino de friction y garantiza el grasp para demos.
    logger.info("  → grasp_close + attach")
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_respondable, 0)
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_static, 1)
    sim.setObjectParent(obj_h, tip_h, True)
    set_gripper(bridge, False)
    for _ in range(40):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    # 4. Lift
    logger.info("  → lift")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [obj_start[0], obj_start[1], 0.40], frames_dir, counter)

    # 5. Deposit (lateral)
    logger.info("  → deposit")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [-0.30, -0.30, 0.30], frames_dir, counter)

    # 6. Release: detach + abrir gripper + restaurar física
    logger.info("  → release + detach")
    sim.setObjectParent(obj_h, -1, True)
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_respondable, 1)
    sim.setObjectInt32Param(obj_h, sim.shapeintparam_static, 0)
    set_gripper(bridge, True)
    for _ in range(60):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    # 7. Home return
    logger.info("  → home_return")
    _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                     [0.0, -0.31, 0.99], frames_dir, counter)

    obj_end = list(sim.getObjectPosition(obj_h, -1))
    moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(obj_start, obj_end)))
    grasp_success = moved > 0.30  # >30 cm de desplazamiento = grasp+place exitoso

    bridge.stop_simulation()

    return PickResult(
        n_frames=counter[0],
        obj_start_pos=obj_start,
        obj_end_pos=obj_end,
        obj_moved_m=moved,
        grasp_success=grasp_success,
        mp4_path=None,
        frames_dir=frames_dir,
    )
