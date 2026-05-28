"""Secuencia pick-and-place reutilizable para escenas tipo bin_base.

Define keyframes UR5 verificados y la función `run_pick_sequence` que
ejecuta el ciclo completo: home → approach → descend → grasp → lift →
deposit → release → home, capturando frames del rgb_camera por step.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logger = logging.getLogger(__name__)

# Keyframes UR5 verificados con bin_base.ttt (UR5 rotado +π/2, bin en +X=0.5).
KEYFRAMES = {
    "home":     [0,           0,            0,            0,            0,           0],
    "approach": [0,          -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
    "descend":  [0,          -math.pi/2.2,  math.pi/2,   -math.pi*0.55, math.pi/2,   0],
    "lift":     [0,          -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
    "deposit":  [-math.pi/2, -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
}

STEPS_PER_SEGMENT = 60
SETTLE_STEPS = 20


@dataclass
class PickResult:
    """Resultado de una corrida de pick sequence."""
    n_frames: int
    obj_start_pos: list[float]
    obj_end_pos: list[float]
    obj_moved_m: float
    mp4_path: Path | None
    frames_dir: Path


def setup_robot_control(bridge: CoppeliaSimBridge) -> None:
    """Configura joints UR5 para control dinámico position-mode + disable script.

    El script threaded interno del UR5.ttm pone los joints en callback mode
    (8), que sobrescribe setJointTargetPosition desde Python.
    """
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


def _capture_frame(bridge: CoppeliaSimBridge, frames_dir: Path, idx: int) -> None:
    from PIL import Image
    sim = bridge.sim
    sim.handleVisionSensor(bridge._camera_rgb_handle)
    img_raw, res = sim.getVisionSensorImg(bridge._camera_rgb_handle)
    w, h = res[0], res[1]
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)
    Image.fromarray(img).save(frames_dir / f"{idx:06d}.png")


def _move_to(bridge: CoppeliaSimBridge, target, frames_dir: Path, counter: list[int],
             n_steps: int = STEPS_PER_SEGMENT) -> None:
    sim = bridge.sim
    for h, t in zip(bridge._joint_handles, target):
        sim.setJointTargetPosition(h, t)
    for _ in range(n_steps + SETTLE_STEPS):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1


def compile_mp4(frames_dir: Path, mp4_path: Path, fps: int = 25) -> Path | None:
    """Compila frames a MP4 con ffmpeg. Devuelve None si ffmpeg no está."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg no encontrado — skip MP4")
        return None
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg falló: {result.stderr[-500:]}")
        return None
    return mp4_path


def run_pick_sequence(
    bridge: CoppeliaSimBridge,
    frames_dir: Path,
    target_object: str = "/object_1",
) -> PickResult:
    """Ejecuta home → approach → descend → grasp → lift → deposit → release → home.

    Pre-condición: bridge ya tiene la escena cargada y simulación EN STEPPED MODE.
    Esta función llama start_simulation y stop_simulation internamente.

    Args:
        bridge: CoppeliaSimBridge conectado con escena cargada.
        frames_dir: directorio para frames PNG (se crea si no existe; los .png
            existentes se eliminan).
        target_object: handle path del objeto que se intenta agarrar.

    Returns:
        PickResult con métricas y referencia a los frames.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    setup_robot_control(bridge)
    sim = bridge.sim
    bridge.start_simulation()

    # Pose inicial del objeto
    obj_h = sim.getObject(target_object)
    obj_start = sim.getObjectPosition(obj_h, -1)

    counter = [0]
    set_gripper(bridge, True)
    for _ in range(15):
        bridge.step()
        _capture_frame(bridge, frames_dir, counter[0])
        counter[0] += 1

    sequence = [
        ("home",         KEYFRAMES["home"],     None),
        ("approach",     KEYFRAMES["approach"], None),
        ("descend",      KEYFRAMES["descend"],  None),
        ("grasp_close",  KEYFRAMES["descend"],  "close"),
        ("lift",         KEYFRAMES["lift"],     None),
        ("deposit",      KEYFRAMES["deposit"],  None),
        ("release_open", KEYFRAMES["deposit"],  "open"),
        ("home_return",  KEYFRAMES["home"],     None),
    ]

    for name, pose, gripper in sequence:
        logger.info(f"  → {name}")
        if gripper == "open":
            set_gripper(bridge, True)
        elif gripper == "close":
            set_gripper(bridge, False)
        _move_to(bridge, pose, frames_dir, counter)

    obj_end = sim.getObjectPosition(obj_h, -1)
    moved = math.sqrt(sum((a-b)**2 for a, b in zip(obj_start, obj_end)))

    bridge.stop_simulation()

    return PickResult(
        n_frames=counter[0],
        obj_start_pos=list(obj_start),
        obj_end_pos=list(obj_end),
        obj_moved_m=moved,
        mp4_path=None,
        frames_dir=frames_dir,
    )
