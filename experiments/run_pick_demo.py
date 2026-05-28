#!/usr/bin/env python3
"""Demo end-to-end: UR5 hace pick-and-place de un objeto del bin.

Secuencia:
    1. HOME       — robot arriba, gripper abierto
    2. APPROACH   — TCP sobre el bin a ~25 cm de altura
    3. DESCEND    — TCP baja al objeto
    4. GRASP      — gripper se cierra
    5. LIFT       — TCP sube con el objeto agarrado
    6. DEPOSIT    — pan rotado, TCP a la zona de depósito
    7. RELEASE    — gripper se abre, objeto cae
    8. HOME       — vuelta al inicio

Captura un frame del rgb_camera por cada paso de simulación.
Compila los frames a `experiments/results/pick_demo/demo.mp4` con ffmpeg.

Uso (requiere CoppeliaSim corriendo + ffmpeg en PATH):
    .venv/bin/python experiments/run_pick_demo.py
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pick_demo")

OUTPUT_DIR = REPO / "experiments" / "results" / "pick_demo"
FRAMES_DIR = OUTPUT_DIR / "frames"
SCENE_PATH = REPO / "data" / "scenes" / "bin_base.ttt"

# Keyframes del UR5 (en orden: pan, lift, elbow, wrist1, wrist2, wrist3).
# Verificados empíricamente con la escena bin_base (UR5 rotado +π/2, bin en +X=0.5).
KEYFRAMES = {
    "home":     [0,           0,            0,            0,            0,           0],
    "approach": [0,          -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
    "descend":  [0,          -math.pi/2.2,  math.pi/2,   -math.pi*0.55, math.pi/2,   0],
    "lift":     [0,          -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
    "deposit":  [-math.pi/2, -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
    "release":  [-math.pi/2, -math.pi/3,    math.pi/2.5, -math.pi*0.6,  math.pi/2,   0],
}

STEPS_PER_SEGMENT = 60  # ~3 segundos sim (~50ms/step) por segmento
SETTLE_STEPS = 20       # steps extra de espera entre keyframes


def setup_robot_control(bridge: CoppeliaSimBridge) -> None:
    """Configura los joints para control dinámico desde Python.

    El script threaded del UR5.ttm los pone en callback mode (8) que
    sobrescribe nuestros setJointTargetPosition. Acá los pasamos a
    dynamic + motor + position y disable del script.
    """
    sim = bridge.sim
    for h in bridge._joint_handles:
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    try:
        scr = sim.getObject("/UR5e/Script")
        sim.setObjectInt32Param(scr, sim.scriptintparam_enabled, 0)
        logger.info("UR5 threaded script disabled")
    except Exception as e:
        logger.warning(f"disable UR5 script: {e}")


def set_gripper(bridge: CoppeliaSimBridge, open_: bool) -> None:
    """Abre/cierra el gripper RG2 vía signal scene-level."""
    bridge.sim.setIntProperty(
        bridge.sim.handle_scene, "signal.RG2_open", 1 if open_ else 0
    )
    logger.info(f"gripper → {'OPEN' if open_ else 'CLOSE'}")


def capture_frame(bridge: CoppeliaSimBridge, frame_idx: int) -> Path:
    """Captura un PNG del rgb_camera y lo guarda como frames/NNNNNN.png."""
    from PIL import Image
    sim = bridge.sim
    sim.handleVisionSensor(bridge._camera_rgb_handle)
    img_raw, res = sim.getVisionSensorImg(bridge._camera_rgb_handle)
    w, h = res[0], res[1]
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)
    out = FRAMES_DIR / f"{frame_idx:06d}.png"
    Image.fromarray(img).save(out)
    return out


def move_to_pose(
    bridge: CoppeliaSimBridge,
    pose_name: str,
    target: list[float],
    frame_counter: list[int],
    n_steps: int = STEPS_PER_SEGMENT,
) -> None:
    """Comanda el target a los joints y avanza la simulación capturando frames."""
    sim = bridge.sim
    logger.info(f"→ {pose_name}: target={[round(t,2) for t in target]}")
    for h, t in zip(bridge._joint_handles, target):
        sim.setJointTargetPosition(h, t)
    for _ in range(n_steps):
        bridge.step()
        capture_frame(bridge, frame_counter[0])
        frame_counter[0] += 1
    # Settle: dejar que termine de llegar
    for _ in range(SETTLE_STEPS):
        bridge.step()
        capture_frame(bridge, frame_counter[0])
        frame_counter[0] += 1


def compile_mp4(fps: int = 25) -> Path | None:
    """Compila frames/*.png en demo.mp4 usando ffmpeg. Si no hay ffmpeg, skip."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg no encontrado en PATH — skip MP4 compilation")
        return None
    mp4_path = OUTPUT_DIR / "demo.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(FRAMES_DIR / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(mp4_path),
    ]
    logger.info(f"ffmpeg: compilando MP4 ({fps} fps)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg falló: {result.stderr[-500:]}")
        return None
    logger.info(f"MP4 escrito: {mp4_path} ({mp4_path.stat().st_size / 1024:.0f} KB)")
    return mp4_path


def main() -> int:
    if not SCENE_PATH.exists():
        logger.error(f"escena no encontrada: {SCENE_PATH}")
        return 1
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    # Limpiar frames anteriores
    for old in FRAMES_DIR.glob("*.png"):
        old.unlink()

    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE_PATH)
        setup_robot_control(bridge)
        bridge.set_stepping(True)
        bridge.start_simulation()

        # Pose inicial del objeto target (object_1, cubo rojo en el bin)
        try:
            obj1 = bridge.sim.getObject("/object_1")
            obj1_start = bridge.sim.getObjectPosition(obj1, -1)
            logger.info(f"object_1 start pos: {[round(p,3) for p in obj1_start]}")
        except Exception:
            obj1_start = None

        frame_counter = [0]

        # Secuencia de pick-and-place
        sequence = [
            ("home",          KEYFRAMES["home"],     "abrir gripper"),
            ("approach",      KEYFRAMES["approach"], None),
            ("descend",       KEYFRAMES["descend"],  None),
            ("grasp_close",   KEYFRAMES["descend"],  "cerrar gripper"),  # mismo pose, cerrar
            ("lift",          KEYFRAMES["lift"],     None),
            ("deposit",       KEYFRAMES["deposit"],  None),
            ("release_open",  KEYFRAMES["release"],  "abrir gripper"),
            ("home_return",   KEYFRAMES["home"],     None),
        ]

        set_gripper(bridge, open_=True)
        for _ in range(15):  # esperar que el gripper se abra
            bridge.step()

        for pose_name, target, gripper_action in sequence:
            if gripper_action == "abrir gripper":
                set_gripper(bridge, open_=True)
            elif gripper_action == "cerrar gripper":
                set_gripper(bridge, open_=False)

            move_to_pose(bridge, pose_name, target, frame_counter)

        # Estado final del object_1
        if obj1_start is not None:
            try:
                obj1_end = bridge.sim.getObjectPosition(obj1, -1)
                moved = math.sqrt(sum((a-b)**2 for a, b in zip(obj1_start, obj1_end)))
                logger.info(
                    f"object_1 end pos:   {[round(p,3) for p in obj1_end]}  "
                    f"|moved={moved:.3f} m|"
                )
            except Exception as e:
                logger.warning(f"no pude leer pose final de object_1: {e}")

        bridge.stop_simulation()

    logger.info(f"frames capturados: {frame_counter[0]}")
    mp4 = compile_mp4(fps=25)

    if mp4 is None:
        logger.info(f"PNGs en {FRAMES_DIR} (compilar manual con ffmpeg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
