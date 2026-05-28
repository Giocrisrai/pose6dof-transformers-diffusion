#!/usr/bin/env python3
"""Demo individual de pick-and-place usando IK + attach.

Wrapper sobre run_pick_sequence del módulo src.simulation.pick_sequence,
para correr UN solo escenario (sin batch) y producir el MP4.

Uso (requiere CoppeliaSim corriendo + ffmpeg):
    .venv/bin/python experiments/run_pick_demo.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import compile_mp4, run_pick_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pick_demo")

OUTPUT_DIR = REPO / "experiments" / "results" / "pick_demo"
FRAMES_DIR = OUTPUT_DIR / "frames"
SCENE_PATH = REPO / "data" / "scenes" / "bin_base.ttt"


def main() -> int:
    if not SCENE_PATH.exists():
        logger.error(f"escena no encontrada: {SCENE_PATH}")
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE_PATH)
        result = run_pick_sequence(bridge, FRAMES_DIR)

    logger.info(f"frames capturados: {result.n_frames}")
    logger.info(f"object_1 start → end: {result.obj_start_pos} → {result.obj_end_pos}")
    logger.info(f"moved: {result.obj_moved_m * 100:.1f} cm")
    logger.info(f"grasp_success: {result.grasp_success}")

    mp4 = compile_mp4(FRAMES_DIR, OUTPUT_DIR / "demo.mp4", fps=25)
    if mp4:
        logger.info(f"MP4: {mp4.relative_to(REPO)} ({mp4.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
