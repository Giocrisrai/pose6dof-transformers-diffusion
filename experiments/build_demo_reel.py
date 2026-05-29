#!/usr/bin/env python3
"""Construye el demo reel curado a partir de los MP4 existentes.

- Anota cada clip fuente con título de etapa + métricas + nota de honestidad
  (overlays cv2, porque ffmpeg local no tiene drawtext).
- Genera tarjetas de intro/cierre.
- Concatena todo en reel_resumen.mp4.

Uso:
    .venv/bin/python experiments/build_demo_reel.py
    .venv/bin/python experiments/build_demo_reel.py --only-clips   # sin reel resumen
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.reel_overlay import (
    normalize_frame, draw_title_bar, draw_metrics, draw_honesty_tag,
    make_title_card,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_demo_reel")

OUT = REPO / "experiments" / "results" / "demo_reel"
CLIPS_OUT = OUT / "clips"
FPS = 24

# Config del reel. metrics = list[(texto, pasa_threshold)].
CLIPS = [
    {
        "key": "01_percepcion",
        "source": REPO / "experiments/results/pick_with_fp_pose/demo.mp4",
        "number": "1.",
        "title": "FoundationPose -> pose 6-DoF",
        "metrics": [("1098 poses YCBV  -  ~4.2 s/pose", True)],
        "honesty": "pose estimada offline (Colab T4)",
    },
    {
        "key": "02_planificacion",
        "source": REPO / "experiments/results/pick_with_diffusion/demo_v3_iter3.mp4",
        "number": "2.",
        "title": "Diffusion Policy v3 -> trayectoria (16 waypoints)",
        "metrics": [
            ("ResNet-18 RGB-D cond  -  IK convergido OK", True),
            ("grasp_plausible_pct_sim 78% (n=50)", True),
        ],
        "honesty": "DP re-entrenada en sim (Iter 3); grasp via attach",
    },
    {
        "key": "03_e2e",
        "source": REPO / "experiments/results/pipeline_e2e/demo_v2.mp4",
        "number": "3.",
        "title": "Pipeline end-to-end",
        "metrics": [
            ("ciclo p95 5.2 s (FP 4.2 / DP 0.2 / sim 1.0)", True),
            ("aceptacion <10 s OK", True),
        ],
        "honesty": "grasp por attach (estandar en sims comerciales)",
    },
    {
        "key": "04_robustez",
        "source": REPO / "experiments/results/pick_battery/base/demo.mp4",
        "number": "4.",
        "title": "Robustez -> 3 escenarios",
        "metrics": [("grasp_proximity 0.8 cm  -  IK OK en los 3", True)],
        "honesty": "grasp por attach (estandar en sims comerciales)",
    },
]

INTRO = [("Bin-picking 6-DoF", 1.3),
         ("Percepcion  ->  Planificacion  ->  Ejecucion", 0.7)]
OUTRO = [("Validado end-to-end  -  ciclo <10 s", 1.0),
         ("honestidad declarada", 0.7),
         ("potencial: grasp fisico / FoundationPose en vivo", 0.55)]


def _compile(frames_dir: Path, mp4_path: Path) -> Path | None:
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg no encontrado — skip MP4")
        return None
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
           "-i", str(frames_dir / "%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           str(mp4_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"ffmpeg falló: {r.stderr[-500:]}")
        return None
    return mp4_path


def annotate_clip(clip: dict) -> Path | None:
    """Lee el MP4 fuente, aplica overlays y compila el clip anotado."""
    src = clip["source"]
    if not src.exists():
        logger.warning(f"[{clip['key']}] fuente faltante: {src} — skip")
        return None
    cap = cv2.VideoCapture(str(src))
    out_mp4 = CLIPS_OUT / f"{clip['key']}.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = normalize_frame(frame)
            draw_title_bar(frame, clip["number"], clip["title"])
            draw_metrics(frame, clip["metrics"])
            draw_honesty_tag(frame, clip["honesty"])
            cv2.imwrite(str(td / f"{idx:06d}.png"), frame)
            idx += 1
        cap.release()
        if idx == 0:
            logger.warning(f"[{clip['key']}] 0 frames leídos — skip")
            return None
        CLIPS_OUT.mkdir(parents=True, exist_ok=True)
        result = _compile(td, out_mp4)
    if result:
        logger.info(f"[{clip['key']}] clip anotado: {out_mp4.name} ({idx} frames)")
    return result


def _make_card_clip(lines, key: str, seconds: float = 3.0) -> Path | None:
    """Genera un MP4 corto (card estática) de `seconds` a FPS."""
    card = make_title_card(lines)
    n = int(round(seconds * FPS))
    out_mp4 = CLIPS_OUT / f"{key}.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n):
            cv2.imwrite(str(td / f"{i:06d}.png"), card)
        return _compile(td, out_mp4)


def build_reel(annotated: list[Path]) -> Path | None:
    """Concatena intro + clips anotados + outro en reel_resumen.mp4."""
    intro = _make_card_clip(INTRO, "00_intro")
    outro = _make_card_clip(OUTRO, "99_outro")
    sequence = [p for p in [intro, *annotated, outro] if p is not None]
    if not sequence:
        logger.error("sin clips para concatenar")
        return None
    # ffmpeg concat demuxer (todos mismo codec/res/fps -> -c copy)
    list_file = OUT / "_concat_list.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in sequence))
    reel = OUT / "reel_resumen.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(reel)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if r.returncode != 0:
        logger.error(f"concat falló: {r.stderr[-500:]}")
        return None
    logger.info(f"reel resumen: {reel}")
    return reel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-clips", action="store_true",
                        help="Solo anota clips; no genera el reel resumen.")
    args = parser.parse_args()

    CLIPS_OUT.mkdir(parents=True, exist_ok=True)
    annotated = []
    for clip in CLIPS:
        p = annotate_clip(clip)
        if p:
            annotated.append(p)

    if not annotated:
        logger.error("0 clips anotados — abortando")
        return 1
    logger.info(f"{len(annotated)}/{len(CLIPS)} clips anotados en {CLIPS_OUT}")

    if args.only_clips:
        return 0
    build_reel(annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
