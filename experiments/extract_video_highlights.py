#!/usr/bin/env python3
"""Extrae frames-highlight del video E2E para incluir en el doc TFM."""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "experiments/results/pipeline_e2e/demo_frames"
OUT = REPO / "experiments/results/pipeline_e2e/highlights"
OUT.mkdir(parents=True, exist_ok=True)

# Frames clave: inicio, mid y fin de cada ciclo (3 ciclos × 80 frames = 240 total)
highlights = {
    "cycle1_perception": 5,    # inicio ciclo 1
    "cycle1_planning": 35,     # mid ciclo 1
    "cycle1_grasp": 75,        # fin ciclo 1
    "cycle2_perception": 85,
    "cycle2_planning": 115,
    "cycle2_grasp": 155,
    "cycle3_perception": 165,
    "cycle3_planning": 195,
    "cycle3_grasp": 235,
}

for name, idx in highlights.items():
    src = SRC / f"frame_{idx:05d}.png"
    if src.exists():
        dst = OUT / f"highlight_{name}.png"
        shutil.copy(src, dst)
        print(f"  {dst.name} ← frame_{idx:05d}.png")

# Combinar las 3 fases del ciclo 1 en un solo composite
try:
    import cv2
    import numpy as np
    imgs = [cv2.imread(str(OUT / f"highlight_cycle1_{p}.png"))
            for p in ("perception", "planning", "grasp")]
    if all(img is not None for img in imgs):
        composite = np.hstack(imgs)
        cv2.imwrite(str(OUT / "composite_3phases.png"), composite)
        print(f"  composite_3phases.png ({composite.shape[1]}×{composite.shape[0]})")
except Exception as e:
    print(f"  [warn] composite: {e}")

print(f"\n[OK] {len(highlights)} highlights en {OUT.relative_to(REPO)}")
