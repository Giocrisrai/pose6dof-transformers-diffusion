#!/usr/bin/env python3
"""Pick-and-place usando pose REAL de FoundationPose como target.

Cierra la **Brecha A** documentada en `docs/INTEGRATION_PIPELINE.md`:
en lugar de usar el ground truth de la escena (cheating), carga una
pose estimada por FoundationPose desde el checkpoint del run real
(`experiments/checkpoints/fp_ycbv_checkpoint.json`) y la pasa al
pick_sequence como override.

NOTA HONESTA: las poses de FP son del dataset YCB-V en escenas reales
(no de nuestro `bin_base.ttt`). Las coords no corresponden directamente
al workspace del UR5. Para que sirva como demo:

  1. Tomamos una pose de FP.
  2. Extraemos el centroide (t_pred).
  3. Lo MAPEAMOS al workspace del sim mediante una transformación
     simple (traslación al área del bin + clamp).
  4. Eso simula "FP detectó el objeto a estas coords; ahora el pick
     ejecuta sobre ellas".

Esto demuestra el FLUJO (FP → pick) aunque la pose no es semánticamente
exacta. Es un primer paso para futura integración con FP real corriendo
sobre RGB-D capturada del sim.

Uso (CoppeliaSim corriendo en :23000):
    python experiments/run_pick_with_fp_pose.py
    # Con índice específico del checkpoint:
    python experiments/run_pick_with_fp_pose.py --fp-index 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import compile_mp4, run_pick_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pick_with_fp")

FP_CHECKPOINT = REPO / "experiments" / "checkpoints" / "fp_ycbv_checkpoint.json"
SCENE_PATH = REPO / "data" / "scenes" / "bin_base.ttt"
OUTPUT_DIR = REPO / "experiments" / "results" / "pick_with_fp_pose"


def map_fp_pose_to_sim_workspace(t_pred: list[float]) -> list[float]:
    """Mapea el centroide de una pose FP al workspace del sim.

    Las poses de FP vienen en coords del dataset YCB-V (cámara del dataset,
    típicamente t_pred[2] ≈ 0.5-1.5 m, t_pred[0]/[1] ≈ ±0.1-0.3 m).

    Para el demo, mapeamos:
      - X_sim = 0.46 (centro del bin) + clamp(t_pred[0], -0.05, +0.05)
      - Y_sim = -0.10 + clamp(t_pred[1], -0.05, +0.05)
      - Z_sim = 0.033 (altura del cube center sobre la table)

    El componente XY de la pose FP se usa para *variar* la posición dentro
    del workspace; Z se ignora (en el dataset es la distancia cámara-objeto,
    no aplica al sim).
    """
    x_offset = max(-0.05, min(0.05, t_pred[0]))
    y_offset = max(-0.05, min(0.05, t_pred[1]))
    return [0.46 + x_offset, -0.10 + y_offset, 0.033]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-index", type=int, default=0,
                        help="Índice de la pose FP a usar (0 = primera)")
    args = parser.parse_args()

    if not FP_CHECKPOINT.exists():
        logger.error(f"checkpoint FP no encontrado: {FP_CHECKPOINT}")
        return 1
    if not SCENE_PATH.exists():
        logger.error(f"escena no encontrada: {SCENE_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUTPUT_DIR / "frames"

    # 1. Cargar pose real estimada por FP
    with FP_CHECKPOINT.open() as f:
        ckpt = json.load(f)
    if args.fp_index >= len(ckpt["results"]):
        logger.error(f"fp-index {args.fp_index} fuera de rango (max {len(ckpt['results'])-1})")
        return 1
    pose = ckpt["results"][args.fp_index]
    t_pred = pose["t_pred"]
    logger.info(
        f"FP pose [{args.fp_index}]: obj_id={pose['obj_id']} scene={pose['scene_id']} "
        f"img={pose['img_id']} t_pred={[round(t, 3) for t in t_pred]}"
    )

    # 2. Mapear al workspace del sim
    target_xyz = map_fp_pose_to_sim_workspace(t_pred)
    logger.info(f"target mapeado al sim: {[round(t, 3) for t in target_xyz]}")

    # 3. Pick usando esa pose como target
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE_PATH)
        result = run_pick_sequence(
            bridge, frames_dir,
            pose_override_xyz=target_xyz,
            pose_source=f"foundation_pose_ckpt[{args.fp_index}]",
        )

    # 4. Compilar video + reportar
    mp4 = compile_mp4(frames_dir, OUTPUT_DIR / "demo.mp4", fps=25)

    print()
    print("=== RESULTADOS pick con FP pose ===")
    print(f"  FP pose source:       {result.pose_source}")
    print(f"  target_xyz (mapeado): {result.obj_start_pos}")
    print(f"  cube_end_pos:         {result.obj_end_pos}")
    print(f"  moved:                {result.obj_moved_m * 100:.1f} cm")
    print(f"  grasp_proximity:      {result.tip_grasp_proximity_m * 100:.1f} cm "
          f"({'✓ plausible' if result.grasp_plausible else '✗ IMPLAUSIBLE'})")
    print(f"  deposit_error:        {result.deposit_error_m * 100:.1f} cm "
          f"({'✓ plausible' if result.deposit_plausible else '✗ IMPLAUSIBLE'})")
    print(f"  ik_converged:         {result.ik_converged}")
    if mp4:
        print(f"  video:                {mp4.relative_to(REPO)} ({mp4.stat().st_size/1024:.0f} KB)")

    # Guardar metadata
    metadata = {
        "fp_index": args.fp_index,
        "fp_obj_id": pose["obj_id"],
        "fp_scene_id": pose["scene_id"],
        "fp_t_pred": t_pred,
        "target_xyz_mapped": target_xyz,
        "result": {
            "obj_start": result.obj_start_pos,
            "obj_end": result.obj_end_pos,
            "moved_m": result.obj_moved_m,
            "grasp_proximity_m": result.tip_grasp_proximity_m,
            "grasp_plausible": result.grasp_plausible,
            "deposit_error_m": result.deposit_error_m,
            "deposit_plausible": result.deposit_plausible,
            "ik_converged": result.ik_converged,
            "pose_source": result.pose_source,
        },
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  metadata:             {(OUTPUT_DIR / 'metadata.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
