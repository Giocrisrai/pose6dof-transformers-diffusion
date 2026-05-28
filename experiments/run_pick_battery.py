#!/usr/bin/env python3
"""Battery runner con pick-and-place real por escenario.

Por cada escenario de scenarios.yaml:
    1. Carga bin_base.ttt (o variant según el manifest).
    2. Aplica tweaks (color/luz/visibility).
    3. Ejecuta la secuencia pick-and-place completa.
    4. Captura un frame por step → compila MP4 del escenario.
    5. Mide si el objeto target fue efectivamente movido del bin.

Outputs en experiments/results/pick_battery/:
    - <scenario_id>/frames/*.png
    - <scenario_id>/demo.mp4
    - report.json + report.md (tabla comparativa)

Uso (CoppeliaSim corriendo en :23000):
    .venv/bin/python experiments/run_pick_battery.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import (
    PickResult,
    compile_mp4,
    run_pick_sequence,
)
from src.simulation.scenarios import Scenario, load_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pick_battery")

SCENES_DIR = REPO / "data" / "scenes"
SCENARIOS_YAML = SCENES_DIR / "scenarios.yaml"
OUTPUT_DIR = REPO / "experiments" / "results" / "pick_battery"

def run_scenario(sc: Scenario) -> dict:
    """Ejecuta pick sequence para un escenario. Devuelve dict con métricas honestas.

    Métricas (ver pick_sequence.PickResult):
    - object_moved_m: desplazamiento total del cubo (ruidoso por no-determinismo).
    - grasp_proximity_m: distancia tip↔cubo al momento del attach (REAL).
    - deposit_error_m: distancia obj_end↔deposit_target (REAL).
    - grasp_plausible: True si grasp_proximity_m < 5 cm.
    - deposit_plausible: True si deposit_error_m < 30 cm.
    - ik_converged: True si todas las llamadas a IK convergieron.
    """
    logger.info(f"\n=== escenario {sc.id} ({sc.difficulty}) ===")
    scenario_dir = OUTPUT_DIR / sc.id
    frames_dir = scenario_dir / "frames"
    mp4_path = scenario_dir / "demo.mp4"

    with CoppeliaSimBridge() as bridge:
        bridge.set_stepping(True)
        bridge.load_scene(SCENES_DIR / sc.scene)
        bridge.apply_scenario(sc.to_dict())
        result: PickResult = run_pick_sequence(bridge, frames_dir)

    compiled = compile_mp4(frames_dir, mp4_path, fps=25)

    return {
        "scenario_id": sc.id,
        "scene": sc.scene,
        "difficulty": sc.difficulty,
        "description": sc.description,
        "n_tweaks": len(sc.tweaks),
        "n_frames": result.n_frames,
        "object_start": [round(p, 3) for p in result.obj_start_pos],
        "object_end": [round(p, 3) for p in result.obj_end_pos],
        "deposit_target": result.deposit_target,
        "object_moved_m": round(result.obj_moved_m, 3),
        "tip_grasp_proximity_m": round(result.tip_grasp_proximity_m, 3),
        "deposit_error_m": round(result.deposit_error_m, 3),
        "obj_displaced": result.obj_displaced,
        "grasp_plausible": result.grasp_plausible,
        "deposit_plausible": result.deposit_plausible,
        "ik_converged": result.ik_converged,
        "mp4_path": str(compiled.relative_to(REPO)) if compiled else None,
        "frames_dir": str(frames_dir.relative_to(REPO)),
    }


def save_report(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "report.json"
    with json_path.open("w") as f:
        json.dump({"scenarios": results}, f, indent=2)
    logger.info(f"escrito: {json_path.relative_to(REPO)}")

    md_path = OUTPUT_DIR / "report.md"
    with md_path.open("w") as f:
        f.write("# Pick Battery Report — métricas honestas\n\n")
        f.write("**IMPORTANTE — leé `docs/PICK_LIMITATIONS.md` antes de interpretar.**\n\n")
        f.write("El \"grasp\" usa la técnica de snap+attach (cubo se teletransporta ")
        f.write("al TCP y se parentea al gripper). El gripper físico NO agarra ")
        f.write("por fricción. Esto es estándar en sims comerciales (Pickit, ")
        f.write("Cognex, RoboDK) pero hay que entender las limitaciones.\n\n")
        f.write("## Métricas\n\n")
        f.write("- **moved**: desplazamiento total del cubo (cm). Ruidoso por no-")
        f.write("determinismo de la física post-release.\n")
        f.write("- **grasp_proximity**: distancia tip↔cubo AL momento del attach (cm). ")
        f.write("Si > 5 cm el grasp NO sería físicamente plausible.\n")
        f.write("- **deposit_error**: distancia entre obj_end y el target deposit (cm). ")
        f.write("Mide la precisión del depósito (independiente de no-determinismo).\n")
        f.write("- **ik_converged**: True si todas las llamadas a IK convergieron.\n\n")
        f.write("## Resultados\n\n")
        f.write("| id | diff | frames | grasp_prox | grasp_OK | moved | deposit_err | deposit_OK | ik_ok | video |\n")
        f.write("|---|---|---:|---:|:-:|---:|---:|:-:|:-:|---|\n")
        for r in results:
            grasp_check = "✓" if r["grasp_plausible"] else "✗"
            deposit_check = "✓" if r["deposit_plausible"] else "✗"
            ik_check = "✓" if r["ik_converged"] else "✗"
            video = f"`{r['mp4_path']}`" if r["mp4_path"] else "n/a"
            f.write(
                f"| {r['scenario_id']} | {r['difficulty']} | {r['n_frames']} | "
                f"{r['tip_grasp_proximity_m']*100:.1f}cm | {grasp_check} | "
                f"{r['object_moved_m']*100:.1f}cm | "
                f"{r['deposit_error_m']*100:.1f}cm | {deposit_check} | "
                f"{ik_check} | {video} |\n"
            )
    logger.info(f"escrito: {md_path.relative_to(REPO)}")


def main() -> int:
    if not SCENARIOS_YAML.exists():
        logger.error(f"scenarios.yaml no encontrado: {SCENARIOS_YAML}")
        return 1

    scenarios = load_scenarios(SCENARIOS_YAML)
    logger.info(f"cargados {len(scenarios)} escenarios")

    results = []
    for sc in scenarios:
        try:
            results.append(run_scenario(sc))
        except Exception as e:
            logger.error(f"scenario {sc.id} falló: {e}")
            results.append({
                "scenario_id": sc.id,
                "scene": sc.scene,
                "difficulty": sc.difficulty,
                "error": str(e),
            })

    save_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
