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

# Si el objeto se movió > este threshold, se considera "manipulated":
# el robot lo desplazó (sea por grasp exitoso o por contacto).
MOVE_THRESHOLD_M = 0.02


def run_scenario(sc: Scenario) -> dict:
    """Ejecuta pick sequence para un escenario. Devuelve dict con métricas."""
    logger.info(f"\n=== escenario {sc.id} ({sc.difficulty}) ===")
    scenario_dir = OUTPUT_DIR / sc.id
    frames_dir = scenario_dir / "frames"
    mp4_path = scenario_dir / "demo.mp4"

    with CoppeliaSimBridge() as bridge:
        bridge.set_stepping(True)
        bridge.load_scene(SCENES_DIR / sc.scene)
        bridge.apply_scenario(sc.to_dict())

        result: PickResult = run_pick_sequence(bridge, frames_dir)

    # Compilar MP4 fuera del with (no necesita bridge)
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
        "object_moved_m": round(result.obj_moved_m, 3),
        "object_manipulated": result.obj_moved_m > MOVE_THRESHOLD_M,
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
        f.write("# Pick Battery Report\n\n")
        f.write("Cada escenario ejecuta la secuencia completa pick-and-place ")
        f.write("(home → approach → descend → grasp → lift → deposit → release → home) ")
        f.write("sobre la escena cargada con sus tweaks aplicados.\n\n")
        f.write("**object_manipulated**: el objeto target se desplazó más de ")
        f.write(f"{MOVE_THRESHOLD_M*100:.0f} cm durante la secuencia ")
        f.write("(no garantiza grasp exitoso — solo contacto/desplazamiento).\n\n")
        f.write("| id | difficulty | frames | object_start → end | moved | manipulated | video |\n")
        f.write("|---|---|---:|---|---:|:-:|---|\n")
        for r in results:
            start = ",".join(f"{p:+.2f}" for p in r["object_start"])
            end = ",".join(f"{p:+.2f}" for p in r["object_end"])
            check = "✓" if r["object_manipulated"] else "✗"
            video = f"`{r['mp4_path']}`" if r["mp4_path"] else "n/a"
            f.write(
                f"| {r['scenario_id']} | {r['difficulty']} | {r['n_frames']} | "
                f"({start}) → ({end}) | {r['object_moved_m']*100:.1f} cm | "
                f"{check} | {video} |\n"
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
