#!/usr/bin/env python3
"""Batería de selection-accuracy del pick por lenguaje (en sim, opcional).

La parte de EVALUACIÓN de selección es pura (no requiere CoppeliaSim): por
cada escena se genera una instrucción que describe el target y se mide si el
grounding lo selecciona. Con --sim además ejecuta el pick real por escena.

Uso:
    .venv/bin/python experiments/run_language_battery.py --n-scenes 30
    .venv/bin/python experiments/run_language_battery.py --n-scenes 10 --sim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.language_pick import (  # noqa: E402
    evaluate_selection, plan_language_scene,
)

OUT = REPO / "experiments" / "results" / "language_battery"


def build_cases(rng: np.random.Generator, n_scenes: int, n_objects: int,
                with_shapes: bool) -> list[dict]:
    """Genera escenas + la instrucción que describe su target (obj 0)."""
    cases = []
    for _ in range(n_scenes):
        specs = plan_language_scene(rng, n_objects, with_shapes)
        t = specs[0]
        instruction = f"pick the {t.color} {t.shape}"
        cases.append({"instruction": instruction, "expected_id": 0, "specs": specs})
    return cases


def aggregate(rows: list[dict]) -> dict:
    """Agrega resultados de evaluación en métricas de selection-accuracy."""
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    return {"n": n, "n_correct": n_correct,
            "selection_accuracy": (n_correct / n) if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=30)
    ap.add_argument("--n-objects", type=int, default=3)
    ap.add_argument("--with-shapes", action="store_true")
    ap.add_argument("--sim", action="store_true",
                    help="además ejecuta el pick real (requiere CoppeliaSim)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cases = build_cases(rng, args.n_scenes, args.n_objects, args.with_shapes)
    rows = [evaluate_selection(c["specs"], c["instruction"], c["expected_id"]) for c in cases]
    agg = aggregate(rows)

    if args.sim:
        from src.simulation.language_pick import run_language_pick
        sim_correct = 0
        for c in cases:
            out = run_language_pick(c["instruction"], n_objects=args.n_objects,
                                    with_shapes=args.with_shapes)
            sim_correct += int(out.get("selection_correct", False))
        agg["sim_selection_accuracy"] = sim_correct / len(cases)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps({"aggregate": agg, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
