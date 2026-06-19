#!/usr/bin/env python3
"""Batería de selection-accuracy del pick por lenguaje (en sim, opcional).

La parte de EVALUACIÓN de selección es pura (no requiere CoppeliaSim): por
cada escena se genera una instrucción que describe el target y se mide si el
grounding lo selecciona. Con --sim además ejecuta el pick real por escena.

La batería mezcla tres dificultades para que la métrica tenga señal real:
  - "color"   : el target es distinguible solo por color (caso trivial baseline).
  - "shape"   : el target comparte color con al menos un distractor, pero tiene
                forma única → requiere desambiguación por forma.
  - "spatial" : hay objetos del mismo color Y forma que el target; solo la
                posición x (izquierda) distingue al target.

Uso:
    .venv/bin/python experiments/run_language_battery.py --n-scenes 30
    .venv/bin/python experiments/run_language_battery.py --n-scenes 10 --sim
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.language_pick import (  # noqa: E402
    SimObjectSpec, evaluate_selection, plan_language_scene,
)

OUT = REPO / "experiments" / "results" / "language_battery"

logger = logging.getLogger(__name__)

# ── Constantes de posiciones y pools ────────────────────────────────────────
# Posiciones x fijas para casos espaciales (se necesitan ≥2 distintas dentro
# del bin válido: _BIN_X = (0.38, 0.55)).
_SPATIAL_XS = [0.39, 0.46, 0.53]
_SPATIAL_Y_BASE = -0.10
_SPATIAL_DY = 0.04
_Z = 0.033

# Pares (color_target, color_distractor) para casos 'shape'.
_SHAPE_PAIRS = [
    ("red", "red"),
    ("blue", "blue"),
    ("green", "green"),
]
# Formas disponibles para casos 'shape'.
_SHAPE_POOL_PAIR = [
    ("sphere", "cube"),
    ("cube", "sphere"),
    ("cylinder", "cube"),
]


# ── Constructores de casos ───────────────────────────────────────────────────

def _case_color(rng: np.random.Generator, n_objects: int,
                with_shapes: bool) -> dict:
    """Caso 'color': el target es distinguible solo por color (baseline).

    Usa plan_language_scene directamente; el target (obj 0) tiene color 'red'
    y los distractores solo pueden ser 'blue' o 'green'.
    """
    specs = plan_language_scene(rng, n_objects, with_shapes)
    t = specs[0]
    instruction = f"pick the {t.color} {t.shape}"
    return {"instruction": instruction, "expected_id": 0,
            "specs": specs, "difficulty": "color"}


def _case_shape(rng: np.random.Generator, n_objects: int) -> dict:
    """Caso 'shape': el target comparte COLOR con ≥1 distractor.

    El target tiene forma única en la escena, por lo que la desambiguación
    requiere atender a la forma. Construye los specs manualmente sobreescribiendo
    colores y formas para garantizar la condición.
    """
    # Posiciones: se usa plan_language_scene solo para muestrear posiciones.
    specs_base = plan_language_scene(rng, n_objects, with_shapes=True)

    # Elegir par de colores y formas aleatorio pero determinista.
    pair_idx = int(rng.integers(0, len(_SHAPE_PAIRS)))
    target_color, distractor_color = _SHAPE_PAIRS[pair_idx]

    shape_idx = int(rng.integers(0, len(_SHAPE_POOL_PAIR)))
    target_shape, distractor_shape = _SHAPE_POOL_PAIR[shape_idx]

    # Reconstruir specs: obj 0 = target, resto = distractores con mismo color.
    specs: list[SimObjectSpec] = []
    for i, base in enumerate(specs_base):
        if i == 0:
            # Target: color y forma definidos por el par elegido.
            specs.append(SimObjectSpec(0, base.position, target_color, target_shape, "large"))
        else:
            # Distractores: mismo COLOR que target, distinta forma.
            specs.append(SimObjectSpec(i, base.position, distractor_color,
                                       distractor_shape, "large"))

    t = specs[0]
    instruction = f"pick the {t.color} {t.shape}"
    return {"instruction": instruction, "expected_id": 0,
            "specs": specs, "difficulty": "shape"}


def _case_spatial(n_objects: int, scene_idx: int) -> dict:
    """Caso 'spatial': todos los objetos comparten color y forma.

    La única diferencia es la posición x. El target es el objeto con la x
    más negativa (más a la izquierda). La instrucción incluye "on the left"
    que el parser determinista convierte a la relación left_of.

    Parámetros
    ----------
    n_objects : int
        Número de objetos en la escena (máximo 3 soportado por los xs fijos).
    scene_idx : int
        Índice de la escena; se usa para variar la y base y evitar escenas
        idénticas entre sí.
    """
    n = min(n_objects, len(_SPATIAL_XS))
    # Variar y ligeramente por escena para que no sean idénticas.
    y_base = _SPATIAL_Y_BASE + (scene_idx % 3) * _SPATIAL_DY

    # Construir posiciones: xs distintos, y y z fijos (con pequeño offset).
    xs = sorted(_SPATIAL_XS[:n])  # orden ascendente → xs[0] es el más a la izquierda
    specs: list[SimObjectSpec] = []
    for i, x in enumerate(xs):
        pos = (x, y_base, _Z)
        specs.append(SimObjectSpec(i, pos, "red", "cube", "large"))

    # El target es el objeto con la x más pequeña (leftmost).
    leftmost_id = min(specs, key=lambda s: s.position[0]).obj_id
    t = specs[leftmost_id]  # obj_id coincide con posición en lista gracias al enumerate
    instruction = f"pick the {t.color} {t.shape} on the left"
    return {"instruction": instruction, "expected_id": leftmost_id,
            "specs": specs, "difficulty": "spatial"}


def build_cases(rng: np.random.Generator, n_scenes: int, n_objects: int,
                with_shapes: bool) -> list[dict]:
    """Genera n_scenes escenas con mezcla de dificultades (color/shape/spatial).

    Los tipos de caso se alternan cíclicamente: color→shape→spatial→color…
    para asegurar representación balanceada de cada dificultad.

    Parámetros
    ----------
    rng : np.random.Generator
        Generador aleatorio (determinista dado seed).
    n_scenes : int
        Número total de escenas a generar.
    n_objects : int
        Número de objetos por escena.
    with_shapes : bool
        Si True, los casos 'color' pueden tener distractores con formas variadas.

    Retorna
    -------
    list[dict]
        Lista de dicts con claves: instruction, expected_id, specs, difficulty.
    """
    _DIFICULTADES = ["color", "shape", "spatial"]
    cases: list[dict] = []
    for i in range(n_scenes):
        dif = _DIFICULTADES[i % len(_DIFICULTADES)]
        if dif == "color":
            cases.append(_case_color(rng, n_objects, with_shapes))
        elif dif == "shape":
            cases.append(_case_shape(rng, n_objects))
        else:  # spatial
            cases.append(_case_spatial(n_objects, scene_idx=i))
    return cases


def aggregate(rows: list[dict]) -> dict:
    """Agrega resultados en métricas de selection-accuracy globales y por dificultad.

    Parámetros
    ----------
    rows : list[dict]
        Resultados por caso; cada uno debe tener 'correct' (bool) y
        opcionalmente 'difficulty' (str).

    Retorna
    -------
    dict
        Claves globales: n, n_correct, selection_accuracy.
        Clave adicional by_difficulty: {dif: {n, n_correct, selection_accuracy}}.
    """
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    result: dict = {
        "n": n,
        "n_correct": n_correct,
        "selection_accuracy": (n_correct / n) if n else 0.0,
    }

    # Agrupar por dificultad.
    by_dif: dict[str, dict] = {}
    for r in rows:
        dif = r.get("difficulty", "?")
        if dif not in by_dif:
            by_dif[dif] = {"n": 0, "n_correct": 0}
        by_dif[dif]["n"] += 1
        by_dif[dif]["n_correct"] += int(r["correct"])
    for dif, stats in by_dif.items():
        nd = stats["n"]
        stats["selection_accuracy"] = (stats["n_correct"] / nd) if nd else 0.0
    result["by_difficulty"] = by_dif
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=30)
    ap.add_argument("--n-objects", type=int, default=3)
    ap.add_argument("--with-shapes", action="store_true")
    ap.add_argument("--sim", action="store_true",
                    help="además ejecuta el pick real (requiere CoppeliaSim)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    rng = np.random.default_rng(args.seed)
    cases = build_cases(rng, args.n_scenes, args.n_objects, args.with_shapes)

    rows = []
    for case in cases:
        row = evaluate_selection(case["specs"], case["instruction"], case["expected_id"])
        row["difficulty"] = case["difficulty"]
        rows.append(row)

    agg = aggregate(rows)

    if args.sim:
        from src.simulation.language_pick import run_language_pick
        sim_correct = 0
        for i, c in enumerate(cases):
            try:
                # Usar los atributos del target del caso para que la escena sim
                # contenga el objeto descrito en la instrucción (métrica honesta).
                t0 = c["specs"][0]
                out = run_language_pick(
                    c["instruction"],
                    n_objects=args.n_objects,
                    with_shapes=args.with_shapes,
                    seed=args.seed + i,
                    target_color=t0.color,
                    target_shape=t0.shape,
                )
                sim_correct += int(out.get("selection_correct", False))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Escena %d falló en sim: %s", i, exc)
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
