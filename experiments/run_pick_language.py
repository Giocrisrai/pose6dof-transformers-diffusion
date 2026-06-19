#!/usr/bin/env python3
"""Demo de bin picking guiado por lenguaje natural.

Ejecuta el pipeline end-to-end seleccionando el objeto descrito por una
instrucción en lenguaje natural. Con --dry-run no requiere CoppeliaSim:
muestra el parsing + grounding sobre una escena sintética fija.

Ejemplos:
    python experiments/run_pick_language.py --instruction "dame el cubo rojo" --dry-run
    python experiments/run_pick_language.py --instruction "pick the blue sphere" --scene clutter --render
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.language import make_parser            # noqa: E402
from src.language.grounding import Grounder     # noqa: E402
from src.language.schema import ObjectView      # noqa: E402

RESULTS = REPO / "experiments/results/language_pick"


def _escena_demo() -> list[ObjectView]:
    """Escena sintética fija de 3 objetos (coherente con exp16/24)."""
    return [
        ObjectView(0, (-0.20, 0.0, 0.5), {"color": "red", "shape": "cube", "size": "large"}),
        ObjectView(1, (0.00, 0.0, 0.5), {"color": "blue", "shape": "cube", "size": "small"}),
        ObjectView(2, (0.20, 0.0, 0.5), {"color": "red", "shape": "sphere", "size": "small"}),
    ]


def run_dry(instruction: str, backend: str = "deterministic") -> int:
    """Parsing + grounding sobre la escena demo; imprime y guarda JSON."""
    parser = make_parser(backend)
    grounder = Grounder(method="attribute")
    instr = parser.parse(instruction)
    objs = _escena_demo()
    res = grounder.ground(instr, objs)
    payload = {
        "instruction": instruction,
        "parsed": {
            "color": instr.target.color, "shape": instr.target.shape,
            "size": instr.target.size, "intent": instr.intent,
            "spatial": instr.spatial.relation if instr.spatial else None,
            "backend": instr.backend,
        },
        "grounding": {
            "target_obj_id": res.target_obj_id, "method": res.method,
            "ambiguous": res.ambiguous, "scores": res.scores,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "last_dry_run.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bin picking guiado por lenguaje natural")
    p.add_argument("--instruction", required=True, help='p.ej. "dame el cubo rojo"')
    p.add_argument("--parser-backend", default="deterministic",
                   choices=["deterministic", "llm_local", "llm_api"])
    p.add_argument("--scene", default="multi", choices=["multi", "clutter"])
    p.add_argument("--render", action="store_true", help="render del grounding (requiere sim)")
    p.add_argument("--dry-run", action="store_true",
                   help="solo parsing + grounding, sin CoppeliaSim")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        return run_dry(args.instruction, args.parser_backend)
    # Ruta E2E con CoppeliaSim: el grounding decide el target y se delega en la
    # batería de pick existente. Requiere CoppeliaSim en localhost:23000.
    from experiments.run_pick_battery import run_language_pick
    return run_language_pick(instruction=args.instruction, scene=args.scene,
                             parser_backend=args.parser_backend, render=args.render)


if __name__ == "__main__":
    raise SystemExit(main())
