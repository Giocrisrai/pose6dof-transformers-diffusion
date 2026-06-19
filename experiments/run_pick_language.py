#!/usr/bin/env python3
"""Demo de bin picking guiado por lenguaje natural.

Ejecuta el pipeline end-to-end seleccionando el objeto descrito por una
instrucción en lenguaje natural. Con --dry-run no requiere CoppeliaSim:
muestra el parsing + grounding sobre una escena sintética fija.

La ruta E2E (sin --dry-run) requiere CoppeliaSim en localhost:23000 y
ejecuta el pick completo usando run_language_pick de src.simulation.language_pick.

Ejemplo funcional (sin sim):
    python experiments/run_pick_language.py --instruction "dame el cubo rojo" --dry-run

Ejemplo E2E (requiere CoppeliaSim en :23000):
    python experiments/run_pick_language.py --instruction "dame el cubo rojo"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.language import make_parser            # noqa: E402
from src.language.demo import demo_scene        # noqa: E402
from src.language.grounding import Grounder     # noqa: E402

RESULTS = REPO / "experiments/results/language_pick"


def run_dry(instruction: str, backend: str = "deterministic") -> int:
    """Parsing + grounding sobre la escena demo; imprime y guarda JSON."""
    parser = make_parser(backend)
    grounder = Grounder(method="attribute")
    instr = parser.parse(instruction)
    objs = demo_scene()
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
    p.add_argument("--scene", default="multi", choices=["multi", "clutter"],
                   help="escena de sim (solo afecta la ruta E2E, no --dry-run)")
    p.add_argument("--render", action="store_true", help="render del grounding (requiere sim)")
    p.add_argument("--dry-run", action="store_true",
                   help="solo parsing + grounding, sin CoppeliaSim")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        return run_dry(args.instruction, args.parser_backend)
    # Ruta E2E con CoppeliaSim. Requiere CoppeliaSim en localhost:23000.
    from src.simulation.language_pick import run_language_pick
    payload = run_language_pick(instruction=args.instruction, scene=args.scene,
                                parser_backend=args.parser_backend, render=args.render)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
