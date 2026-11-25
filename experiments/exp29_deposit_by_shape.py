#!/usr/bin/env python3
"""Exp 29 — por qué caía el pick-and-place con apariencia aleatoria, y la corrección.

Las evaluaciones en simulación con apariencia aleatoria (Iter 8 e Iter 9) dan
un agarre cercano al 100 % pero un pick-and-place completo del 36-48 %. Este
análisis separa esas corridas por forma de la pieza y las compara, agarre a
agarre (mismas semillas), con la corrida posterior que añade:

- soltado suave: la pieza queda quieta mientras la pinza se abre, para que no
  recupere su física todavía entre los dedos;
- bandeja de depósito con paredes, para que esferas y cilindros no rueden fuera.

`released_in_zone` mide dónde estaba la pieza al abrir la pinza: si la política
la llevó a la zona de depósito, con independencia de lo que pase después.

No necesita simulador: lee los resultados versionados.

Uso:
    python experiments/exp29_deposit_by_shape.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results/pick_with_diffusion"
OUT = REPO / "experiments/results/exp29_deposit_by_shape"
FORMAS = ("cubo", "esfera", "cilindro")

# (archivo, condición) de las corridas con apariencia aleatoria de la memoria
CORRIDAS_MEMORIA = [
    ("eval_v8_robustez.json", "v8_randomized__randomizada"),
    ("eval_v9_clutter.json", "v9_clutter__sin_distractores"),
    ("eval_v9_clutter.json", "v9_clutter__con_distractores"),
]
# corrida posterior con soltado suave y bandeja; mismas semillas que eval_v9_clutter.json
CORRIDA_CORREGIDA = "eval_v9_bandeja.json"
CONDICIONES_PAREADAS = ("v9_clutter__sin_distractores", "v9_clutter__con_distractores")


def _validos(archivo: str, condicion: str) -> list[dict]:
    datos = json.loads((RES / archivo).read_text())
    return [r for r in datos["condiciones"][condicion]["resultados"] if "error" not in r]


def _pct(valores) -> float:
    return round(100 * float(np.mean(valores)), 1) if len(valores) else float("nan")


def _exito(r: dict) -> bool:
    return bool(r["grasp_plausible"] and r["deposit_plausible"] and r["ik_converged"])


def por_forma(resultados: list[dict]) -> dict:
    salida = {}
    con_soltado = all(r.get("released_in_zone") is not None for r in resultados)
    for forma in (*FORMAS, "todas"):
        rs = [r for r in resultados if forma == "todas" or r["forma"] == forma]
        if not rs:
            continue
        fila = {
            "n": len(rs),
            "grasp_pct": _pct([r["grasp_plausible"] for r in rs]),
            "deposit_pct": _pct([r["deposit_plausible"] for r in rs]),
            "pick_place_pct": _pct([_exito(r) for r in rs]),
            "mediana_deposit_error_m": round(float(np.median([r["deposit_error_m"] for r in rs])), 3),
        }
        if con_soltado:
            fila["released_in_zone_pct"] = _pct([r["released_in_zone"] for r in rs])
            fila["mediana_release_error_m"] = round(
                float(np.median([r["release_error_m"] for r in rs])), 3)
        salida[forma] = fila
    return salida


def main() -> int:
    memoria = [r for archivo, cond in CORRIDAS_MEMORIA for r in _validos(archivo, cond)]
    salida: dict = {
        "corridas_memoria": [f"{a}:{c}" for a, c in CORRIDAS_MEMORIA],
        "memoria_por_forma": por_forma(memoria),
    }
    if (RES / CORRIDA_CORREGIDA).exists():
        antes = [r for c in CONDICIONES_PAREADAS for r in _validos("eval_v9_clutter.json", c)]
        despues = [r for c in CONDICIONES_PAREADAS for r in _validos(CORRIDA_CORREGIDA, c)]
        salida["pareado"] = {
            "condiciones": list(CONDICIONES_PAREADAS),
            "antes_por_forma": por_forma(antes),
            "corregido_por_forma": por_forma(despues),
            "por_condicion": {
                c: {
                    "antes_pick_place_pct": _pct([_exito(r) for r in _validos("eval_v9_clutter.json", c)]),
                    "corregido_pick_place_pct": _pct([_exito(r) for r in _validos(CORRIDA_CORREGIDA, c)]),
                }
                for c in CONDICIONES_PAREADAS
            },
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp29_results.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))

    def imprimir(titulo: str, tabla: dict) -> None:
        print(titulo)
        for forma, f in tabla.items():
            extra = (f"  soltada en zona {f['released_in_zone_pct']:5.1f} %"
                     if "released_in_zone_pct" in f else "")
            print(f"  {forma:9s} n={f['n']:3d}  agarre {f['grasp_pct']:5.1f} %  depósito "
                  f"{f['deposit_pct']:5.1f} %  pick-and-place {f['pick_place_pct']:5.1f} %{extra}")

    imprimir("Corridas de la memoria (apariencia aleatoria), por forma:", salida["memoria_por_forma"])
    if "pareado" in salida:
        imprimir("Mismas semillas, antes:", salida["pareado"]["antes_por_forma"])
        imprimir("Mismas semillas, con soltado suave y bandeja:", salida["pareado"]["corregido_por_forma"])
    print(f"→ {OUT / 'exp29_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
