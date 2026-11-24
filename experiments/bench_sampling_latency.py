#!/usr/bin/env python3
"""Latencia de muestreo por trayectoria: DDIM-25 frente al modelo destilado (1 NFE).

El criterio de H2 pide un muestreo por debajo de 50 ms por trayectoria. En la
memoria del TFM, Diffusion Policy con DDIM-25 no lo alcanzaba (118 ms en el
profiling del pipeline). La exploración 2 destiló ese modelo a un solo paso
(`ultra_fast`). Este script mide ambos por el mismo camino que usa la API
(`POST /plan-grasp`, una trayectoria por llamada) y guarda el resultado.

    python experiments/bench_sampling_latency.py            # n = 100 por modelo

Escribe experiments/results/sampling_latency/latency.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SALIDA = REPO / "experiments/results/sampling_latency/latency.json"
UMBRAL_H2_MS = 50.0
PETICION = {"object_position": [0.0, 0.0, 0.8], "n_samples": 1, "n_diffusion_steps": 25}


def medir(cliente, modelo: str, n: int, calentamiento: int = 5) -> dict:
    peticion = {**PETICION, "model": modelo}
    for _ in range(calentamiento):
        cliente.post("/plan-grasp", json=peticion)
    ms = sorted(cliente.post("/plan-grasp", json=peticion).json()["latency_ms"] for _ in range(n))
    return {
        "n": n,
        "median_ms": round(statistics.median(ms), 2),
        "p95_ms": round(ms[int(0.95 * n) - 1], 2),
        "max_ms": round(ms[-1], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=100, help="llamadas medidas por modelo")
    args = parser.parse_args()

    import api_server
    from fastapi.testclient import TestClient

    cliente = TestClient(api_server.app)
    resultado = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": api_server.get_device(),
        "machine": f"{platform.system()} {platform.machine()}",
        "h2_threshold_ms": UMBRAL_H2_MS,
        "request": PETICION,
        "models": {
            "ultra": {"sampler": "DDIM-25", **medir(cliente, "ultra", args.n)},
            "ultra_fast": {"sampler": "distilled, 1 NFE", **medir(cliente, "ultra_fast", args.n)},
        },
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resultado, indent=2) + "\n")
    for nombre, r in resultado["models"].items():
        cumple = "cumple" if r["p95_ms"] < UMBRAL_H2_MS else "no cumple"
        print(f"{nombre:10s} {r['sampler']:18s} mediana {r['median_ms']:7.2f} ms · "
              f"p95 {r['p95_ms']:7.2f} ms · {cumple} el umbral de {UMBRAL_H2_MS:.0f} ms")
    print(f"→ {SALIDA.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
