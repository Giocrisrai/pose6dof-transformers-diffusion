"""Smoke test reproducible de la simulación CoppeliaSim (migrado al bridge).

Migración del runner original (que usaba RemoteAPIClient directo) a
CoppeliaSimBridge. La creación dinámica de sensores (createVisionSensor)
sigue usando la API ZMQ cruda vía bridge.sim — es un caso fuera del scope
de los wrappers del bridge.

Salidas en experiments/results/coppelia_smoke/.

Uso:
  1. CoppeliaSim Edu V4.10 en /Applications/CoppeliaSim_Edu.app, abierto.
  2. python experiments/run_coppelia_smoke_test.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "coppelia_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENE_PATH = Path(
    "/Applications/CoppeliaSim_Edu.app/Contents/Resources/scenes/pickAndPlaceDemo.ttt"
)


def main() -> int:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("[FAIL] Faltan numpy / Pillow en el venv")
        return 1

    print("[INFO] conectando a localhost:23000 vía bridge ...")
    t0 = time.perf_counter()
    with CoppeliaSimBridge() as bridge:
        connect_ms = (time.perf_counter() - t0) * 1000
        print(f"[OK]   conectado en {connect_ms:.0f} ms")

        server_version = bridge.sim.getInt32Param(bridge.sim.intparam_program_version)
        print(f"[INFO] sim version: {server_version}")

        # Cargar la escena pickAndPlaceDemo
        if SCENE_PATH.exists():
            print(f"[INFO] cargando escena {SCENE_PATH.name}")
            bridge.load_scene(SCENE_PATH)
            time.sleep(1.0)
        else:
            print(f"[WARN] escena no encontrada: {SCENE_PATH}")
            print("       seguimos con la escena por defecto cargada en CoppeliaSim")

        # Crear vision sensor cenital — escape hatch (createVisionSensor no
        # tiene wrapper en el bridge; ver spec sección 4, decisión I1)
        print("[INFO] creando vision sensor cenital ...")
        sim = bridge.sim
        options = 1 + 2  # explicit handling + perspective
        int_params = [640, 480, 0, 0]
        float_params = [0.05, 4.0, math.radians(60),
                        0.1, 0.1, 0.1,
                        0.0, 0.0, 0.0, 0.0, 0.0]
        vs_handle = sim.createVisionSensor(options, int_params, float_params)
        sim.setObjectAlias(vs_handle, "tfm_overview_sensor")
        sim.setObjectPosition(vs_handle, -1, [0.0, 0.0, 1.5])
        sim.setObjectOrientation(vs_handle, -1, [math.pi, 0.0, 0.0])

        # Inventario de objetos relevantes
        discovered = {}
        for query in ("/Floor", "/genericConveyorTypeA", "/genericDetectionWindow"):
            try:
                h = sim.getObject(query)
                discovered[query] = {
                    "handle": int(h),
                    "alias": sim.getObjectAlias(h),
                }
                print(f"[OK]   {query:30s} → handle={h}")
            except Exception:
                discovered[query] = {"handle": None}
                print(f"[--]   {query:30s} → no encontrado")

        # Iniciar simulación stepped y avanzar
        print("[INFO] iniciando simulación stepped, 100 pasos (5 s) ...")
        bridge.set_stepping(True)
        bridge.start_simulation()

        step_times_ms = []
        for _ in range(100):
            t = time.perf_counter()
            bridge.step()
            step_times_ms.append((time.perf_counter() - t) * 1000)

        final_sim_time = bridge.get_simulation_time()

        # Renderizar y guardar PNG
        print("[INFO] renderizando vision sensor ...")
        sim.handleVisionSensor(vs_handle)
        img_raw, res = sim.getVisionSensorImg(vs_handle)
        w, h = res[0], res[1]
        img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
        img = np.flipud(img)
        out_png = OUTPUT_DIR / "coppelia_overview_pickandplace.png"
        Image.fromarray(img).save(out_png)
        print(f"[OK]   {out_png.name}  {w}x{h} (mean={img.mean():.1f}, std={img.std():.1f})")

        # Cleanup explícito (el __exit__ del context manager también lo hace)
        bridge.stop_simulation()

    # Exportar resumen (fuera del with — los datos ya están capturados)
    summary = {
        "connect_ms": round(connect_ms, 2),
        "server_version": server_version,
        "scene_loaded": SCENE_PATH.name if SCENE_PATH.exists() else None,
        "discovered_handles": discovered,
        "vision_sensor": {
            "handle": int(vs_handle),
            "resolution": [w, h],
            "image_mean_intensity": float(img.mean()),
            "image_std": float(img.std()),
            "image_max": int(img.max()),
        },
        "stepping": {
            "n_steps": 100,
            "sim_time_advanced_s": round(final_sim_time, 4),
            "step_ms_mean": round(sum(step_times_ms) / len(step_times_ms), 3),
            "step_ms_min": round(min(step_times_ms), 3),
            "step_ms_max": round(max(step_times_ms), 3),
        },
        "outputs": {
            "overview_png": str(out_png.relative_to(REPO_ROOT)),
        },
    }
    out_json = OUTPUT_DIR / "smoke_test_result.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK]   {out_json.relative_to(REPO_ROOT)}")
    print("\n=== RESUMEN ===")
    print(f"  Conexión:          {summary['connect_ms']} ms")
    print(f"  Escena:            {summary['scene_loaded']}")
    print(f"  Step latency:      {summary['stepping']['step_ms_mean']} ms (mean)")
    print(f"  Sim time advanced: {summary['stepping']['sim_time_advanced_s']} s")
    print(f"  Render:            {w}x{h}, mean intensity {img.mean():.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
