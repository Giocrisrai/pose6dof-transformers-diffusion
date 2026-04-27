"""Smoke test reproducible de la simulación CoppeliaSim.

Valida que CoppeliaSim Edu V4.10.0 + el cliente Python ZMQ Remote API
están operativos para el pipeline de simulación de bin picking del TFM.

Pasos:
  1. Conectar al puerto 23000.
  2. Cargar la escena `pickAndPlaceDemo.ttt` (bundle de la app).
  3. Crear un vision sensor en cenital (1.5 m sobre el origen, 60° FOV,
     resolución 640×480) — pieza canónica de evidencia para el cap. 5.
  4. Iniciar simulación stepped, avanzar 100 pasos (5 s sim time).
  5. Renderizar el sensor y guardar PNG.
  6. Detener simulación, exportar JSON con métricas y handles
     descubiertos.

Salidas en `experiments/results/coppelia_smoke/`.

Uso:
  1. Asegurar CoppeliaSim Edu V4.10.0 en `/Applications/CoppeliaSim_Edu.app`.
  2. Lanzarlo (open -a CoppeliaSim_Edu) — la addon ZMQ Remote API se carga
     por defecto y escucha en localhost:23000.
  3. python experiments/run_coppelia_smoke_test.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "coppelia_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENE_PATH = (
    "/Applications/CoppeliaSim_Edu.app/Contents/Resources/scenes/"
    "pickAndPlaceDemo.ttt"
)


def main() -> int:
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        print("[FAIL] Instala antes:  uv pip install coppeliasim_zmqremoteapi_client")
        return 1

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("[FAIL] Faltan numpy / Pillow en el venv")
        return 1

    print("[INFO] conectando a localhost:23000 ...")
    t0 = time.perf_counter()
    client = RemoteAPIClient("localhost", 23000)
    sim = client.require("sim")
    connect_ms = (time.perf_counter() - t0) * 1000
    print(f"[OK]   conectado en {connect_ms:.0f} ms")

    server_version = sim.getInt32Param(sim.intparam_program_version)
    print(f"[INFO] sim version: {server_version}")

    # Cargar la escena pickAndPlaceDemo
    if Path(SCENE_PATH).exists():
        print(f"[INFO] cargando escena {Path(SCENE_PATH).name}")
        sim.loadScene(SCENE_PATH)
        time.sleep(1.0)
    else:
        print(f"[WARN] escena no encontrada: {SCENE_PATH}")
        print("       seguimos con la escena por defecto cargada en CoppeliaSim")

    # Crear vision sensor cenital para captura del overview
    print("[INFO] creando vision sensor cenital ...")
    options = 1 + 2  # explicit handling + perspective
    int_params = [640, 480, 0, 0]                          # resX, resY
    float_params = [0.05, 4.0, math.radians(60),            # near, far, fov
                    0.1, 0.1, 0.1,                          # cube size
                    0.0, 0.0, 0.0, 0.0, 0.0]                # padding + color
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
    sim.setStepping(True)
    sim.startSimulation()

    step_times_ms = []
    for _ in range(100):
        t = time.perf_counter()
        client.step()
        step_times_ms.append((time.perf_counter() - t) * 1000)

    final_sim_time = sim.getSimulationTime()

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

    # Cleanup
    sim.stopSimulation()
    sim.setStepping(False)
    for _ in range(40):
        if sim.getSimulationState() == sim.simulation_stopped:
            break
        time.sleep(0.1)

    # Exportar resumen
    summary = {
        "connect_ms": round(connect_ms, 2),
        "server_version": server_version,
        "scene_loaded": Path(SCENE_PATH).name if Path(SCENE_PATH).exists() else None,
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
    print(f"\n=== RESUMEN ===")
    print(f"  Conexión:          {summary['connect_ms']} ms")
    print(f"  Escena:            {summary['scene_loaded']}")
    print(f"  Step latency:      {summary['stepping']['step_ms_mean']} ms (mean)")
    print(f"  Sim time advanced: {summary['stepping']['sim_time_advanced_s']} s")
    print(f"  Render:            {w}x{h}, mean intensity {img.mean():.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
