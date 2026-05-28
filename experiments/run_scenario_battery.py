#!/usr/bin/env python3
"""Runner de batería de escenarios.

Itera sobre data/scenes/scenarios.yaml. Por cada escenario:
    1. Carga .ttt via bridge.
    2. Aplica tweaks (apply_scenario).
    3. Inicia simulación stepped.
    4. Captura RGB-D inicial → snapshot PNG.
    5. Diffusion planning sobre la pose nominal (FP no se re-ejecuta).
    6. Avanza N steps simulando ejecución del trajectory.
    7. Lee is_grasping() (con estabilización previa).
    8. Detiene simulación.

Outputs en experiments/results/scenario_battery/:
    - report.json
    - report.md
    - snapshots/<scenario_id>.png

Uso (requiere CoppeliaSim corriendo en :23000):
    .venv/bin/python experiments/run_scenario_battery.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.scenarios import Scenario, load_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SCENES_DIR = REPO / "data" / "scenes"
SCENARIOS_YAML = SCENES_DIR / "scenarios.yaml"
OUTPUT_DIR = REPO / "experiments" / "results" / "scenario_battery"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"

# Tiempos nominales (consistentes con run_pipeline_e2e.py)
NOMINAL_FP_MS = 4154.0          # mediana YCB-V del run real 2026-04-27
SIM_STEPS_PER_INSTANCE = 50
STABILIZATION_STEPS = 5


def save_snapshot(rgb: np.ndarray, scenario_id: str) -> Path:
    """Guarda RGB como PNG."""
    from PIL import Image
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOTS_DIR / f"{scenario_id}.png"
    Image.fromarray(rgb).save(out)
    return out


def load_planner():
    """Carga la Diffusion Policy entrenada (mismo patrón que run_e2e_live.py)."""
    import torch

    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=128).to(device)
    weights_path = REPO / "data/models/diffusion_policy_grasp.pth"

    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                if key in ckpt:
                    planner.load_state_dict(ckpt[key])
                    break
            else:
                planner.load_state_dict(ckpt)
        else:
            planner.load_state_dict(ckpt)
        logger.info(f"pesos cargados: {weights_path.name}")
    else:
        logger.warning("sin pesos entrenados — usando random init")

    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)
    return planner, scheduler, device


def run_diffusion_planning(planner, scheduler, device) -> float:
    """Corre diffusion sampling con conditioning random. Devuelve elapsed_ms."""
    import torch

    t0 = time.time()
    cond = torch.zeros(1, 64, dtype=torch.float32, device=device)
    horizon = 16
    action_dim = 7
    x = torch.randn(1, horizon, action_dim, device=device)

    with torch.no_grad():
        for step in reversed(range(scheduler.num_timesteps)):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = planner(x, t_tensor, cond)
            alpha = scheduler.alphas[step]
            alpha_bar = scheduler.alpha_bar[step]
            beta = scheduler.betas[step]
            x = (1.0 / np.sqrt(alpha)) * (x - beta / np.sqrt(1 - alpha_bar) * noise_pred)
            if step > 0:
                x = x + np.sqrt(beta) * torch.randn_like(x)

    return (time.time() - t0) * 1000.0


def run_scenario(bridge: CoppeliaSimBridge, sc: Scenario, planner, scheduler, device) -> dict:
    """Ejecuta un escenario. Devuelve dict con métricas."""
    logger.info(f"--- escenario {sc.id} ({sc.difficulty}) ---")

    scene_path = SCENES_DIR / sc.scene
    bridge.load_scene(scene_path)
    bridge.apply_scenario(sc.to_dict())
    bridge.set_stepping(True)
    bridge.start_simulation()

    # Warm-up: el vision sensor necesita al menos un step para renderizar
    # tras start_simulation en modo stepped.
    for _ in range(3):
        bridge.step()

    # Forzar render explícito del vision sensor antes de capturar (sin esto
    # getVisionSensorImg devuelve un buffer sin inicializar = imagen negra).
    # handleVisionSensor no está envuelto por el bridge → escape hatch público.
    if bridge._camera_rgb_handle is not None:
        try:
            bridge.sim.handleVisionSensor(bridge._camera_rgb_handle)
        except Exception as e:
            logger.debug(f"handleVisionSensor skipped: {e}")
    else:
        logger.warning("rgb_camera handle no inicializado; capture_rgbd devolverá ceros")

    # Snapshot inicial
    rgb, _ = bridge.capture_rgbd()
    snapshot_path = save_snapshot(rgb, sc.id)
    logger.info(f"snapshot: {snapshot_path.relative_to(REPO)}")

    # Diffusion planning
    diff_ms = run_diffusion_planning(planner, scheduler, device)

    # Simulación: N steps
    sim_t0 = time.time()
    for _ in range(SIM_STEPS_PER_INSTANCE):
        bridge.step()
    sim_ms = (time.time() - sim_t0) * 1000.0

    # Estabilización antes de leer grasp (mitigación spec sección 3)
    try:
        bridge.actuate_gripper(open=False)
    except Exception as e:
        logger.debug(f"actuate_gripper skipped (no gripper): {e}")
    for _ in range(STABILIZATION_STEPS):
        bridge.step()

    gripper_present = bridge._gripper_handle is not None
    grasp_success = bridge.is_grasping() if gripper_present else False

    bridge.stop_simulation()

    return {
        "scenario_id": sc.id,
        "scene": sc.scene,
        "difficulty": sc.difficulty,
        "description": sc.description,
        "n_tweaks": len(sc.tweaks),
        "cycle_total_ms": NOMINAL_FP_MS + diff_ms + sim_ms,
        "fp_ms": NOMINAL_FP_MS,
        "diff_ms": diff_ms,
        "sim_ms": sim_ms,
        "gripper_present": gripper_present,
        "grasp_success": grasp_success,
        "snapshot": str(snapshot_path.relative_to(REPO)),
    }


def save_report(results: list[dict]) -> None:
    """Guarda report.json + report.md."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "report.json"
    with json_path.open("w") as f:
        json.dump({"scenarios": results}, f, indent=2)
    logger.info(f"escrito: {json_path.relative_to(REPO)}")

    md_path = OUTPUT_DIR / "report.md"
    with md_path.open("w") as f:
        f.write("# Scenario Battery Report\n\n")
        f.write("**fp_ms** = tiempo nominal de FoundationPose (no se re-ejecuta sin GPU dedicada).\n")
        f.write("**grasp_success**: bool basado en `is_grasping()` tras estabilización. ")
        f.write("Si `gripper_present=false`, el campo se interpreta como 'no aplica'.\n\n")
        f.write("| id | difficulty | cycle_ms | diff_ms | sim_ms | gripper | grasp_ok | snapshot |\n")
        f.write("|---|---|---:|---:|---:|:-:|:-:|---|\n")
        for r in results:
            grip = "✓" if r["gripper_present"] else "—"
            grasp = ("✓" if r["grasp_success"] else "✗") if r["gripper_present"] else "n/a"
            f.write(
                f"| {r['scenario_id']} | {r['difficulty']} | "
                f"{r['cycle_total_ms']:.0f} | {r['diff_ms']:.0f} | {r['sim_ms']:.0f} | "
                f"{grip} | {grasp} | `{r['snapshot']}` |\n"
            )
    logger.info(f"escrito: {md_path.relative_to(REPO)}")


def main() -> int:
    scenarios = load_scenarios(SCENARIOS_YAML)
    logger.info(f"cargados {len(scenarios)} escenarios desde {SCENARIOS_YAML.relative_to(REPO)}")

    planner, scheduler, device = load_planner()

    results = []
    for sc in scenarios:
        try:
            with CoppeliaSimBridge() as bridge:
                result = run_scenario(bridge, sc, planner, scheduler, device)
                results.append(result)
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
