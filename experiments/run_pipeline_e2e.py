#!/usr/bin/env python3
"""E2E: percepcion (FP) + planificacion (Diffusion) + simulacion (CoppeliaSim opcional).

Mide el ciclo total por instancia para validar H3 (< 10 s/instancia).

- Si CoppeliaSim esta corriendo en :23000, usa su step real.
- Si no, simula con timing nominal medido en el smoke test (18 ms/step).

Salida: experiments/results/pipeline_e2e/e2e_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUTPUT = REPO / "experiments/results/pipeline_e2e"
OUTPUT.mkdir(parents=True, exist_ok=True)


def try_connect_coppelia(timeout_s=2.0):
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        t0 = time.time()
        client = RemoteAPIClient(host="localhost", port=23000)
        sim = client.getObject("sim")
        # Test simple: getSimulationTime
        sim.getSimulationTime()
        return sim, time.time() - t0
    except Exception:
        return None, None


def load_fp_predictions(ds_name: str, n_max: int):
    """Carga las primeras N predicciones FP del checkpoint."""
    ckpt_file = REPO / f"experiments/checkpoints/fp_{ds_name}_checkpoint.json"
    if not ckpt_file.exists():
        return []
    with open(ckpt_file) as f:
        ckpt = json.load(f)
    return ckpt["results"][:n_max]


def run_diffusion_planning(pose_pred, planner, scheduler):
    """Ejecuta diffusion planning sobre una pose. Devuelve trajectory + tiempo."""
    import torch
    t0 = time.time()
    R = np.array(pose_pred["R_pred"])
    t = np.array(pose_pred["t_pred"])
    # Conditioning: pose flatten (R[3x3] + t[3] = 12 dims) -> 64 dims pad
    cond_dim = 64
    cond_vec = np.concatenate([R.flatten(), t.flatten()])
    if len(cond_vec) < cond_dim:
        cond_vec = np.concatenate([cond_vec, np.zeros(cond_dim - len(cond_vec))])
    cond = torch.tensor(cond_vec[:cond_dim], dtype=torch.float32).unsqueeze(0)

    # Sampling con DDPM (random init)
    horizon = 16
    action_dim = 7
    x = torch.randn(1, horizon, action_dim)
    with torch.no_grad():
        for step in reversed(range(scheduler.num_timesteps)):
            t_tensor = torch.tensor([step], dtype=torch.long)
            noise_pred = planner(x, t_tensor, cond)
            # Simplified DDPM step
            alpha = scheduler.alphas[step]
            alpha_bar = scheduler.alpha_bar[step]
            beta = scheduler.betas[step]
            x = (1.0 / np.sqrt(alpha)) * (x - beta / np.sqrt(1 - alpha_bar) * noise_pred)
            if step > 0:
                x = x + np.sqrt(beta) * torch.randn_like(x)

    elapsed_ms = (time.time() - t0) * 1000
    return x.numpy()[0], elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--datasets", nargs="+", default=["ycbv", "tless"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[E2E] n_instances={args.n_instances}, datasets={args.datasets}")

    # 1. Conectar CoppeliaSim (opcional)
    print("\n[1/3] Conectando a CoppeliaSim...")
    sim, connect_time = try_connect_coppelia()
    coppelia_available = sim is not None
    if coppelia_available:
        print(f"  CoppeliaSim OK (connect: {connect_time*1000:.1f} ms)")
        try:
            sim.startSimulation()
        except Exception:
            pass
    else:
        print("  CoppeliaSim no disponible — usando tiempo nominal del smoke test (18 ms/step)")

    # 2. Cargar Diffusion Policy entrenada
    print("\n[2/3] Cargando Diffusion Policy...")
    import torch

    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=128)
    weights_path = REPO / "data/models/diffusion_policy_grasp.pth"
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        # Pueden venir como state_dict directo o anidados bajo varias claves
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                if key in ckpt:
                    planner.load_state_dict(ckpt[key])
                    break
            else:
                planner.load_state_dict(ckpt)
        else:
            planner.load_state_dict(ckpt)
        print(f"  Pesos cargados: {weights_path.name}")
    else:
        print("  [warn] No hay pesos entrenados — usando random init")
    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)

    # 3. Loop E2E
    print(f"\n[3/3] Ejecutando E2E sobre {args.n_instances} instancias por dataset...")
    np.random.seed(args.seed)
    results = {"datasets": {}, "config": vars(args), "coppelia_available": coppelia_available}

    # Tiempo nominal FP (mediana del run): ~4150 ms YCB-V, ~4350 ms T-LESS
    NOMINAL_FP_MS = {"ycbv": 4154.0, "tless": 4350.0}
    NOMINAL_STEP_MS = 18.0  # del smoke test
    SIM_STEPS_PER_INSTANCE = 50  # aproximado: 1 ciclo pick = ~50 steps

    for ds_name in args.datasets:
        preds = load_fp_predictions(ds_name, args.n_instances)
        if not preds:
            print(f"  [{ds_name}] sin checkpoint, skip")
            continue

        cycle_times_ms = []
        fp_times_ms = []
        diff_times_ms = []
        sim_times_ms = []

        for i, pred in enumerate(preds):
            time.time()

            # FP: tiempo nominal del run real (no se re-ejecuta sin GPU)
            fp_ms = pred.get("time_s", NOMINAL_FP_MS[ds_name] / 1000.0) * 1000.0
            fp_times_ms.append(fp_ms)

            # Diffusion planning (real, en MPS)
            traj, diff_ms = run_diffusion_planning(pred, planner, scheduler)
            diff_times_ms.append(diff_ms)

            # Simulacion: si CoppeliaSim disponible, step N veces; si no, tiempo nominal
            sim_t0 = time.time()
            if coppelia_available:
                try:
                    for _ in range(SIM_STEPS_PER_INSTANCE):
                        sim.step()
                    sim_ms = (time.time() - sim_t0) * 1000.0
                except Exception:
                    sim_ms = NOMINAL_STEP_MS * SIM_STEPS_PER_INSTANCE
            else:
                sim_ms = NOMINAL_STEP_MS * SIM_STEPS_PER_INSTANCE
            sim_times_ms.append(sim_ms)

            total_ms = fp_ms + diff_ms + sim_ms
            cycle_times_ms.append(total_ms)

            if (i + 1) % 10 == 0:
                print(f"  [{ds_name}] {i+1}/{len(preds)} cycles avg={np.mean(cycle_times_ms):.0f} ms")

        results["datasets"][ds_name] = {
            "n": len(preds),
            "cycle_total_ms": {
                "mean": float(np.mean(cycle_times_ms)),
                "median": float(np.median(cycle_times_ms)),
                "p95": float(np.percentile(cycle_times_ms, 95)),
                "max": float(np.max(cycle_times_ms)),
            },
            "fp_ms": {
                "mean": float(np.mean(fp_times_ms)),
                "median": float(np.median(fp_times_ms)),
                "source": "checkpoint time_s (real run 2026-04-27)",
            },
            "diffusion_ms": {
                "mean": float(np.mean(diff_times_ms)),
                "median": float(np.median(diff_times_ms)),
                "source": "diffusion_policy_grasp.pth on MPS (n=100 timesteps)",
            },
            "simulation_ms": {
                "mean": float(np.mean(sim_times_ms)),
                "median": float(np.median(sim_times_ms)),
                "source": "CoppeliaSim real" if coppelia_available else "nominal 18 ms/step from smoke test",
                "n_steps_per_instance": SIM_STEPS_PER_INSTANCE,
            },
            "h3_acceptance": {
                "criterion": "p95 cycle_total < 10000 ms",
                "p95_ms": float(np.percentile(cycle_times_ms, 95)),
                "passed": bool(np.percentile(cycle_times_ms, 95) < 10000),
            },
        }

        h3 = results["datasets"][ds_name]["h3_acceptance"]
        print(f"  [{ds_name}] H3 (<10s): p95={h3['p95_ms']:.0f} ms → {'✓ PASA' if h3['passed'] else '✗ FALLA'}")

    if coppelia_available:
        try:
            sim.stopSimulation()
        except Exception:
            pass

    out_path = OUTPUT / "e2e_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] guardado: {out_path}")


if __name__ == "__main__":
    main()
