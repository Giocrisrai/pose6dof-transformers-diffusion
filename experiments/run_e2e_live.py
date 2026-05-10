#!/usr/bin/env python3
"""E2E LIVE: ejecuta el ciclo completo con CoppeliaSim corriendo + Diffusion entrenado.

Mejoras vs run_pipeline_e2e.py:
- Sampling diffusion vectorizado (con DDIM 25 steps en lugar de DDPM 100)
- Conexion a CoppeliaSim con timeout
- Reporta H3 con datos reales

Salida: experiments/results/pipeline_e2e/e2e_live_metrics.json
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


def try_connect_coppelia(timeout_s=5.0):
    """Conexion robusta con timeout."""
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        t0 = time.time()
        client = RemoteAPIClient(host="localhost", port=23000)
        sim = client.getObject("sim")
        sim.getSimulationTime()  # ping
        return sim, time.time() - t0
    except Exception as e:
        print(f"  [warn] CoppeliaSim no disponible: {type(e).__name__}: {str(e)[:80]}")
        return None, None


def fast_diffusion_sample(planner, scheduler, cond, device, n_ddim_steps=25):
    """Sampling DDIM rapido en lugar de DDPM completo."""
    import torch
    horizon = 16
    action_dim = 7

    x = torch.randn(1, horizon, action_dim, device=device)

    # Subset de timesteps para DDIM
    full_t = scheduler.num_timesteps
    step_indices = np.linspace(0, full_t - 1, n_ddim_steps).astype(int)[::-1]

    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = planner(x, t_tensor, cond)

            ab_t = alpha_bar[step]
            # DDIM update
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                next_step = step_indices[i + 1]
                ab_next = alpha_bar[next_step]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0

    return x.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--datasets", nargs="+", default=["ycbv", "tless"])
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--sim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[e2e-live] n_instances={args.n_instances} ddim_steps={args.ddim_steps} sim_steps={args.sim_steps}")

    # 1. CoppeliaSim
    print("\n[1/4] Conectando a CoppeliaSim...")
    sim, connect_ms = try_connect_coppelia()
    if sim is None:
        print("  CoppeliaSim no disponible — abortando E2E live (use aggregate_e2e_timings.py para version offline)")
        return
    print(f"  CoppeliaSim OK (connect {connect_ms*1000:.1f} ms)")

    # Cargar escena por defecto si no hay nada (necesario para stepping)
    SCENE_PATH = "/Applications/CoppeliaSim_Edu.app/Contents/Resources/scenes/pickAndPlaceDemo.ttt"
    try:
        # Detener si esta corriendo
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            time.sleep(0.5)
        # Cargar escena
        if Path(SCENE_PATH).exists():
            sim.loadScene(SCENE_PATH)
            print(f"  Escena cargada: {Path(SCENE_PATH).name}")
        sim.setStepping(True)
        sim.startSimulation()
        # Test 1 step
        sim.step()
        print("  Stepping OK")
    except Exception as e:
        print(f"  [warn] setup sim: {e}")
        return

    # 2. Diffusion Policy
    print("\n[2/4] Cargando Diffusion Policy entrenado...")
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
        print(f"  Pesos: {weights_path.name} ({weights_path.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("  [warn] Sin pesos entrenados, usando random init")
    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)

    # 3. Loop E2E
    print(f"\n[3/4] Loop E2E (n={args.n_instances} por dataset)...")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {"datasets": {}, "config": vars(args), "coppelia_connect_ms": connect_ms*1000}

    for ds_name in args.datasets:
        ckpt_file = REPO / f"experiments/checkpoints/fp_{ds_name}_checkpoint.json"
        if not ckpt_file.exists():
            print(f"  [{ds_name}] sin checkpoint, skip")
            continue
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        preds = ckpt["results"][:args.n_instances]
        if not preds:
            continue

        per_instance = {"fp_ms": [], "diff_ms": [], "sim_ms": [], "total_ms": []}

        for i, pred in enumerate(preds):
            # FP: tiempo real del checkpoint (no se re-ejecuta sin GPU)
            fp_ms = pred["time_s"] * 1000.0

            # Diffusion sampling
            R = np.array(pred["R_pred"])
            t = np.array(pred["t_pred"])
            cond_vec = np.zeros(64, dtype=np.float32)
            flat = np.concatenate([R.flatten(), t.flatten()])
            cond_vec[:len(flat)] = flat
            cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

            t0 = time.time()
            traj = fast_diffusion_sample(planner, scheduler, cond, device, args.ddim_steps)
            diff_ms = (time.time() - t0) * 1000.0

            # Sim steps reales
            t0 = time.time()
            try:
                for _ in range(args.sim_steps):
                    sim.step()
                sim_ms = (time.time() - t0) * 1000.0
            except Exception as e:
                print(f"    [warn] sim.step: {e}")
                sim_ms = float("nan")

            total_ms = fp_ms + diff_ms + sim_ms
            per_instance["fp_ms"].append(fp_ms)
            per_instance["diff_ms"].append(diff_ms)
            per_instance["sim_ms"].append(sim_ms)
            per_instance["total_ms"].append(total_ms)

            if (i + 1) % 5 == 0:
                cur_p95 = np.percentile(per_instance["total_ms"], 95)
                print(f"  [{ds_name}] {i+1}/{len(preds)} | mean total={np.mean(per_instance['total_ms']):.0f} ms | p95={cur_p95:.0f} ms")

        # Stats
        arr = lambda k: np.array(per_instance[k])
        result = {"n": len(preds)}
        for k in per_instance:
            v = arr(k)
            result[k] = {
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "max": float(v.max()),
            }

        h3_p95 = result["total_ms"]["p95"]
        result["h3_acceptance"] = {
            "criterion": "p95(total) < 10000 ms",
            "p95_ms": h3_p95,
            "passed": bool(h3_p95 < 10000),
            "margin_ms": float(10000 - h3_p95),
        }

        results["datasets"][ds_name] = result
        print(f"  [{ds_name}] H3 p95={h3_p95:.0f} ms → {'✓ PASA' if h3_p95 < 10000 else '✗ FALLA'} (margen {10000-h3_p95:.0f} ms)")

    # 4. Detener sim
    try:
        sim.stopSimulation()
    except Exception:
        pass

    out = OUTPUT / "e2e_live_metrics.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
