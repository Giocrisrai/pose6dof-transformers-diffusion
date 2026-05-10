#!/usr/bin/env python3
"""Ablation: n_diffusion_steps en {25, 50, 100} sobre el modelo Diffusion Policy entrenado.

Evalua trade-off latencia vs calidad para los 3 niveles canonicos de DDIM steps.
Usa el modelo entrenado en local (data/models/diffusion_policy_grasp.pth).

Salida:
    experiments/results/exp5_diffusion_steps/exp5_results.json
    experiments/results/exp5_diffusion_steps/latency_vs_steps.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUTPUT = REPO / "experiments/results/exp5_diffusion_steps"
OUTPUT.mkdir(parents=True, exist_ok=True)

import torch
from src.planning.diffusion_policy import SimpleDDPMScheduler, ConditionalUNet1D


def load_planner(device):
    planner = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=128).to(device)
    weights = REPO / "data/models/diffusion_policy_grasp.pth"
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    planner.load_state_dict(sd)
    planner.eval()
    return planner


def ddim_sample(planner, scheduler, cond, device, n_steps):
    horizon, action_dim = 16, 7
    x = torch.randn(1, horizon, action_dim, device=device)

    full_t = scheduler.num_timesteps
    step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = planner(x, t_tensor, cond)
            ab_t = alpha_bar[step]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                next_step = step_indices[i + 1]
                ab_next = alpha_bar[next_step]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0

    return x.cpu().numpy()[0]


def trajectory_smoothness(traj):
    """Jerk RMS como medida de suavidad."""
    if len(traj) < 4:
        return 0.0
    jerk = np.diff(traj, n=3, axis=0)
    return float(np.sqrt(np.mean(jerk ** 2)))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[exp5] device: {device}")

    planner = load_planner(device)
    scheduler = SimpleDDPMScheduler(num_timesteps=100)
    print(f"[exp5] modelo cargado: data/models/diffusion_policy_grasp.pth")

    # Cargar 30 poses reales del checkpoint (mismo subset que H2)
    ckpt = REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json"
    with open(ckpt) as f:
        preds = json.load(f)["results"][:30]
    print(f"[exp5] poses condicionantes: {len(preds)}")

    levels = [25, 50, 100]
    n_warmup = 3  # warmup para evitar overhead de compilacion MPS

    results = {"device": device, "n_poses": len(preds), "levels": {}}

    for n_steps in levels:
        print(f"\n=== n_diffusion_steps = {n_steps} ===")
        latencies = []
        smoothness = []

        # Warmup
        for k in range(n_warmup):
            R = np.array(preds[k]["R_pred"])
            t = np.array(preds[k]["t_pred"])
            cond_vec = np.zeros(64, dtype=np.float32)
            flat = np.concatenate([R.flatten(), t.flatten()])
            cond_vec[:len(flat)] = flat
            cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
            _ = ddim_sample(planner, scheduler, cond, device, n_steps)

        # Medicion
        for i, pred in enumerate(preds):
            R = np.array(pred["R_pred"])
            t = np.array(pred["t_pred"])
            cond_vec = np.zeros(64, dtype=np.float32)
            flat = np.concatenate([R.flatten(), t.flatten()])
            cond_vec[:len(flat)] = flat
            cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

            t0 = time.time()
            traj = ddim_sample(planner, scheduler, cond, device, n_steps)
            elapsed_ms = (time.time() - t0) * 1000.0

            latencies.append(elapsed_ms)
            smoothness.append(trajectory_smoothness(traj))

        latencies = np.array(latencies)
        smoothness = np.array(smoothness)

        result = {
            "n_steps": n_steps,
            "latency_ms": {
                "mean": float(latencies.mean()),
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "std": float(latencies.std()),
            },
            "smoothness_jerk_rms": {
                "mean": float(smoothness.mean()),
                "median": float(np.median(smoothness)),
                "std": float(smoothness.std()),
            },
        }
        results["levels"][str(n_steps)] = result
        print(f"  latencia: mean={result['latency_ms']['mean']:.1f} ms, p95={result['latency_ms']['p95']:.1f} ms")
        print(f"  jerk RMS: mean={result['smoothness_jerk_rms']['mean']:.4f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        levels_arr = np.array(levels)
        means = [results["levels"][str(n)]["latency_ms"]["mean"] for n in levels]
        p95s = [results["levels"][str(n)]["latency_ms"]["p95"] for n in levels]
        jerks = [results["levels"][str(n)]["smoothness_jerk_rms"]["mean"] for n in levels]

        ax1.plot(levels_arr, means, 'o-', color='#0098CD', label='latencia mean (ms)', linewidth=2, markersize=8)
        ax1.plot(levels_arr, p95s, 's--', color='#006C8F', label='latencia p95 (ms)', linewidth=2, markersize=7)
        ax1.set_xlabel('n_diffusion_steps')
        ax1.set_ylabel('Latencia (ms)', color='#0098CD')
        ax1.tick_params(axis='y', labelcolor='#0098CD')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(levels_arr, jerks, '^-', color='#FF6B35', label='jerk RMS', linewidth=2, markersize=8)
        ax2.set_ylabel('Jerk RMS (suavidad)', color='#FF6B35')
        ax2.tick_params(axis='y', labelcolor='#FF6B35')
        ax2.legend(loc='upper right')

        plt.title('Ablation: n_diffusion_steps — latencia vs suavidad de trayectoria\n(M1 Pro / MPS, n=30 poses condicionantes YCB-V)')
        plt.tight_layout()
        out_png = OUTPUT / 'latency_vs_steps.png'
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] {out_png}")
    except Exception as e:
        print(f"[warn] plot fallido: {e}")

    out_json = OUTPUT / 'exp5_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] {out_json}")


if __name__ == '__main__':
    main()
