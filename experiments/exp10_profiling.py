#!/usr/bin/env python3
"""Profiling detallado del pipeline para identificar cuellos de botella.

Mide tiempo por componente y por sub-operacion del pipeline:
- FoundationPose: tiempo total reportado por el checkpoint (no se reejecuta sin GPU)
- Diffusion DDIM-25: tiempo desglosado por timestep
- CoppeliaSim: tiempo del paso fisico
- Total agregado vs cada componente

Salida: experiments/results/exp10_profiling/exp10_results.json + figura
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp10_profiling"
OUT.mkdir(parents=True, exist_ok=True)

import torch

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler


def profile_diffusion(planner, scheduler, cond, device, n_steps=25, n_warmup=5, n_iters=20):
    """Profiling fino del DDIM sampling."""
    horizon, action_dim = 16, 7

    # Warmup
    for _ in range(n_warmup):
        x = torch.randn(1, horizon, action_dim, device=device)
        full_t = scheduler.num_timesteps
        step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
        alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
        with torch.no_grad():
            for i, step in enumerate(step_indices):
                t_tensor = torch.tensor([step], dtype=torch.long, device=device)
                _ = planner(x, t_tensor, cond)

    # Profiling
    forward_times = []
    full_sample_times = []
    overhead_times = []

    for trial in range(n_iters):
        x = torch.randn(1, horizon, action_dim, device=device)
        full_t = scheduler.num_timesteps
        step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
        alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
        forward_t_per_step = []

        t_start = time.perf_counter()
        with torch.no_grad():
            for i, step in enumerate(step_indices):
                t_tensor = torch.tensor([step], dtype=torch.long, device=device)
                t_fwd = time.perf_counter()
                noise_pred = planner(x, t_tensor, cond)
                if device == "mps":
                    torch.mps.synchronize()
                forward_t_per_step.append((time.perf_counter() - t_fwd) * 1000)
                ab_t = alpha_bar[step]
                pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
                if i < len(step_indices) - 1:
                    next_step = step_indices[i + 1]
                    ab_next = alpha_bar[next_step]
                    x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
                else:
                    x = pred_x0
        if device == "mps":
            torch.mps.synchronize()
        total_ms = (time.perf_counter() - t_start) * 1000
        full_sample_times.append(total_ms)
        forward_total = sum(forward_t_per_step)
        forward_times.append(forward_total)
        overhead_times.append(total_ms - forward_total)

    return {
        "n_iters": n_iters,
        "n_steps_ddim": n_steps,
        "full_sample_ms": {
            "mean": float(np.mean(full_sample_times)),
            "median": float(np.median(full_sample_times)),
            "std": float(np.std(full_sample_times)),
            "p95": float(np.percentile(full_sample_times, 95)),
        },
        "forward_only_ms": {
            "mean": float(np.mean(forward_times)),
            "fraction_of_total": float(np.mean(forward_times) / np.mean(full_sample_times)),
        },
        "overhead_ms": {
            "mean": float(np.mean(overhead_times)),
            "fraction_of_total": float(np.mean(overhead_times) / np.mean(full_sample_times)),
        },
        "per_forward_pass_ms": float(np.mean(forward_times) / n_steps),
    }


def main():
    print("[exp10] Profiling detallado del pipeline")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    planner = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=128).to(device)
    weights = REPO / "data/models/diffusion_policy_grasp.pth"
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    planner.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)

    cond = torch.randn(1, 64, device=device)

    print("\n  Profiling Diffusion DDIM con distintos n_steps:")
    diff_profiles = {}
    for n_steps in [10, 25, 50, 100]:
        prof = profile_diffusion(planner, scheduler, cond, device, n_steps=n_steps,
                                  n_warmup=3, n_iters=15)
        diff_profiles[n_steps] = prof
        print(f"    n_steps={n_steps:3d}: total {prof['full_sample_ms']['mean']:6.1f} ms, "
              f"forward {prof['forward_only_ms']['fraction_of_total']*100:.0f}%, "
              f"overhead {prof['overhead_ms']['fraction_of_total']*100:.0f}%, "
              f"per_pass {prof['per_forward_pass_ms']:.1f} ms")

    # Tiempos FP del checkpoint real
    print("\n  Tiempos FoundationPose (del checkpoint real, no reejecutados):")
    fp_times = {}
    for ds in ["ycbv", "tless"]:
        ckpt_path = REPO / f"experiments/checkpoints/fp_{ds}_checkpoint.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                results = json.load(f)["results"]
            times_ms = [r["time_s"] * 1000 for r in results]
            fp_times[ds] = {
                "n": len(times_ms),
                "mean": float(np.mean(times_ms)),
                "median": float(np.median(times_ms)),
                "p95": float(np.percentile(times_ms, 95)),
            }
            print(f"    {ds.upper()}: n={len(times_ms)}, mean={fp_times[ds]['mean']:.0f} ms, "
                  f"p95={fp_times[ds]['p95']:.0f} ms")

    # CoppeliaSim del smoke test
    sim_path = REPO / "experiments/results/coppelia_smoke/smoke_test_result.json"
    sim_data = {}
    if sim_path.exists():
        with open(sim_path) as f:
            smoke = json.load(f)
        sim_data = {
            "step_ms_mean": smoke["stepping"]["step_ms_mean"],
            "step_ms_min": smoke["stepping"]["step_ms_min"],
            "step_ms_max": smoke["stepping"]["step_ms_max"],
        }
        print(f"\n  CoppeliaSim sim.step (smoke): mean={sim_data['step_ms_mean']:.1f} ms, "
              f"max={sim_data['step_ms_max']:.0f} ms")

    # Cuello de botella analysis
    fp_avg = fp_times.get("ycbv", {}).get("mean", 4154)
    diff_25 = diff_profiles.get(25, {}).get("full_sample_ms", {}).get("mean", 200)
    sim_50_steps = sim_data.get("step_ms_mean", 18) * 50
    total_estimated = fp_avg + diff_25 + sim_50_steps

    bottleneck = {
        "FoundationPose_ms": fp_avg,
        "Diffusion_DDIM25_ms": diff_25,
        "CoppeliaSim_50steps_ms": sim_50_steps,
        "Total_estimated_ms": total_estimated,
        "FP_fraction_pct": fp_avg / total_estimated * 100,
        "Diff_fraction_pct": diff_25 / total_estimated * 100,
        "Sim_fraction_pct": sim_50_steps / total_estimated * 100,
    }
    print("\n=== CUELLO DE BOTELLA ===")
    print(f"  FoundationPose: {bottleneck['FP_fraction_pct']:.1f}% del ciclo (DOMINANTE)")
    print(f"  Diffusion DDIM-25: {bottleneck['Diff_fraction_pct']:.1f}% del ciclo")
    print(f"  CoppeliaSim 50 steps: {bottleneck['Sim_fraction_pct']:.1f}% del ciclo")

    summary = {
        "device": device,
        "diffusion_profiles_by_steps": diff_profiles,
        "foundationpose_real_times": fp_times,
        "coppelia_step_times": sim_data,
        "bottleneck_analysis": bottleneck,
    }
    with open(OUT / 'exp10_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] {OUT}/exp10_results.json")

    # Plot pie chart cuello de botella
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart distribucion del ciclo
        labels = ['FoundationPose', 'Diffusion DDIM-25', 'CoppeliaSim (50 pasos)']
        sizes = [fp_avg, diff_25, sim_50_steps]
        colors = ['#0098CD', '#35876B', '#E66B00']
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                    startangle=90, textprops={'fontsize': 11})
        axes[0].set_title(f'Distribución del tiempo de ciclo E2E\nTotal estimado: {total_estimated:.0f} ms', fontsize=12)

        # Bar chart latency vs n_steps DDIM
        steps = sorted(diff_profiles.keys())
        means = [diff_profiles[s]['full_sample_ms']['mean'] for s in steps]
        p95s = [diff_profiles[s]['full_sample_ms']['p95'] for s in steps]
        x = np.arange(len(steps))
        w = 0.35
        axes[1].bar(x - w/2, means, w, label='Mean', color='#0098CD')
        axes[1].bar(x + w/2, p95s, w, label='p95', color='#FF6B35')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(s) for s in steps])
        axes[1].set_xlabel('n_diffusion_steps')
        axes[1].set_ylabel('Latencia (ms)')
        axes[1].set_title(f'Latencia DDIM Diffusion Policy en {device.upper()}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, (m, p) in enumerate(zip(means, p95s)):
            axes[1].text(i - w/2, m + 5, f'{m:.0f}', ha='center', fontsize=9)
            axes[1].text(i + w/2, p + 5, f'{p:.0f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(OUT / 'fig_profiling.png', dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[OK] {OUT}/fig_profiling.png")
    except Exception as e:
        print(f"[warn] plot fail: {e}")


if __name__ == '__main__':
    main()
