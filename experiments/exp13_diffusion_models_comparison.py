#!/usr/bin/env python3
"""Comparativa rigurosa de los 3 modelos Diffusion Policy entrenados.

Modelos:
- Original:  30 epochs, 2K trayectorias, hidden=128
- Extended:  50 epochs, 5K trayectorias, hidden=192
- Ultra:    100 epochs, 10K trayectorias, hidden=256

Metricas:
- MSE final reportado
- Calidad de trayectoria muestreada (jerk, suavidad)
- Diversidad multimodal (dispersion endpoint)
- Latencia DDIM sampling
- Coherencia con pose objetivo (distancia endpoint - target)

Bootstrap CI 95% sobre 50 muestras por modelo.

Salida: experiments/results/exp13_model_comparison/exp13_results.json + figura
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp13_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

MODELS = {
    "original": {
        "path": REPO / "data/models/diffusion_policy_grasp.pth",
        "hidden_dim": 128,
        "label": "Original (30ep, 2K)",
        "reference_mse": 0.020,
    },
    "extended_mps": {
        "path": REPO / "data/models/diffusion_policy_extended_mps.pth",
        "hidden_dim": 192,
        "label": "Extended MPS (50ep, 5K)",
        "reference_mse": 0.01288,
    },
    "ultra": {
        "path": REPO / "data/models/diffusion_policy_ultra.pth",
        "hidden_dim": 256,
        "label": "Ultra (100ep, 10K)",
        "reference_mse": None,  # se lee del checkpoint
    },
}


def load_model(name, info, device):
    model = ConditionalUNet1D(
        action_dim=7, horizon=16, cond_dim=64, hidden_dim=info["hidden_dim"]
    ).to(device)
    if not info["path"].exists():
        return None, None
    ckpt = torch.load(info["path"], map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        val_loss = ckpt.get("val_loss") if "val_loss" in ckpt else None
    else:
        sd = ckpt
        val_loss = None
    model.load_state_dict(sd)
    model.eval()
    return model, val_loss


def ddim_sample(model, scheduler, cond, device, n_steps=25):
    horizon, action_dim = 16, 7
    x = torch.randn(1, horizon, action_dim, device=device)
    full_t = scheduler.num_timesteps
    step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = model(x, t_tensor, cond)
            ab_t = alpha_bar[step]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                next_step = step_indices[i + 1]
                ab_next = alpha_bar[next_step]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0
    return x.cpu().numpy()[0]


def jerk_rms(traj):
    if len(traj) < 4:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(traj[:, :3], n=3, axis=0) ** 2)))


def main():
    print("[exp13] Comparativa de modelos Diffusion Policy")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")

    scheduler = SimpleDDPMScheduler(num_timesteps=100)

    # Pose objetivo fija para comparabilidad
    cond = torch.zeros(1, 64, device=device)
    cond[0, :3] = torch.tensor([0.0, 0.0, 0.8])

    N_SAMPLES = 50

    results = {}
    for name, info in MODELS.items():
        print(f"\n=== {info['label']} ===")
        model, val_loss = load_model(name, info, device)
        if model is None:
            print("  [skip] modelo no disponible")
            continue

        # Warmup
        for _ in range(3):
            ddim_sample(model, scheduler, cond, device, 25)

        # Muestreo + medicion
        trajs = []
        latencies = []
        jerks = []
        for _ in range(N_SAMPLES):
            t0 = time.time()
            traj = ddim_sample(model, scheduler, cond, device, 25)
            if device == "mps":
                torch.mps.synchronize()
            latencies.append((time.time() - t0) * 1000)
            trajs.append(traj)
            jerks.append(jerk_rms(traj))

        trajs = np.array(trajs)
        endpoints = trajs[:, -1, :3]
        # Distancia entre endpoints (diversidad)
        n = len(endpoints)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.linalg.norm(endpoints[i] - endpoints[j]))
        # Bootstrap CI sobre el jerk
        rng = np.random.default_rng(42)
        B = 1000
        boot_jerk = np.empty(B)
        for k in range(B):
            sample = rng.choice(jerks, size=len(jerks), replace=True)
            boot_jerk[k] = np.mean(sample)

        results[name] = {
            "label": info["label"],
            "reference_mse": info["reference_mse"] if info["reference_mse"] is not None else val_loss,
            "hidden_dim": info["hidden_dim"],
            "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
            },
            "jerk_rms": {
                "mean": float(np.mean(jerks)),
                "ci95_lo": float(np.percentile(boot_jerk, 2.5)),
                "ci95_hi": float(np.percentile(boot_jerk, 97.5)),
            },
            "diversity_endpoint_dist_m": {
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
            },
        }

        r = results[name]
        print(f"  MSE ref: {r['reference_mse']:.5f}")
        print(f"  Latencia: mean={r['latency_ms']['mean']:.1f} ms, p95={r['latency_ms']['p95']:.1f} ms")
        print(f"  Jerk: mean={r['jerk_rms']['mean']:.4f} [CI 95% {r['jerk_rms']['ci95_lo']:.4f}, {r['jerk_rms']['ci95_hi']:.4f}]")
        print(f"  Diversidad: {r['diversity_endpoint_dist_m']['mean']*100:.1f} ± {r['diversity_endpoint_dist_m']['std']*100:.1f} cm")

    # Plot comparativo
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        names = list(results.keys())
        labels = [results[n]["label"] for n in names]
        colors = ['#0098CD', '#35876B', '#FF6B35']

        # MSE
        mse = [results[n]["reference_mse"] for n in names]
        axes[0].bar(range(len(names)), mse, color=colors)
        for i, v in enumerate(mse):
            axes[0].text(i, v + 0.0008, f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        axes[0].set_ylabel('Best validation MSE')
        axes[0].set_title('Calidad del entrenamiento (menor = mejor)')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Latencia con error bars
        means = [results[n]['latency_ms']['mean'] for n in names]
        p95s = [results[n]['latency_ms']['p95'] for n in names]
        x = np.arange(len(names))
        w = 0.35
        axes[1].bar(x - w/2, means, w, label='mean', color='#0098CD')
        axes[1].bar(x + w/2, p95s, w, label='p95', color='#FF6B35')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        axes[1].set_ylabel('Latencia DDIM-25 (ms)')
        axes[1].set_title('Latencia sampling')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # Diversidad
        div = [results[n]['diversity_endpoint_dist_m']['mean']*100 for n in names]
        axes[2].bar(range(len(names)), div, color=colors)
        for i, v in enumerate(div):
            axes[2].text(i, v + 1, f"{v:.1f} cm", ha='center', fontsize=10, fontweight='bold')
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        axes[2].set_ylabel('Distancia endpoint promedio (cm)')
        axes[2].set_title('Diversidad multimodal')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Comparativa de modelos Diffusion Policy (n=50 muestras, bootstrap CI 95%)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUT / 'fig_model_comparison.png', dpi=180, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] {OUT}/fig_model_comparison.png")
    except Exception as e:
        print(f"[warn] plot fail: {e}")

    with open(OUT / 'exp13_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] {OUT}/exp13_results.json")

    # Resumen
    print("\n=== RESUMEN COMPARATIVO ===")
    print(f"{'Modelo':30} {'MSE':>10} {'Latencia':>12} {'Jerk':>10} {'Diversidad':>15}")
    for name in names:
        r = results[name]
        print(f"{r['label'][:30]:30} {r['reference_mse']:>10.5f} "
              f"{r['latency_ms']['mean']:>9.1f} ms "
              f"{r['jerk_rms']['mean']:>10.4f} "
              f"{r['diversity_endpoint_dist_m']['mean']*100:>12.1f} cm")


if __name__ == "__main__":
    main()
