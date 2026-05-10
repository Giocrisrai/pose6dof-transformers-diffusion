#!/usr/bin/env python3
"""Visualizacion 3D de trayectorias generadas por Diffusion Policy.

Genera figura 3D mostrando:
- 50 trayectorias muestreadas por Diffusion Policy para misma pose objetivo
- Trayectoria heuristica (linea recta) como baseline
- Pose objetivo destacada
- Distintas vistas (perspectiva, top, side)

Salida: experiments/results/exp9_3d_viz/fig_trajectories_3d.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp9_3d_viz"
OUT.mkdir(parents=True, exist_ok=True)

import torch

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler


def ddim_sample(planner, scheduler, cond, device, n_steps=25):
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


def main():
    print("[exp9] Visualizacion 3D de trayectorias multimodales")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    planner = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=128).to(device)
    weights = REPO / "data/models/diffusion_policy_grasp.pth"
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    planner.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)

    # Pose objetivo desde checkpoint real
    with open(REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json") as f:
        pred = json.load(f)["results"][0]
    R = np.array(pred["R_pred"])
    t_pose = np.array(pred["t_pred"])
    if np.linalg.norm(t_pose) < 5.0:
        t_pose = t_pose * 1000.0
    t_obj = t_pose / 1000.0  # m

    cond_vec = np.zeros(64, dtype=np.float32)
    flat = np.concatenate([R.flatten(), t_obj])
    cond_vec[:len(flat)] = flat

    # Muestrear N trayectorias
    N = 50
    print(f"  Muestreando {N} trayectorias diffusion...")
    trajs = []
    torch.manual_seed(42)
    np.random.seed(42)
    for i in range(N):
        cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
        traj = ddim_sample(planner, scheduler, cond, device, 25)
        trajs.append(traj)
    trajs = np.array(trajs)
    print(f"  Trayectorias: {trajs.shape}")

    # Trayectoria heuristica baseline
    horizon = 16
    heur = np.zeros((horizon, 7))
    start = np.array([0.0, 0.0, 0.5])
    end = t_obj
    for k in range(horizon):
        a = k / max(horizon - 1, 1)
        heur[k, :3] = (1 - a) * start + a * end

    # Plot 3D con multiples vistas
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 6))

    views = [
        ('Perspectiva 3D', 30, 45),
        ('Vista superior (XY)', 90, -90),
        ('Vista lateral (XZ)', 0, -90),
    ]

    for v_idx, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, v_idx + 1, projection='3d')

        # Plot N trayectorias diffusion (alpha bajo para superposicion)
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='#0098CD', alpha=0.25, linewidth=0.8)

        # Promedio diffusion
        traj_mean = trajs.mean(axis=0)
        ax.plot(traj_mean[:, 0], traj_mean[:, 1], traj_mean[:, 2],
                color='#0098CD', linewidth=3, label=f'Diffusion mean (N={N})')

        # Heuristico baseline
        ax.plot(heur[:, 0], heur[:, 1], heur[:, 2],
                color='#FF6B35', linewidth=3, linestyle='--', label='Heurístico determinista')

        # Pose objetivo
        ax.scatter([t_obj[0]], [t_obj[1]], [t_obj[2]],
                    s=220, c='#35876B', marker='*', edgecolor='black',
                    label='Pose objetivo (FP)', zorder=10)
        ax.scatter([0.0], [0.0], [0.5], s=120, c='black', marker='o',
                    label='Start', zorder=10)

        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.view_init(elev=elev, azim=azim)
        if v_idx == 0:
            ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Trayectorias Diffusion Policy multimodales ({N} muestras) vs heurístico — pose objetivo obj_id={pred["obj_id"]} (YCB-V)',
                  fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'fig_trajectories_3d.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] {OUT}/fig_trajectories_3d.png")

    # Estadisticas finales
    endpoint_dispersion = float(np.std(trajs[:, -1, :3], axis=0).mean() * 100)  # cm
    print(f"  Dispersion std endpoint: {endpoint_dispersion:.2f} cm")
    summary = {
        "n_trajectories": N,
        "obj_id": pred["obj_id"],
        "endpoint_std_cm": endpoint_dispersion,
        "endpoint_range_x_cm": float((trajs[:, -1, 0].max() - trajs[:, -1, 0].min()) * 100),
        "endpoint_range_y_cm": float((trajs[:, -1, 1].max() - trajs[:, -1, 1].min()) * 100),
        "endpoint_range_z_cm": float((trajs[:, -1, 2].max() - trajs[:, -1, 2].min()) * 100),
    }
    with open(OUT / 'exp9_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] {OUT}/exp9_results.json")


if __name__ == '__main__':
    main()
