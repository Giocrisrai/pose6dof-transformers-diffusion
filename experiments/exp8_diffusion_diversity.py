#!/usr/bin/env python3
"""Analisis de diversidad multimodal de Diffusion Policy vs heuristico determinista.

Objetivo: cuantificar la capacidad multimodal del modelo entrenado mediante
K-means sobre N trayectorias muestreadas para una misma pose, y compararlo
con un planificador heuristico que produce una unica solucion.

Metricas:
- Numero efectivo de modos (silhouette + elbow K-means)
- Distancia inter-cluster media (cm)
- Distancia intra-cluster media (compactidad)
- Jerk RMS por trayectoria (suavidad)

Salidas:
    experiments/results/exp8_diversity/exp8_results.json
    experiments/results/exp8_diversity/fig_diversity_pca.png
    experiments/results/exp8_diversity/fig_jerk_comparison.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp8_diversity"
OUT.mkdir(parents=True, exist_ok=True)

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


def jerk_rms(traj):
    """Jerk RMS de una trayectoria (suavidad)."""
    if len(traj) < 4:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(traj, n=3, axis=0) ** 2)))


def heuristic_trajectory(t_obj, horizon=16, action_dim=7):
    """Trayectoria heuristica deterministica: linea recta hacia objetivo + apertura gripper."""
    traj = np.zeros((horizon, action_dim))
    start = np.array([0.0, 0.0, 0.5, 0, 0, 0, 1.0])  # gripper abierto
    end = np.array([t_obj[0], t_obj[1], t_obj[2], 0, 0, 0, 0.0])  # gripper cerrado
    for k in range(horizon):
        alpha = k / max(horizon - 1, 1)
        traj[k] = (1 - alpha) * start + alpha * end
    return traj


def main():
    print("[exp8] Analisis de diversidad multimodal")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = load_planner(device)
    scheduler = SimpleDDPMScheduler(num_timesteps=100)

    # Cargar pose condicional real
    ckpt = REPO / "experiments/checkpoints/fp_ycbv_checkpoint.json"
    with open(ckpt) as f:
        preds = json.load(f)["results"][:5]  # 5 escenas distintas

    N_SAMPLES = 30  # trayectorias por escena
    print(f"  Muestras por escena: {N_SAMPLES} | escenas: {len(preds)}")

    all_diff_trajs = []
    all_jerk_diff = []
    all_jerk_heur = []
    per_scene_results = []

    for i, pred in enumerate(preds):
        R = np.array(pred["R_pred"])
        t_pose = np.array(pred["t_pred"])
        if np.linalg.norm(t_pose) < 5.0:
            t_pose = t_pose * 1000.0  # m -> mm
        t_pose_m = t_pose / 1000.0  # m

        cond_vec = np.zeros(64, dtype=np.float32)
        flat = np.concatenate([R.flatten(), t_pose_m])
        cond_vec[:len(flat)] = flat

        # Diffusion: N trayectorias distintas para misma pose
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        diff_trajs_scene = []
        for _ in range(N_SAMPLES):
            cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
            traj = ddim_sample(planner, scheduler, cond, device, 25)
            diff_trajs_scene.append(traj)
            all_jerk_diff.append(jerk_rms(traj))
        diff_trajs_scene = np.array(diff_trajs_scene)
        all_diff_trajs.append(diff_trajs_scene)

        # Heuristico (1 sola, deterministico)
        heur = heuristic_trajectory(t_pose_m)
        all_jerk_heur.append(jerk_rms(heur))

        # Distancia inter-trayectoria diffusion (variabilidad para misma pose)
        # Comparamos los puntos finales (waypoint final 6-DoF + gripper)
        endpoints = diff_trajs_scene[:, -1, :3]  # (N, 3)
        dists = []
        for a in range(N_SAMPLES):
            for b in range(a+1, N_SAMPLES):
                dists.append(np.linalg.norm(endpoints[a] - endpoints[b]))
        per_scene_results.append({
            "scene": i,
            "obj_id": pred["obj_id"],
            "n_trajs": N_SAMPLES,
            "endpoint_dist_mean_cm": float(np.mean(dists) * 100),
            "endpoint_dist_std_cm": float(np.std(dists) * 100),
            "endpoint_dist_max_cm": float(np.max(dists) * 100),
        })
        print(f"  Escena {i+1}: dist endpoint mean = {np.mean(dists)*100:.2f} cm")

    # Agregado global de jerk (Diffusion vs Heuristico)
    summary = {
        "n_scenes": len(preds),
        "n_samples_per_scene": N_SAMPLES,
        "diffusion_jerk_rms": {
            "mean": float(np.mean(all_jerk_diff)),
            "median": float(np.median(all_jerk_diff)),
            "std": float(np.std(all_jerk_diff)),
        },
        "heuristic_jerk_rms": {
            "mean": float(np.mean(all_jerk_heur)),
            "median": float(np.median(all_jerk_heur)),
            "std": float(np.std(all_jerk_heur)),
        },
        "per_scene": per_scene_results,
    }

    # Numero efectivo de modos via K-means + silhouette
    print("\n  K-means sobre trayectorias agregadas (todas las escenas):")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        all_endpoints = np.vstack([t[:, -1, :3] for t in all_diff_trajs])
        scores = {}
        for k in [2, 3, 4, 5, 6, 8]:
            if k >= len(all_endpoints):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(all_endpoints)
            sil = silhouette_score(all_endpoints, labels)
            scores[k] = float(sil)
            print(f"    K={k}: silhouette = {sil:.3f}")
        best_k = max(scores.items(), key=lambda x: x[1])[0]
        summary["best_n_modes_silhouette"] = best_k
        summary["silhouette_scores"] = scores
        print(f"  Mejor K (silhouette): {best_k}")
    except Exception as e:
        print(f"  [warn] sklearn fallo: {e}")
        summary["best_n_modes_silhouette"] = None

    # Plot 1: PCA de trayectorias para visualizar multimodalidad
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        all_trajs_flat = np.vstack([t.reshape(t.shape[0], -1) for t in all_diff_trajs])
        pca = PCA(n_components=2)
        pts_2d = pca.fit_transform(all_trajs_flat)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot izq: PCA agregado coloreado por escena
        for i, scene_trajs in enumerate(all_diff_trajs):
            n = len(scene_trajs)
            start = i * n
            end = start + n
            axes[0].scatter(pts_2d[start:end, 0], pts_2d[start:end, 1],
                            label=f'Escena {i+1} (obj_id={preds[i]["obj_id"]})', alpha=0.6, s=40)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0].set_title(f'Diversidad multimodal Diffusion Policy\n({N_SAMPLES} muestras × {len(preds)} escenas)')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Plot der: jerk box-plot Diffusion vs Heuristico
        axes[1].boxplot([all_jerk_diff, all_jerk_heur],
                         tick_labels=['Diffusion Policy', 'Heurístico determinista'],
                         widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor='#0098CD', alpha=0.7))
        axes[1].set_ylabel('Jerk RMS (suavidad)')
        axes[1].set_title(f'Suavidad de trayectoria — Diffusion (n={len(all_jerk_diff)}) vs Heurístico (n={len(all_jerk_heur)})')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(OUT / 'fig_diversity_pca.png', dpi=180, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] {OUT}/fig_diversity_pca.png")
    except Exception as e:
        print(f"[warn] plot fail: {e}")

    out_json = OUT / 'exp8_results.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] {out_json}")

    # Imprimir resumen
    print(f"\n=== RESUMEN ===")
    print(f"  Diffusion jerk RMS mean: {summary['diffusion_jerk_rms']['mean']:.4f}")
    print(f"  Heuristico jerk RMS mean: {summary['heuristic_jerk_rms']['mean']:.4f}")
    print(f"  Distancia endpoint media: {np.mean([s['endpoint_dist_mean_cm'] for s in per_scene_results]):.2f} cm")
    print(f"  Mejor K modos (silhouette): {summary.get('best_n_modes_silhouette')}")


if __name__ == '__main__':
    main()
