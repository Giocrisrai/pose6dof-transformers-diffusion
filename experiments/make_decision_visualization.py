#!/usr/bin/env python3
"""Genera una visualizacion clara del 'proceso de decision' del pipeline.

Salida: para cada objeto seleccionado, una figura compuesta que muestra
visualmente lo que el sistema *decide* en cada etapa, en lugar de solo
una vista cenital generica.

Panel A: nombre del objeto + tipo de dataset (YCB-V / T-LESS)
Panel B: pose 6-DoF predicha por FoundationPose (frame visualizado como ejes XYZ)
Panel C: trayectoria Diffusion (10 muestras, target marcado)
Panel D: linea de tiempo con las 3 fases del pipeline y latencias reales

Uso:
    .venv/bin/python experiments/make_decision_visualization.py --n 6 --out decision_grid.png
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Mapas de nombre de objeto (BOP nomenclature)
YCBV_NAMES = {
    1: "master_chef_can", 2: "cracker_box", 3: "sugar_box", 4: "tomato_soup_can",
    5: "mustard_bottle", 6: "tuna_fish_can", 7: "pudding_box", 8: "gelatin_box",
    9: "potted_meat_can", 10: "banana", 11: "pitcher_base", 12: "bleach_cleanser",
    13: "bowl", 14: "mug", 15: "power_drill", 16: "wood_block", 17: "scissors",
    18: "large_marker", 19: "large_clamp", 20: "extra_large_clamp", 21: "foam_brick",
}
TLESS_NAMES = {i: f"obj_{i:02d}_industrial_part" for i in range(1, 31)}


def load_diffusion():
    import torch
    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=256).to(device)
    ckpt = torch.load(REPO / "data/models/diffusion_policy_ultra.pth",
                      map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.eval()
    return model, SimpleDDPMScheduler(num_timesteps=100), device


def ddim_sample(model, scheduler, cond, device, n_steps=25):
    import torch
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


def draw_pose_axes(ax, R, t, axis_length=0.04):
    """Dibuja los 3 ejes de un frame 6-DoF en 3D."""
    origin = t.flatten()
    colors = ['#E63946', '#2A9D8F', '#1D3557']  # X rojo, Y verde, Z azul
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        end = origin + R[:, i] * axis_length
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
                color=colors[i], linewidth=2.5, solid_capstyle='round')
        ax.text(end[0], end[1], end[2], labels[i], color=colors[i],
                fontsize=10, fontweight='bold')


def build_decision_card(pred, dataset, model, scheduler, device):
    """Genera figura 2x2 explicando la decision del pipeline para 1 objeto."""
    import torch
    obj_id = pred["obj_id"]
    name_map = YCBV_NAMES if dataset == "ycbv" else TLESS_NAMES
    obj_name = name_map.get(obj_id, f"obj_{obj_id}")
    R = np.array(pred["R_pred"])
    t_pose = np.array(pred["t_pred"]).flatten()
    fp_ms = pred["time_s"] * 1000

    # Diffusion sampling
    cond_vec = np.zeros(64, dtype=np.float32)
    cond_vec[:9] = R.flatten()
    cond_vec[9:12] = t_pose if np.linalg.norm(t_pose) > 5 else t_pose * 1000
    cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.time()
    trajs = np.array([ddim_sample(model, scheduler, cond, device, 25) for _ in range(10)])
    diff_ms = (time.time() - t0) * 1000 / 10  # ms/traj

    NOMINAL_SIM_MS = 906
    total_ms = fp_ms + diff_ms + NOMINAL_SIM_MS

    fig = plt.figure(figsize=(14, 8), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # PANEL A — Identificacion del objeto
    axA = fig.add_subplot(gs[0, 0])
    axA.set_axis_off()
    color = '#FF6B35' if dataset == 'ycbv' else '#0098CD'
    rect = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                    boxstyle="round,pad=0.02",
                                    linewidth=3, edgecolor=color, facecolor='#FAFAFA',
                                    transform=axA.transAxes)
    axA.add_patch(rect)
    axA.text(0.5, 0.85, "1. DETECCION", fontsize=13, fontweight='bold',
             color=color, ha='center', transform=axA.transAxes)
    axA.text(0.5, 0.68, f"Dataset: {dataset.upper()}", fontsize=14,
             ha='center', transform=axA.transAxes)
    axA.text(0.5, 0.55, f"Objeto detectado: #{obj_id}", fontsize=18,
             fontweight='bold', ha='center', transform=axA.transAxes)
    axA.text(0.5, 0.43, f'"{obj_name}"', fontsize=14, style='italic',
             color='#444', ha='center', transform=axA.transAxes)
    axA.text(0.5, 0.28, f"Escena: {pred.get('scene_id', '?')} | Imagen: {pred.get('img_id', '?')}",
             fontsize=10, color='#666', ha='center', transform=axA.transAxes)
    axA.text(0.5, 0.18, f"Tiempo de deteccion: {fp_ms:.0f} ms",
             fontsize=11, color='#333', ha='center', transform=axA.transAxes)
    axA.text(0.5, 0.08, "(FoundationPose Transformer)",
             fontsize=9, color='#888', ha='center', transform=axA.transAxes)

    # PANEL B — Pose 6-DoF
    axB = fig.add_subplot(gs[0, 1], projection='3d')
    axB.set_title("2. POSE 6-DoF estimada\n(posicion + orientacion del objeto)",
                  fontsize=11, fontweight='bold', color='#FF6B35', pad=10)
    # Normalizar t a metros si viene en mm
    t_m = t_pose / 1000 if np.linalg.norm(t_pose) > 5 else t_pose
    draw_pose_axes(axB, R, t_m, axis_length=0.08)
    # Marcar el origen del objeto
    axB.scatter([t_m[0]], [t_m[1]], [t_m[2]], s=120, c='black', marker='o',
                edgecolor='white', linewidth=2, zorder=20)
    axB.text(t_m[0], t_m[1], t_m[2] + 0.03,
             f"({t_m[0]:.2f}, {t_m[1]:.2f}, {t_m[2]:.2f}) m",
             fontsize=9, ha='center')
    # Range visual
    margin = 0.15
    axB.set_xlim(t_m[0] - margin, t_m[0] + margin)
    axB.set_ylim(t_m[1] - margin, t_m[1] + margin)
    axB.set_zlim(t_m[2] - margin, t_m[2] + margin)
    axB.set_xlabel('X (m)'); axB.set_ylabel('Y (m)'); axB.set_zlabel('Z (m)')
    axB.grid(True, alpha=0.3)

    # PANEL C — Trayectorias Diffusion
    axC = fig.add_subplot(gs[1, 0], projection='3d')
    axC.set_title(f"3. PLANIFICACION (Diffusion Policy)\n10 trayectorias multimodales · "
                  f"{diff_ms:.0f} ms/trayectoria",
                  fontsize=11, fontweight='bold', color='#35876B', pad=10)
    for traj in trajs:
        axC.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='#35876B', alpha=0.5, linewidth=1.3)
    # Punto de inicio (todos coinciden)
    start = trajs[:, 0, :3].mean(axis=0)
    axC.scatter([start[0]], [start[1]], [start[2]], s=130, c='black', marker='^',
                edgecolor='white', linewidth=1.5, zorder=10, label='Posicion inicial robot')
    # Endpoints (donde se acerca al objeto)
    endpoints = trajs[:, -1, :3]
    axC.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2],
                s=80, c='#E63946', marker='*', edgecolor='white', linewidth=0.8,
                zorder=15, label='Punto de agarre')
    endpoint_std_cm = float(np.std(endpoints, axis=0).mean() * 100)
    axC.set_xlabel('X (m)'); axC.set_ylabel('Y (m)'); axC.set_zlabel('Z (m)')
    axC.legend(loc='upper left', fontsize=8)
    axC.text2D(0.02, 0.02, f"Dispersion final entre opciones: {endpoint_std_cm:.1f} cm",
               transform=axC.transAxes, fontsize=9, color='#444', style='italic')
    axC.grid(True, alpha=0.3)

    # PANEL D — Timeline de fases
    axD = fig.add_subplot(gs[1, 1])
    axD.set_title("4. CICLO COMPLETO (timeline)",
                  fontsize=11, fontweight='bold', color='#1D3557', pad=10)
    phases = [
        ("FoundationPose\n(Transformer)", fp_ms, '#FF6B35'),
        ("Diffusion Policy\n(planificacion)", diff_ms, '#35876B'),
        ("Simulacion\n(CoppeliaSim)", NOMINAL_SIM_MS, '#1D3557'),
    ]
    cumulative = 0
    for label, dur, c in phases:
        axD.barh(0, dur, left=cumulative, color=c, edgecolor='white', linewidth=2, height=0.5)
        axD.text(cumulative + dur/2, 0, f"{label}\n{dur:.0f} ms",
                 ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        cumulative += dur
    axD.axvline(10000, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axD.text(10000, 0.6, '  Limite H3 (10 s)', fontsize=9, color='red')
    axD.text(cumulative + 200, 0, f"  TOTAL: {total_ms:.0f} ms",
             fontsize=11, fontweight='bold', va='center',
             color='green' if total_ms < 10000 else 'red')
    axD.set_xlim(0, max(11000, cumulative * 1.15))
    axD.set_ylim(-0.6, 0.9)
    axD.set_xlabel('Tiempo (ms)')
    axD.set_yticks([])
    axD.grid(True, axis='x', alpha=0.3)

    fig.suptitle(f"Decision del pipeline para 1 instancia  ·  "
                 f"{dataset.upper()} obj_id={obj_id}  ·  {obj_name}",
                 fontsize=14, fontweight='bold', y=0.99)
    fig.text(0.5, 0.005,
             "1) Detecta y mide la pose 6-DoF  →  2) Decide la pose exacta  →  "
             "3) Genera multiples caminos posibles  →  4) Ejecuta en el robot",
             ha='center', fontsize=10, color='#555', style='italic')
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ycbv", "tless"])
    ap.add_argument("--n-per-dataset", type=int, default=2,
                    help="Cuantos objetos distintos visualizar por dataset")
    ap.add_argument("--out-dir", default="experiments/results/pipeline_e2e/decision_cards")
    args = ap.parse_args()

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Cargando modelo Diffusion ultra...")
    model, scheduler, device = load_diffusion()
    print(f"  device={device}")

    for dataset in args.datasets:
        ckpt = json.loads((REPO / f"experiments/checkpoints/fp_{dataset}_checkpoint.json").read_text())
        # Tomar un objeto representativo distinto cada vez
        seen_objs = set()
        picked = []
        for r in ckpt["results"]:
            if r["obj_id"] not in seen_objs:
                picked.append(r)
                seen_objs.add(r["obj_id"])
            if len(picked) >= args.n_per_dataset:
                break

        for pred in picked:
            print(f"\n[{dataset}] obj_id={pred['obj_id']} ...")
            fig = build_decision_card(pred, dataset, model, scheduler, device)
            out_path = out / f"decision_{dataset}_obj{pred['obj_id']:02d}.png"
            fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  -> {out_path.relative_to(REPO)}")

    print("\nListo. Imagenes en", out.relative_to(REPO))


if __name__ == "__main__":
    main()
