#!/usr/bin/env python3
"""Experimento 21: simulaciones visuales del modelo multi-objeto (exp 20).

Renders 3D de escenas con N=2..5 objetos, mostrando:
- Cada objeto con su forma real
- Etiquetas color+shape
- Trayectoria al objeto seleccionado
- Gate probabilities (bar chart) sobre todos los candidatos

Salida:
    experiments/results/exp21_visual_multi/scene_*.png  (10 escenas)
    experiments/results/exp21_visual_multi/grid_overview.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

HORIZON = 16
MAX_OBJ = 5
ATTR_DIM = 7
COND_DIM = 64

COLORS_RGB_HEX = {"red": "#E63946", "blue": "#0098CD", "green": "#2A9D8F"}
COLORS_RGB_VEC = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}
SHAPES = ["cube", "sphere", "cylinder", "box"]
SHAPE_ENC = {s: [1.0 if s == x else 0.0 for x in SHAPES] for s in SHAPES}

OUTPUT = REPO / "experiments/results/exp21_visual_multi"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_model():
    from transformers import CLIPTextModel, CLIPTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_mod = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    ckpt = torch.load(REPO / "data/models/diffusion_policy_clip_multi.pth",
                       map_location=device, weights_only=True)

    class Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(512, 64), nn.Mish(), nn.Linear(64, 32))
        def forward(self, x): return self.net(x)

    class MultiObjectGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(7+512, 128), nn.Mish(),
                                          nn.Linear(128, 128), nn.Mish(),
                                          nn.Linear(128, 1))
        def forward(self, ce, attrs, mask):
            B, N, A = attrs.shape
            x = torch.cat([attrs, ce.unsqueeze(1).expand(-1, N, -1)], -1)
            logits = self.score(x).squeeze(-1).masked_fill(mask == 0, -1e9)
            return F.softmax(logits, dim=-1)

    model = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=256).to(device).eval()
    proj = Proj().to(device).eval()
    gate = MultiObjectGate().to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    proj.load_state_dict(ckpt["projector_state_dict"])
    gate.load_state_dict(ckpt["gate_state_dict"])
    return tok, clip_mod, model, proj, gate, device


def draw_cube(ax, center, color, size=0.05):
    r = size / 2; x, y, z = center
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = [[x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r],
         [x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r]]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        ax.plot([v[e[0]][0],v[e[1]][0]],[v[e[0]][1],v[e[1]][1]],[v[e[0]][2],v[e[1]][2]],color=color,linewidth=2)
    faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
             [v[2],v[3],v[7],v[6]],[v[0],v[3],v[7],v[4]],[v[1],v[2],v[6],v[5]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor=color))


def draw_sphere(ax, center, color, radius=0.035):
    u = np.linspace(0, 2*np.pi, 18); v = np.linspace(0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.6, edgecolor=color, linewidth=0.3)


def draw_cylinder(ax, center, color, radius=0.028, height=0.07):
    th = np.linspace(0, 2*np.pi, 22); z = np.linspace(center[2]-height/2, center[2]+height/2, 8)
    T, Z = np.meshgrid(th, z)
    ax.plot_surface(center[0] + radius*np.cos(T), center[1] + radius*np.sin(T), Z,
                      color=color, alpha=0.55, edgecolor=color, linewidth=0.2)


def draw_box(ax, center, color, size=(0.07, 0.04, 0.035)):
    sx, sy, sz = [s/2 for s in size]; x, y, z = center
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = [[x-sx,y-sy,z-sz],[x+sx,y-sy,z-sz],[x+sx,y+sy,z-sz],[x-sx,y+sy,z-sz],
         [x-sx,y-sy,z+sz],[x+sx,y-sy,z+sz],[x+sx,y+sy,z+sz],[x-sx,y+sy,z+sz]]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        ax.plot([v[e[0]][0],v[e[1]][0]],[v[e[0]][1],v[e[1]][1]],[v[e[0]][2],v[e[1]][2]],color=color,linewidth=2)
    faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
             [v[2],v[3],v[7],v[6]],[v[0],v[3],v[7],v[4]],[v[1],v[2],v[6],v[5]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor=color))


def draw_object(ax, center, color, shape):
    c = COLORS_RGB_HEX[color]
    {"cube": draw_cube, "sphere": draw_sphere, "cylinder": draw_cylinder, "box": draw_box}[shape](ax, center, c)


def render_scene(scene, traj, gate_probs, text, target_idx, chosen_idx, save_path):
    """Layout vertical: 3D arriba (fila completa) + barras + explicacion abajo."""
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    })
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0],
                              width_ratios=[1.3, 1.0],
                              hspace=0.25, wspace=0.15)

    target = scene["objects"][target_idx]
    chosen = scene["objects"][chosen_idx]
    ok = chosen_idx == target_idx
    fig.suptitle(f'Instrucción: "{text}"   (N={scene["n_obj"]} objetos)',
                  fontsize=22, fontweight="bold", y=0.985)

    # === ESCENA 3D (fila superior, completa) ===
    ax = fig.add_subplot(gs[0, :], projection="3d")
    xs = np.linspace(-0.5, 0.5, 2); ys = np.linspace(-0.5, 0.5, 2)
    xx, yy = np.meshgrid(xs, ys)
    ax.plot_surface(xx, yy, np.full_like(xx, 0.7), alpha=0.08, color="gray")
    for i, obj in enumerate(scene["objects"]):
        draw_object(ax, obj["pos"], obj["color"], obj["shape"])
        edge_c = "#16a34a" if i == target_idx else COLORS_RGB_HEX[obj["color"]]
        lw = 3 if i == target_idx else 1.5
        ax.text(obj["pos"][0], obj["pos"][1], obj["pos"][2] + 0.10,
                  f"#{i+1}: {obj['color']} {obj['shape']}",
                  ha="center", fontsize=14,
                  fontweight="bold" if i == target_idx else "normal",
                  bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=COLORS_RGB_HEX[obj["color"]] + "30",
                              edgecolor=edge_c, linewidth=lw))
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#FF6B35", linewidth=4,
              alpha=0.9, label="Trayectoria planificada", zorder=10)
    ax.scatter([traj[0,0]], [traj[0,1]], [traj[0,2]], s=260, c="black",
                  marker="^", edgecolor="white", linewidth=2,
                  label="Posición inicial del robot", zorder=12)
    ax.scatter([traj[-1,0]], [traj[-1,1]], [traj[-1,2]], s=300, c="#FF6B35",
                  marker="*", edgecolor="white", linewidth=2,
                  label="Punto de agarre final", zorder=12)
    ax.set_xlabel("X (m)", fontsize=15, labelpad=10)
    ax.set_ylabel("Y (m)", fontsize=15, labelpad=10)
    ax.set_zlabel("Z (m)", fontsize=15, labelpad=10)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0.65, 1.18)
    ax.legend(loc="upper left", fontsize=14, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=22, azim=-45)

    # === BARRAS DE DECISION (fila inferior izquierda) ===
    ax2 = fig.add_subplot(gs[1, 0])
    labels = [f"#{i+1}\n{o['color']}\n{o['shape']}" for i, o in enumerate(scene["objects"])]
    colors_bar = [COLORS_RGB_HEX[o["color"]] for o in scene["objects"]]
    probs_n = gate_probs[:scene["n_obj"]]
    bars = ax2.bar(labels, probs_n, color=colors_bar,
                       edgecolor="black", linewidth=2, width=0.6)
    bars[target_idx].set_edgecolor("#16a34a")
    bars[target_idx].set_linewidth(5)
    ax2.set_ylim(0, 1.18)
    ax2.set_ylabel("Confianza del gate (softmax)", fontsize=15, labelpad=10)
    ax2.set_title("Decisión del modelo (target con borde verde)",
                    fontsize=17, fontweight="bold", pad=12)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=13)
    for i, v in enumerate(probs_n):
        ax2.text(i, v + 0.04, f"{v:.0%}",
                   ha="center", fontweight="bold", fontsize=16,
                   color="#16a34a" if i == target_idx else "#444")
    ax2.grid(True, alpha=0.3, axis="y")

    # === EXPLICACION (fila inferior derecha) ===
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_axis_off()
    info = (
        f"INSTRUCCIÓN\n"
        f'   "{text}"\n\n'
        f"ESCENA\n"
        f"   {scene['n_obj']} objetos colocados sobre la mesa\n\n"
        f"TARGET ESPERADO\n"
        f"   Objeto #{target_idx+1}  →  {target['color']} {target['shape']}\n\n"
        f"DECISIÓN DEL MODELO\n"
        f"   Objeto #{chosen_idx+1}  →  {chosen['color']} {chosen['shape']}\n"
        f"   Confianza: {probs_n[chosen_idx]:.1%}\n\n"
        f"DISTANCIA al target  →  {np.linalg.norm(traj[-1, :3] - target['pos'])*100:.1f} cm\n\n"
        f"RESULTADO  →  {'ACIERTA' if ok else 'ERROR'}"
    )
    ax3.text(0.02, 0.97, info, transform=ax3.transAxes, fontsize=15,
              verticalalignment="top", family="DejaVu Sans", linespacing=1.5,
              bbox=dict(boxstyle="round,pad=1.0",
                          facecolor="#f0fdf4" if ok else "#fee2e2",
                          edgecolor="#16a34a" if ok else "#dc2626", linewidth=3))

    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                 facecolor="white", pad_inches=0.3)
    plt.close()


def run_inference(text, scene, tok, clip_mod, model, proj, gate, device):
    attrs_pad = np.zeros((MAX_OBJ, ATTR_DIM), dtype=np.float32)
    pos_pad = np.zeros((MAX_OBJ, 3), dtype=np.float32)
    mask = np.zeros(MAX_OBJ, dtype=np.float32)
    for i, obj in enumerate(scene["objects"]):
        attrs_pad[i, :3] = COLORS_RGB_VEC[obj["color"]]
        attrs_pad[i, 3:7] = SHAPE_ENC[obj["shape"]]
        pos_pad[i] = obj["pos"]
        mask[i] = 1.0

    with torch.no_grad():
        ins = tok([text], padding=True, return_tensors="pt", truncation=True)
        ins = {k: v.to(device) for k, v in ins.items()}
        ce = clip_mod(**ins).pooler_output
        attrs_t = torch.tensor(attrs_pad, device=device, dtype=torch.float32).unsqueeze(0)
        pos_t = torch.tensor(pos_pad, device=device, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)
        gates = gate(ce, attrs_t, mask_t)  # (1, MAX_OBJ)
        selected = (gates.unsqueeze(-1) * pos_t).sum(dim=1)  # (1, 3)
        clip_proj = proj(ce)

        cb = torch.zeros(1, COND_DIM, device=device, dtype=torch.float32)
        cb[0, :3] = selected[0]
        cb[0, 3:18] = pos_t[0].flatten()[:15]
        cb[0, 18:50] = clip_proj[0]
        cb[0, 50:50+ATTR_DIM] = attrs_t[0].max(dim=0).values

        scheduler = SimpleDDPMScheduler(num_timesteps=100)
        x = torch.randn(1, HORIZON, 7, device=device, dtype=torch.float32)
        si = np.linspace(0, 99, 25).astype(int)[::-1]
        ab = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
        for i, step in enumerate(si):
            t = torch.full((1,), int(step), dtype=torch.long, device=device)
            e = model(x, t, cb)
            ab_t = ab[step]
            x0 = (x - torch.sqrt(1 - ab_t) * e) / torch.sqrt(ab_t)
            if i < len(si) - 1:
                ab_n = ab[si[i+1]]
                x = torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * e
            else:
                x = x0
    return x.cpu().numpy()[0], gates.cpu().numpy()[0]


def make_scene(text, objects_def, target_idx):
    """objects_def: list of (color, shape, [x, y, z])."""
    return {
        "text": text,
        "n_obj": len(objects_def),
        "target_idx": target_idx,
        "objects": [{"color": c, "shape": s, "pos": np.array(p)} for c, s, p in objects_def],
    }


# Escenas curadas: cobertura N=2..5
DEMO_SCENES = [
    # N=2
    make_scene("pick the red sphere",
                [("blue", "sphere", [-0.25, 0.10, 0.80]),
                 ("red", "sphere", [0.25, -0.10, 0.85])], 1),
    # N=3
    make_scene("grab the blue cylinder",
                [("red", "cube", [-0.30, 0.15, 0.80]),
                 ("blue", "cylinder", [0.00, -0.10, 0.85]),
                 ("green", "sphere", [0.30, 0.20, 0.78])], 1),
    make_scene("select the green box",
                [("red", "box", [-0.30, -0.15, 0.85]),
                 ("blue", "box", [0.00, 0.20, 0.80]),
                 ("green", "box", [0.30, -0.05, 0.82])], 2),
    # N=4
    make_scene("take the red cube",
                [("blue", "cube", [-0.35, 0.10, 0.82]),
                 ("red", "cube", [-0.10, -0.20, 0.85]),
                 ("green", "cylinder", [0.15, 0.15, 0.80]),
                 ("blue", "sphere", [0.35, -0.05, 0.78])], 1),
    make_scene("fetch the green sphere",
                [("red", "cube", [-0.30, -0.10, 0.82]),
                 ("blue", "cylinder", [-0.05, 0.20, 0.85]),
                 ("green", "sphere", [0.20, -0.15, 0.80]),
                 ("red", "box", [0.35, 0.10, 0.78])], 2),
    # N=5
    make_scene("pick the blue sphere",
                [("red", "cube", [-0.40, 0.00, 0.82]),
                 ("blue", "cylinder", [-0.20, 0.20, 0.80]),
                 ("green", "box", [0.00, -0.15, 0.85]),
                 ("blue", "sphere", [0.20, 0.10, 0.78]),
                 ("red", "cylinder", [0.40, -0.10, 0.82])], 3),
    make_scene("grab the red cylinder",
                [("green", "cube", [-0.35, 0.10, 0.85]),
                 ("blue", "sphere", [-0.10, -0.20, 0.80]),
                 ("red", "cylinder", [0.10, 0.20, 0.82]),
                 ("green", "box", [0.30, 0.00, 0.78]),
                 ("blue", "cube", [0.40, -0.15, 0.85])], 2),
    make_scene("select the green cube",
                [("red", "sphere", [-0.35, 0.10, 0.82]),
                 ("blue", "box", [-0.10, -0.15, 0.85]),
                 ("green", "cube", [0.10, 0.10, 0.80]),
                 ("red", "cylinder", [0.30, 0.20, 0.85]),
                 ("blue", "sphere", [0.35, -0.10, 0.78])], 2),
    # Casos limite
    make_scene("the round green one",
                [("red", "sphere", [-0.30, 0.15, 0.82]),
                 ("green", "cube", [-0.05, -0.10, 0.80]),
                 ("green", "sphere", [0.20, 0.20, 0.85]),
                 ("blue", "cylinder", [0.35, -0.10, 0.78])], 2),
    make_scene("pick anything blue",
                [("red", "sphere", [-0.35, 0.10, 0.82]),
                 ("green", "cube", [-0.10, -0.15, 0.85]),
                 ("blue", "cylinder", [0.15, 0.10, 0.80]),
                 ("red", "box", [0.35, 0.20, 0.78])], 2),
]


def main():
    print("[exp21] Generando simulaciones visuales multi-objeto")
    tok, clip_mod, model, proj, gate, device = load_model()
    print(f"  device={device}, {len(DEMO_SCENES)} escenas")

    results = {"scenes": []}
    correct = 0
    for i, scene in enumerate(DEMO_SCENES):
        traj, gates = run_inference(scene["text"], scene, tok, clip_mod, model, proj, gate, device)
        chosen_idx = int(np.argmax(gates[:scene["n_obj"]]))
        ok = chosen_idx == scene["target_idx"]
        if ok:
            correct += 1
        save_path = OUTPUT / f"scene_{i+1:02d}.png"
        render_scene(scene, traj, gates, scene["text"], scene["target_idx"], chosen_idx, save_path)
        flag = "✓" if ok else "✗"
        endpoint = traj[-1, :3]
        d = np.linalg.norm(endpoint - scene["objects"][scene["target_idx"]]["pos"])
        print(f"  {flag} scene {i+1:02d} (N={scene['n_obj']}): '{scene['text']}' | "
              f"target={scene['target_idx']+1} chose={chosen_idx+1} | "
              f"conf={gates[chosen_idx]:.1%} | d={d*100:.1f}cm")
        results["scenes"].append({
            "id": i+1, "text": scene["text"], "n_obj": scene["n_obj"],
            "target_idx": scene["target_idx"], "chosen_idx": chosen_idx,
            "gate_probs": [float(g) for g in gates[:scene["n_obj"]]],
            "correct": ok, "distance_cm": float(d * 100),
        })

    print(f"\n  Total: {correct}/{len(DEMO_SCENES)} = {correct/len(DEMO_SCENES):.1%}")
    results["summary"] = {"n_scenes": len(DEMO_SCENES), "n_correct": correct,
                              "accuracy": correct / len(DEMO_SCENES)}

    # Grid overview con escenas a tamano grande para legibilidad
    print("  Generando grid overview...")
    n_cols = 2
    rows = (len(DEMO_SCENES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(rows, n_cols, figsize=(30, 7 * rows))
    axes_flat = axes.flat if rows > 1 else [axes]
    for ax, scene, i in zip(axes_flat, DEMO_SCENES, range(len(DEMO_SCENES))):
        from matplotlib.image import imread
        img = imread(OUTPUT / f"scene_{i+1:02d}.png")
        ax.imshow(img)
        ax.set_title(f'Escena {i+1} (N={scene["n_obj"]} objetos): "{scene["text"]}"',
                       fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
    plt.suptitle(f"VLA-lite multi-objeto — {len(DEMO_SCENES)} escenas demostrativas "
                  f"({correct}/{len(DEMO_SCENES)} aciertos)",
                  fontsize=18, fontweight="bold", y=0.998)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    overview_path = OUTPUT / "grid_overview.png"
    plt.savefig(overview_path, dpi=70, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {overview_path}")

    with open(OUTPUT / "exp21_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT / 'exp21_results.json'}")


if __name__ == "__main__":
    main()
