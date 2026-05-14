#!/usr/bin/env python3
"""Experimento 19: simulaciones visuales del VLA-lite multi-atributo.

Genera imagenes PNG mostrando escenas 3D con objetos REALES (cubos, esferas,
cilindros, cajas) coloreados, la trayectoria que el modelo planifica y la
decision del gate. Para cada escena se genera una tarjeta visual que explica
QUE entendio el sistema y POR QUE escogio ese objeto.

Salida:
    experiments/results/exp19_visual_sims/scene_*.png  (12 escenas)
    experiments/results/exp19_visual_sims/grid_overview.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

HORIZON = 16
COLORS_RGB_HEX = {"red": "#E63946", "blue": "#0098CD", "green": "#2A9D8F"}
COLORS_RGB_VEC = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}
SHAPES = ["cube", "sphere", "cylinder", "box"]
SHAPE_ENC = {s: [1.0 if s == x else 0.0 for x in SHAPES] for s in SHAPES}

OUTPUT = REPO / "experiments/results/exp19_visual_sims"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_model():
    from transformers import CLIPTokenizer, CLIPTextModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_mod = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    ckpt = torch.load(REPO / "data/models/diffusion_policy_clip_shapes.pth",
                       map_location=device, weights_only=True)

    class CLIPProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(512, 64), nn.Mish(), nn.Linear(64, 32))
        def forward(self, x): return self.net(x)

    class MultiAttributeGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(7+512, 128), nn.Mish(),
                                          nn.Linear(128, 128), nn.Mish(),
                                          nn.Linear(128, 1))
        def forward(self, c, aa, ab):
            sa = self.score(torch.cat([aa, c], -1)).squeeze(-1)
            sb = self.score(torch.cat([ab, c], -1)).squeeze(-1)
            return F.softmax(torch.stack([sa, sb], -1), dim=-1).unbind(-1)

    model = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=256).to(device).eval()
    proj = CLIPProjector().to(device).eval()
    gate = MultiAttributeGate().to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    proj.load_state_dict(ckpt["projector_state_dict"])
    gate.load_state_dict(ckpt["gate_state_dict"])
    return tok, clip_mod, model, proj, gate, device


def draw_cube(ax, center, color, size=0.06):
    r = size / 2
    x, y, z = center
    vertices = [
        [x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r],
        [x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        v1, v2 = vertices[e[0]], vertices[e[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=color, linewidth=2.5)
    # Caras semi-transparentes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
    ]
    pc = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor=color)
    ax.add_collection3d(pc)


def draw_sphere(ax, center, color, radius=0.04):
    u, v = np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 15)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.6, edgecolor=color, linewidth=0.5)


def draw_cylinder(ax, center, color, radius=0.035, height=0.08):
    z_min = center[2] - height/2
    z_max = center[2] + height/2
    theta = np.linspace(0, 2*np.pi, 25)
    z = np.linspace(z_min, z_max, 10)
    th, zz = np.meshgrid(theta, z)
    x = center[0] + radius * np.cos(th)
    y = center[1] + radius * np.sin(th)
    ax.plot_surface(x, y, zz, color=color, alpha=0.55, edgecolor=color, linewidth=0.3)


def draw_box(ax, center, color, size=(0.08, 0.05, 0.04)):
    """Caja rectangular alargada (no cubica)."""
    sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    x, y, z = center
    vertices = [
        [x-sx, y-sy, z-sz], [x+sx, y-sy, z-sz], [x+sx, y+sy, z-sz], [x-sx, y+sy, z-sz],
        [x-sx, y-sy, z+sz], [x+sx, y-sy, z+sz], [x+sx, y+sy, z+sz], [x-sx, y+sy, z+sz],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        v1, v2 = vertices[e[0]], vertices[e[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=color, linewidth=2.5)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
    ]
    pc = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor=color)
    ax.add_collection3d(pc)


def draw_object(ax, center, color, shape):
    color_hex = COLORS_RGB_HEX[color]
    if shape == "cube":
        draw_cube(ax, center, color_hex)
    elif shape == "sphere":
        draw_sphere(ax, center, color_hex)
    elif shape == "cylinder":
        draw_cylinder(ax, center, color_hex)
    elif shape == "box":
        draw_box(ax, center, color_hex)


def render_scene(scene, traj, gate_probs, text, save_path):
    """Genera figura con la escena 3D + decision del modelo."""
    # Estilo global mas legible
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1.1, 1.4], wspace=0.25)

    # PANEL A: Escena 3D
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    # Mesa
    xs = np.linspace(-0.5, 0.5, 2)
    ys = np.linspace(-0.5, 0.5, 2)
    xx, yy = np.meshgrid(xs, ys)
    ax.plot_surface(xx, yy, np.full_like(xx, 0.7), alpha=0.1, color="gray")

    # Objetos A y B
    draw_object(ax, scene["p_a"], scene["c_a"], scene["s_a"])
    draw_object(ax, scene["p_b"], scene["c_b"], scene["s_b"])

    # Etiquetas
    ax.text(scene["p_a"][0], scene["p_a"][1], scene["p_a"][2] + 0.08,
              f"A:\n{scene['c_a']} {scene['s_a']}", ha="center", fontsize=9, fontweight="bold")
    ax.text(scene["p_b"][0], scene["p_b"][1], scene["p_b"][2] + 0.08,
              f"B:\n{scene['c_b']} {scene['s_b']}", ha="center", fontsize=9, fontweight="bold")

    # Trayectoria
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#FF6B35", linewidth=3, alpha=0.85,
              label="Trayectoria planificada", zorder=10)
    # Inicio
    ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], s=180, c="black", marker="^",
                  edgecolor="white", linewidth=1.5, label="Posicion inicial", zorder=12)
    # Endpoint
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], s=200, c="#FF6B35", marker="*",
                  edgecolor="white", linewidth=1.5, label="Punto de agarre", zorder=12)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0.65, 1.15)
    ax.set_title(f'Instruccion: "{text}"', fontsize=15, pad=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # PANEL B: Gate probabilities
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [f"A\n{scene['c_a']}\n{scene['s_a']}",
                f"B\n{scene['c_b']}\n{scene['s_b']}"]
    colors_bar = [COLORS_RGB_HEX[scene["c_a"]], COLORS_RGB_HEX[scene["c_b"]]]
    ax2.bar(labels, gate_probs, color=colors_bar, edgecolor="black", linewidth=1.8)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Confianza del gate", fontsize=13)
    ax2.set_title("Decision del modelo", fontsize=14, fontweight="bold", pad=10)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    for i, v in enumerate(gate_probs):
        ax2.text(i, v + 0.04, f"{v:.0%}", ha="center", fontweight="bold", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    # PANEL C: Explicacion
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_axis_off()
    chosen = "A" if gate_probs[0] > gate_probs[1] else "B"
    chosen_color = scene["c_a"] if chosen == "A" else scene["c_b"]
    chosen_shape = scene["s_a"] if chosen == "A" else scene["s_b"]
    target_match = chosen == ("A" if scene["target_idx"] == 0 else "B")
    correct = "ACIERTA" if target_match else "ERROR"

    text_box = (
        f"ATRIBUTOS DE LA ESCENA\n"
        f"\n"
        f"  Objeto A : {scene['c_a']} {scene['s_a']}\n"
        f"  Objeto B : {scene['c_b']} {scene['s_b']}\n"
        f"\n"
        f"INSTRUCCION RECIBIDA\n"
        f"\n"
        f'  "{text}"\n'
        f"\n"
        f"DECISION DEL MODELO\n"
        f"\n"
        f"  Eligio   : Objeto {chosen}\n"
        f"             ({chosen_color} {chosen_shape})\n"
        f"  Confianza: {max(gate_probs):.1%}\n"
        f"\n"
        f"TARGET ESPERADO\n"
        f"\n"
        f"  Objeto {'A' if scene['target_idx'] == 0 else 'B'}\n"
        f"\n"
        f"RESULTADO\n"
        f"\n"
        f"  {correct}"
    )
    ax3.text(0.0, 0.97, text_box, transform=ax3.transAxes, fontsize=12,
              verticalalignment="top", family="monospace",
              linespacing=1.4,
              bbox=dict(boxstyle="round,pad=0.8",
                          facecolor="#f0fdf4" if target_match else "#fee2e2",
                          edgecolor="#16a34a" if target_match else "#dc2626",
                          linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close()


def run_inference(text, scene, tok, clip_mod, model, proj, gate, device):
    """Inferencia completa: text -> gate -> trajectory."""
    with torch.no_grad():
        ins = tok([text], padding=True, return_tensors="pt", truncation=True)
        ins = {k: v.to(device) for k, v in ins.items()}
        ce = clip_mod(**ins).pooler_output

        attr_a = torch.tensor([[*COLORS_RGB_VEC[scene["c_a"]], *SHAPE_ENC[scene["s_a"]]]],
                                device=device, dtype=torch.float32)
        attr_b = torch.tensor([[*COLORS_RGB_VEC[scene["c_b"]], *SHAPE_ENC[scene["s_b"]]]],
                                device=device, dtype=torch.float32)
        g_a, g_b = gate(ce, attr_a, attr_b)
        p_a = torch.tensor(scene["p_a"], device=device, dtype=torch.float32)
        p_b = torch.tensor(scene["p_b"], device=device, dtype=torch.float32)
        selected = g_a * p_a + g_b * p_b
        clip_proj = proj(ce)

        # Cond
        cb = torch.zeros(1, 64, device=device, dtype=torch.float32)
        cb[0, :3] = selected
        cb[0, 4:7] = p_a
        cb[0, 7:10] = p_b
        cb[0, 10:13] = torch.tensor(COLORS_RGB_VEC[scene["c_a"]], device=device, dtype=torch.float32)
        cb[0, 13:17] = torch.tensor(SHAPE_ENC[scene["s_a"]], device=device, dtype=torch.float32)
        cb[0, 17:20] = torch.tensor(COLORS_RGB_VEC[scene["c_b"]], device=device, dtype=torch.float32)
        cb[0, 20:24] = torch.tensor(SHAPE_ENC[scene["s_b"]], device=device, dtype=torch.float32)
        cb[0, 24:56] = clip_proj[0]

        # DDIM
        scheduler = SimpleDDPMScheduler(num_timesteps=100)
        x = torch.randn(1, 16, 7, device=device, dtype=torch.float32)
        si = np.linspace(0, 99, 25).astype(int)[::-1]
        ab = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
        for i, step in enumerate(si):
            t = torch.full((1,), int(step), dtype=torch.long, device=device)
            e = model(x, t, cb)
            ab_t = ab[step]
            x0 = (x - torch.sqrt(1 - ab_t) * e) / torch.sqrt(ab_t)
            if i < len(si) - 1:
                ab_n = ab[si[i + 1]]
                x = torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * e
            else:
                x = x0
    return x.cpu().numpy()[0], float(g_a.item()), float(g_b.item())


# Catalogo de escenas curadas para demostrar la potencialidad
DEMO_SCENES = [
    # === Color-only ===
    {"text": "pick the red object",
     "c_a": "blue", "s_a": "cube", "p_a": [-0.25, 0.10, 0.80],
     "c_b": "red", "s_b": "cube", "p_b": [0.25, -0.10, 0.85],
     "target_idx": 1},
    {"text": "select the green one",
     "c_a": "green", "s_a": "sphere", "p_a": [-0.20, -0.15, 0.78],
     "c_b": "red", "s_b": "sphere", "p_b": [0.30, 0.10, 0.85],
     "target_idx": 0},
    # === Shape-only ===
    {"text": "pick the sphere",
     "c_a": "blue", "s_a": "cube", "p_a": [-0.25, 0.10, 0.80],
     "c_b": "blue", "s_b": "sphere", "p_b": [0.25, -0.10, 0.85],
     "target_idx": 1},
    {"text": "grab the cylinder",
     "c_a": "red", "s_a": "cylinder", "p_a": [-0.30, 0.00, 0.82],
     "c_b": "red", "s_b": "box", "p_b": [0.25, 0.15, 0.85],
     "target_idx": 0},
    {"text": "take the box",
     "c_a": "green", "s_a": "sphere", "p_a": [-0.25, -0.20, 0.85],
     "c_b": "green", "s_b": "box", "p_b": [0.30, 0.10, 0.78],
     "target_idx": 1},
    {"text": "select the cube",
     "c_a": "red", "s_a": "sphere", "p_a": [-0.20, 0.20, 0.82],
     "c_b": "red", "s_b": "cube", "p_b": [0.30, -0.15, 0.85],
     "target_idx": 1},
    # === Combinacion ===
    {"text": "pick the red sphere",
     "c_a": "blue", "s_a": "sphere", "p_a": [-0.25, 0.15, 0.85],
     "c_b": "red", "s_b": "sphere", "p_b": [0.30, -0.10, 0.80],
     "target_idx": 1},
    {"text": "grab the blue cube",
     "c_a": "blue", "s_a": "cube", "p_a": [-0.30, -0.10, 0.85],
     "c_b": "red", "s_b": "cube", "p_b": [0.25, 0.20, 0.80],
     "target_idx": 0},
    {"text": "select the green cylinder",
     "c_a": "red", "s_a": "cylinder", "p_a": [-0.20, 0.10, 0.82],
     "c_b": "green", "s_b": "cylinder", "p_b": [0.30, -0.10, 0.85],
     "target_idx": 1},
    {"text": "take the green box",
     "c_a": "green", "s_a": "box", "p_a": [-0.25, 0.05, 0.78],
     "c_b": "blue", "s_b": "box", "p_b": [0.30, -0.15, 0.85],
     "target_idx": 0},
    # === Casos limite / interesantes ===
    {"text": "the round one",
     "c_a": "red", "s_a": "cube", "p_a": [-0.25, 0.10, 0.85],
     "c_b": "red", "s_b": "sphere", "p_b": [0.30, -0.05, 0.80],
     "target_idx": 1},
    {"text": "pick anything blue",
     "c_a": "green", "s_a": "cylinder", "p_a": [-0.30, 0.15, 0.82],
     "c_b": "blue", "s_b": "cylinder", "p_b": [0.30, -0.10, 0.85],
     "target_idx": 1},
]


def main():
    print("[exp19] Generando simulaciones visuales del VLA-lite multi-atributo")
    print(f"[exp19] {len(DEMO_SCENES)} escenas de demostracion")

    tok, clip_mod, model, proj, gate, device = load_model()
    print(f"  device={device}")

    # Convertir posiciones a numpy
    for s in DEMO_SCENES:
        s["p_a"] = np.array(s["p_a"])
        s["p_b"] = np.array(s["p_b"])
        s["target_pos"] = s["p_a"] if s["target_idx"] == 0 else s["p_b"]
        s["distractor_pos"] = s["p_b"] if s["target_idx"] == 0 else s["p_a"]

    results = {"scenes": []}
    correct_count = 0

    for i, scene in enumerate(DEMO_SCENES):
        text = scene["text"]
        traj, g_a, g_b = run_inference(text, scene, tok, clip_mod, model, proj, gate, device)
        chosen_idx = 0 if g_a > g_b else 1
        ok = chosen_idx == scene["target_idx"]
        if ok:
            correct_count += 1
        save_path = OUTPUT / f"scene_{i+1:02d}.png"
        render_scene(scene, traj, [g_a, g_b], text, save_path)
        flag = "✓" if ok else "✗"
        endpoint = traj[-1, :3]
        d_target = np.linalg.norm(endpoint - scene["target_pos"])
        print(f"  {flag} scene {i+1:02d}: '{text}' | A=({scene['c_a']},{scene['s_a']}) "
              f"B=({scene['c_b']},{scene['s_b']}) | gate A={g_a:.1%} B={g_b:.1%} | "
              f"d_target={d_target*100:.1f}cm")
        results["scenes"].append({
            "id": i + 1, "text": text, "A": f"{scene['c_a']} {scene['s_a']}",
            "B": f"{scene['c_b']} {scene['s_b']}",
            "target": scene["target_idx"], "chosen": chosen_idx,
            "gate_a": g_a, "gate_b": g_b, "correct": ok,
            "distance_to_target_cm": float(d_target * 100),
        })

    print(f"\n  Total: {correct_count}/{len(DEMO_SCENES)} correctos = {correct_count/len(DEMO_SCENES):.1%}")
    results["summary"] = {
        "n_scenes": len(DEMO_SCENES),
        "n_correct": correct_count,
        "accuracy": correct_count / len(DEMO_SCENES),
    }

    # Grid overview con miniaturas (2 columnas para que cada escena se vea legible)
    print("\n  Generando grid overview...")
    n_cols = 2
    n_rows = (len(DEMO_SCENES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 7 * n_rows))
    axes_flat = axes.flat if n_rows > 1 else [axes] if n_cols == 1 else axes
    for i, (ax, scene) in enumerate(zip(axes_flat, DEMO_SCENES)):
        from matplotlib.image import imread
        img = imread(OUTPUT / f"scene_{i+1:02d}.png")
        ax.imshow(img)
        ax.set_title(f'Escena {i+1}: "{scene["text"]}"',
                       fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
    plt.suptitle(f"VLA-lite multi-atributo — {len(DEMO_SCENES)} escenas demostrativas "
                  f"({correct_count}/{len(DEMO_SCENES)} aciertos)",
                  fontsize=18, fontweight="bold", y=0.998)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    overview_path = OUTPUT / "grid_overview.png"
    plt.savefig(overview_path, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {overview_path}")

    with open(OUTPUT / "exp19_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT / 'exp19_results.json'}")


if __name__ == "__main__":
    main()
