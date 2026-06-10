#!/usr/bin/env python3
"""Demo minimalista para charlas en vivo: el público elige dónde está la pieza
y la Diffusion Policy genera N trayectorias en tiempo real.

Pensado para proyector: un solo control que importa (posición), un botón grande,
un gráfico 3D grande con la estética de la charla (fondo navy, trayectorias cian).

Uso:
    .venv/bin/python scripts/demo_charla.py     # → http://127.0.0.1:7860
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELO = "diffusion_policy_ultra.pth"  # el de mejor MSE (0.00221)
HIDDEN_DIM = 256
NAVY, CIAN, AMARILLO, ROJO, VERDE = "#0F2A43", "#7FD4F0", "#FFD166", "#EF476F", "#00E08F"

PRESETS = {
    "Centro de la mesa": (0.00, 0.00, 0.80),
    "A la izquierda": (-0.30, 0.10, 0.80),
    "Esquina lejana": (0.35, -0.30, 0.75),
}

_cache: dict = {}


def _load():
    if "m" in _cache:
        return _cache["m"]
    import torch
    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet1D(action_dim=7, horizon=16, cond_dim=64, hidden_dim=HIDDEN_DIM).to(device)
    ckpt = torch.load(REPO / "data/models" / MODELO, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.eval()
    _cache["m"] = (model, SimpleDDPMScheduler(num_timesteps=100), device)
    return _cache["m"]


def _ddim(model, scheduler, cond, device, n_steps=25):
    import torch

    x = torch.randn(1, 16, 7, device=device)
    steps = np.linspace(0, scheduler.num_timesteps - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, step in enumerate(steps):
            t = torch.tensor([step], dtype=torch.long, device=device)
            eps = model(x, t, cond)
            ab = alpha_bar[step]
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            if i < len(steps) - 1:
                ab_n = alpha_bar[steps[i + 1]]
                x = torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * eps
            else:
                x = x0
    return x.cpu().numpy()[0]


def generar(x, y, z, n):
    import torch

    model, scheduler, device = _load()
    cond_vec = np.zeros(64, dtype=np.float32)
    cond_vec[:3] = [x, y, z]
    cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

    t0 = time.time()
    trajs = np.array([_ddim(model, scheduler, cond, device) for _ in range(int(n))])
    ms = (time.time() - t0) * 1000

    fig = plt.figure(figsize=(11, 7.6), facecolor=NAVY)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(NAVY)
    pts = trajs[:, :, :3]
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=CIAN, alpha=0.55, lw=2.2)
        ax.scatter(*tr[0, :3], color="#FFFFFF", s=22, alpha=0.9)   # inicio
        ax.scatter(*tr[-1, :3], color=VERDE, s=30, alpha=0.95)     # final
    ax.scatter([x], [y], [z], color=AMARILLO, s=700, marker="*",
               edgecolors="white", linewidths=1.5, zorder=10, label="la pieza (objetivo)")
    ax.plot([], [], color=CIAN, lw=2.2, label="caminos generados por la IA")
    ax.scatter([], [], color=VERDE, s=30, label="dónde termina cada camino")

    # zoom a la acción: límites = nube + pieza, con margen
    todos = np.vstack([pts.reshape(-1, 3), [[x, y, z]]])
    lo, hi = todos.min(axis=0), todos.max(axis=0)
    margen = 0.15 * (hi - lo).max() + 1e-3
    for setlim, lo_i, hi_i in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
        setlim(lo_i - margen, hi_i + margen)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.04)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.grid(False)
    ax.legend(facecolor=NAVY, labelcolor="white", framealpha=0.25, fontsize=15,
              loc="upper left", borderpad=1.0)
    ax.set_title(f"{int(n)} caminos distintos · generados en {ms/1000:.1f} s · mismo objetivo",
                 color="white", fontsize=20, fontweight="bold", pad=16)
    fig.tight_layout()

    spread = np.std(trajs[:, -1, :3], axis=0).mean() * 100
    resumen = (f"**{int(n)} trayectorias** en **{ms:.0f} ms** ({ms / n:.0f} ms c/u, en `{device}`) · "
               f"los finales coinciden dentro de **{spread:.1f} cm** — diversidad con propósito.")
    return fig, resumen


CSS = """
.gradio-container {max-width: 1500px !important; font-size: 18px;}
button.primary {font-size: 26px !important; padding: 18px !important;}
h1 {font-size: 42px !important;}
"""

with gr.Blocks(title="¿Dónde está la pieza?") as demo:
    gr.Markdown("# 🤖 ¿Dónde está la pieza? — la IA genera el movimiento")
    with gr.Row():
        with gr.Column(scale=1):
            sx = gr.Slider(-0.4, 0.4, value=0.0, step=0.05, label="x — izquierda / derecha (m)")
            sy = gr.Slider(-0.4, 0.4, value=0.0, step=0.05, label="y — cerca / lejos (m)")
            sz = gr.Slider(0.6, 1.1, value=0.8, step=0.05, label="z — altura (m)")
            sn = gr.Slider(5, 50, value=20, step=5, label="¿cuántas trayectorias?")
            btn = gr.Button("✨ Generar trayectorias", variant="primary")
            with gr.Row():
                botones = [gr.Button(p, size="sm") for p in PRESETS]
            resumen = gr.Markdown()
        with gr.Column(scale=2):
            plot = gr.Plot(label="", container=False)

    btn.click(generar, [sx, sy, sz, sn], [plot, resumen])
    for b, (nombre, (px, py, pz)) in zip(botones, PRESETS.items()):
        b.click(lambda px=px, py=py, pz=pz: (px, py, pz), outputs=[sx, sy, sz])
    # ejemplo al abrir: el público nunca ve un panel vacío
    demo.load(generar, [sx, sy, sz, sn], [plot, resumen])

if __name__ == "__main__":
    print("Cargando modelo (una vez)...")
    _load()
    print("Listo → http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                css=CSS, theme=gr.themes.Soft(primary_hue="cyan"))
