#!/usr/bin/env python3
"""Demo minimalista para charlas en vivo: el público elige dónde está la pieza
y la Diffusion Policy genera N trayectorias en tiempo real.

Tres vistas:
  1. Nube de caminos (multimodalidad) — gráfico 3D.
  2. 🎬 Animación: un brazo estilizado ejecuta el mejor camino y levanta la pieza.
  3. 🤖 Ejecución REAL en CoppeliaSim (si está abierto): el UR del simulador
     hace el pick con la pieza donde el público la puso (stack Iter 7c:
     policy v7a_phase2 + best-of-8 + fix IK).

Uso:
    .venv/bin/python scripts/demo_charla.py     # → http://127.0.0.1:7860
"""
from __future__ import annotations

import math
import socket
import subprocess
import sys
import tempfile
import threading
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
GRIS_ARM, AZUL_ARM = "#C9D4DE", "#7EC8E3"

PRESETS = {
    "Centro de la mesa": (0.00, 0.00, 0.80),
    "A la izquierda": (-0.30, 0.10, 0.80),
    "Esquina lejana": (0.35, -0.30, 0.75),
}

_cache: dict = {}
_sim_lock = threading.Lock()


# ──────────────────────────── modelo web ────────────────────────────
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


# ─────────────────────── nube de caminos (3D) ───────────────────────
def _fig_nube(trajs, x, y, z, n, ms):
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
    return fig


# ─────────────────── animación del brazo estilizado ───────────────────
L1, L2, Z_BASE = 0.68, 0.68, 0.14


def _ik_brazo(p, base_xy):
    """IK analítica: giro de base + 2 eslabones (codo arriba), base en base_xy.
    Devuelve los puntos 3D: base, hombro, codo, efector y el yaw."""
    bx, by = base_xy
    x, y, z = p
    yaw = math.atan2(y - by, x - bx)
    r = max(math.hypot(x - bx, y - by), 1e-6)
    dz = z - Z_BASE
    d = min(max(math.hypot(r, dz), abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4)
    a_int = math.acos((L1**2 + d**2 - L2**2) / (2 * L1 * d))
    ang_h = math.atan2(dz, r) + a_int
    codo = np.array([bx + math.cos(ang_h) * math.cos(yaw) * L1,
                     by + math.cos(ang_h) * math.sin(yaw) * L1,
                     Z_BASE + math.sin(ang_h) * L1])
    # efector exactamente donde alcanza la cadena (clampeado si p está fuera)
    ee = np.array([bx + math.cos(yaw) * r, by + math.sin(yaw) * r, Z_BASE + dz])
    base, hombro = np.array([bx, by, 0.0]), np.array([bx, by, Z_BASE])
    return base, hombro, codo, ee, yaw


def _suavizar(path, sub=3):
    """Subdivide waypoints con interpolación lineal + media móvil ligera."""
    dense = []
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, sub, endpoint=False):
            dense.append((1 - t) * a + t * b)
    dense.append(path[-1])
    arr = np.array(dense)
    if len(arr) > 4:
        k = np.array([0.2, 0.6, 0.2])
        for j in range(arr.shape[1]):
            arr[1:-1, j] = np.convolve(arr[:, j], k, mode="same")[1:-1]
    return arr


def _video_brazo(trajs, x, y, z):
    """Renderiza la animación del brazo siguiendo el mejor camino → mp4."""
    target = np.array([x, y, z])
    best = trajs[np.argmin(np.linalg.norm(trajs[:, -1, :3] - target, axis=1))]
    wps = best[:, :3]

    # base del robot 0.75 m "detrás" de la pieza (alcance frontal natural)
    dir_xy = target[:2] if np.linalg.norm(target[:2]) > 0.05 else np.array([1.0, 0.0])
    dir_xy = dir_xy / np.linalg.norm(dir_xy)
    base_xy = target[:2] - dir_xy * 0.75

    inicio = np.array([*(base_xy + dir_xy * 0.30), 0.70])
    aproximacion = np.linspace(inicio, wps[0], 10)
    descenso = _suavizar(wps, sub=3)
    contacto = np.linspace(descenso[-1], target, 8)
    levante = np.linspace(target, target + [0, 0, 0.22], 12)
    path = np.vstack([aproximacion, descenso, contacto, levante])
    n_pre = len(aproximacion) + len(descenso) + len(contacto)  # frame donde agarra

    lims_pts = np.vstack([path, [[*base_xy, 0]], [target]])
    lo, hi = lims_pts.min(0), lims_pts.max(0)
    margen = 0.18 * (hi - lo).max()
    azim = math.degrees(math.atan2(dir_xy[1], dir_xy[0])) + 105  # perfil del brazo

    tmpdir = Path(tempfile.mkdtemp(prefix="brazo_"))
    fig = plt.figure(figsize=(8.8, 6.2), facecolor=NAVY)
    ax = fig.add_subplot(111, projection="3d")

    # piso de referencia
    fx = np.linspace(lo[0] - margen, hi[0] + margen, 2)
    fy = np.linspace(lo[1] - margen, hi[1] + margen, 2)
    FX, FY = np.meshgrid(fx, fy)

    for i, p in enumerate(path):
        ax.cla()
        ax.set_facecolor(NAVY)
        ax.view_init(elev=16, azim=azim)
        agarrada = i >= n_pre
        ax.plot_surface(FX, FY, np.zeros_like(FX), color=CIAN, alpha=0.06)
        for tr in trajs[:10]:  # nube de fondo, tenue
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=CIAN, alpha=0.08, lw=1.0)
        ax.plot(path[: i + 1, 0], path[: i + 1, 1], path[: i + 1, 2],
                color=VERDE, alpha=0.85, lw=2.4)  # rastro recorrido
        base, hombro, codo, ee, yaw = _ik_brazo(p, base_xy)
        ax.plot(*zip(base, hombro), color=GRIS_ARM, lw=11, solid_capstyle="round")
        ax.plot(*zip(hombro, codo), color=GRIS_ARM, lw=9, solid_capstyle="round")
        ax.plot(*zip(codo, ee), color=GRIS_ARM, lw=7, solid_capstyle="round")
        for joint, s in ((base, 180), (hombro, 150), (codo, 120)):
            ax.scatter(*joint, color=AZUL_ARM, s=s, zorder=9, edgecolors="white", linewidths=0.8)
        # pinza: dos dedos que se cierran al agarrar
        apertura = 0.022 if agarrada else 0.065
        lateral = np.array([-math.sin(yaw), math.cos(yaw), 0.0]) * apertura
        eje = (ee - codo) / (np.linalg.norm(ee - codo) + 1e-9) * 0.10
        for sgn in (+1, -1):
            dedo = ee + sgn * lateral
            ax.plot(*zip(dedo, dedo + eje), color=AZUL_ARM, lw=5, solid_capstyle="round")
        pieza = ee + eje if agarrada else target
        ax.scatter(*pieza, color=AMARILLO, s=420, marker="s",
                   edgecolors="white", linewidths=1.5, zorder=10)
        for setlim, lo_i, hi_i in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
            setlim(lo_i - margen, hi_i + margen)
        ax.set_zlim(0, hi[2] + margen)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.02)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
        ax.grid(False)
        ax.set_box_aspect((1, 1, 0.85))
        fase = "acercándose..." if i < n_pre - 8 else ("tomando la pieza" if not agarrada else "¡la tiene!")
        ax.set_title(f"El robot ejecuta el mejor camino — {fase}",
                     color="white", fontsize=15, fontweight="bold", pad=10)
        fig.savefig(tmpdir / f"f_{i:04d}.png", dpi=85, facecolor=NAVY)
    plt.close(fig)

    mp4 = tmpdir / "brazo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "16",
         "-i", str(tmpdir / "f_%04d.png"),
         "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        check=True,
    )
    return str(mp4)


def generar(x, y, z, n):
    import torch

    model, scheduler, device = _load()
    cond_vec = np.zeros(64, dtype=np.float32)
    cond_vec[:3] = [x, y, z]
    cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

    t0 = time.time()
    trajs = np.array([_ddim(model, scheduler, cond, device) for _ in range(int(n))])
    ms = (time.time() - t0) * 1000

    fig = _fig_nube(trajs, x, y, z, n, ms)
    video = _video_brazo(trajs, x, y, z)

    spread = np.std(trajs[:, -1, :3], axis=0).mean() * 100
    resumen = (f"**{int(n)} trayectorias** en **{ms:.0f} ms** ({ms / n:.0f} ms c/u, en `{device}`) · "
               f"los finales coinciden dentro de **{spread:.1f} cm** — diversidad con propósito.")
    return fig, video, resumen


# ──────────────── ejecución real en CoppeliaSim (Iter 7c) ────────────────
def _coppelia_disponible() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 23000), timeout=1.0):
            return True
    except OSError:
        return False


def _load_sim_stack():
    if "sim" in _cache:
        return _cache["sim"]
    import torch
    from src.planning.diffusion_policy import DiffusionGraspPlanner
    from src.planning.visual_encoder import ResNet18RGBDEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(REPO / "data/models/diffusion_policy_v7a_phase2.pth",
                      map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    enc_state = torch.load(REPO / "data/models/visual_encoder_iter5.pth",
                           map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])
    _cache["sim"] = (planner, encoder)
    return _cache["sim"]


def ejecutar_en_sim(sx, sy, rot_deg):
    """Generador: va informando el progreso mientras el UR ejecuta el pick."""
    if not _coppelia_disponible():
        yield ("❌ **CoppeliaSim no está corriendo.** Ábrelo primero "
               "(app CoppeliaSim, puerto ZMQ 23000) y vuelve a intentar.")
        return
    if not _sim_lock.acquire(blocking=False):
        yield "⏳ Ya hay una ejecución en curso — espera a que termine."
        return
    try:
        yield "🔌 Conectando con CoppeliaSim y cargando el modelo Iter 7c..."
        from experiments.run_pick_with_diffusion import pick_with_dp
        from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

        planner, encoder = _load_sim_stack()
        theta = math.radians(float(rot_deg))
        c, s = math.cos(theta), math.sin(theta)
        pose = np.eye(4)
        pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pose[:3, 3] = [float(sx), float(sy), 0.033]

        yield (f"🤖 **Ejecutando en el simulador** — pieza en x={sx:.2f}, y={sy:.2f}, "
               f"rotación {int(float(rot_deg))}°.\n\n👀 **Miren la ventana de CoppeliaSim**: "
               "el robot ve la pieza, genera 8 trayectorias, ejecuta la mejor y la deposita (~1 min).")
        t0 = time.time()
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(REPO / "data/scenes/bin_base.ttt")
            r = pick_with_dp(planner, pose, bridge, frames_dir=None,
                             visual_encoder=encoder, best_of_n=8)
        dt = time.time() - t0
        ok = r["grasp_plausible"] and r["deposit_plausible"] and r["ik_converged"]
        icono = "✅" if ok else "⚠️"
        yield (f"{icono} **Pick {'completado' if ok else 'terminado con observaciones'}** en {dt:.0f} s\n\n"
               f"- Precisión del agarre: **{r['grasp_proximity_m']*100:.1f} cm** de la pieza\n"
               f"- Depósito a **{r['deposit_error_m']*100:.1f} cm** del objetivo\n"
               f"- Brazo (IK): {'convergió ✓' if r['ik_converged'] else 'no convergió'}\n\n"
               "_Mismo pipeline del estudio: percepción → difusión best-of-8 → control._")
    except Exception as e:  # noqa: BLE001 — en vivo, cualquier fallo se informa y se sigue
        yield f"❌ Algo falló: `{e}`\n\nPlan B: video grabado en la slide 13 de la charla."
    finally:
        _sim_lock.release()


# ──────────────────────────────── UI ────────────────────────────────
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

            with gr.Accordion("🤖 Ejecutarlo de verdad — CoppeliaSim", open=False):
                gr.Markdown("La pieza se coloca donde ustedes digan **sobre la mesa del simulador** "
                            "y el brazo UR hace el pick completo (pipeline Iter 7c).")
                cx = gr.Slider(0.40, 0.55, value=0.47, step=0.01, label="posición — cerca / lejos (m)")
                cy = gr.Slider(-0.15, -0.05, value=-0.10, step=0.01, label="posición — izquierda / derecha (m)")
                crot = gr.Radio(["0", "45", "90"], value="0", label="rotación de la pieza (°)")
                btn_sim = gr.Button("🤖 Ejecutar en el simulador", variant="primary")
                estado_sim = gr.Markdown()
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🌀 Los caminos posibles"):
                    plot = gr.Plot(label="", container=False)
                with gr.Tab("🎬 El robot en acción"):
                    video = gr.Video(label="", autoplay=True, loop=True, container=False)

    btn.click(generar, [sx, sy, sz, sn], [plot, video, resumen])
    for b, (nombre, (px, py, pz)) in zip(botones, PRESETS.items()):
        b.click(lambda px=px, py=py, pz=pz: (px, py, pz), outputs=[sx, sy, sz])
    btn_sim.click(ejecutar_en_sim, [cx, cy, crot], [estado_sim])
    # ejemplo al abrir: el público nunca ve un panel vacío
    demo.load(generar, [sx, sy, sz, sn], [plot, video, resumen])

if __name__ == "__main__":
    print("Cargando modelo (una vez)...")
    _load()
    print("Listo → http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                css=CSS, theme=gr.themes.Soft(primary_hue="cyan"))
