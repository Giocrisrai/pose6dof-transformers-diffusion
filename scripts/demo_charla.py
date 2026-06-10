#!/usr/bin/env python3
"""Demo minimalista para charlas en vivo: el público elige dónde está la pieza
y la Diffusion Policy genera N trayectorias en tiempo real.

Tres vistas:
  1. Nube de caminos (multimodalidad) — gráfico 3D.
  2. 🎬 Visor 3D interactivo (three.js, vendorizado — funciona sin internet):
     un brazo robótico con luces y sombras ejecuta el mejor camino en tiempo
     real; se puede rotar/zoomear con el mouse durante la charla.
  3. 🤖 Ejecución REAL en CoppeliaSim (si está abierto): el UR del simulador
     hace el pick con la pieza donde el público la puso (stack Iter 7c:
     policy v7a_phase2 + best-of-8 + fix IK).

Uso:
    .venv/bin/python scripts/demo_charla.py     # → http://127.0.0.1:7860
"""
from __future__ import annotations

import html as html_mod
import json
import math
import socket
import sys
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

ASSETS = REPO / "scripts/assets_demo_charla"

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


# ─────────────── brazo: IK + visor 3D interactivo (three.js) ───────────────
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


def _a3(p):
    """Coordenadas robot (Z arriba) → three.js (Y arriba)."""
    return [round(float(p[0]), 4), round(float(p[2]), 4), round(float(-p[1]), 4)]


_PLANTILLA_VISOR = """<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0;overflow:hidden;background:#0F2A43;font-family:-apple-system,sans-serif}
.hud{position:fixed;z-index:9;color:#fff;font-weight:600}</style>
<script type="importmap">{"imports":{"three":"__THREE__"}}</script></head>
<body>
<div class="hud" style="left:14px;top:10px;font-size:16px">El robot ejecuta el mejor camino</div>
<div class="hud" id="fase" style="right:14px;top:10px;color:#FFD166;font-size:15px"></div>
<div class="hud" style="left:14px;bottom:8px;color:#9FD8EE;font-size:12px;font-weight:400">
arrastra para girar · rueda para acercar</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from '__ORBIT__';
const D = __DATA__;
const v = a => new THREE.Vector3(a[0], a[1], a[2]);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0F2A43);
scene.fog = new THREE.Fog(0x0F2A43, 4.0, 8.0);
const cam = new THREE.PerspectiveCamera(42, innerWidth/innerHeight, .01, 60);
cam.position.copy(v(D.cam));
const ren = new THREE.WebGLRenderer({antialias:true});
ren.setSize(innerWidth, innerHeight);
ren.setPixelRatio(Math.min(devicePixelRatio, 2));
ren.shadowMap.enabled = true; ren.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(ren.domElement);
const ctl = new OrbitControls(cam, ren.domElement);
ctl.target.copy(v(D.look)); ctl.enableDamping = true; ctl.maxPolarAngle = 1.52;

scene.add(new THREE.HemisphereLight(0xbfdcec, 0x16324a, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.7);
sun.position.set(2.2, 4.5, 1.8); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {left:-2.5, right:2.5, top:2.5, bottom:-2.5});
scene.add(sun);

const piso = new THREE.Mesh(new THREE.CircleGeometry(2.6, 64),
  new THREE.MeshStandardMaterial({color:0x16324a, roughness:.92}));
piso.rotation.x = -Math.PI/2; piso.receiveShadow = true; scene.add(piso);
const grid = new THREE.GridHelper(5, 32, 0x2a5878, 0x1c4360);
grid.position.y = .002; scene.add(grid);

for (const tr of D.cloud) {
  const g = new THREE.BufferGeometry().setFromPoints(tr.map(v));
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial(
    {color:0x7FD4F0, transparent:true, opacity:.22})));
}
const trailG = new THREE.BufferGeometry().setFromPoints(D.trail.map(v));
trailG.setDrawRange(0, 1);
scene.add(new THREE.Line(trailG, new THREE.LineBasicMaterial(
  {color:0x00E08F, transparent:true, opacity:.95})));

const pieza = new THREE.Mesh(new THREE.BoxGeometry(.072, .072, .072),
  new THREE.MeshStandardMaterial({color:0xFFD166, emissive:0x553f08, roughness:.45}));
pieza.castShadow = true; pieza.position.copy(v(D.target)); scene.add(pieza);

const matArm = new THREE.MeshStandardMaterial({color:0xD8DEE6, metalness:.62, roughness:.34});
const matAcc = new THREE.MeshStandardMaterial({color:0x7EC8E3, metalness:.45, roughness:.30});
const mk = (geo, mat) => {const m = new THREE.Mesh(geo, mat); m.castShadow = true; scene.add(m); return m;};
const segs = [[.058,.050],[.048,.040],[.036,.028]].map(
  ([r1,r2]) => mk(new THREE.CylinderGeometry(r2, r1, 1, 24), matArm));
const joints = [.085,.072,.058,.045].map(r => mk(new THREE.SphereGeometry(r, 26, 18), matAcc));
const dedos = [0,1].map(() => mk(new THREE.BoxGeometry(.018, .105, .03), matAcc));
const ped = mk(new THREE.CylinderGeometry(.13, .16, .05, 32), matAcc);
ped.position.copy(v(D.frames[0].j[0])); ped.position.y = .025;

const up = new THREE.Vector3(0,1,0), q = new THREE.Quaternion(), dir = new THREE.Vector3();
function setSeg(m, a, b) {
  dir.subVectors(b, a); const L = Math.max(dir.length(), 1e-5);
  m.position.copy(a).addScaledVector(dir, .5);
  q.setFromUnitVectors(up, dir.normalize());
  m.quaternion.copy(q); m.scale.set(1, L, 1);
}
const F = D.frames, N = F.length, FPS = 26, HOLD = 1100;
const total = N / FPS * 1000;
const J = [0,1,2,3].map(() => new THREE.Vector3());
const t0 = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  const t = Math.min(((now - t0) % (total + HOLD)) / 1000 * FPS, N - 1);
  const i = Math.floor(t), f = t - i;
  const A = F[i], B = F[Math.min(i + 1, N - 1)];
  for (let k = 0; k < 4; k++) J[k].copy(v(A.j[k])).lerp(v(B.j[k]), f);
  setSeg(segs[0], J[0], J[1]); setSeg(segs[1], J[1], J[2]); setSeg(segs[2], J[2], J[3]);
  J.forEach((p, k) => joints[k].position.copy(p));
  const lat = v(A.lat), eje = J[3].clone().sub(J[2]).normalize();
  const ap = A.g ? .046 : .075;
  dedos.forEach((d, k) => {
    d.position.copy(J[3]).addScaledVector(lat, k ? -ap : ap).addScaledVector(eje, .055);
    q.setFromUnitVectors(up, eje); d.quaternion.copy(q);
  });
  if (A.g) pieza.position.copy(J[3]).addScaledVector(eje, .095);
  else pieza.position.copy(v(D.target));
  trailG.setDrawRange(0, Math.max(2, Math.floor((i + f) / N * D.trail.length)));
  document.getElementById('fase').textContent =
    A.g ? '¡la tiene!' : (i < D.nPre - 10 ? 'siguiendo el camino de la IA…' : 'tomando la pieza…');
  ctl.update(); ren.render(scene, cam);
}
requestAnimationFrame(animate);
addEventListener('resize', () => {
  cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix();
  ren.setSize(innerWidth, innerHeight);
});
</script></body></html>"""


def _visor_3d(trajs, x, y, z):
    """Visor three.js interactivo: el brazo ejecuta el mejor camino → iframe HTML."""
    target = np.array([x, y, z])
    best = trajs[np.argmin(np.linalg.norm(trajs[:, -1, :3] - target, axis=1))]
    wps = best[:, :3]

    # base del robot 0.75 m "detrás" de la pieza (alcance frontal natural)
    dir_xy = target[:2] if np.linalg.norm(target[:2]) > 0.05 else np.array([1.0, 0.0])
    dir_xy = dir_xy / np.linalg.norm(dir_xy)
    base_xy = target[:2] - dir_xy * 0.75

    inicio = np.array([*(base_xy + dir_xy * 0.30), 0.70])
    aproximacion = np.linspace(inicio, wps[0], 12)
    descenso = _suavizar(wps, sub=3)
    contacto = np.linspace(descenso[-1], target, 8)
    levante = np.linspace(target, target + [0, 0, 0.24], 14)
    path = np.vstack([aproximacion, descenso, contacto, levante])
    n_pre = len(aproximacion) + len(descenso) + len(contacto)

    frames = []
    for i, p in enumerate(path):
        base, hombro, codo, ee, yaw = _ik_brazo(p, base_xy)
        frames.append({
            "j": [_a3(base), _a3(hombro), _a3(codo), _a3(ee)],
            "lat": _a3([-math.sin(yaw), math.cos(yaw), 0.0]),
            "g": 1 if i >= n_pre else 0,
        })

    centro = np.array([*((base_xy + target[:2]) / 2), 0.45])
    perp = np.array([-dir_xy[1], dir_xy[0]])
    cam_pos = np.array([*(centro[:2] + perp * 1.55 - dir_xy * 0.25), 0.95])

    data = {
        "frames": frames,
        "cloud": [[_a3(pt) for pt in tr] for tr in trajs[:10, :, :3].tolist()],
        "trail": [_a3(p) for p in path],
        "target": _a3(target),
        "nPre": n_pre,
        "cam": _a3(cam_pos),
        "look": _a3(centro),
    }
    pagina = (_PLANTILLA_VISOR
              .replace("__THREE__", f"/gradio_api/file={ASSETS}/three.module.min.js")
              .replace("__ORBIT__", f"/gradio_api/file={ASSETS}/OrbitControls.js")
              .replace("__DATA__", json.dumps(data)))
    return (f'<iframe srcdoc="{html_mod.escape(pagina, quote=True)}" '
            'style="width:100%;height:640px;border:none;border-radius:12px;'
            'background:#0F2A43"></iframe>')


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
    visor = _visor_3d(trajs, x, y, z)

    spread = np.std(trajs[:, -1, :3], axis=0).mean() * 100
    resumen = (f"**{int(n)} trayectorias** en **{ms:.0f} ms** ({ms / n:.0f} ms c/u, en `{device}`) · "
               f"los finales coinciden dentro de **{spread:.1f} cm** — diversidad con propósito.")
    return fig, visor, resumen


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
                with gr.Tab("🎬 El robot en acción (3D interactivo)"):
                    visor = gr.HTML()
                with gr.Tab("🌀 Los caminos posibles"):
                    plot = gr.Plot(label="", container=False)

    btn.click(generar, [sx, sy, sz, sn], [plot, visor, resumen])
    for b, (nombre, (px, py, pz)) in zip(botones, PRESETS.items()):
        b.click(lambda px=px, py=py, pz=pz: (px, py, pz), outputs=[sx, sy, sz])
    btn_sim.click(ejecutar_en_sim, [cx, cy, crot], [estado_sim])
    # ejemplo al abrir: el público nunca ve un panel vacío
    demo.load(generar, [sx, sy, sz, sn], [plot, visor, resumen])

if __name__ == "__main__":
    gr.set_static_paths(paths=[ASSETS])
    print("Cargando modelo (una vez)...")
    _load()
    print("Listo → http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                css=CSS, theme=gr.themes.Soft(primary_hue="cyan"))
