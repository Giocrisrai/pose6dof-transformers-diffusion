#!/usr/bin/env python3
"""Demo minimalista para charlas en vivo: el público elige dónde está la pieza
y la Diffusion Policy genera N trayectorias en tiempo real.

Tres vistas:
  1. Nube de caminos (multimodalidad) — gráfico 3D.
  2. 🎬 Visor 3D interactivo (three.js, vendorizado — funciona sin internet):
     ciclo didáctico VER → IMAGINAR CAMINOS → ELEGIR → EJECUTAR → DEPOSITAR,
     con física de caída en la bandeja, 4 cámaras conmutables y un botón de
     PERTURBACIÓN: empuja la pieza en plena ejecución y el sistema re-planifica
     (nueva nube de difusión hacia la nueva posición) y completa el pick igual.
  3. 🤖 Ejecución REAL en CoppeliaSim (si está abierto): el UR del simulador
     hace el pick con la pieza donde el público la puso (stack Iter 7c).

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


def _muestrear(xyz, n):
    """Genera n trayectorias condicionadas a la posición xyz."""
    import torch

    model, scheduler, device = _load()
    cond_vec = np.zeros(64, dtype=np.float32)
    cond_vec[:3] = xyz
    cond = torch.tensor(cond_vec, device=device).unsqueeze(0)
    return np.array([_ddim(model, scheduler, cond, device) for _ in range(int(n))])


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
    """IK analítica: giro de base + 2 eslabones (codo arriba), base en base_xy."""
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


def _ciclo(target, wps, base_xy, tray, inicio):
    """Construye el ciclo completo de frames (joints + apertura de pinza) para
    un objetivo: aproximación → descenso → cierre → levante → traslado → suelta."""
    aproximacion = np.linspace(inicio, wps[0], 12)
    descenso = _suavizar(wps, sub=3)
    # el efector frena ANTES del centro de la pieza: el punto de agarre de la
    # pinza queda 0.095 m más adelante sobre el eje del último eslabón, así la
    # pieza no "salta" al engancharse
    _, _, codo_g, ee_g, _ = _ik_brazo(target, base_xy)
    eje_g = (ee_g - codo_g) / (np.linalg.norm(ee_g - codo_g) + 1e-9)
    ee_grasp = target - eje_g * 0.095
    contacto = np.linspace(descenso[-1], ee_grasp, 10)
    cierre = np.repeat(ee_grasp[None, :], 7, axis=0)   # quieto mientras cierra
    levante = np.linspace(ee_grasp, ee_grasp + [0, 0, 0.26], 14)
    sobre_bandeja = np.array([tray[0], tray[1], 0.42])
    traslado = []
    for t in np.linspace(0, 1, 24):
        p = (1 - t) * levante[-1] + t * sobre_bandeja
        p[2] += 0.12 * math.sin(math.pi * t)
        traslado.append(p)
    traslado = np.array(traslado)
    retirada = np.linspace(sobre_bandeja, sobre_bandeja + [0, 0, 0.16], 10)
    path = np.vstack([aproximacion, descenso, contacto, cierre, levante, traslado, retirada])

    n_pre = len(aproximacion) + len(descenso) + len(contacto) + len(cierre)
    lift_idx = n_pre + len(levante)
    release_idx = lift_idx + len(traslado)

    frames = []
    for i, p in enumerate(path):
        base, hombro, codo, ee, yaw = _ik_brazo(p, base_xy)
        if i < n_pre - len(cierre):
            ap = 1.0                                       # abierta hasta llegar
        elif i < n_pre:
            ap = 1.0 - (i - (n_pre - len(cierre))) / len(cierre)  # cierra quieta
        elif i < release_idx:
            ap = 0.0
        else:
            ap = min(1.0, (i - release_idx) / 5)
        frames.append({
            "j": [_a3(base), _a3(hombro), _a3(codo), _a3(ee)],
            "lat": _a3([-math.sin(yaw), math.cos(yaw), 0.0]),
            "ap": round(float(ap), 3),
        })
    trail = [_a3(p) for p in path[:lift_idx]]
    return frames, trail, n_pre, lift_idx, release_idx


_PLANTILLA_VISOR = """<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;overflow:hidden;background:#0F2A43;font-family:-apple-system,sans-serif}
.hud{position:fixed;z-index:9;color:#fff;font-weight:600}
#fases{position:fixed;z-index:9;left:50%;bottom:10px;transform:translateX(-50%);
display:flex;gap:6px;font-size:12.5px;font-weight:600}
#fases span{padding:4px 10px;border-radius:12px;background:rgba(127,212,240,.10);
color:#7FD4F0;transition:all .3s;white-space:nowrap}
#fases span.on{background:#FFD166;color:#0F2A43}
#cams{position:fixed;z-index:9;right:12px;top:38px;display:flex;flex-direction:column;gap:5px}
#cams button{font:600 12px -apple-system;color:#9FD8EE;background:rgba(127,212,240,.12);
border:1px solid rgba(127,212,240,.25);border-radius:8px;padding:5px 10px;cursor:pointer}
#cams button.on{background:#7FD4F0;color:#0F2A43}
#perturbar{position:fixed;z-index:9;left:12px;top:40px;font:700 14px -apple-system;
padding:9px 14px;border-radius:10px;border:1px solid rgba(239,71,111,.5);cursor:pointer;
background:rgba(239,71,111,.15);color:#ffb3c4;opacity:.45;transition:all .3s}
#perturbar.armed{opacity:1;background:#EF476F;color:#fff;box-shadow:0 0 18px rgba(239,71,111,.6)}
</style>
<script type="importmap">{"imports":{"three":"__THREE__"}}</script></head>
<body>
<div class="hud" style="left:14px;top:10px;font-size:16px">Así actúa el sistema</div>
<div class="hud" id="fase" style="right:12px;top:10px;color:#FFD166;font-size:15px"></div>
<button id="perturbar">🫳 perturbar la pieza</button>
<div id="cams">
<button id="cG" class="on">🎥 general</button><button id="cP">perfil</button>
<button id="cC">cenital</button><button id="cS">seguir pinza</button></div>
<div id="fases">
<span id="f0">1 ver</span><span id="f1">2 imaginar caminos</span>
<span id="f2">3 elegir el mejor</span><span id="f3">4 ejecutar</span><span id="f4">5 depositar</span></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from '__ORBIT__';
const D = __DATA__;
const v = a => new THREE.Vector3(a[0], a[1], a[2]);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0F2A43);
scene.fog = new THREE.Fog(0x0F2A43, 4.5, 9.5);
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

const piso = new THREE.Mesh(new THREE.CircleGeometry(2.9, 64),
  new THREE.MeshStandardMaterial({color:0x16324a, roughness:.92}));
piso.rotation.x = -Math.PI/2; piso.receiveShadow = true; scene.add(piso);
const grid = new THREE.GridHelper(5.8, 36, 0x2a5878, 0x1c4360);
grid.position.y = .002; scene.add(grid);

// ── escenografía: mesa bajo la pieza (cubre target y target2) y bandeja ──
const tMid = [(D.target[0]+D.target2[0])/2, D.target[1], (D.target[2]+D.target2[2])/2];
const hMesa = D.target[1] - D.mitadPieza - .001;
const matMesa = new THREE.MeshStandardMaterial({color:0x4a7390, roughness:.55, metalness:.15});
const fuste = new THREE.Mesh(new THREE.CylinderGeometry(.055, .09, hMesa - .03, 28), matMesa);
fuste.position.set(tMid[0], (hMesa - .03)/2, tMid[2]);
const tapa = new THREE.Mesh(new THREE.CylinderGeometry(.18, .18, .03, 40), matMesa);
tapa.position.set(tMid[0], hMesa - .015, tMid[2]);
for (const m of [fuste, tapa]) { m.castShadow = m.receiveShadow = true; scene.add(m); }
const bandeja = new THREE.Group();
const bBase = new THREE.Mesh(new THREE.BoxGeometry(.30, .035, .30),
  new THREE.MeshStandardMaterial({color:0x1f4a63, roughness:.6}));
bBase.castShadow = bBase.receiveShadow = true; bandeja.add(bBase);
for (const [dx,dz] of [[1,0],[-1,0],[0,1],[0,-1]]) {
  const lado = new THREE.Mesh(new THREE.BoxGeometry(dx? .02:.30, .07, dz? .02:.30),
    new THREE.MeshStandardMaterial({color:0x2a5878, roughness:.6}));
  lado.position.set(dx*.14, .035, dz*.14); lado.castShadow = true; bandeja.add(lado);
}
bandeja.position.set(D.tray[0], .02, D.tray[2]); scene.add(bandeja);

// anillo de "percepción"
const anillo = new THREE.Mesh(new THREE.RingGeometry(.09, .115, 48),
  new THREE.MeshBasicMaterial({color:0xFFD166, transparent:true, side:THREE.DoubleSide}));
anillo.rotation.x = -Math.PI/2;
anillo.position.set(D.target[0], D.target[1]+.04, D.target[2]); scene.add(anillo);

// nubes de caminos (nominal y re-plan) y rastros
function hazNube(cloud, color) {
  return cloud.map(tr => {
    const g = new THREE.BufferGeometry().setFromPoints(tr.map(v));
    g.setDrawRange(0, 0);
    const ln = new THREE.Line(g, new THREE.LineBasicMaterial(
      {color, transparent:true, opacity:.45}));
    scene.add(ln); return ln;
  });
}
const nube = hazNube(D.cloud, 0x7FD4F0);
const nube2 = hazNube(D.cloud2, 0xffa3b8);
function hazTrail(pts) {
  const g = new THREE.BufferGeometry().setFromPoints(pts.map(v));
  g.setDrawRange(0, 0);
  const m = new THREE.LineBasicMaterial({color:0x00E08F, transparent:true, opacity:.95});
  scene.add(new THREE.Line(g, m)); return [g, m];
}
const [trailG, trailM] = hazTrail(D.trail);
const [trail2G, trail2M] = hazTrail(D.trail2);

const geoPieza = [new THREE.BoxGeometry(.072, .072, .072),
  new THREE.SphereGeometry(.045, 28, 20),
  new THREE.CylinderGeometry(.04, .04, .076, 28)][D.forma];
const colPieza = new THREE.Color(D.colorPieza);
const pieza = new THREE.Mesh(geoPieza,
  new THREE.MeshStandardMaterial({color:colPieza,
    emissive:colPieza.clone().multiplyScalar(.35), roughness:.45}));
pieza.castShadow = true; pieza.position.copy(v(D.target)); scene.add(pieza);

// ── brazo ──
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
const J = [0,1,2,3].map(() => new THREE.Vector3());
let latAct = new THREE.Vector3(), ejeAct = new THREE.Vector3(0,1,0), apAct = 1;
function setSeg(m, a, b) {
  dir.subVectors(b, a); const L = Math.max(dir.length(), 1e-5);
  m.position.copy(a).addScaledVector(dir, .5);
  q.setFromUnitVectors(up, dir.normalize());
  m.quaternion.copy(q); m.scale.set(1, L, 1);
}
function dibujaBrazo() {
  setSeg(segs[0], J[0], J[1]); setSeg(segs[1], J[1], J[2]); setSeg(segs[2], J[2], J[3]);
  J.forEach((p, k) => joints[k].position.copy(p));
  ejeAct.copy(J[3]).sub(J[2]).normalize();
  const ap = .046 + apAct * .032;
  dedos.forEach((d, k) => {
    d.position.copy(J[3]).addScaledVector(latAct, k ? -ap : ap).addScaledVector(ejeAct, .055);
    q.setFromUnitVectors(up, ejeAct); d.quaternion.copy(q);
  });
}
function poseDe(frames, i, f) {
  const A = frames[i], B = frames[Math.min(i+1, frames.length-1)];
  for (let k = 0; k < 4; k++) J[k].copy(v(A.j[k])).lerp(v(B.j[k]), f);
  latAct.copy(v(A.lat)).lerp(v(B.lat), f);
  apAct = A.ap + (B.ap - A.ap) * f;
  dibujaBrazo();
}

// ── cámaras conmutables ──
const centro = v(D.look);
const dGen = v(D.cam).sub(centro);
const CAMS = {
  cG: {pos: () => centro.clone().addScaledVector(dGen, 1.22).add(new THREE.Vector3(0,.5,0)), look: () => centro},
  cP: {pos: () => centro.clone().add(dGen), look: () => centro},
  cC: {pos: () => centro.clone().add(new THREE.Vector3(.01, 2.8, .01)), look: () => centro},
  cS: {pos: () => J[3].clone().addScaledVector(dGen.clone().normalize(), .85).add(new THREE.Vector3(0,.32,0)),
       look: () => J[3].clone()},
};
let camMode = 'cG', camTween = 0;
for (const id of Object.keys(CAMS)) {
  document.getElementById(id).onclick = () => {
    camMode = id; camTween = 0;
    document.querySelectorAll('#cams button').forEach(b => b.classList.toggle('on', b.id === id));
  };
}

// ── máquina de estados del ciclo ──
const Q = new URLSearchParams(location.search || location.hash.slice(1));
const SKIP = Q.get('skip') === '1', VEL = parseFloat(Q.get('vel') || '1');
const AUTOPERTURB = Q.get('perturb') === '1';
if (Q.get('wait') === '1') { const im = new Image(); im.src = 'http://10.255.255.1/x'; }
const DUR = SKIP ? {ver:60, nube:90, elegir:60} : {ver:1400, nube:1800, elegir:1200};
const FPS = 26;
const fase = document.getElementById('fase');
const chips = [0,1,2,3,4].map(i => document.getElementById('f' + i));
const btnPerturbar = document.getElementById('perturbar');
function marca(i, txt) {
  chips.forEach((c, k) => c.classList.toggle('on', k === i));
  fase.textContent = txt;
}
const sueloBandeja = .02 + .0175 + D.mitadPieza;
let estado = 'ver', tEstado = performance.now(), fIdx = 0, cae = null;
let perturbado = false, slide0 = null, jSnap = null, latSnap = null;
function pasaA(e) { estado = e; tEstado = performance.now(); }

btnPerturbar.onclick = () => {
  if (estado === 'exec' && fIdx < D.nPre - 8 && !perturbado) {
    perturbado = true; slide0 = pieza.position.clone();
    jSnap = J.map(p => p.clone()); latSnap = latAct.clone();
    btnPerturbar.classList.remove('armed');
    pasaA('replan');
  }
};

function reset() {
  cae = null; perturbado = false; fIdx = 0;
  pieza.position.copy(v(D.target));
  nube.forEach(l => l.geometry.setDrawRange(0, 0));
  nube2.forEach(l => { l.geometry.setDrawRange(0, 0); l.material.opacity = .5; });
  trailG.setDrawRange(0, 0); trail2G.setDrawRange(0, 0);
  anillo.position.set(D.target[0], D.target[1]+.04, D.target[2]);
}

function fisicaCaida(dt) {
  if (!cae.reposo) {
    cae.v.y -= 3.2 * dt; cae.p.addScaledVector(cae.v, dt * 3.2);
    if (cae.p.y <= sueloBandeja) {
      cae.p.y = sueloBandeja;
      if (Math.abs(cae.v.y) > .25) cae.v.y = -cae.v.y * .35;
      else { cae.v.set(0, 0, 0); cae.reposo = true; }
    }
  }
  pieza.position.copy(cae.p);
}

let prev = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - prev) / 1000, .05); prev = now;
  const te = now - tEstado;
  btnPerturbar.classList.toggle('armed',
    estado === 'exec' && fIdx < D.nPre - 8 && !perturbado);

  if (estado === 'ver') {
    marca(0, 'el robot VE la pieza (pose 6-DoF)');
    const p = .5 + .5 * Math.sin(te / 140);
    anillo.material.opacity = .35 + .55 * p;
    anillo.scale.setScalar(1 + .25 * p);
    pieza.material.emissiveIntensity = 1 + 1.6 * p;
    poseDe(D.frames, 0, 0);
    if (te > DUR.ver) pasaA('nube');
  } else if (estado === 'nube') {
    marca(1, D.cloud.length + ' caminos imaginados por difusión');
    anillo.material.opacity = .15; anillo.scale.setScalar(1);
    pieza.material.emissiveIntensity = 1;
    const u = te / DUR.nube;
    nube.forEach((l, k) => {
      const uk = Math.min(Math.max(u * 1.6 - k * .05, 0), 1);
      l.geometry.setDrawRange(0, Math.floor(uk * D.cloud[k].length));
    });
    if (te > DUR.nube) pasaA('elegir');
  } else if (estado === 'elegir') {
    marca(2, 'elige el de mejor agarre (best-of-N)');
    const u = Math.min(te / DUR.elegir, 1);
    nube.forEach(l => { l.material.opacity = .45 - .33 * u; });
    trailG.setDrawRange(0, D.trail.length);
    trailM.opacity = .4 + .6 * Math.abs(Math.sin(u * 6));
    if (te > DUR.elegir) { trailM.opacity = .95; trailG.setDrawRange(0, 2); pasaA('exec'); }
  } else if (estado === 'exec') {
    if (AUTOPERTURB && fIdx > 22 && !perturbado) btnPerturbar.onclick();
    fIdx = Math.min(fIdx + dt * FPS * VEL, D.frames.length - 1);
    const i = Math.floor(fIdx);
    poseDe(D.frames, i, fIdx - i);
    trailG.setDrawRange(0, Math.max(2, Math.floor(fIdx / D.frames.length * D.trail.length)));
    if (i >= D.releaseIdx && !cae)
      cae = {p: pieza.position.clone(), v: new THREE.Vector3(0,0,0), reposo: false};
    if (cae) { fisicaCaida(dt); marca(4, 'pieza depositada — caída con gravedad'); }
    else if (i >= D.nPre) {
      pieza.position.copy(J[3]).addScaledVector(ejeAct, .095);
      marca(i < D.liftIdx ? 3 : 4, i < D.liftIdx ? 'la levanta' : 'llevándola a la bandeja');
    } else marca(3, i > D.nPre - 12 ? 'cerrando la pinza…' : 'ejecutando el camino elegido');
    if (fIdx >= D.frames.length - 1) pasaA('hold');
  } else if (estado === 'replan') {                      // ⚡ perturbación
    const u = Math.min(te / 500, 1);                     // la pieza se desliza
    pieza.position.lerpVectors(slide0, v(D.target2), u);
    pieza.position.y = D.target[1] + .06 * Math.sin(Math.PI * u);
    anillo.position.set(pieza.position.x, D.target[1]+.04, pieza.position.z);
    anillo.material.opacity = .7; anillo.scale.setScalar(1.1);
    nube.forEach(l => { l.material.opacity = Math.max(.02, .12 - u * .1); });
    if (te > 500) {                                      // nueva nube de difusión
      const u2 = Math.min((te - 500) / 1100, 1);
      nube2.forEach((l, k) => {
        const uk = Math.min(Math.max(u2 * 1.6 - k * .05, 0), 1);
        l.geometry.setDrawRange(0, Math.floor(uk * D.cloud2[k].length));
      });
      trail2G.setDrawRange(0, D.trail2.length);
      trail2M.opacity = .4 + .6 * Math.abs(Math.sin(u2 * 6));
    }
    marca(2, '⚡ ¡la pieza se movió! re-planificando…');
    if (te > 1700) { fIdx = 0; trail2M.opacity = .95; trail2G.setDrawRange(0, 2); pasaA('puente'); }
  } else if (estado === 'puente') {                      // transición suave al plan 2
    const u = Math.min(te / 450, 1), s = u * u * (3 - 2 * u);
    const F0 = D.frames2[0];
    for (let k = 0; k < 4; k++) J[k].copy(jSnap[k]).lerp(v(F0.j[k]), s);
    latAct.copy(latSnap).lerp(v(F0.lat), s); apAct = 1;
    dibujaBrazo();
    nube2.forEach(l => { l.material.opacity = .45 - .3 * u; });
    marca(3, 'nuevo plan listo — ejecutando');
    if (te > 450) { fIdx = 0; pasaA('exec2'); }
  } else if (estado === 'exec2') {
    fIdx = Math.min(fIdx + dt * FPS * VEL, D.frames2.length - 1);
    const i = Math.floor(fIdx);
    poseDe(D.frames2, i, fIdx - i);
    trail2G.setDrawRange(0, Math.max(2, Math.floor(fIdx / D.frames2.length * D.trail2.length)));
    if (i >= D.releaseIdx2 && !cae)
      cae = {p: pieza.position.clone(), v: new THREE.Vector3(0,0,0), reposo: false};
    if (cae) { fisicaCaida(dt); marca(4, 'pieza depositada — caída con gravedad'); }
    else if (i >= D.nPre2) {
      pieza.position.copy(J[3]).addScaledVector(ejeAct, .095);
      marca(i < D.liftIdx2 ? 3 : 4, i < D.liftIdx2 ? 'la levanta (pese a la perturbación)'
                                                   : 'llevándola a la bandeja');
    } else {
      pieza.position.copy(v(D.target2));
      marca(3, i > D.nPre2 - 12 ? 'cerrando la pinza…' : 'ejecutando el plan re-calculado');
    }
    if (fIdx >= D.frames2.length - 1) pasaA('hold');
  } else if (estado === 'hold') {
    marca(4, perturbado ? 'ciclo completo ✓ — superó la perturbación' : 'ciclo completo ✓ — se repite');
    if (cae) fisicaCaida(dt);
    if (te > 2200) { reset(); pasaA('ver'); }
  }

  if (camTween < 1) camTween = Math.min(camTween + .02, 1);
  const C = CAMS[camMode];
  if (camMode === 'cS' || camTween < 1) {
    cam.position.lerp(C.pos(), camMode === 'cS' ? .08 : .06);
    ctl.target.lerp(C.look(), camMode === 'cS' ? .12 : .06);
  }
  ctl.update(); ren.render(scene, cam);
}
requestAnimationFrame(animate);
addEventListener('resize', () => {
  cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix();
  ren.setSize(innerWidth, innerHeight);
});
</script></body></html>"""


FORMAS_WEB = {"cubo": 0, "esfera": 1, "cilindro": 2}
COLORES_WEB = {"amarillo": "#FFD166", "rojo": "#EF6461", "verde": "#39C77F", "azul": "#5AA9E6"}
# mitad de la altura de la pieza por forma (para la física de la caída)
MITAD_PIEZA = {"cubo": 0.036, "esfera": 0.045, "cilindro": 0.038}


def _visor_3d(trajs, x, y, z, forma="cubo", color="amarillo"):
    """Visor three.js: ciclo VER → IMAGINAR → ELEGIR → EJECUTAR → DEPOSITAR,
    con perturbación interactiva (re-planificación con una segunda nube real)."""
    target = np.array([x, y, z])
    best = trajs[np.argmin(np.linalg.norm(trajs[:, -1, :3] - target, axis=1))]

    dir_xy = target[:2] if np.linalg.norm(target[:2]) > 0.05 else np.array([1.0, 0.0])
    dir_xy = dir_xy / np.linalg.norm(dir_xy)
    base_xy = target[:2] - dir_xy * 0.75
    perp = np.array([-dir_xy[1], dir_xy[0]])
    tray = np.array([*(base_xy + perp * 0.62), 0.0])
    inicio = np.array([*(base_xy + dir_xy * 0.30), 0.70])

    frames, trail, n_pre, lift_idx, release_idx = _ciclo(target, best[:, :3], base_xy, tray, inicio)

    # objetivo perturbado (sobre la misma mesa) + re-plan REAL con la difusión
    target2 = target + np.array([*(perp * 0.13 - dir_xy * 0.05), 0.0])
    trajs2 = _muestrear(target2, 12)
    best2 = trajs2[np.argmin(np.linalg.norm(trajs2[:, -1, :3] - target2, axis=1))]
    frames2, trail2, n_pre2, lift_idx2, release_idx2 = _ciclo(
        target2, best2[:, :3], base_xy, tray, inicio)

    centro = np.array([*((base_xy + target[:2] + tray[:2]) / 3), 0.42])
    cam_pos = np.array([*(centro[:2] + perp * 1.7 - dir_xy * 0.35), 1.00])

    data = {
        "frames": frames, "trail": trail,
        "nPre": n_pre, "liftIdx": lift_idx, "releaseIdx": release_idx,
        "frames2": frames2, "trail2": trail2,
        "nPre2": n_pre2, "liftIdx2": lift_idx2, "releaseIdx2": release_idx2,
        "cloud": [[_a3(pt) for pt in tr] for tr in trajs[:14, :, :3].tolist()],
        "cloud2": [[_a3(pt) for pt in tr] for tr in trajs2[:, :, :3].tolist()],
        "target": _a3(target), "target2": _a3(target2), "tray": _a3(tray),
        "cam": _a3(cam_pos), "look": _a3(centro),
        "forma": FORMAS_WEB.get(forma, 0),
        "colorPieza": COLORES_WEB.get(color, "#FFD166"),
        "mitadPieza": MITAD_PIEZA.get(forma, 0.036),
    }
    pagina = (_PLANTILLA_VISOR
              .replace("__THREE__", f"/gradio_api/file={ASSETS}/three.module.min.js")
              .replace("__ORBIT__", f"/gradio_api/file={ASSETS}/OrbitControls.js")
              .replace("__DATA__", json.dumps(data)))
    return (f'<iframe srcdoc="{html_mod.escape(pagina, quote=True)}" '
            'style="width:100%;height:660px;border:none;border-radius:12px;'
            'background:#0F2A43"></iframe>')


def generar(x, y, z, n, forma="cubo", color="amarillo"):
    t0 = time.time()
    trajs = _muestrear([x, y, z], n)
    ms = (time.time() - t0) * 1000

    fig = _fig_nube(trajs, x, y, z, n, ms)
    visor = _visor_3d(trajs, x, y, z, forma, color)

    spread = np.std(trajs[:, -1, :3], axis=0).mean() * 100
    resumen = (f"**{int(n)} trayectorias** en **{ms:.0f} ms** ({ms / n:.0f} ms c/u) · "
               f"los finales coinciden dentro de **{spread:.1f} cm** — diversidad con propósito. "
               "En el visor 3D: prueba **🫳 perturbar la pieza** durante la ejecución.")
    return fig, visor, resumen


# ──────────────── ejecución real en CoppeliaSim (Iter 7c) ────────────────
def _coppelia_disponible() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 23000), timeout=1.0):
            return True
    except OSError:
        return False


# Cada política con el encoder de SU entrenamiento: la cabeza de proyección se
# inicializa al azar en precompute_visual_cond, los embeddings no son
# intercambiables entre iteraciones.
POLITICAS_SIM = {
    "original (Iter 7c — solo cubos rojos)":
        ("diffusion_policy_v7a_phase2", "visual_encoder_iter5"),
    "robusta (Iter 8 — formas y colores variados)":
        ("diffusion_policy_v8_randomized", "visual_encoder_iter8rand"),
    "clutter (Iter 9 — elige entre varias piezas)":
        ("diffusion_policy_v9_clutter", "visual_encoder_iter9clut"),
}
POLITICA_ORIGINAL, POLITICA_ROBUSTA, POLITICA_CLUTTER = tuple(POLITICAS_SIM)


def _load_sim_stack(politica: str):
    clave = f"sim_{politica}"
    if clave in _cache:
        return _cache[clave]
    import torch
    from src.planning.diffusion_policy import DiffusionGraspPlanner
    from src.planning.visual_encoder import ResNet18RGBDEncoder

    nombre_pol, nombre_enc = POLITICAS_SIM[politica]
    path_pol = REPO / f"data/models/{nombre_pol}.pth"
    if not path_pol.exists():
        raise FileNotFoundError(
            f"{path_pol.name} no está (es regenerable — ver Iter 8 en "
            "docs/INTEGRATION_PIPELINE.md); usa la política original")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(path_pol, map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    enc_state = torch.load(REPO / f"data/models/{nombre_enc}.pth",
                           map_location=device, weights_only=True)
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])
    _cache[clave] = (planner, encoder)
    return _cache[clave]


COLORES_SIM = {
    "rojo": (0.85, 0.15, 0.15), "verde": (0.20, 0.75, 0.20),
    "azul": (0.15, 0.30, 0.85), "amarillo": (0.95, 0.78, 0.18),
}


def _preparar_pieza_sim(sim, forma: str, color: str):
    """Deja /object_1 con la forma y color pedidos (con la política original,
    otras formas/colores son un test de robustez; la robusta los maneja)."""
    h = sim.getObject("/object_1")
    if forma != "cubo":
        # estacionar el cubo original y crear una primitiva dinámica en su lugar
        sim.setObjectAlias(h, "object_1_off")
        sim.setObjectPosition(h, -1, [-1.0, -1.0, -1.0])
        tipo = sim.primitiveshape_spheroid if forma == "esfera" else sim.primitiveshape_cylinder
        try:
            h = sim.createPrimitiveShape(tipo, [0.05, 0.05, 0.05], 0)
        except Exception:                      # API antigua (<4.5)
            h = sim.createPureShape(1 if forma == "esfera" else 2, 8, [0.05, 0.05, 0.05], 0.05)
        sim.setObjectAlias(h, "object_1")
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        try:
            sim.setShapeMass(h, 0.05)
        except Exception:
            pass
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(COLORES_SIM[color]))


FORMAS_SIM = ("cubo", "esfera", "cilindro")


def _crear_distractores_sim(sim, n: int, target_xy, target_app, rng):
    """Crea n piezas extra de apariencia ≠ a la pedida, separadas ≥9 cm del
    objetivo y entre sí. El robot debe ir por la pieza indicada (su pose),
    ignorando estas — el test de selección en clutter."""
    puestos = [tuple(target_xy)]
    creados = []
    while len(creados) < n:
        x = float(rng.uniform(0.38, 0.58))
        y = float(rng.uniform(-0.19, -0.01))
        if any((x - px) ** 2 + (y - py) ** 2 < 0.09**2 for px, py in puestos):
            continue
        while True:
            forma = FORMAS_SIM[int(rng.integers(0, 3))]
            color = list(COLORES_SIM)[int(rng.integers(0, len(COLORES_SIM)))]
            if (forma, color) != tuple(target_app):
                break
        try:
            tipo = {"cubo": sim.primitiveshape_cuboid,
                    "esfera": sim.primitiveshape_spheroid,
                    "cilindro": sim.primitiveshape_cylinder}[forma]
            h = sim.createPrimitiveShape(tipo, [0.05, 0.05, 0.05], 0)
        except Exception:                      # API antigua (<4.5)
            h = sim.createPureShape({"cubo": 0, "esfera": 1, "cilindro": 2}[forma],
                                    8, [0.05, 0.05, 0.05], 0.05)
        sim.setObjectAlias(h, "distractor")
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        try:
            sim.setShapeMass(h, 0.05)
        except Exception:
            pass
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse,
                          list(COLORES_SIM[color]))
        sim.setObjectPosition(h, -1, [x, y, 0.033])
        puestos.append((x, y))
        creados.append(f"{forma} {color}")
    return creados


def chequear_conexion():
    if _coppelia_disponible():
        return "🟢 **CoppeliaSim detectado** (puerto 23000) — listo para ejecutar."
    return ("🔴 **CoppeliaSim no detectado.** Abre la app CoppeliaSim y vuelve a "
            "entrar a esta pestaña.")


def ejecutar_en_sim(sx, sy, rot_deg, forma, color, politica, n_dist):
    """Generador: va informando el progreso mientras el UR ejecuta el pick."""
    if not _coppelia_disponible():
        yield ("❌ **CoppeliaSim no está corriendo.** Ábrelo primero "
               "(app CoppeliaSim, puerto ZMQ 23000) y vuelve a intentar.")
        return
    if not _sim_lock.acquire(blocking=False):
        yield "⏳ Ya hay una ejecución en curso — espera a que termine."
        return
    try:
        robusta = politica != POLITICA_ORIGINAL
        yield f"🔌 Conectando con CoppeliaSim y cargando la política {politica}..."
        from experiments.run_pick_with_diffusion import pick_with_dp
        from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

        planner, encoder = _load_sim_stack(politica)
        theta = math.radians(float(rot_deg))
        c, s = math.cos(theta), math.sin(theta)
        pose = np.eye(4)
        pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pose[:3, 3] = [float(sx), float(sy), 0.033]

        if (forma, color) == ("cubo", "rojo"):
            nota = ""
        elif robusta:
            nota = ("\n\n_Esta política se entrenó con formas y colores variados "
                    "(domain randomization) — debería manejar esta pieza._")
        else:
            nota = ("\n\n_Nota: esta política se entrenó solo con cubos rojos — esta "
                    "forma/color es un test de robustez del encoder visual._")
        n_dist = int(n_dist)
        objetivo = (f"la **{forma} {color}**" if n_dist
                    else f"{forma} {color}")
        clutter_txt = (f" La mesa tendrá además **{n_dist} pieza(s) distractora(s)** — "
                       "el robot debe ir SOLO por la indicada." if n_dist else "")
        if n_dist and politica != POLITICA_CLUTTER:
            clutter_txt += (" _Esta política no se entrenó con varias piezas en escena "
                            "— la de clutter (Iter 9) es la especialista._")
        yield (f"🤖 **Ejecutando en el simulador** — {objetivo} en x={sx:.2f}, y={sy:.2f}, "
               f"rotación {int(float(rot_deg))}°.{clutter_txt}{nota}\n\n"
               "👀 **Miren la ventana de CoppeliaSim**: el robot ve la escena, genera "
               "8 trayectorias, ejecuta la mejor y la deposita (~1 min).")
        t0 = time.time()
        with CoppeliaSimBridge() as bridge:
            # robustez: si quedó una simulación corriendo (run interrumpido o
            # play manual en la GUI), detenerla y esperar el estado 'stopped'
            if bridge.get_simulation_state() != 0:
                bridge.stop_simulation()
                for _ in range(60):
                    if bridge.get_simulation_state() == 0:
                        break
                    time.sleep(0.1)
                else:
                    yield ("⚠️ El simulador no terminó de detenerse. Pulsa el botón "
                           "■ (stop) en CoppeliaSim y vuelve a intentar.")
                    return
            bridge.load_scene(REPO / "data/scenes/bin_base.ttt")
            _preparar_pieza_sim(bridge.sim, forma, color)
            distractores = []
            if n_dist:
                distractores = _crear_distractores_sim(
                    bridge.sim, n_dist, (float(sx), float(sy)), (forma, color),
                    np.random.default_rng())
            r = pick_with_dp(planner, pose, bridge, frames_dir=None,
                             visual_encoder=encoder, best_of_n=8)
        dt = time.time() - t0
        ok = r["grasp_plausible"] and r["deposit_plausible"] and r["ik_converged"]
        ood = (forma, color) != ("cubo", "rojo")
        metricas = (f"- Precisión del agarre: **{r['grasp_proximity_m']*100:.1f} cm** de la pieza\n"
                    f"- Depósito a **{r['deposit_error_m']*100:.1f} cm** del objetivo\n"
                    f"- Brazo (IK): {'convergió ✓' if r['ik_converged'] else 'no convergió'}\n")
        if distractores:
            metricas += f"- En mesa también había: {', '.join(distractores)}\n"
        metricas += "\n"
        if ok:
            extra = (" Esta política se entrenó con apariencias randomizadas — agarra "
                     "lo que se le pida." if (ood and robusta) else "")
            sel = (f" **Fue por la pieza indicada ({forma} {color})** entre "
                   f"{1 + len(distractores)} piezas." if distractores else "")
            yield (f"✅ **Pick completado** en {dt:.0f} s\n\n" + metricas +
                   "_Mismo pipeline del estudio: percepción → difusión best-of-8 → control._"
                   + sel + extra)
        elif ood and robusta:
            rodo = r["grasp_plausible"] and forma in ("esfera", "cilindro")
            detalle = ("**La agarró bien** pero la pieza **rodó al depositarla** — limitación "
                       "física del depósito plano (las esferas y cilindros ruedan), no de la "
                       "percepción. En la eval pareada, la precisión de agarre de esta política "
                       "es ~1.4 cm con cualquier forma y color." if rodo else
                       "Puede pasar; reintenta con otra posición. En la eval pareada esta "
                       "política agarra el 96 % de las piezas randomizadas (~1.4 cm).")
            yield (f"🔬 **Resultado** ({dt:.0f} s):\n\n" + metricas + detalle)
        elif ood:
            yield (f"🔬 **Resultado del experimento de robustez** ({dt:.0f} s): el sistema se degradó.\n\n"
                   + metricas +
                   "**La lección de IA**: esta política se entrenó SOLO con cubos rojos — con una pieza "
                   "fuera de esa distribución, la percepción visual pierde precisión y el agarre falla "
                   "o la pieza sale empujada (física real). **El antes/después**: cambia a la política "
                   "**robusta (Iter 8)** y repite — esa se entrenó con formas y colores variados. "
                   "_Así de importante es la distribución de entrenamiento._")
        else:
            yield (f"⚠️ **Pick terminado con observaciones** en {dt:.0f} s\n\n" + metricas +
                   "_Puede pasar (~16 % de los casos según la eval); reintenta con otra posición._")
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
            with gr.Row():
                sforma = gr.Radio(["cubo", "esfera", "cilindro"], value="cubo", label="forma de la pieza")
                scolor = gr.Radio(["amarillo", "rojo", "verde", "azul"], value="amarillo", label="color")
            btn = gr.Button("✨ Generar trayectorias", variant="primary")
            with gr.Row():
                botones = [gr.Button(p, size="sm") for p in PRESETS]
            resumen = gr.Markdown()

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🎬 El robot en acción (3D interactivo)"):
                    visor = gr.HTML()
                with gr.Tab("🌀 Los caminos posibles"):
                    plot = gr.Plot(label="", container=False)
                with gr.Tab("🤖 CoppeliaSim — ejecución real") as tab_sim:
                    estado_con = gr.Markdown()
                    gr.Markdown("**El pick de verdad**: la pieza se coloca sobre la mesa del "
                                "simulador donde ustedes digan, y el brazo UR ejecuta el pipeline "
                                "completo (percepción → difusión best-of-8 → control). "
                                "👀 *Miren la ventana de CoppeliaSim mientras corre (~35 s).*")
                    with gr.Row():
                        cx = gr.Slider(0.40, 0.55, value=0.47, step=0.01,
                                       label="posición — cerca / lejos (m)")
                        cy = gr.Slider(-0.15, -0.05, value=-0.10, step=0.01,
                                       label="posición — izquierda / derecha (m)")
                    with gr.Row():
                        crot = gr.Radio(["0", "45", "90"], value="0", label="rotación (°)")
                        cforma = gr.Radio(["cubo", "esfera", "cilindro"], value="cubo",
                                          label="forma 🔬 (≠cubo = test de robustez)")
                        ccolor = gr.Radio(["rojo", "verde", "azul", "amarillo"], value="rojo",
                                          label="color 🔬 (≠rojo = test de robustez)")
                    _pols_disp = [p for p, (m, _) in POLITICAS_SIM.items()
                                  if (REPO / f"data/models/{m}.pth").exists()] \
                        or list(POLITICAS_SIM)
                    with gr.Row():
                        cpol = gr.Radio(_pols_disp, value=_pols_disp[0], scale=3,
                                        label="política 🧠 (antes/después del entrenamiento robusto)")
                        cdist = gr.Radio(["0", "1", "2"], value="0", scale=1,
                                         label="piezas distractoras 🎯 (va solo por la indicada)")
                    btn_sim = gr.Button("🤖 Ejecutar en el simulador", variant="primary")
                    estado_sim = gr.Markdown()

    btn.click(generar, [sx, sy, sz, sn, sforma, scolor], [plot, visor, resumen])
    for b, (nombre, (px, py, pz)) in zip(botones, PRESETS.items()):
        b.click(lambda px=px, py=py, pz=pz: (px, py, pz), outputs=[sx, sy, sz])
    btn_sim.click(ejecutar_en_sim, [cx, cy, crot, cforma, ccolor, cpol, cdist], [estado_sim])
    tab_sim.select(chequear_conexion, outputs=[estado_con])
    # ejemplo al abrir: el público nunca ve un panel vacío
    demo.load(generar, [sx, sy, sz, sn, sforma, scolor], [plot, visor, resumen])

if __name__ == "__main__":
    gr.set_static_paths(paths=[ASSETS])
    print("Cargando modelo (una vez)...")
    _load()
    print("Listo → http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                css=CSS, theme=gr.themes.Soft(primary_hue="cyan"))
