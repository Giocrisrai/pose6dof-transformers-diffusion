# Hero Showcase Reel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Producir `reel_showcase.mp4` (~90-120 s): toma cinematográfica del mejor pick-and-place de Iter 7c (cámara dedicada con coreografía órbita→seguimiento→retroceso) + segmento narrativo de valor/aplicaciones.

**Architecture:** Una cámara `vision sensor` dedicada de 1280×720 creada en runtime (separada de la de percepción, que queda fija para la policy), posicionada por una coreografía basada en el progreso del pick. Un `frame_hook` opcional inyectado en el loop de ejecución captura desde esa cámara. La parte de valor se arma con los overlays cv2 existentes. Todo se concatena con ffmpeg.

**Tech Stack:** CoppeliaSim ZMQ Remote API (`sim.createVisionSensor`, `setExplicitHandling`, `handleVisionSensor`, `getVisionSensorImg`), numpy, OpenCV (cv2, vía `reel_overlay.py`), ffmpeg.

**API validado en vivo (2026-06-08):** `sim.createVisionSensor(0,[1280,720,0,0],[near,far,fov,...])` → handle; requiere `sim.setExplicitHandling(h,1)` antes de `handleVisionSensor`; `getVisionSensorImg(h)` → (bytes, [w,h]); `sim.removeObjects([h])` para limpiar.

---

## File Structure

| Archivo | Responsabilidad |
|---------|-----------------|
| `src/simulation/cine_camera.py` (crear) | Helpers de geometría puros (lerp, orbit_position, look_at_euler, choreograph) + clase `CineCamera` (ciclo de vida del vision sensor + captura) |
| `tests/test_cine_camera.py` (crear) | Tests unitarios de los helpers de geometría (sin sim) |
| `src/simulation/pick_sequence.py` (modificar) | Añadir param opcional `frame_hook` a `_move_tcp_via_ik` (no cambia comportamiento si es None) |
| `experiments/run_pick_with_diffusion.py` (modificar) | Añadir param opcional `frame_hook` a `pick_with_dp`, propagado a `_move_tcp_via_ik` |
| `experiments/make_showcase_reel.py` (crear) | Orquesta: hero pick con captura cine (parte A) + tarjetas de valor (parte B) + concatenación → `reel_showcase.mp4` |

---

## Task 1: Helpers de geometría de cámara (puros, TDD)

**Files:**
- Create: `src/simulation/cine_camera.py`
- Test: `tests/test_cine_camera.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_cine_camera.py
"""Tests para los helpers de geometría de src/simulation/cine_camera.py."""
import math

import numpy as np

from src.simulation.cine_camera import lerp, orbit_position, look_at_euler, choreograph


def test_lerp_endpoints_and_mid():
    assert lerp(0.0, 10.0, 0.0) == 0.0
    assert lerp(0.0, 10.0, 1.0) == 10.0
    assert lerp(0.0, 10.0, 0.5) == 5.0


def test_orbit_position_radius_and_height():
    center = (0.0, 0.0, 0.0)
    p = orbit_position(center, radius=0.8, angle_rad=0.0, height=0.5)
    # angle 0 → +x del centro, altura = z
    assert abs(p[0] - 0.8) < 1e-6
    assert abs(p[1] - 0.0) < 1e-6
    assert abs(p[2] - 0.5) < 1e-6
    # distancia radial en el plano xy se mantiene a 90°
    p90 = orbit_position(center, radius=0.8, angle_rad=math.pi / 2, height=0.5)
    assert abs(math.hypot(p90[0], p90[1]) - 0.8) < 1e-6


def test_look_at_euler_points_down_toward_target():
    # cámara arriba mirando al origen → pitch apunta hacia abajo
    eul = look_at_euler(cam_pos=(0.0, 0.0, 1.0), target=(0.0, 0.0, 0.0))
    assert len(eul) == 3
    assert all(isinstance(v, float) for v in eul)


def test_choreograph_phases_move_camera_closer():
    tcp = (0.45, -0.12, 0.20)
    center = (0.3, -0.3, 0.1)
    cam_far, tgt0 = choreograph(0.0, tcp, center)      # establecimiento (órbita amplia)
    cam_near, tgt1 = choreograph(0.5, tcp, center)     # seguimiento (cerca del TCP)
    cam_pull, tgt2 = choreograph(1.0, tcp, center)     # retroceso
    d_far = math.dist(cam_far, center)
    d_near = math.dist(cam_near, tcp)
    d_pull = math.dist(cam_pull, center)
    # en seguimiento la cámara está más cerca del TCP que en establecimiento
    assert d_near < math.dist(cam_far, tcp)
    # en retroceso vuelve a alejarse respecto al seguimiento
    assert d_pull > d_near
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv/bin/python -m pytest tests/test_cine_camera.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.simulation.cine_camera'`

- [ ] **Step 3: Implementar los helpers**

```python
# src/simulation/cine_camera.py
"""Cámara cinematográfica dedicada para el showcase reel.

Vision sensor de alta resolución, SEPARADO de la cámara de percepción
(/rgb_camera), creado en runtime. La de percepción queda fija para no
corromper el RGB-D que recibe la Diffusion Policy.

Dos partes:
- Helpers de geometría puros (lerp, orbit_position, look_at_euler, choreograph):
  testeables sin simulador.
- Clase CineCamera: ciclo de vida del vision sensor en CoppeliaSim + captura.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

Vec3 = tuple[float, float, float]


def lerp(a: float, b: float, t: float) -> float:
    """Interpolación lineal a→b para t en [0,1]."""
    return a + (b - a) * t


def orbit_position(center: Vec3, radius: float, angle_rad: float, height: float) -> Vec3:
    """Posición sobre un círculo de radio `radius` alrededor de `center` en el
    plano xy, a altura absoluta `height`. angle_rad=0 → dirección +x."""
    return (
        center[0] + radius * math.cos(angle_rad),
        center[1] + radius * math.sin(angle_rad),
        height,
    )


def look_at_euler(cam_pos: Vec3, target: Vec3) -> Vec3:
    """Ángulos de Euler (alpha,beta,gamma, convención XYZ de CoppeliaSim) para
    que una cámara en `cam_pos` mire hacia `target`. El eje -Z de la cámara
    apunta al target."""
    d = np.array(target, dtype=float) - np.array(cam_pos, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    d /= n
    # yaw alrededor de Z, luego pitch. -Z mira al target.
    yaw = math.atan2(d[1], d[0])
    pitch = math.asin(max(-1.0, min(1.0, d[2])))
    # En CoppeliaSim una cámara con orientación (0,0,0) mira hacia -Z (abajo).
    # alpha = -(90°+pitch) inclina desde mirar-abajo hacia el target; gamma=yaw+90°.
    alpha = -(math.pi / 2 + pitch)
    beta = 0.0
    gamma = yaw + math.pi / 2
    return (float(alpha), float(beta), float(gamma))


def choreograph(progress: float, tcp: Vec3, workspace_center: Vec3) -> tuple[Vec3, Vec3]:
    """Devuelve (posición_cámara, target_lookat) para un `progress` en [0,1]
    del pick. Tres fases:
      - [0.00,0.35) establecimiento: órbita amplia mirando al workspace.
      - [0.35,0.70) seguimiento: acercamiento mirando al TCP.
      - [0.70,1.00] retroceso: alejamiento mirando al workspace.
    """
    p = max(0.0, min(1.0, progress))
    if p < 0.35:
        a = p / 0.35
        angle = lerp(math.radians(20), math.radians(80), a)
        pos = orbit_position(workspace_center, radius=lerp(1.0, 0.7, a),
                             angle_rad=angle, height=lerp(0.9, 0.7, a))
        return pos, workspace_center
    if p < 0.70:
        a = (p - 0.35) / 0.35
        angle = lerp(math.radians(80), math.radians(110), a)
        pos = orbit_position(tcp, radius=lerp(0.55, 0.32, a),
                             angle_rad=angle, height=lerp(0.55, 0.35, a))
        return pos, tcp
    a = (p - 0.70) / 0.30
    angle = lerp(math.radians(110), math.radians(150), a)
    pos = orbit_position(workspace_center, radius=lerp(0.5, 1.0, a),
                         angle_rad=angle, height=lerp(0.45, 0.85, a))
    return pos, workspace_center
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `.venv/bin/python -m pytest tests/test_cine_camera.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/simulation/cine_camera.py tests/test_cine_camera.py
git commit -m "feat(cine): helpers de geometría de cámara cinematográfica (orbit/look-at/choreograph)"
```

---

## Task 2: Clase CineCamera (ciclo de vida del vision sensor)

**Files:**
- Modify: `src/simulation/cine_camera.py` (añadir la clase al final)

> No hay test unitario CI (requiere CoppeliaSim). Verificación: smoke manual al final de la task.

- [ ] **Step 1: Implementar la clase CineCamera**

Añadir al final de `src/simulation/cine_camera.py`:

```python
class CineCamera:
    """Vision sensor cinematográfico dedicado en CoppeliaSim.

    Uso:
        cam = CineCamera(bridge, res=(1280, 720))
        cam.create()
        ...  # por frame: cam.aim(progress, tcp, center); cam.capture(frames_dir, idx)
        cam.remove()
    """

    def __init__(self, bridge, res: tuple[int, int] = (1280, 720),
                 fov_deg: float = 60.0):
        self.bridge = bridge
        self.res = res
        self.fov_deg = fov_deg
        self.handle: Optional[int] = None

    def create(self) -> int:
        """Crea el vision sensor con explicit handling. Devuelve el handle."""
        sim = self.bridge.sim
        floatp = [0.01, 10.0, math.radians(self.fov_deg),
                  0.05, 0.05, 0.05, 0, 0, 0, 0, 0]
        self.handle = sim.createVisionSensor(0, [self.res[0], self.res[1], 0, 0], floatp)
        sim.setExplicitHandling(self.handle, 1)
        return self.handle

    def aim(self, progress: float, tcp: Vec3, workspace_center: Vec3) -> None:
        """Posiciona/orienta la cámara según la coreografía para `progress`."""
        sim = self.bridge.sim
        pos, target = choreograph(progress, tcp, workspace_center)
        sim.setObjectPosition(self.handle, -1, list(pos))
        sim.setObjectOrientation(self.handle, -1, list(look_at_euler(pos, target)))

    def capture(self, frames_dir: Path, idx: int) -> None:
        """Renderiza y guarda un PNG del frame actual."""
        from PIL import Image
        sim = self.bridge.sim
        sim.handleVisionSensor(self.handle)
        img_raw, res = sim.getVisionSensorImg(self.handle)
        w, h = res[0], res[1]
        arr = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
        arr = np.flipud(arr)
        frames_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(frames_dir / f"{idx:06d}.png")

    def remove(self) -> None:
        """Elimina el vision sensor de la escena."""
        if self.handle is not None:
            try:
                self.bridge.sim.removeObjects([self.handle])
            except Exception:
                pass
            self.handle = None
```

- [ ] **Step 2: Smoke manual (requiere CoppeliaSim en :23000)**

Run:
```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,'.')
from pathlib import Path
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.cine_camera import CineCamera
with CoppeliaSimBridge() as b:
    b.load_scene("data/scenes/bin_base.ttt")
    b.set_stepping(True); b.start_simulation()
    cam = CineCamera(b); cam.create()
    fd = Path("experiments/results/_cine_smoke")
    for i in range(3):
        b.step()
        cam.aim(i/2, (0.45,-0.12,0.2), (0.3,-0.3,0.1))
        cam.capture(fd, i)
    cam.remove(); b.stop_simulation()
    print("frames:", len(list(fd.glob('*.png'))))
PY
rm -rf experiments/results/_cine_smoke
```
Expected: `frames: 3` (sin excepción).

- [ ] **Step 3: Commit**

```bash
git add src/simulation/cine_camera.py
git commit -m "feat(cine): clase CineCamera (create/aim/capture/remove) con explicit handling"
```

---

## Task 3: frame_hook opcional en el loop de ejecución

**Files:**
- Modify: `src/simulation/pick_sequence.py` (`_move_tcp_via_ik`)
- Modify: `experiments/run_pick_with_diffusion.py` (`pick_with_dp`)

> El cambio es aditivo: con `frame_hook=None` el comportamiento es idéntico al actual (no regresiona el reel técnico). Sin test CI (sim).

- [ ] **Step 1: Añadir `frame_hook` a `_move_tcp_via_ik`**

En `src/simulation/pick_sequence.py`, la firma actual:
```python
def _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                      target_xyz, frames_dir, counter,
                      n_substeps: int = 40, steps_per_substep: int = 3,
                      convergence_tracker: Optional[list] = None) -> None:
```
Cambiar a (añadir `frame_hook` al final):
```python
def _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                      target_xyz, frames_dir, counter,
                      n_substeps: int = 40, steps_per_substep: int = 3,
                      convergence_tracker: Optional[list] = None,
                      frame_hook=None) -> None:
```
Y en el loop interno de captura, la línea actual:
```python
            bridge.step()
            _capture_frame(bridge, frames_dir, counter[0])
            counter[0] += 1
```
cambiar a:
```python
            bridge.step()
            if frame_hook is not None:
                frame_hook()
            else:
                _capture_frame(bridge, frames_dir, counter[0])
            counter[0] += 1
```
(Hay DOS bloques de captura en la función: el del loop de substeps y el de "Settle". Aplicar el mismo cambio a ambos.)

- [ ] **Step 2: Propagar `frame_hook` desde `pick_with_dp`**

En `experiments/run_pick_with_diffusion.py`, añadir `frame_hook=None` a la firma de `pick_with_dp` (al final, junto a `best_of_n`):
```python
    visual_encoder=None,
    best_of_n: int = 1,
    frame_hook=None,
):
```
Y en CADA llamada a `_move_tcp_via_ik(...)` dentro de `pick_with_dp`, pasar `frame_hook=frame_hook` como último argumento.

- [ ] **Step 3: Verificar que no regresiona (smoke, sim)**

Run (pick normal con frames de percepción, frame_hook=None):
```bash
.venv/bin/python -c "import ast; ast.parse(open('src/simulation/pick_sequence.py').read()); ast.parse(open('experiments/run_pick_with_diffusion.py').read()); print('sintaxis OK')"
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -1
```
Expected: `sintaxis OK` y `222 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/simulation/pick_sequence.py experiments/run_pick_with_diffusion.py
git commit -m "feat(sim): frame_hook opcional en pick loop (None preserva comportamiento)"
```

---

## Task 4: Orquestador — parte A (hero pick con captura cine)

**Files:**
- Create: `experiments/make_showcase_reel.py`

- [ ] **Step 1: Escribir el script (parte A)**

```python
# experiments/make_showcase_reel.py
"""Genera reel_showcase.mp4: hero pick cinematográfico (Iter 7c) + valor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.cine_camera import CineCamera
from experiments.run_pick_with_diffusion import pick_with_dp, compile_mp4
from experiments.eval_diffusion_iter2_sim import sample_pose_eval, EVAL_SEED

SCENE = REPO / "data/scenes/bin_base.ttt"
OUT = REPO / "experiments/results/demo_reel"
WORKSPACE_CENTER = (0.0, -0.30, 0.10)  # centro aproximado bin/deposit
POSE_INDEX = 49
TORCH_SEED = 3
TOTAL_FRAMES_ESTIM = 16 * 8 + 30  # 16 waypoints × n_substeps(8) + settle


def run_hero_pick(frames_dir: Path) -> dict:
    torch.manual_seed(TORCH_SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(REPO / "data/models/diffusion_policy_v7a_phase2.pth",
                      map_location=device, weights_only=True)
    planner = DiffusionGraspPlanner(action_dim=7, horizon=16, n_diffusion_steps=100,
                                    device=device, hidden_dim=ckpt["config"]["hidden_dim"])
    planner.model.load_state_dict(ckpt["model_state_dict"]); planner.model.eval()
    es = torch.load(REPO / "data/models/visual_encoder_iter5.pth",
                    map_location=device, weights_only=True)
    enc = ResNet18RGBDEncoder(out_dim=es["out_dim"]).to(device).eval()
    enc.load_state_dict(es["state_dict"])

    rng = np.random.default_rng(EVAL_SEED)
    pose = None
    for _ in range(POSE_INDEX + 1):
        pose = sample_pose_eval(rng)

    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        cam = CineCamera(bridge); cam.create()
        tip = bridge.sim.getObject("/tip")
        state = {"i": 0}

        def hook():
            tcp = tuple(bridge.sim.getObjectPosition(tip, -1))
            progress = min(1.0, state["i"] / TOTAL_FRAMES_ESTIM)
            cam.aim(progress, tcp, WORKSPACE_CENTER)
            cam.capture(frames_dir, state["i"])
            state["i"] += 1

        result = pick_with_dp(planner, pose, bridge, frames_dir=None,
                              visual_encoder=enc, best_of_n=8, frame_hook=hook)
        cam.remove()
    result["frames_captured"] = state["i"]
    return result
```

- [ ] **Step 2: Smoke (sim) — verificar parte A**

Run:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
import experiments.make_showcase_reel as m
fd = m.OUT / 'frames_showcase'
import shutil; shutil.rmtree(fd, ignore_errors=True)
r = m.run_hero_pick(fd)
print('frames:', r['frames_captured'], 'grasp_plausible:', r['grasp_plausible'], 'deposit_plausible:', r['deposit_plausible'], 'ik:', r['ik_converged'])
assert r['frames_captured'] > 100, 'pocos frames'
"
```
Expected: `frames: ~150+`, `grasp_plausible: True`, `deposit_plausible: True`, `ik: True`. Si grasp/deposit no son True, ajustar TORCH_SEED (probar 3, 11, 2026) hasta una corrida limpia (la coreografía no afecta el pick, solo la cámara).

- [ ] **Step 3: Commit**

```bash
git add experiments/make_showcase_reel.py
git commit -m "feat(reel): orquestador showcase parte A — hero pick con captura cine"
```

---

## Task 5: Parte B (tarjetas de valor) + ensamblaje final

**Files:**
- Modify: `experiments/make_showcase_reel.py` (añadir parte B + main)

- [ ] **Step 1: Añadir tarjetas de valor y `main()`**

Añadir a `experiments/make_showcase_reel.py`:

```python
from src.simulation.reel_overlay import make_title_card, draw_honesty_tag

# Cada tupla es (texto, escala_de_fuente) — la escala la consume make_title_card
# (NO es duración). La duración por tarjeta es fija (SECONDS_PER_CARD).
VALUE_CARDS = [
    [("Sistema bin-picking 6-DoF", 1.3),
     ("FoundationPose  +  Diffusion Policy", 0.7)],
    [("Lo que se logro", 1.1),
     ("pick-and-place E2E 84%", 0.8),
     ("IK 100%  -  ciclo p95 < 10 s", 0.7)],
    [("Por que es mejor", 1.1),
     ("corre en hardware accesible (~USD 1.920)", 0.7),
     ("vs setups industriales USD 15k-150k", 0.7)],
    [("Aplicaciones", 1.1),
     ("logistica / e-commerce  -  manufactura  -  clasificacion", 0.65)],
    [("Honestidad declarada", 1.0),
     ("grasp por snap+attach (estandar en sims)", 0.7),
     ("siguiente: grasp fisico + robot real", 0.65)],
]
SECONDS_PER_CARD = 4.0


def build_value_clip(frames_dir: Path, fps: int = 30) -> int:
    """Genera frames de las tarjetas de valor (SECONDS_PER_CARD c/u).
    Devuelve nº de frames escritos."""
    from PIL import Image
    frames_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    n_per_card = int(round(SECONDS_PER_CARD * fps))
    for lines in VALUE_CARDS:
        card = make_title_card(lines)  # (H,W,3) uint8; el float de cada línea = escala
        for _ in range(n_per_card):
            Image.fromarray(card).save(frames_dir / f"{idx:06d}.png")
            idx += 1
    return idx


def main() -> int:
    fps = 30
    frames_a = OUT / "frames_showcase"
    import shutil
    shutil.rmtree(frames_a, ignore_errors=True)
    result = run_hero_pick(frames_a)
    assert result["grasp_plausible"] and result["deposit_plausible"] and result["ik_converged"], \
        f"pick no limpio: {result}"

    # parte A → mp4
    part_a = OUT / "_showcase_partA.mp4"
    compile_mp4(frames_a, part_a, fps=fps)
    # parte B → mp4
    frames_b = OUT / "frames_value"
    shutil.rmtree(frames_b, ignore_errors=True)
    build_value_clip(frames_b, fps=fps)
    part_b = OUT / "_showcase_partB.mp4"
    compile_mp4(frames_b, part_b, fps=fps)

    # concatenar A + B con ffmpeg
    import subprocess
    listf = OUT / "_showcase_concat.txt"
    listf.write_text(f"file '{part_a.resolve()}'\nfile '{part_b.resolve()}'\n")
    final = OUT / "reel_showcase.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(final)],
                   check=True, capture_output=True)
    # limpieza de intermedios
    for p in (part_a, part_b, listf):
        p.unlink(missing_ok=True)
    print(f"reel_showcase.mp4 listo: {final}")
    print(f"  hero frames={result['frames_captured']} grasp={result['grasp_proximity_m']*100:.1f}cm "
          f"deposit={result['deposit_error_m']*100:.1f}cm ik={result['ik_converged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> Confirmado (reel_overlay.py:107): `make_title_card(lines, w=1280, h=720, accent) -> np.ndarray` de shape `(720,1280,3)` uint8; el float de cada `(texto, escala)` es la escala de fuente. Por eso `build_value_clip` usa `SECONDS_PER_CARD` para la duración, no el float de la tupla.

- [ ] **Step 2: Generar el reel completo (sim)**

Run:
```bash
.venv/bin/python experiments/make_showcase_reel.py
```
Expected: `reel_showcase.mp4 listo: .../reel_showcase.mp4`, con grasp/deposit/ik plausibles.

- [ ] **Step 3: Verificar el video**

Run:
```bash
.venv/bin/python -c "import cv2; c=cv2.VideoCapture('experiments/results/demo_reel/reel_showcase.mp4'); n=c.get(7); fps=c.get(5); print(f'{n/fps:.1f}s, {int(n)} frames @ {fps:.0f}fps, {c.get(3):.0f}x{c.get(4):.0f}')"
```
Expected: duración ~90-120 s, resolución 1280x720.

- [ ] **Step 4: Commit**

```bash
git add experiments/make_showcase_reel.py
git commit -m "feat(reel): showcase parte B (valor/aplicaciones) + ensamblaje reel_showcase.mp4"
```

---

## Task 6: gitignore + docs + cierre

**Files:**
- Modify: `.gitignore`
- Modify: `experiments/results/demo_reel/README.md`

- [ ] **Step 1: Ignorar intermedios y el reel showcase (regenerable)**

Verificar que `experiments/results/demo_reel/` ya está gitignored (lo está, salvo README.md). `reel_showcase.mp4` y `frames_showcase/`, `frames_value/` quedan cubiertos. Añadir a `.gitignore` si hiciera falta una entrada explícita para los intermedios `_showcase_*`. Confirmar con:
```bash
git check-ignore experiments/results/demo_reel/reel_showcase.mp4
```
Expected: imprime la ruta (está ignorada).

- [ ] **Step 2: Documentar el showcase en el README del reel**

Añadir a `experiments/results/demo_reel/README.md` una sección:
```markdown
## Reel showcase (alto impacto)

`reel_showcase.mp4` — hero pick cinematográfico de Iter 7c (cámara dedicada
1280×720, coreografía órbita→seguimiento→retroceso) + segmento de valor y
aplicaciones. Regenerar: `.venv/bin/python experiments/make_showcase_reel.py`.
Para producto/pitch; el `reel_resumen.mp4` técnico queda para la defensa formal.
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore experiments/results/demo_reel/README.md
git commit -m "docs(reel): documentar reel_showcase + gitignore de intermedios"
```

---

## Self-Review

**1. Spec coverage:**
- Cámara dedicada separada de percepción → Task 2 (CineCamera.create) ✓
- Coreografía órbita→seguimiento→retroceso → Task 1 (choreograph) ✓
- Más tiempo/fluidez (30 fps, más frames) → Task 4/5 (fps=30, captura por substep) ✓
- Segmento de valor (logros/costo/aplicaciones/honestidad) → Task 5 (VALUE_CARDS) ✓
- Salida reel_showcase.mp4 gitignored → Task 6 ✓
- No corromper percepción → frame_hook captura de cine_camera, /rgb_camera intacta (Task 3) ✓
- Pick limpio (assert métricas) → Task 5 main() assert ✓
- Fallback creación cámara → validado en vivo (createVisionSensor+setExplicitHandling funciona), fallback de escena innecesario; documentado en header.

**2. Placeholder scan:** sin TBD/TODO. La nota de `make_title_card` pide confirmar firma real (no es placeholder: instrucción de verificación con adaptación concreta).

**3. Type consistency:** `CineCamera(bridge)`, `.create()/.aim(progress,tcp,center)/.capture(frames_dir,idx)/.remove()` consistentes entre Task 2 y Task 4. `choreograph(progress,tcp,center)→(pos,target)` consistente entre Task 1 y Task 2. `frame_hook` (callable sin args) consistente entre Task 3 y Task 4. `compile_mp4(frames_dir,mp4_path,fps)` coincide con la firma real de run_pick_with_diffusion.py.

**Riesgo residual:** ninguno bloqueante. API de cámara validado en vivo; firma de `make_title_card` confirmada (escala vs duración resuelto). La única incógnita menor es estética (encuadre exacto de la coreografía), ajustable iterando los parámetros de `choreograph` tras ver el primer render.
