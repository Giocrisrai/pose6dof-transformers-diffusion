# Diffusion Policy retraining (Iter 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-entrenar la Diffusion Policy con un dataset de 230 trayectorias (200 heurísticas + 30 ejecutadas en sim), conectar la policy entrenada al pick demo de CoppeliaSim, y producir métricas honestas que cierren la Brecha B del pipeline.

**Architecture:** 3 fases secuenciales: Phase A (data collection — generamos dataset .pt), Phase B (fine-tune del checkpoint actual sobre dataset), Phase C (nuevo runner que usa la policy entrenada para mover el TCP vía IK). Todo en MPS (Apple Silicon) sin requerir GPU dedicada.

**Tech Stack:** Python 3.12, PyTorch (MPS backend), CoppeliaSim 4.10 vía ZMQ, simIK, numpy. Reutiliza `ConditionalUNet1D` y `SimpleDDPMScheduler` existentes (`src/planning/diffusion_policy.py`).

**Spec:** `docs/superpowers/specs/2026-05-28-diffusion-retrain-sim-design.md`

---

## File Structure

**Create:**
- `src/planning/diffusion_dataset.py` — `SimPickDataset` (PyTorch Dataset)
- `experiments/collect_diffusion_dataset.py` — data collector (200 heur + 30 sim → train.pt / val.pt)
- `experiments/train_diffusion_on_sim.py` — train loop, 50 epochs, MPS
- `experiments/run_pick_with_diffusion.py` — runner que usa la policy entrenada en sim
- `src/simulation/utils.py` — helper `map_fp_pose_to_sim_workspace` extraído
- `tests/test_diffusion_dataset.py` — smoke test del dataset

**Modify:**
- `src/planning/diffusion_policy.py` — agregar `SimpleDDPMScheduler.add_noise_batch` (versión batch del existente)
- `src/simulation/pick_sequence.py` — permitir `frames_dir=None` para skip captura (collection rápida)
- `experiments/run_pick_with_fp_pose.py` — importar el helper desde `src/simulation/utils.py`

**Persisted outputs (no commitear los .pt grandes; agregar a .gitignore):**
- `data/datasets/sim_pick_v1/{heuristic,executed,train,val}.pt`
- `data/models/diffusion_policy_sim_v1.pth`
- `experiments/results/pick_with_diffusion/{frames,demo.mp4,metadata.json,eval_summary.json}`

---

## Pre-flight checks (before starting)

- [ ] Confirmar working directory + branch limpio:

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git checkout main
git pull --ff-only
git status --short
```

Expected: working tree clean.

- [ ] Crear branch de feature:

```bash
git checkout -b feat/diffusion-retrain-iter1
```

- [ ] Verificar CoppeliaSim corriendo (Phase A.2 lo necesita):

```bash
.venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient('localhost', 23000).require('sim')
print(f'sim version: {sim.getInt32Param(sim.intparam_program_version)}')
"
```

Expected: `sim version: 41000`. Si falla: `open -a CoppeliaSim_Edu && sleep 3` y reintentar.

- [ ] Verificar checkpoint actual existe:

```bash
ls -la data/models/diffusion_policy_grasp.pth
```

Expected: archivo de ~1.5 MB.

---

## Task 1: Agregar `SimpleDDPMScheduler.add_noise_batch` para training

El scheduler actual tiene `add_noise(x_0, t)` que toma UN solo timestep. Para training necesitamos sample DIFERENTES timesteps por elemento del batch. Agregamos versión batch.

**Files:**
- Modify: `src/planning/diffusion_policy.py`

### Step 1.1: Crear test del método batch

- [ ] Crear `tests/test_diffusion_scheduler.py`:

```python
"""Tests para SimpleDDPMScheduler.add_noise_batch."""
import numpy as np
import torch

from src.planning.diffusion_policy import SimpleDDPMScheduler


def test_add_noise_batch_shape():
    sch = SimpleDDPMScheduler(num_timesteps=100)
    # Batch de 4 trayectorias de horizon=16, action_dim=7
    x_0 = torch.randn(4, 16, 7)
    t = torch.tensor([10, 20, 30, 50], dtype=torch.long)
    x_t, eps = sch.add_noise_batch(x_0, t)
    assert x_t.shape == x_0.shape
    assert eps.shape == x_0.shape
    assert x_t.dtype == torch.float32


def test_add_noise_batch_t_zero_returns_almost_clean():
    sch = SimpleDDPMScheduler(num_timesteps=100)
    x_0 = torch.randn(2, 16, 7)
    t = torch.tensor([0, 0], dtype=torch.long)
    x_t, eps = sch.add_noise_batch(x_0, t)
    # En t=0, alpha_bar ≈ 0.9999; x_t debe estar cerca de x_0
    diff = (x_t - x_0).abs().mean().item()
    assert diff < 0.5  # tolerancia generosa
```

### Step 1.2: Correr test — debe fallar

```bash
.venv/bin/pytest tests/test_diffusion_scheduler.py -v
```

Expected: `AttributeError: 'SimpleDDPMScheduler' object has no attribute 'add_noise_batch'`.

### Step 1.3: Implementar `add_noise_batch`

- [ ] Editar `src/planning/diffusion_policy.py`. Después del método `add_noise` (línea ~56), agregar:

```python
    def add_noise_batch(self, x_0, t):
        """Versión batch de add_noise. Acepta torch tensors.

        Args:
            x_0: (B, horizon, action_dim) tensor torch (clean data).
            t:   (B,) tensor torch long (timestep por batch element).

        Returns:
            x_t: (B, horizon, action_dim) tensor noisy data.
            eps: (B, horizon, action_dim) tensor noise applied.
        """
        import torch
        device = x_0.device
        # alpha_bar es numpy de shape (T,); indexamos por t.
        alpha_bar_np = self.alpha_bar[t.cpu().numpy()]  # (B,)
        alpha_bar = torch.tensor(alpha_bar_np, dtype=torch.float32, device=device)
        # broadcast a (B, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * eps
        return x_t, eps
```

### Step 1.4: Correr test — debe pasar

```bash
.venv/bin/pytest tests/test_diffusion_scheduler.py -v
```

Expected: 2 passed.

### Step 1.5: Commit

```bash
git add tests/test_diffusion_scheduler.py src/planning/diffusion_policy.py
git commit -m "feat(planning): SimpleDDPMScheduler.add_noise_batch para training

Versión batch del existente add_noise. Acepta torch tensors y aplica
ruido con un alpha_bar distinto por elemento del batch (lo necesario
para DDPM training donde sample t ~ Uniform(0, T-1) por step).

Refs: spec sección 'Phase B — Fine-tune' (Task 1 del plan)."
```

---

## Task 2: Extraer helper `map_fp_pose_to_sim_workspace` a `src/simulation/utils.py`

Hoy está duplicado en `run_pick_with_fp_pose.py`. Lo movemos a un módulo compartido para que `run_pick_with_diffusion.py` lo reuse.

**Files:**
- Create: `src/simulation/utils.py`
- Modify: `experiments/run_pick_with_fp_pose.py`

### Step 2.1: Crear `src/simulation/utils.py`

- [ ] Crear archivo:

```python
"""Utilidades compartidas del subsistema de simulación."""
from __future__ import annotations


def map_fp_pose_to_sim_workspace(t_pred):
    """Mapea el centroide de una pose FP (YCB-V dataset frame) al
    workspace del UR5 en bin_base.ttt.

    Las poses FP están en el frame de la cámara del dataset (típicamente
    t_pred[2] ≈ 0.5-1.5 m). Para el demo del sim, usamos solo la
    componente XY de la pose para variar dentro del workspace (Z = cube
    height fija).

    Args:
        t_pred: lista o array de 3 floats (translation de la pose FP).

    Returns:
        list[float] de 3 elementos: [x, y, z] en coords mundo del sim.
    """
    x_offset = max(-0.05, min(0.05, t_pred[0]))
    y_offset = max(-0.05, min(0.05, t_pred[1]))
    return [0.46 + x_offset, -0.10 + y_offset, 0.033]
```

### Step 2.2: Actualizar `run_pick_with_fp_pose.py` para importar el helper

- [ ] Editar `experiments/run_pick_with_fp_pose.py`. Reemplazar la definición local de `map_fp_pose_to_sim_workspace` por un import:

Buscar:

```python
def map_fp_pose_to_sim_workspace(t_pred: list[float]) -> list[float]:
    """Mapea el centroide de una pose FP al workspace del sim.
    ...
    """
    x_offset = max(-0.05, min(0.05, t_pred[0]))
    y_offset = max(-0.05, min(0.05, t_pred[1]))
    return [0.46 + x_offset, -0.10 + y_offset, 0.033]
```

Reemplazar TODO ese bloque por:

```python
from src.simulation.utils import map_fp_pose_to_sim_workspace
```

(colocar el import junto a los otros imports al inicio del archivo, no aquí en mitad del código)

### Step 2.3: Verificar que el runner sigue funcionando

```bash
.venv/bin/python experiments/run_pick_with_fp_pose.py --fp-index 0 2>&1 | tail -8
```

Expected: mismo output que antes (`FP pose source: foundation_pose_ckpt[0]`, métricas plausible).

### Step 2.4: Commit

```bash
git add src/simulation/utils.py experiments/run_pick_with_fp_pose.py
git commit -m "refactor(sim): extraer map_fp_pose_to_sim_workspace a utils

Helper para mapear poses de FoundationPose al workspace del sim.
Movido desde run_pick_with_fp_pose.py a src/simulation/utils.py
para que run_pick_with_diffusion.py también lo use.

Refs: plan Task 2."
```

---

## Task 3: Permitir `frames_dir=None` en `run_pick_sequence` para collection rápida

La Phase A.2 (sim-executed trajectories) corre `pick_sequence` 30 veces. Sin captura de frames cada corrida tarda ~50% menos.

**Files:**
- Modify: `src/simulation/pick_sequence.py`

### Step 3.1: Editar `_capture_frame` para no-op si frames_dir es None

- [ ] Buscar `_capture_frame` (alrededor de línea 75):

```python
def _capture_frame(bridge, frames_dir: Path, idx: int) -> None:
    from PIL import Image
    sim = bridge.sim
    sim.handleVisionSensor(bridge._camera_rgb_handle)
    img_raw, res = sim.getVisionSensorImg(bridge._camera_rgb_handle)
    w, h = res[0], res[1]
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)
    Image.fromarray(img).save(frames_dir / f"{idx:06d}.png")
```

- [ ] Reemplazar por:

```python
def _capture_frame(bridge, frames_dir, idx: int) -> None:
    """Captura un frame del rgb_camera. Si frames_dir es None, no-op
    (modo collection rápida sin overhead de PNG)."""
    if frames_dir is None:
        return
    from PIL import Image
    sim = bridge.sim
    sim.handleVisionSensor(bridge._camera_rgb_handle)
    img_raw, res = sim.getVisionSensorImg(bridge._camera_rgb_handle)
    w, h = res[0], res[1]
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
    img = np.flipud(img)
    Image.fromarray(img).save(frames_dir / f"{idx:06d}.png")
```

### Step 3.2: Actualizar `run_pick_sequence` para aceptar `frames_dir=None`

- [ ] Buscar la signatura (línea ~179):

```python
def run_pick_sequence(
    bridge: CoppeliaSimBridge,
    frames_dir: Path,
    target_object: str = "/object_1",
    pose_override_xyz: Optional[list[float]] = None,
    pose_source: str = "scene_groundtruth",
) -> PickResult:
```

- [ ] Cambiar el tipo de `frames_dir` a aceptar None:

```python
def run_pick_sequence(
    bridge: CoppeliaSimBridge,
    frames_dir: Optional[Path],
    target_object: str = "/object_1",
    pose_override_xyz: Optional[list[float]] = None,
    pose_source: str = "scene_groundtruth",
) -> PickResult:
```

- [ ] Buscar el bloque que crea `frames_dir` y elimina PNGs viejos (línea ~189):

```python
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
```

- [ ] Reemplazar por:

```python
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*.png"):
            old.unlink()
```

### Step 3.3: Verificar que `run_pick_demo.py` y `run_pick_battery.py` siguen funcionando

```bash
# pick_demo (with frames)
.venv/bin/python experiments/run_pick_demo.py 2>&1 | grep -E "frames cap|grasp_proximity" | tail -3
```

Expected: `frames capturados: 870`, `grasp_proximity: 0.8 cm (plausible)`.

### Step 3.4: Commit

```bash
git add src/simulation/pick_sequence.py
git commit -m "feat(sim): run_pick_sequence acepta frames_dir=None (modo rápido)

Para data collection masiva (Phase A.2: 30 sim-executed trajectories),
saltearse la captura de PNG reduce ~50% el tiempo por corrida.
Backwards-compatible: si frames_dir se pasa, comportamiento idéntico.

Refs: plan Task 3."
```

---

## Task 4: Phase A.1 — Generar 200 trayectorias heurísticas

**Files:**
- Create: `experiments/collect_diffusion_dataset.py` (parcial — solo Phase A.1 en este task)

### Step 4.1: Crear esqueleto del collector

- [ ] Crear `experiments/collect_diffusion_dataset.py`:

```python
#!/usr/bin/env python3
"""Genera dataset para re-entrenamiento de la Diffusion Policy.

Phase A.1: 200 trayectorias heurísticas (rápido, ~1s c/u).
Phase A.2: 30 trayectorias ejecutadas en sim (lento, ~50s c/u).
Phase A.3: combina + split 80/20 → train.pt + val.pt.

Uso:
    python experiments/collect_diffusion_dataset.py --phase heuristic
    python experiments/collect_diffusion_dataset.py --phase executed
    python experiments/collect_diffusion_dataset.py --phase split
    python experiments/collect_diffusion_dataset.py --phase all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("collect_dp")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v1"
N_HEURISTIC = 200
N_EXECUTED = 30
SEED = 42

# Workspace bounds (matching bin_base.ttt geometry)
X_RANGE = (0.40, 0.55)
Y_RANGE = (-0.15, -0.05)
Z_FIXED = 0.033


def sample_pose(rng: np.random.Generator) -> np.ndarray:
    """Sample una pose SE(3) random dentro del workspace.

    Returns: (4, 4) matriz SE(3).
    """
    x = rng.uniform(*X_RANGE)
    y = rng.uniform(*Y_RANGE)
    z = Z_FIXED
    # Rotación: una de [identity, 45° Z, 90° Z]
    rot_choices = [0.0, np.pi / 4, np.pi / 2]
    theta = rng.choice(rot_choices)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    return pose


def phase_heuristic(n: int = N_HEURISTIC) -> None:
    """Genera n trayectorias heurísticas y las guarda."""
    logger.info(f"Phase A.1: generando {n} trayectorias heurísticas")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, cond_dim=64, hidden_dim=128, n_timesteps=100,
        device="cpu",  # heurística no usa la red
    )

    rng = np.random.default_rng(SEED)
    conds = np.zeros((n, 64), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)

    for i in range(n):
        pose = sample_pose(rng)
        traj = planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)
        # plan_grasp_heuristic devuelve (1, 16, 7); extraer
        trajs[i] = traj[0]
        cond = planner.encode_observation(pose)  # (1, 64) torch
        conds[i] = cond.cpu().numpy()[0]

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{n}")

    out = DATASET_DIR / "heuristic.pt"
    torch.save({
        "conds": torch.from_numpy(conds),
        "trajs": torch.from_numpy(trajs),
        "source": "heuristic",
        "seed": SEED,
    }, out)
    logger.info(f"escrito: {out} ({n} trayectorias)")


def phase_executed(n: int = N_EXECUTED) -> None:
    """Genera n trayectorias ejecutadas en CoppeliaSim. STUB — implementado en Task 5."""
    raise NotImplementedError("Phase A.2 se implementa en Task 5 del plan")


def phase_split() -> None:
    """Combina heurístic + executed y hace train/val split. STUB — Task 6."""
    raise NotImplementedError("Phase A.3 se implementa en Task 6 del plan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["heuristic", "executed", "split", "all"],
                        default="all")
    args = parser.parse_args()

    if args.phase in ("heuristic", "all"):
        phase_heuristic()
    if args.phase in ("executed", "all"):
        phase_executed()
    if args.phase in ("split", "all"):
        phase_split()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 4.2: Correr la Phase heurística

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase heuristic 2>&1 | tail -10
```

Expected: `escrito: data/datasets/sim_pick_v1/heuristic.pt (200 trayectorias)`. Tarda ~30 s.

### Step 4.3: Verificar el dataset cargado

```bash
.venv/bin/python -c "
import torch
d = torch.load('data/datasets/sim_pick_v1/heuristic.pt', weights_only=True)
print('keys:', list(d.keys()))
print('conds:', d['conds'].shape, d['conds'].dtype)
print('trajs:', d['trajs'].shape, d['trajs'].dtype)
print('trajs[0, 0]:', d['trajs'][0, 0].tolist())  # primer waypoint de primera traj
"
```

Expected: `conds: torch.Size([200, 64])`, `trajs: torch.Size([200, 16, 7])`.

### Step 4.4: Agregar `data/datasets/sim_pick_v1/` a `.gitignore`

- [ ] Agregar línea al final de `.gitignore`:

```
# Datasets de Iter1 retrain (regenerables con collect_diffusion_dataset.py)
data/datasets/sim_pick_v1/
```

### Step 4.5: Commit

```bash
git add experiments/collect_diffusion_dataset.py .gitignore
git commit -m "feat(planning): collector Phase A.1 — 200 trayectorias heurísticas

experiments/collect_diffusion_dataset.py genera trayectorias heurísticas
para entrenar la Diffusion Policy. Cada trayectoria: pose SE(3) random
en workspace bounds (X [0.40, 0.55], Y [-0.15, -0.05], Z=0.033) +
rotación de un conjunto discreto [0, π/4, π/2]. Output: heuristic.pt
con dict {conds (200, 64), trajs (200, 16, 7)}.

.gitignore ahora excluye data/datasets/sim_pick_v1/ (regenerable).

Refs: plan Task 4."
```

---

## Task 5: Phase A.2 — Generar 30 trayectorias ejecutadas en sim

**Files:**
- Modify: `experiments/collect_diffusion_dataset.py` (implementar `phase_executed`)

### Step 5.1: Pre-flight — CoppeliaSim corriendo

```bash
.venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient('localhost', 23000).require('sim')
print(f'OK v{sim.getInt32Param(sim.intparam_program_version)}')
"
```

Expected: `OK v41000`. Si falla, abrir CoppeliaSim Edu.

### Step 5.2: Implementar `phase_executed`

- [ ] Editar `experiments/collect_diffusion_dataset.py`. Reemplazar la versión STUB:

```python
def phase_executed(n: int = N_EXECUTED) -> None:
    """Genera n trayectorias ejecutadas en CoppeliaSim. STUB — implementado en Task 5."""
    raise NotImplementedError("Phase A.2 se implementa en Task 5 del plan")
```

por la implementación completa:

```python
def phase_executed(n: int = N_EXECUTED) -> None:
    """Genera n trayectorias ejecutadas en CoppeliaSim.

    Por cada pose:
      1. Mueve /object_1 a la pose target.
      2. Corre pick_sequence con frames_dir=None (rápido).
      3. Durante el run, captura TCP pose por step → submuestrea a 16
         waypoints uniformemente espaciados.
      4. Acción 7-D por waypoint: [x, y, z, rx, ry, rz, gripper] donde
         (rx, ry, rz) es so3_log de la rotación del tip.

    Persiste: data/datasets/sim_pick_v1/executed.pt
    """
    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
    from src.simulation.pick_sequence import (
        _setup_ik, _move_tcp_via_ik, set_gripper, setup_robot_control,
    )
    from src.utils.lie_groups import so3_log

    logger.info(f"Phase A.2: generando {n} trayectorias ejecutadas en sim")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    planner_aux = DiffusionGraspPlanner(
        action_dim=7, horizon=16, cond_dim=64, hidden_dim=128, n_timesteps=100,
        device="cpu",
    )
    rng = np.random.default_rng(SEED + 1)
    conds = np.zeros((n, 64), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)
    skipped = 0

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

    for i in range(n):
        pose = sample_pose(rng)
        conds[i] = planner_aux.encode_observation(pose).cpu().numpy()[0]

        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                # Mover /object_1 a la pose target en el sim
                sim = bridge.sim
                obj1 = sim.getObject("/object_1")
                sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

                setup_robot_control(bridge)
                env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
                bridge.set_stepping(True)
                bridge.start_simulation()
                tip_h = sim.getObject("/tip")

                # Captura TCP pose por step en una lista
                tip_log = []  # list of (xyz, rot_matrix_flat, gripper_open)

                def log_tip(gripper_open):
                    p = sim.getObjectPosition(tip_h, -1)
                    M = sim.getObjectMatrix(tip_h, -1)  # 12 elementos (3x4)
                    R = np.array([M[0:3], M[4:7], M[8:11]])
                    tip_log.append((p, R, gripper_open))

                set_gripper(bridge, True)
                for _ in range(20):
                    bridge.step()
                    log_tip(1.0)

                # Approach
                _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                                 [pose[0, 3], pose[1, 3], 0.30], frames_dir=None, counter=[0])
                # Log post-approach
                for _ in range(5): log_tip(1.0)

                # Descend (cubo no-respondable)
                sim.setObjectInt32Param(obj1, sim.shapeintparam_respondable, 0)
                _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                                 [pose[0, 3], pose[1, 3], pose[2, 3]], frames_dir=None, counter=[0])
                for _ in range(5): log_tip(1.0)

                # Grasp + close
                set_gripper(bridge, False)
                for _ in range(20):
                    bridge.step()
                    log_tip(0.0)

                # Lift
                _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                                 [pose[0, 3], pose[1, 3], 0.40], frames_dir=None, counter=[0])
                for _ in range(5): log_tip(0.0)

                # Simulación end
                bridge.stop_simulation()
                try: simIK.eraseEnvironment(env)
                except: pass

            # Submuestrear a 16 waypoints
            n_logged = len(tip_log)
            if n_logged < 16:
                logger.warning(f"  [{i}] solo {n_logged} steps logged, skipping")
                skipped += 1
                continue

            indices = np.linspace(0, n_logged - 1, 16).astype(int)
            for k, idx in enumerate(indices):
                p, R, g = tip_log[idx]
                rot_vec = so3_log(R)
                trajs[i, k] = [p[0], p[1], p[2], rot_vec[0], rot_vec[1], rot_vec[2], g]

            if (i + 1) % 5 == 0:
                logger.info(f"  {i+1}/{n} (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    if skipped > 0:
        logger.warning(f"Phase A.2: {skipped}/{n} trayectorias saltadas")

    out = DATASET_DIR / "executed.pt"
    torch.save({
        "conds": torch.from_numpy(conds[:n - skipped]),
        "trajs": torch.from_numpy(trajs[:n - skipped]),
        "source": "executed",
        "seed": SEED + 1,
    }, out)
    logger.info(f"escrito: {out} ({n - skipped} trayectorias)")
```

### Step 5.3: Correr la phase

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase executed 2>&1 | tail -15
```

Expected: 30 trayectorias ejecutadas (~25 min). El último mensaje debe ser `escrito: ...executed.pt (30 trayectorias)` (o menos si hay skipped).

### Step 5.4: Verificar dataset ejecutado

```bash
.venv/bin/python -c "
import torch
d = torch.load('data/datasets/sim_pick_v1/executed.pt', weights_only=True)
print('conds:', d['conds'].shape)
print('trajs:', d['trajs'].shape)
print('trajs[0, 0]:', d['trajs'][0, 0].tolist())
print('trajs[0, 8]:', d['trajs'][0, 8].tolist())  # mid-trajectory
"
```

Expected: `conds: torch.Size([30, 64])` (o menos si hubo skipped), `trajs: torch.Size([30, 16, 7])`.

### Step 5.5: Commit

```bash
git add experiments/collect_diffusion_dataset.py
git commit -m "feat(planning): collector Phase A.2 — 30 trayectorias ejecutadas en sim

Por cada pose: mueve /object_1, corre pick_sequence con frames_dir=None
para minimizar overhead, loguea TCP pose por step (xyz + so3_log de
rotation + gripper signal), submuestrea a 16 waypoints uniformemente
espaciados. Skip robusto si algún pick falla; reporta count.

Refs: plan Task 5."
```

---

## Task 6: Phase A.3 — Combinar + split 80/20

**Files:**
- Modify: `experiments/collect_diffusion_dataset.py` (implementar `phase_split`)

### Step 6.1: Implementar `phase_split`

- [ ] Editar `experiments/collect_diffusion_dataset.py`. Reemplazar:

```python
def phase_split() -> None:
    """Combina heurístic + executed y hace train/val split. STUB — Task 6."""
    raise NotImplementedError("Phase A.3 se implementa en Task 6 del plan")
```

por:

```python
def phase_split() -> None:
    """Combina heuristic.pt + executed.pt y hace 80/20 split."""
    logger.info("Phase A.3: combinando + split 80/20")
    heur_path = DATASET_DIR / "heuristic.pt"
    exec_path = DATASET_DIR / "executed.pt"
    if not heur_path.exists() or not exec_path.exists():
        raise FileNotFoundError(
            "Faltan datasets parciales. Corré --phase heuristic y --phase executed primero."
        )

    h = torch.load(heur_path, weights_only=True)
    e = torch.load(exec_path, weights_only=True)
    all_conds = torch.cat([h["conds"], e["conds"]], dim=0)
    all_trajs = torch.cat([h["trajs"], e["trajs"]], dim=0)
    n = len(all_conds)

    rng = np.random.default_rng(SEED + 2)
    indices = rng.permutation(n)
    train_n = int(0.8 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    torch.save({
        "conds": all_conds[train_idx],
        "trajs": all_trajs[train_idx],
        "split": "train",
    }, DATASET_DIR / "train.pt")
    torch.save({
        "conds": all_conds[val_idx],
        "trajs": all_trajs[val_idx],
        "split": "val",
    }, DATASET_DIR / "val.pt")

    logger.info(f"escrito: train.pt ({train_n}), val.pt ({n - train_n})")
```

### Step 6.2: Correr el split

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase split
```

Expected: `escrito: train.pt (184), val.pt (46)` (números pueden variar si hubo trayectorias skipped en A.2).

### Step 6.3: Verificar

```bash
.venv/bin/python -c "
import torch
for split in ('train', 'val'):
    d = torch.load(f'data/datasets/sim_pick_v1/{split}.pt', weights_only=True)
    print(f'{split}: conds={tuple(d[\"conds\"].shape)}, trajs={tuple(d[\"trajs\"].shape)}')
"
```

Expected: shapes consistentes con un 80/20 split del total.

### Step 6.4: Commit

```bash
git add experiments/collect_diffusion_dataset.py
git commit -m "feat(planning): collector Phase A.3 — combinar + train/val split

Une heuristic.pt + executed.pt, shuffle con seed determinista,
y guarda 80% en train.pt y 20% en val.pt. Listo para Phase B.

Refs: plan Task 6."
```

---

## Task 7: Dataset class + smoke test

**Files:**
- Create: `src/planning/diffusion_dataset.py`
- Create: `tests/test_diffusion_dataset.py`

### Step 7.1: Crear test

- [ ] Crear `tests/test_diffusion_dataset.py`:

```python
"""Smoke tests para SimPickDataset."""
from pathlib import Path

import pytest
import torch


def _make_dummy_pt(path, n=10):
    """Crea un .pt con n elementos dummy."""
    torch.save({
        "conds": torch.randn(n, 64),
        "trajs": torch.randn(n, 16, 7),
        "split": "test",
    }, path)


def test_sim_pick_dataset_len(tmp_path):
    from src.planning.diffusion_dataset import SimPickDataset
    pt = tmp_path / "test.pt"
    _make_dummy_pt(pt, n=23)
    ds = SimPickDataset(pt)
    assert len(ds) == 23


def test_sim_pick_dataset_getitem_shapes(tmp_path):
    from src.planning.diffusion_dataset import SimPickDataset
    pt = tmp_path / "test.pt"
    _make_dummy_pt(pt, n=5)
    ds = SimPickDataset(pt)
    cond, traj = ds[0]
    assert cond.shape == (64,)
    assert traj.shape == (16, 7)
    assert cond.dtype == torch.float32
    assert traj.dtype == torch.float32


def test_sim_pick_dataset_dataloader(tmp_path):
    from torch.utils.data import DataLoader

    from src.planning.diffusion_dataset import SimPickDataset
    pt = tmp_path / "test.pt"
    _make_dummy_pt(pt, n=16)
    ds = SimPickDataset(pt)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    assert batch[0].shape == (4, 64)
    assert batch[1].shape == (4, 16, 7)
```

### Step 7.2: Correr test — debe fallar

```bash
.venv/bin/pytest tests/test_diffusion_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.planning.diffusion_dataset'`.

### Step 7.3: Implementar el dataset

- [ ] Crear `src/planning/diffusion_dataset.py`:

```python
"""Dataset PyTorch para fine-tune de la Diffusion Policy sobre datos del sim."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset


class SimPickDataset(Dataset):
    """Dataset de pares (cond, trajectory) para training de DiffusionPolicy.

    Carga un .pt con dict {conds: (N, 64), trajs: (N, 16, 7)}.
    """

    def __init__(self, pt_path):
        pt_path = Path(pt_path)
        data = torch.load(pt_path, weights_only=True)
        self.conds = data["conds"].to(torch.float32)
        self.trajs = data["trajs"].to(torch.float32)
        assert self.conds.shape[0] == self.trajs.shape[0], (
            f"mismatched len: conds={self.conds.shape}, trajs={self.trajs.shape}"
        )

    def __len__(self) -> int:
        return self.conds.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conds[i], self.trajs[i]
```

### Step 7.4: Correr test — debe pasar

```bash
.venv/bin/pytest tests/test_diffusion_dataset.py -v
```

Expected: 3 passed.

### Step 7.5: Commit

```bash
git add src/planning/diffusion_dataset.py tests/test_diffusion_dataset.py
git commit -m "feat(planning): SimPickDataset + smoke tests

PyTorch Dataset que carga .pt con {conds, trajs}. Hace asserts de
consistencia interna (mismatched len) y enfuerza dtype float32.

Refs: plan Task 7."
```

---

## Task 8: Train loop

**Files:**
- Create: `experiments/train_diffusion_on_sim.py`

### Step 8.1: Crear el train script

- [ ] Crear `experiments/train_diffusion_on_sim.py`:

```python
#!/usr/bin/env python3
"""Fine-tune de la Diffusion Policy sobre dataset del sim (Iter 1).

Carga el checkpoint existente, entrena 50 epochs con MSE noise loss,
guarda nuevo checkpoint y curvas train/val.

Uso (CoppeliaSim NO requerido para esta phase):
    python experiments/train_diffusion_on_sim.py
    python experiments/train_diffusion_on_sim.py --epochs 50 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_dataset import SimPickDataset
from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_dp")

REPO_OUT_MODELS = REPO / "data" / "models"
DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-in", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_grasp.pth")
    parser.add_argument("--checkpoint-out", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_sim_v1.pth")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"device: {device}")

    # Dataset
    train_ds = SimPickDataset(DATASET_DIR / "train.pt")
    val_ds = SimPickDataset(DATASET_DIR / "val.pt")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f"train={len(train_ds)}, val={len(val_ds)}")

    # Model + scheduler
    config = {"action_dim": 7, "horizon": 16, "cond_dim": 64,
              "hidden_dim": 128, "n_timesteps": 100, "n_epochs": args.epochs}
    model = ConditionalUNet1D(
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        cond_dim=config["cond_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
    scheduler = SimpleDDPMScheduler(num_timesteps=config["n_timesteps"])

    # Cargar checkpoint existente si está
    if args.checkpoint_in.exists():
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"checkpoint cargado: {args.checkpoint_in.name}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_train = 0.0
        n_batches = 0
        for cond, traj in train_dl:
            cond, traj = cond.to(device), traj.to(device)
            B = cond.shape[0]
            t = torch.randint(0, config["n_timesteps"], (B,), device=device, dtype=torch.long)
            traj_noisy, noise = scheduler.add_noise_batch(traj, t)
            noise_pred = model(traj_noisy, t, cond)
            loss = loss_fn(noise_pred, noise)
            if torch.isnan(loss):
                logger.error(f"NaN loss en epoch {epoch}, abortando")
                return 1
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_train += loss.item()
            n_batches += 1
        epoch_train /= max(n_batches, 1)

        # Val
        model.eval()
        epoch_val = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for cond, traj in val_dl:
                cond, traj = cond.to(device), traj.to(device)
                B = cond.shape[0]
                t = torch.randint(0, config["n_timesteps"], (B,), device=device, dtype=torch.long)
                traj_noisy, noise = scheduler.add_noise_batch(traj, t)
                noise_pred = model(traj_noisy, t, cond)
                loss = loss_fn(noise_pred, noise)
                epoch_val += loss.item()
                n_val_batches += 1
        epoch_val /= max(n_val_batches, 1)

        train_losses.append(epoch_train)
        val_losses.append(epoch_val)
        logger.info(f"epoch {epoch+1:3d}/{args.epochs}  "
                    f"train_loss={epoch_train:.4f}  val_loss={epoch_val:.4f}")

    # Guardar checkpoint
    REPO_OUT_MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config,
    }, args.checkpoint_out)
    logger.info(f"checkpoint escrito: {args.checkpoint_out}")

    # Resumen final
    summary = {
        "epochs": args.epochs,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "min_val_loss": min(val_losses),
        "min_val_epoch": val_losses.index(min(val_losses)) + 1,
    }
    summary_path = args.checkpoint_out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 8.2: Correr training

```bash
.venv/bin/python experiments/train_diffusion_on_sim.py 2>&1 | tail -20
```

Expected: 50 epochs procesados (~10 min en MPS), `train_loss` decreciente, `val_loss` cerca de `train_loss`. Output final: `checkpoint escrito: data/models/diffusion_policy_sim_v1.pth`.

### Step 8.3: Verificar checkpoint

```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('data/models/diffusion_policy_sim_v1.pth', map_location='cpu', weights_only=True)
print('keys:', list(ckpt.keys()))
print('final train loss:', ckpt['train_losses'][-1])
print('final val loss:', ckpt['val_losses'][-1])
print('min val:', min(ckpt['val_losses']))
"
cat data/models/diffusion_policy_sim_v1.summary.json
```

Expected: train/val losses convergentes, no NaN.

### Step 8.4: Agregar checkpoints a .gitignore

- [ ] Agregar línea al `.gitignore`:

```
# Checkpoints de Iter1 (regenerables; el original sí se commit)
data/models/diffusion_policy_sim_v1.pth
data/models/diffusion_policy_sim_v1.summary.json
```

### Step 8.5: Commit

```bash
git add experiments/train_diffusion_on_sim.py .gitignore
git commit -m "feat(planning): train loop Iter 1 — fine-tune DP sobre dataset del sim

experiments/train_diffusion_on_sim.py carga el checkpoint actual y hace
50 epochs de fine-tune en MPS. Loss: MSE noise prediction (DDPM estándar).
Output: diffusion_policy_sim_v1.pth + summary.json con losses.

NaN guardrail: aborta si loss diverge.

Refs: plan Task 8."
```

---

## Task 9: Connect to sim — `run_pick_with_diffusion.py`

**Files:**
- Create: `experiments/run_pick_with_diffusion.py`

### Step 9.1: Crear el runner

- [ ] Crear `experiments/run_pick_with_diffusion.py`:

```python
#!/usr/bin/env python3
"""Pick-and-place usando la Diffusion Policy entrenada (Iter 1).

Cierra la Brecha B del pipeline: la DP entrenada en Phase B genera
una trayectoria de 16 waypoints, los pasamos a _move_tcp_via_ik para
ejecutarlos en CoppeliaSim.

Uso:
    python experiments/run_pick_with_diffusion.py
    python experiments/run_pick_with_diffusion.py --pose-source fp_ckpt --fp-index 0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.pick_sequence import (
    _capture_frame, _move_tcp_via_ik, _setup_ik,
    compile_mp4, set_gripper, setup_robot_control,
)
from src.simulation.utils import map_fp_pose_to_sim_workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pick_with_dp")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
POLICY = REPO / "data" / "models" / "diffusion_policy_sim_v1.pth"
FP_CKPT = REPO / "experiments" / "checkpoints" / "fp_ycbv_checkpoint.json"


def get_target_pose(args) -> tuple[np.ndarray, str]:
    """Obtiene la pose target según --pose-source. Devuelve (pose 4x4, source_label)."""
    if args.pose_source == "groundtruth":
        # Pose por default del object_1 (en bin_base.ttt)
        pose = np.eye(4)
        pose[:3, 3] = [0.46, -0.10, 0.033]
        return pose, "scene_groundtruth"
    elif args.pose_source == "fp_ckpt":
        ckpt = json.loads(FP_CKPT.read_text())
        entry = ckpt["results"][args.fp_index]
        t_mapped = map_fp_pose_to_sim_workspace(entry["t_pred"])
        pose = np.eye(4)
        pose[:3, 3] = t_mapped
        return pose, f"foundation_pose_ckpt[{args.fp_index}]"
    raise ValueError(f"pose_source {args.pose_source} no soportado")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-source", choices=["groundtruth", "fp_ckpt"], default="groundtruth")
    parser.add_argument("--fp-index", type=int, default=0)
    args = parser.parse_args()

    if not POLICY.exists():
        logger.error(f"policy no encontrada: {POLICY}. Corré train_diffusion_on_sim.py primero.")
        return 1

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = REPO_OUT / "frames"

    # 1. Cargar policy + scheduler
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, cond_dim=64, hidden_dim=128, n_timesteps=100,
        device=device,
    )
    ckpt = torch.load(POLICY, map_location=device, weights_only=True)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    logger.info(f"policy cargada: {POLICY.name}")

    # 2. Obtener target pose
    pose, source_label = get_target_pose(args)
    logger.info(f"target: t={pose[:3,3].tolist()}, source={source_label}")

    # 3. Generar trayectoria con DP
    traj = planner.plan_grasp(pose, n_samples=1)  # (1, 16, 7)
    waypoints = traj[0]  # (16, 7)
    logger.info(f"trayectoria DP: shape={waypoints.shape}, first={waypoints[0].tolist()}")

    # 4. Ejecutar en sim
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        setup_robot_control(bridge)
        env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
        bridge.set_stepping(True)
        bridge.start_simulation()
        sim = bridge.sim
        obj1 = sim.getObject("/object_1")

        # Mover cubo a la pose target (para alinear el "cube center" con el target)
        sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

        counter = [0]
        ik_convergence = []
        # Limpiar frames anteriores
        if frames_dir.exists():
            for f in frames_dir.glob("*.png"): f.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Ejecutar cada waypoint
        prev_gripper = 1.0
        for i, wp in enumerate(waypoints):
            x, y, z, _, _, _, gripper = wp.tolist()
            # Toggle gripper si cambió
            if (gripper > 0.5) != (prev_gripper > 0.5):
                set_gripper(bridge, gripper > 0.5)
                prev_gripper = gripper
            _move_tcp_via_ik(bridge, env, ik_group, target_dummy, ik_joints, simIK,
                             [x, y, z], frames_dir, counter,
                             n_substeps=8, steps_per_substep=2,
                             convergence_tracker=ik_convergence)
            logger.info(f"  waypoint {i+1}/16: xyz=[{x:.3f},{y:.3f},{z:.3f}] gripper={gripper:.1f}")

        # Métricas finales
        cube_end = sim.getObjectPosition(obj1, -1)
        tip_end = sim.getObjectPosition(sim.getObject("/tip"), -1)
        ik_converged = len(ik_convergence) > 0 and all(ik_convergence)

        bridge.stop_simulation()
        try: simIK.eraseEnvironment(env)
        except: pass

    # 5. Compilar MP4
    mp4_path = REPO_OUT / "demo.mp4"
    compiled = compile_mp4(frames_dir, mp4_path, fps=25)

    # 6. Reporte
    metadata = {
        "policy": str(POLICY.relative_to(REPO)),
        "pose_source": source_label,
        "target_pose_t": pose[:3, 3].tolist(),
        "cube_end": cube_end,
        "tip_end": tip_end,
        "ik_converged": ik_converged,
        "n_waypoints": len(waypoints),
        "mp4": str(compiled.relative_to(REPO)) if compiled else None,
    }
    (REPO_OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print("=== RESULTADOS pick con Diffusion Policy ===")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 9.2: Correr el runner (CoppeliaSim debe estar corriendo)

```bash
.venv/bin/python experiments/run_pick_with_diffusion.py 2>&1 | tail -25
```

Expected: 16 waypoints ejecutados, MP4 generado, `metadata.json` con `ik_converged: True`.

### Step 9.3: Verificar outputs

```bash
ls -la experiments/results/pick_with_diffusion/
cat experiments/results/pick_with_diffusion/metadata.json
```

Expected: `demo.mp4` >50 KB, metadata con todos los campos.

### Step 9.4: Commit

```bash
git add experiments/run_pick_with_diffusion.py
git commit -m "feat(sim): run_pick_with_diffusion.py — Brecha B cerrada

Carga la DP entrenada (diffusion_policy_sim_v1.pth), genera 16
waypoints con planner.plan_grasp(pose), y los ejecuta vía
_move_tcp_via_ik en CoppeliaSim. Captura frames + MP4 + metadata.

Soporta --pose-source {groundtruth, fp_ckpt} para conectar también
con la Brecha A (FP poses reales).

Refs: plan Task 9, spec Phase C."
```

---

## Task 10: Eval set + summary

Después del training, ejecutar 20 picks con poses nuevas (seed diferente) para medir la DP entrenada vs la heurística.

**Files:**
- Create: `experiments/eval_diffusion_iter1.py`

### Step 10.1: Crear eval script

- [ ] Crear `experiments/eval_diffusion_iter1.py`:

```python
#!/usr/bin/env python3
"""Evalúa la DP entrenada (Iter 1) sobre 20 poses no vistas.

Métricas:
- mse_dp_vs_heuristic: distancia promedio entre trayectoria DP y heurística
- dp_avg_proximity_at_grasp_phase: distancia del waypoint medio (k=8) al cubo
  (proxy de plausibilidad sin ejecutar en sim).

NO ejecuta en sim (eso sería caro por 20 picks). Solo compara trayectorias.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
POLICY = REPO / "data" / "models" / "diffusion_policy_sim_v1.pth"
N_EVAL = 20
SEED = 999  # distinto del training (42) y del collector A.2 (43)


def sample_pose_eval(rng: np.random.Generator) -> np.ndarray:
    """Mismo workspace que training, distinta seed."""
    x = rng.uniform(0.40, 0.55)
    y = rng.uniform(-0.15, -0.05)
    z = 0.033
    theta = rng.choice([0.0, np.pi / 4, np.pi / 2])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    return pose


def main() -> int:
    if not POLICY.exists():
        print(f"policy no encontrada: {POLICY}")
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, cond_dim=64, hidden_dim=128, n_timesteps=100,
        device=device,
    )
    ckpt = torch.load(POLICY, map_location=device, weights_only=True)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()

    rng = np.random.default_rng(SEED)
    mses = []
    proximities = []
    for i in range(N_EVAL):
        pose = sample_pose_eval(rng)
        traj_dp = planner.plan_grasp(pose, n_samples=1)[0]  # (16, 7)
        traj_heur = planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)[0]  # (16, 7)

        # MSE entre trayectorias (componentes XYZ y gripper; ignoramos rot)
        mse = float(np.mean((traj_dp[:, :3] - traj_heur[:, :3]) ** 2))
        mses.append(mse)

        # Proximity en waypoint k=8 (mid-trajectory, donde la heurística está en grasp pose)
        target_xyz = pose[:3, 3]
        prox = float(np.linalg.norm(traj_dp[8, :3] - target_xyz))
        proximities.append(prox)

    summary = {
        "n_eval": N_EVAL,
        "policy": str(POLICY.relative_to(REPO)),
        "mse_dp_vs_heuristic_mean": float(np.mean(mses)),
        "mse_dp_vs_heuristic_max": float(np.max(mses)),
        "dp_grasp_proximity_mean_m": float(np.mean(proximities)),
        "dp_grasp_proximity_max_m": float(np.max(proximities)),
        "dp_grasp_plausible_pct": 100.0 * sum(p < 0.05 for p in proximities) / N_EVAL,
        "thresholds_passed": {
            "mse_dp_vs_heuristic_mean < 0.10": float(np.mean(mses)) < 0.10,
            "dp_grasp_plausible_pct >= 70": (100.0 * sum(p < 0.05 for p in proximities) / N_EVAL) >= 70,
        },
    }

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    out = REPO_OUT / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 10.2: Correr eval

```bash
.venv/bin/python experiments/eval_diffusion_iter1.py
```

Expected: JSON con métricas. Si `thresholds_passed` muestra todo `true`, Iter 1 cumple los criterios del spec.

### Step 10.3: Commit

```bash
git add experiments/eval_diffusion_iter1.py
git commit -m "feat(planning): eval Iter 1 — DP vs heurística sobre 20 poses

experiments/eval_diffusion_iter1.py compara trayectorias generadas por
la DP entrenada vs la heurística sobre 20 poses con seed nuevo. Métricas:
mse promedio, proximity en waypoint medio, % plausibles. Reporta
thresholds del spec.

Refs: plan Task 10, spec sección 'Métricas de éxito'."
```

---

## Task 11: Documentar resultados + actualizar INTEGRATION_PIPELINE

**Files:**
- Modify: `docs/INTEGRATION_PIPELINE.md`

### Step 11.1: Actualizar INTEGRATION_PIPELINE.md

- [ ] Editar `docs/INTEGRATION_PIPELINE.md`. Buscar la sección "Brecha B" (cerca de la tabla del roadmap) y agregar antes:

Texto a agregar (al final del archivo o antes del roadmap):

```markdown
## Iter 1 (cerrado 2026-05-28): retrain de la Diffusion Policy

Estado: **Brecha B cerrada en su versión mínima viable**.

Pipeline ahora ejecuta:
1. Cargar pose (groundtruth | FoundationPose checkpoint).
2. `DiffusionGraspPlanner` (re-entrenado en `diffusion_policy_sim_v1.pth`) genera 16 waypoints.
3. Cada waypoint XYZ se ejecuta en CoppeliaSim vía `_move_tcp_via_ik`.
4. Captura frames + MP4 + métricas (proximity, deposit_error, ik_converged).

Demo entry point: `experiments/run_pick_with_diffusion.py`.

Métricas de la Iter 1 (ver `experiments/results/pick_with_diffusion/eval_summary.json`):
- 50 epochs de fine-tune sobre 230 trayectorias (200 heurísticas + 30 ejecutadas).
- `mse_dp_vs_heuristic_mean`: <métrica del corrido>
- `dp_grasp_plausible_pct`: <métrica>

**Honestidad declarada**: la DP en Iter 1 IMITA la heurística (200/230 trayectorias del dataset son output de la heurística). NO genera trayectorias novedosas — replica el comportamiento que ya teníamos pero con la infraestructura de DP funcionando end-to-end. La Iter 2 (fuera de scope de este PR) podría escalar con trayectorias más diversas o RL.
```

(Reemplazar `<métrica del corrido>` con los valores reales del `eval_summary.json` cuando se llegue a este punto.)

### Step 11.2: Commit

```bash
git add docs/INTEGRATION_PIPELINE.md
git commit -m "docs: Iter 1 completa — Brecha B cerrada con DP entrenada en sim

INTEGRATION_PIPELINE.md actualizado con los resultados de Iter 1:
fine-tune de la DP sobre 230 trayectorias del sim + connect via IK.
Métricas finales y declaración honesta sobre las limitaciones
(la DP imita la heurística; Iter 2 escalaría diversidad).

Refs: plan Task 11."
```

---

## Verificación final

### Step F.1: Suite completa pasa

```bash
.venv/bin/pytest tests/test_coppeliasim_bridge.py tests/test_scenarios_loader.py tests/test_diffusion_scheduler.py tests/test_diffusion_dataset.py -v
```

Expected: tests existentes + nuevos pasan. ~48 passed.

### Step F.2: Cycle completo end-to-end

```bash
# Pre: CoppeliaSim corriendo
.venv/bin/python experiments/run_pick_with_diffusion.py --pose-source groundtruth 2>&1 | tail -10
.venv/bin/python experiments/run_pick_with_diffusion.py --pose-source fp_ckpt --fp-index 0 2>&1 | tail -10
```

Expected: ambos corren OK, generan `metadata.json` con `ik_converged: True`.

### Step F.3: Push del branch

```bash
git log --oneline main..HEAD | head -15
git push -u origin feat/diffusion-retrain-iter1
```

Expected: 11+ commits pushed.

---

## Checklist final del plan

- [ ] **Task 1:** `SimpleDDPMScheduler.add_noise_batch` + tests
- [ ] **Task 2:** Extraer `map_fp_pose_to_sim_workspace` a utils
- [ ] **Task 3:** `run_pick_sequence` acepta `frames_dir=None`
- [ ] **Task 4:** Phase A.1 — 200 heurísticas
- [ ] **Task 5:** Phase A.2 — 30 ejecutadas en sim
- [ ] **Task 6:** Phase A.3 — combine + split 80/20
- [ ] **Task 7:** `SimPickDataset` + tests
- [ ] **Task 8:** Train loop (50 epochs MPS) + checkpoint
- [ ] **Task 9:** `run_pick_with_diffusion.py` — Brecha B cerrada
- [ ] **Task 10:** Eval Iter 1 + summary
- [ ] **Task 11:** Documentar resultados en INTEGRATION_PIPELINE
- [ ] **F.1–F.3:** verificación + push
