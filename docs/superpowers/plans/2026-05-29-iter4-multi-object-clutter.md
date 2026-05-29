# Iter 4 Implementation Plan — Multi-object clutter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development o superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Demostrar que DP v4 con conditioning visual evita distractores mejor que el heurístico geométrico. Threshold: `distractor_collision_pct (DP) < heurístico − 20 pp` con `grasp_plausible_pct_sim (DP) ≥ 60 %`.

**Architecture:** Reusa v3 (ResNet-18 frozen + DP UNet hidden_dim=256). Cambio principal: dataset con escenas multi-objeto + nuevas métricas de colisión.

**Tech Stack:** CoppeliaSim + simIK + ZMQ; PyTorch MPS; torchvision.

---

## File Structure

**Nuevos**:
- `src/simulation/multi_object_scene.py` — spawn/paint/place cubos
- `tests/test_multi_object_scene.py` — unit tests
- `experiments/eval_heuristic_baseline_multi_sim.py`
- `experiments/eval_diffusion_iter4_multi_sim.py`

**Modificados**:
- `experiments/collect_diffusion_dataset.py` — `DATASET_VERSION = "v4"`, multi-object spawn
- `.gitignore` — `sim_pick_v4/`, `diffusion_policy_sim_v4.pth`, `visual_encoder_iter4.pth`

---

## Pre-flight

- [ ] CoppeliaSim corriendo en :23000.
- [ ] Branch: `feat/multi-object-clutter` (ya creada).
- [ ] Espacio: ≥ 3 GB libres en disco (dataset + ckpt).

---

## Task 1: `multi_object_scene.py` — helpers + tests

**Files:**
- Create: `src/simulation/multi_object_scene.py`
- Test: `tests/test_multi_object_scene.py`

### Step 1.1: Write failing tests

```python
# tests/test_multi_object_scene.py
import numpy as np
from src.simulation.multi_object_scene import (
    sample_non_overlapping_positions, BIN_X_RANGE, BIN_Y_RANGE, MIN_DIST_M,
)


def test_sample_positions_count():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=5, rng=rng)
    assert positions.shape == (5, 3)  # x, y, z


def test_sample_positions_min_dist():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=5, rng=rng)
    for i in range(5):
        for j in range(i + 1, 5):
            d = np.linalg.norm(positions[i, :2] - positions[j, :2])
            assert d >= MIN_DIST_M, f"too close: {d:.4f} between {i},{j}"


def test_sample_positions_within_bin():
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=8, rng=rng)
    for p in positions:
        assert BIN_X_RANGE[0] <= p[0] <= BIN_X_RANGE[1]
        assert BIN_Y_RANGE[0] <= p[1] <= BIN_Y_RANGE[1]


def test_sample_positions_8_extreme():
    """n=8 in 15×10 cm with min_dist=4cm es feasible."""
    rng = np.random.default_rng(0)
    positions = sample_non_overlapping_positions(n=8, rng=rng, max_retries=200)
    assert positions.shape == (8, 3)
```

### Step 1.2: Run tests — should fail (ModuleNotFoundError)

```bash
.venv/bin/python -m pytest tests/test_multi_object_scene.py -v 2>&1 | tail -5
```

### Step 1.3: Implement `multi_object_scene.py`

```python
"""Helpers para escenas multi-object: spawn, paint, sample posiciones."""
from __future__ import annotations
import numpy as np

BIN_X_RANGE = (0.38, 0.55)
BIN_Y_RANGE = (-0.17, -0.02)
Z_FIXED = 0.033
MIN_DIST_M = 0.04


def sample_non_overlapping_positions(
    n: int, rng: np.random.Generator, max_retries: int = 50
) -> np.ndarray:
    """Sample n posiciones (x, y, z) en el bin con distancia mínima MIN_DIST_M."""
    out = []
    for _ in range(n):
        for _ in range(max_retries):
            x = rng.uniform(*BIN_X_RANGE)
            y = rng.uniform(*BIN_Y_RANGE)
            p = np.array([x, y, Z_FIXED])
            if all(np.linalg.norm(p[:2] - q[:2]) >= MIN_DIST_M for q in out):
                out.append(p)
                break
        else:
            raise RuntimeError(f"sample failed after {max_retries} retries (n={n})")
    return np.array(out)


# Colores por convención: index 0 = target rojo, resto distractor azul/verde
COLOR_TARGET = (0.85, 0.15, 0.15)  # rojo
COLOR_DISTRACTOR_POOL = [
    (0.15, 0.30, 0.85),  # azul
    (0.20, 0.75, 0.20),  # verde
]


def paint_cube(sim, handle: int, color: tuple) -> None:
    """Pinta un cubo en CoppeliaSim. color es (r,g,b) en [0,1]."""
    sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, list(color))


def ensure_n_cubes(sim, n_needed: int, base_handle: int) -> list[int]:
    """Devuelve handles de n_needed cubos. Clona /object_1 si hace falta más allá de los pre-existentes."""
    pre_existing = []
    for k in range(1, 6):
        try:
            h = sim.getObject(f"/object_{k}")
            pre_existing.append(h)
        except Exception:
            break
    handles = list(pre_existing[:n_needed])
    while len(handles) < n_needed:
        new = sim.copyPasteObjects([base_handle], 0)
        if not new:
            raise RuntimeError("copyPasteObjects falló")
        handles.append(new[0])
    return handles


def setup_multi_object_scene(
    sim, n_cubes: int, rng: np.random.Generator
) -> tuple[list[int], np.ndarray]:
    """Spawn + paint n_cubes cubos. Devuelve (handles, positions). handles[0]=target rojo."""
    base = sim.getObject("/object_1")
    handles = ensure_n_cubes(sim, n_cubes, base)
    positions = sample_non_overlapping_positions(n_cubes, rng)
    for h, pos in zip(handles, positions):
        sim.setObjectPosition(h, -1, list(pos))
    paint_cube(sim, handles[0], COLOR_TARGET)
    for h in handles[1:]:
        color = COLOR_DISTRACTOR_POOL[int(rng.integers(0, len(COLOR_DISTRACTOR_POOL)))]
        paint_cube(sim, h, color)
    return handles, positions
```

### Step 1.4: Run tests — should pass

```bash
.venv/bin/python -m pytest tests/test_multi_object_scene.py -v 2>&1 | tail -10
```
Expected: 4 passed.

### Step 1.5: Smoke test integration con sim live

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.simulation.multi_object_scene import setup_multi_object_scene
from pathlib import Path
SCENE = Path('data/scenes/bin_base.ttt')
rng = np.random.default_rng(42)
with CoppeliaSimBridge() as br:
    br.load_scene(SCENE)
    br.set_stepping(True); br.start_simulation()
    handles, positions = setup_multi_object_scene(br.sim, n_cubes=7, rng=rng)
    print('handles:', handles)
    print('positions:')
    for p in positions: print(' ', p.tolist())
    rgb, depth = br.capture_rgbd()
    print(f'RGB mean: {rgb.mean():.1f}, depth range: {depth.min():.2f}-{depth.max():.2f}')
    br.stop_simulation()
"
```
Expected: handles list de 7 + posiciones razonables + RGB y depth con stats reales.

### Step 1.6: Commit

```bash
git add src/simulation/multi_object_scene.py tests/test_multi_object_scene.py
git commit -m "feat(sim): multi_object_scene helpers — spawn/paint/sample 3-8 cubos

sample_non_overlapping_positions: rejection sampling con min_dist 4 cm
en bin 15x10 cm. setup_multi_object_scene paint target en rojo y
distractors en azul/verde random. 4 tests unitarios.

Refs: spec Iter 4, plan Task 1."
```

---

## Task 2: Collector v4 — multi-object

**Files:**
- Modify: `experiments/collect_diffusion_dataset.py`
- Modify: `.gitignore`

### Step 2.1: Update DATASET_VERSION + constantes

```python
DATASET_VERSION = "v4"
N_HEURISTIC = 2000
N_EXECUTED = 200
# Multi-object
N_CUBES_RANGE = (3, 8)
```

### Step 2.2: Update `phase_heuristic`

En vez de `sample_pose(rng)` único:
- Per trayectoria, sample n_cubes ∈ [3,8].
- `setup_multi_object_scene(sim, n_cubes, rng)` posiciona y pinta.
- `target_pose = pose 4×4 de handles[0]` (cubo rojo).
- `_capture_rgbd_for_pose` ya no mueve el cubo (ya está posicionado); en vez, captura rgbd directo.
- Genera traj heurística con `plan_grasp_heuristic(target_pose, ...)`.
- Guarda además `n_distractors = n_cubes - 1`, `distractor_positions: (max_distractors, 3)` padded con NaN.

Pseudocódigo del cambio:
```python
def phase_heuristic(n=N_HEURISTIC):
    from src.simulation.multi_object_scene import setup_multi_object_scene
    ...
    n_distractors_arr = np.zeros((n,), dtype=np.int32)
    distractor_pos = np.full((n, 7, 3), np.nan, dtype=np.float32)  # max 7 distractors
    with bridge:
        bridge.load_scene(SCENE)
        bridge.set_stepping(True); bridge.start_simulation()
        for i in range(n):
            n_cubes = int(rng.integers(*N_CUBES_RANGE, endpoint=True))
            handles, positions = setup_multi_object_scene(sim, n_cubes, rng)
            target_pose = pose_from_position(positions[0])  # 4x4
            for _ in range(3): bridge.step()  # settle
            rgbd = capture_rgbd_only(bridge)  # NO setObjectPosition, ya están
            traj = planner.plan_grasp_heuristic(target_pose, ...)
            trajs[i] = traj[0]; rgbds[i] = rgbd
            poses[i] = target_pose.flatten()
            n_distractors_arr[i] = n_cubes - 1
            distractor_pos[i, :n_cubes-1] = positions[1:]
        bridge.stop_simulation()
    torch.save({
        "poses": ..., "rgbds": ..., "trajs": ...,
        "n_distractors": torch.from_numpy(n_distractors_arr),
        "distractor_positions": torch.from_numpy(distractor_pos),
        ...
    }, DATASET_DIR / "heuristic.pt")
```

Necesita helper `capture_rgbd_only(bridge)` que NO mueve el cubo. Es esencialmente `_capture_rgbd_for_pose` sin el `setObjectPosition` inicial. Refactorizar a:

```python
def _capture_rgbd_only(bridge) -> np.ndarray:
    """Captura rgbd y resize. NO toca posiciones (asume escena ya seteada)."""
    import torch.nn.functional as F
    for _ in range(3): bridge.step()  # settle
    rgb, depth = bridge.capture_rgbd()
    # ... mismo procesamiento que _capture_rgbd_for_pose
```

Y `_capture_rgbd_for_pose(bridge, pose)` lo llama después de hacer `setObjectPosition`.

### Step 2.3: Update `phase_executed`

Similar: agregar setup multi-object antes del bloque que ya tiene.

### Step 2.4: Update `phase_split`

Agregar `n_distractors` y `distractor_positions` al concat + split.

### Step 2.5: Smoke test n=3 heuristic

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from experiments.collect_diffusion_dataset import phase_heuristic
phase_heuristic(n=3)
"
```
Verificar que `heuristic.pt` contiene los nuevos campos.

### Step 2.6: Add v4 to .gitignore

```
data/datasets/sim_pick_v4/
data/models/diffusion_policy_sim_v4.pth
data/models/diffusion_policy_sim_v4.summary.json
data/models/visual_encoder_iter4.pth
```

### Step 2.7: Commit

```bash
git add experiments/collect_diffusion_dataset.py .gitignore
git commit -m "feat(planning): collector v4 — multi-object clutter (3-8 cubos)

DATASET_VERSION=v4, N_HEURISTIC=2000, N_EXECUTED=200, N_CUBES_RANGE=[3,8].
Cada trayectoria spawn n_cubes con distancias mínimas, pinta target
en rojo, distractors en azul/verde. Guarda n_distractors + distractor_positions
para análisis posterior.

Refs: spec Iter 4, plan Task 2."
```

---

## Task 3: Run collector v4

### Step 3.1: Phase heuristic (~25 min, 2000 con multi-obj)

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase heuristic > /tmp/collect_v4_heur.log 2>&1 &
```

Expected: 2000/2000 sin errores, ~1.5 GB.

### Step 3.2: Phase executed (~1h, 200 picks)

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase executed > /tmp/collect_v4_exec.log 2>&1 &
```

**No matar.** Expected: 200/200 (skipped <10), ~150 MB.

### Step 3.3: Phase split

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase split
```

Expected: `train.pt (1980), val.pt (220)`.

### Step 3.4: Verify

```bash
.venv/bin/python -c "
import torch
d = torch.load('data/datasets/sim_pick_v4/train.pt', weights_only=True)
print('keys:', list(d.keys()))
print('poses:', d['poses'].shape)
print('rgbds:', d['rgbds'].shape)
print('n_distractors hist:', torch.bincount(d['n_distractors'].long()))
"
```

---

## Task 4: Precompute embeddings v4

### Step 4.1: Modify precompute_visual_cond.py

Cambiar:
```python
DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v3"
ENCODER_CKPT = REPO / "data" / "models" / "visual_encoder_iter3.pth"
```
a:
```python
DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v4"
ENCODER_CKPT = REPO / "data" / "models" / "visual_encoder_iter4.pth"
```

(Para evitar re-trabajo, considerar `--dataset-version` CLI flag.)

### Step 4.2: Run

```bash
.venv/bin/python experiments/precompute_visual_cond.py 2>&1 | tail -5
```

Expected: ~30s, train+val overwritten con `visual_emb`.

### Step 4.3: Commit

```bash
git add experiments/precompute_visual_cond.py
git commit -m "feat(planning): precompute v4 — encoder y dataset Iter 4

Refs: plan Task 4."
```

---

## Task 5: Train v4

```bash
.venv/bin/python experiments/train_diffusion_on_sim.py \
    --dataset-dir data/datasets/sim_pick_v4 \
    --hidden-dim 256 --from-scratch --epochs 150 \
    --checkpoint-out data/models/diffusion_policy_sim_v4.pth \
    2>&1 | tail -10
```

Expected: ~10 min, `final_val_loss < 0.05`. Ckpt gitignored.

---

## Task 6: Eval Iter 4 — DP v4 y heurístico baseline en multi-obj

**Files:**
- Create: `experiments/eval_heuristic_baseline_multi_sim.py`
- Create: `experiments/eval_diffusion_iter4_multi_sim.py`

### Step 6.1: Common helper — métricas de colisión

Agregar a `src/simulation/multi_object_scene.py`:

```python
def measure_collision(
    sim, distractor_handles: list[int], initial_positions: np.ndarray,
    threshold_m: float = 0.01,
) -> tuple[bool, float]:
    """Mide si algún distractor se movió > threshold_m al final del pick.
    Devuelve (collided, max_displacement)."""
    max_disp = 0.0
    for h, pos0 in zip(distractor_handles, initial_positions):
        pos = sim.getObjectPosition(h, -1)
        d = float(np.linalg.norm(np.array(pos) - np.array(pos0)))
        max_disp = max(max_disp, d)
    return max_disp > threshold_m, max_disp
```

### Step 6.2: `eval_heuristic_baseline_multi_sim.py`

Copia de `eval_heuristic_baseline_sim.py` con:
- `setup_multi_object_scene` antes del pick.
- Guardar posiciones iniciales de distractores.
- Después del pick, llamar `measure_collision`.
- Reportar `distractor_collision_pct`, `mean_max_displacement`.

### Step 6.3: `eval_diffusion_iter4_multi_sim.py`

Copia de `eval_diffusion_iter3_sim.py` con:
- Carga `diffusion_policy_sim_v4.pth` + `visual_encoder_iter4.pth`.
- `setup_multi_object_scene` por escena.
- Mismas métricas de colisión.

### Step 6.4: Smoke test n=2 ambos

```bash
.venv/bin/python experiments/eval_heuristic_baseline_multi_sim.py --n 2 2>&1 | tail -10
.venv/bin/python experiments/eval_diffusion_iter4_multi_sim.py --n 2 2>&1 | tail -10
```

Expected: ambos completan 2/2 con números razonables.

### Step 6.5: Eval completo n=50 ambos (paralelizar si CoppeliaSim lo permite — secuencialmente seguro)

```bash
.venv/bin/python experiments/eval_heuristic_baseline_multi_sim.py --n 50 > /tmp/eval_heur_multi.log 2>&1
.venv/bin/python experiments/eval_diffusion_iter4_multi_sim.py --n 50 > /tmp/eval_v4_multi.log 2>&1
```

Expected: ~50 min cada uno.

### Step 6.6: Commit + verify

```bash
git add experiments/eval_*_multi_sim.py \
        experiments/results/pick_with_diffusion/eval_*_multi_sim.json \
        src/simulation/multi_object_scene.py  # measure_collision
git commit -m "feat(eval): Iter 4 multi-object — DP v4 y heurístico baseline

50 picks en escenas con 3-8 cubos. Métricas nuevas:
distractor_collision_pct, mean_max_displacement.

Refs: spec Iter 4, plan Task 6."
```

---

## Task 7: Documentar resultados + comparación

### Step 7.1: Update INTEGRATION_PIPELINE.md

Sección nueva: `## Iter 4 (cerrado 2026-MM-DD): multi-object clutter`. Estructura paralela a Iter 3 con tabla comparativa cuádruple (heurístico simple, DP v3, heurístico multi, DP v4).

Lectura honesta:
- Si DP v4 < heurístico en colisiones por 20 pp ✓: claim "DP justifica su existencia para clutter".
- Si no: documentar honestamente, proponer Iter 5.

### Step 7.2: Update README.md

Agregar fila a la tabla de iteraciones.

### Step 7.3: Commit + push + PR

```bash
git add docs/INTEGRATION_PIPELINE.md README.md
git commit -m "docs(integration): cerrar Iter 4 — multi-object clutter

Resultados eval n=50 escenas (3-8 cubos):
- heurístico: <X%> grasp, <X%> collision
- DP v4: <X%> grasp, <X%> collision (delta vs heurístico)

Refs: spec Iter 4, plan Task 7."
git push -u origin feat/multi-object-clutter
gh pr create --base main --head feat/multi-object-clutter \
    --title "Iter 4: multi-object clutter — DP v4 vs heurístico" \
    --body "..."
```

---

## Final verification

### F.1: Run all tests

```bash
.venv/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -5
```

Expected: 211+/211+ (los 4 nuevos de multi_object_scene).

### F.2: Confirmar CI verde post-PR

```bash
gh pr checks <num>
```
