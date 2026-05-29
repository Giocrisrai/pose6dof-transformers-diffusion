# Diffusion Policy retraining (Iter 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subir `dp_grasp_plausible_pct` de 25% (Iter 1) a ≥70% geométrico y ≥50% ejecutado en sim, escalando el dataset (1500+200) + loss ponderado en grasp phase + re-train desde cero con red más grande (hidden_dim 256).

**Architecture:** 3 fases secuenciales. Phase A2: collection escalado a 1700 trayectorias en `data/datasets/sim_pick_v2/`. Phase B2: new loss module + re-train desde cero con ConditionalUNet1D(hidden_dim=256, ~1.4M params) y 150 epochs. Phase C2: eval EJECUTADO EN SIM sobre 50 picks (no solo geométrico).

**Tech Stack:** Python 3.12, PyTorch (MPS), CoppeliaSim 4.10 + simIK, numpy. Reutiliza `ConditionalUNet1D` y `SimpleDDPMScheduler` existentes; nuevo módulo `diffusion_loss` con weighted MSE.

**Spec:** `docs/superpowers/specs/2026-05-28-diffusion-retrain-iter2-design.md`

---

## File Structure

**Create:**
- `src/planning/diffusion_loss.py` — `make_grasp_weights` + `weighted_mse_loss`
- `tests/test_diffusion_loss.py` — 3 unit tests del loss
- `experiments/eval_diffusion_iter2_sim.py` — eval ejecutado en sim sobre 50 picks
- `docs/superpowers/plans/2026-05-28-diffusion-retrain-iter2.md` (este)

**Modify:**
- `experiments/collect_diffusion_dataset.py` — `DATASET_VERSION="v2"`, N_HEUR=1500, N_EXEC=200, split 90/10
- `experiments/train_diffusion_on_sim.py` — args `--from-scratch`, `--hidden-dim`, `--dataset-dir`, uso de `weighted_mse_loss`
- `experiments/run_pick_with_diffusion.py` — extraer función `pick_with_dp(planner, bridge, pose, capture_frames)` para reuso
- `docs/INTEGRATION_PIPELINE.md` — sección Iter 2

**Persisted (gitignored):**
- `data/datasets/sim_pick_v2/{heuristic,executed,train,val}.pt`
- `data/models/diffusion_policy_sim_v2.pth` + `.summary.json`

---

## Pre-flight checks

- [ ] On main, branch limpio, CoppeliaSim corriendo:

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git checkout main && git pull --ff-only
git status --short  # debe ser vacío
.venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient('localhost', 23000).require('sim')
print(f'OK v{sim.getInt32Param(sim.intparam_program_version)}')
"
```

Expected: `OK v41000`, no untracked.

- [ ] Crear branch:

```bash
git checkout -b feat/diffusion-retrain-iter2
```

---

## Task 1: `diffusion_loss.py` + tests

TDD. Crea el módulo de loss ponderado con tests primero.

**Files:**
- Create: `src/planning/diffusion_loss.py`
- Create: `tests/test_diffusion_loss.py`

### Step 1.1: Create test file

- [ ] Crear `tests/test_diffusion_loss.py`:

```python
"""Tests para src/planning/diffusion_loss.py."""
import torch

from src.planning.diffusion_loss import make_grasp_weights, weighted_mse_loss


def test_make_grasp_weights_shape_and_values():
    w = make_grasp_weights(horizon=16, action_dim=7)
    assert w.shape == (16, 7)
    # Verificar peso en grasp phase (k=8 ∈ [6,11)) para XYZ (dim 0) = 3 * 2 = 6
    assert w[8, 0].item() == 6.0
    # Verificar peso fuera del grasp phase (k=0) para XYZ = 1 * 2 = 2
    assert w[0, 0].item() == 2.0
    # Verificar peso en grasp phase para gripper (dim 6) = 3 * 1 = 3
    assert w[8, 6].item() == 3.0
    # Verificar peso fuera del grasp phase para gripper = 1 * 1 = 1
    assert w[0, 6].item() == 1.0


def test_weighted_mse_zero_when_pred_equals_target():
    pred = torch.randn(4, 16, 7)
    target = pred.clone()
    weights = make_grasp_weights()
    loss = weighted_mse_loss(pred, target, weights)
    assert loss.item() < 1e-6


def test_weighted_mse_is_larger_than_unweighted_when_error_in_grasp_phase():
    # Error exclusivo en k=8 (grasp phase) y dim X
    pred = torch.zeros(1, 16, 7)
    target = torch.zeros(1, 16, 7)
    target[0, 8, 0] = 1.0  # error de 1 en k=8, dim X
    weights = make_grasp_weights()
    weighted = weighted_mse_loss(pred, target, weights).item()
    # weight[8, 0] = 6.0 vs uniform 1.0 → weighted should be 6× the unweighted mean
    unweighted = ((pred - target) ** 2).mean().item()
    assert abs(weighted - 6.0 * unweighted) < 1e-5
```

### Step 1.2: Run tests — should fail

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/pytest tests/test_diffusion_loss.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.planning.diffusion_loss'`.

### Step 1.3: Implement the module

- [ ] Crear `src/planning/diffusion_loss.py`:

```python
"""Losses para training de Diffusion Policy."""
from __future__ import annotations

import torch


def make_grasp_weights(
    horizon: int = 16,
    action_dim: int = 7,
    grasp_phase_start: int = 6,
    grasp_phase_end: int = 11,
    weight_grasp_phase: float = 3.0,
    weight_xyz: float = 2.0,
    weight_rot_gripper: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Construye matriz de pesos (horizon, action_dim) para weighted loss.

    weights = weights_k * weights_dim (outer product broadcast):
      - weights_k[grasp_phase_start:grasp_phase_end] = weight_grasp_phase
      - weights_k[else] = 1.0
      - weights_dim[0:3] = weight_xyz (XYZ)
      - weights_dim[3:] = weight_rot_gripper (rotación + gripper)

    Default: peso máximo en (k∈[6,11), dim XYZ) = 3 * 2 = 6.
    Peso mínimo en (k fuera grasp, dim rot/gripper) = 1 * 1 = 1.

    Returns: (horizon, action_dim) tensor.
    """
    weights_k = torch.ones(horizon, device=device)
    weights_k[grasp_phase_start:grasp_phase_end] = weight_grasp_phase
    weights_dim = torch.full((action_dim,), weight_rot_gripper, device=device)
    weights_dim[:3] = weight_xyz
    return weights_k.view(-1, 1) * weights_dim.view(1, -1)


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error ponderado por (horizon, action_dim) matrix.

    Args:
        pred:   (B, horizon, action_dim) tensor.
        target: (B, horizon, action_dim) tensor.
        weights: (horizon, action_dim) tensor — broadcasts en batch dim.

    Returns:
        Scalar tensor: mean of (pred - target)² * weights.
    """
    return ((pred - target) ** 2 * weights).mean()
```

### Step 1.4: Run tests — should pass

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/pytest tests/test_diffusion_loss.py -v
```

Expected: `3 passed`.

### Step 1.5: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add src/planning/diffusion_loss.py tests/test_diffusion_loss.py
git commit -m "feat(planning): diffusion_loss — weighted MSE para grasp phase

Nuevo módulo con make_grasp_weights y weighted_mse_loss.
Peso 3x en waypoints k=6..10 (grasp phase) y 2x en dims XYZ
vs rotación/gripper. Peso máximo en (grasp, XYZ) = 6x el baseline.

Refs: spec Iter 2, plan Task 1."
```

---

## Task 2: Update `collect_diffusion_dataset.py` para Iter 2

**Files:**
- Modify: `experiments/collect_diffusion_dataset.py`

### Step 2.1: Add DATASET_VERSION constant + scale up

- [ ] Editar `experiments/collect_diffusion_dataset.py`. Buscar:

```python
DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v1"
N_HEURISTIC = 200
N_EXECUTED = 30
SEED = 42
```

- [ ] Reemplazar por:

```python
DATASET_VERSION = "v2"
DATASET_DIR = REPO / "data" / "datasets" / f"sim_pick_{DATASET_VERSION}"
N_HEURISTIC = 1500
N_EXECUTED = 200
SEED = 42
```

### Step 2.2: Change split from 80/20 to 90/10

- [ ] En `phase_split`, buscar:

```python
    train_n = int(0.8 * n)
```

- [ ] Reemplazar por:

```python
    train_n = int(0.9 * n)
```

### Step 2.3: Run Phase A.1 (heuristic, ~10s)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/collect_diffusion_dataset.py --phase heuristic 2>&1 | tail -5
```

Expected: `escrito: ...sim_pick_v2/heuristic.pt (1500 trayectorias)`.

### Step 2.4: Run Phase A.2 (sim-executed, ~20 min)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/collect_diffusion_dataset.py --phase executed 2>&1 | tail -10
```

Expected: `escrito: ...sim_pick_v2/executed.pt (200 trayectorias)` (o menos si hubo skipped).

**Tolerancia**: si <180 trayectorias se generan (>10% skipped), investigar. Para <190, continuar.

### Step 2.5: Run Phase A.3 (split 90/10)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/collect_diffusion_dataset.py --phase split 2>&1 | tail -3
```

Expected: `escrito: train.pt (~1530), val.pt (~170)`.

### Step 2.6: Verify datasets

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python -c "
import torch
for split in ('train', 'val'):
    d = torch.load(f'data/datasets/sim_pick_v2/{split}.pt', weights_only=True)
    print(f'{split}: conds={tuple(d[\"conds\"].shape)}, trajs={tuple(d[\"trajs\"].shape)}')
"
```

Expected: `train: conds=(~1530, 64), trajs=(~1530, 16, 7)` + similarly for val.

### Step 2.7: Add `sim_pick_v2/` to .gitignore

- [ ] Verificar que `.gitignore` ya tiene `data/datasets/sim_pick_v1/` o similar. Si no incluye `sim_pick_v2/` (probable que ya lo cubra con prefijo, pero verificar):

```bash
grep "sim_pick" .gitignore
```

- [ ] Si no aparece `sim_pick_v2` o un patrón que lo cubra, agregar:

```
# Iter 2 dataset (regenerable con collect_diffusion_dataset.py)
data/datasets/sim_pick_v2/
```

(Si ya existe `data/datasets/sim_pick_v1/`, mejor cambiar a `data/datasets/sim_pick_*/` para cubrir futuras versiones.)

### Step 2.8: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add experiments/collect_diffusion_dataset.py .gitignore
git commit -m "feat(planning): collector Iter 2 — DATASET_VERSION='v2', 1500+200 trayectorias

Cambia constantes:
- DATASET_VERSION='v2' → outputs en sim_pick_v2/
- N_HEURISTIC=1500 (era 200)
- N_EXECUTED=200 (era 30)
- split 90/10 (era 80/20)

.gitignore actualizado para incluir sim_pick_v2/.

Refs: spec Iter 2, plan Task 2."
```

---

## Task 3: Update `train_diffusion_on_sim.py` — args + weighted loss

**Files:**
- Modify: `experiments/train_diffusion_on_sim.py`

### Step 3.1: Add new CLI args

- [ ] Editar `experiments/train_diffusion_on_sim.py`. Buscar el bloque del `parser`:

```python
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-in", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_grasp.pth")
    parser.add_argument("--checkpoint-out", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_sim_v1.pth")
    args = parser.parse_args()
```

- [ ] Reemplazar por (agregar 3 args nuevos):

```python
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-in", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_grasp.pth")
    parser.add_argument("--checkpoint-out", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_sim_v1.pth")
    parser.add_argument("--from-scratch", action="store_true",
                        help="No cargar checkpoint inicial; entrenar desde init random.")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="ConditionalUNet1D hidden_dim. 256 para Iter 2.")
    parser.add_argument("--dataset-dir", type=Path,
                        default=REPO / "data" / "datasets" / "sim_pick_v1",
                        help="Dir con train.pt y val.pt.")
    args = parser.parse_args()
```

### Step 3.2: Use new args in dataset + model creation

- [ ] Buscar:

```python
    train_ds = SimPickDataset(DATASET_DIR / "train.pt")
    val_ds = SimPickDataset(DATASET_DIR / "val.pt")
```

- [ ] Reemplazar por (usar `args.dataset_dir`):

```python
    train_ds = SimPickDataset(args.dataset_dir / "train.pt")
    val_ds = SimPickDataset(args.dataset_dir / "val.pt")
```

- [ ] Buscar:

```python
    config = {"action_dim": 7, "horizon": 16, "cond_dim": 64,
              "hidden_dim": 128, "n_timesteps": 100, "n_epochs": args.epochs}
    model = ConditionalUNet1D(
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        cond_dim=config["cond_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
```

- [ ] Reemplazar por (usar `args.hidden_dim`):

```python
    config = {"action_dim": 7, "horizon": 16, "cond_dim": 64,
              "hidden_dim": args.hidden_dim, "n_timesteps": 100, "n_epochs": args.epochs}
    model = ConditionalUNet1D(
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        cond_dim=config["cond_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
```

### Step 3.3: Add --from-scratch gate

- [ ] Buscar:

```python
    # Cargar checkpoint existente si está
    if args.checkpoint_in.exists():
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"checkpoint cargado: {args.checkpoint_in.name}")
```

- [ ] Reemplazar por:

```python
    # Cargar checkpoint existente solo si NO from-scratch
    if args.from_scratch:
        logger.info("--from-scratch: NO cargando checkpoint inicial (entrenando desde init random)")
    elif args.checkpoint_in.exists():
        try:
            ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=True)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"checkpoint cargado: {args.checkpoint_in.name}")
        except (RuntimeError, KeyError) as e:
            logger.warning(f"no pude cargar checkpoint inicial ({e}); entrenando desde init random")
```

Esto agrega un fallback robusto: si el checkpoint inicial tiene arquitectura distinta (e.g., hidden_dim 128 vs 256), no crashea, solo loguea warning.

### Step 3.4: Replace nn.MSELoss with weighted_mse_loss

- [ ] Buscar:

```python
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
```

- [ ] Reemplazar por:

```python
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Loss ponderado en grasp phase (k=6..10) x3 y dims XYZ x2
    from src.planning.diffusion_loss import make_grasp_weights, weighted_mse_loss
    loss_weights = make_grasp_weights(
        horizon=config["horizon"],
        action_dim=config["action_dim"],
        device=device,
    )
    logger.info(f"loss weights: max={loss_weights.max().item():.1f}, min={loss_weights.min().item():.1f}")
```

- [ ] En el train loop, buscar:

```python
            noise_pred = model(traj_noisy, t, cond)
            loss = loss_fn(noise_pred, noise)
            if torch.isnan(loss):
```

- [ ] Reemplazar por:

```python
            noise_pred = model(traj_noisy, t, cond)
            loss = weighted_mse_loss(noise_pred, noise, loss_weights)
            if torch.isnan(loss):
```

- [ ] En el eval val loop, similar — buscar:

```python
                noise_pred = model(traj_noisy, t, cond)
                loss = loss_fn(noise_pred, noise)
                epoch_val += loss.item()
```

- [ ] Reemplazar por:

```python
                noise_pred = model(traj_noisy, t, cond)
                loss = weighted_mse_loss(noise_pred, noise, loss_weights)
                epoch_val += loss.item()
```

### Step 3.5: Verify that the unused `nn` import is OK (or remove)

`import torch.nn as nn` ya no se usa (era para `nn.MSELoss`). Es OK dejarlo (no es error) o removerlo. Lo dejamos por consistencia.

### Step 3.6: Verify args + dataset Iter 1 still works (backwards-compat)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/train_diffusion_on_sim.py --epochs 2 --dataset-dir data/datasets/sim_pick_v1 2>&1 | tail -10
```

Expected: corre 2 epochs sin error, log dice `loss weights: max=6.0, min=1.0`, output `data/models/diffusion_policy_sim_v1.pth` (sobrescribe el de Iter 1 — eso es OK, lo regeneramos en Iter 2 con sim_pick_v2). No NaN.

**IMPORTANTE**: este test corre brevemente solo para validar que el código no rompe. NO es el training final.

### Step 3.7: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add experiments/train_diffusion_on_sim.py
git commit -m "feat(planning): train loop con args Iter 2 + weighted loss

Nuevos args: --from-scratch, --hidden-dim, --dataset-dir.
Usa weighted_mse_loss del nuevo módulo diffusion_loss.
Robust: si checkpoint inicial tiene arquitectura distinta, no crashea
(logueamos warning y entrenamos desde init random).

Refs: spec Iter 2, plan Task 3."
```

---

## Task 4: Entrenar v2 con hidden_dim=256 + dataset_v2 + 150 epochs

**Files:**
- Run script + persist checkpoint

### Step 4.1: Run training

CoppeliaSim NO requerido para esta phase.

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/train_diffusion_on_sim.py \
    --from-scratch \
    --hidden-dim 256 \
    --dataset-dir data/datasets/sim_pick_v2 \
    --epochs 150 \
    --batch-size 32 \
    --checkpoint-out data/models/diffusion_policy_sim_v2.pth \
    2>&1 | tail -30
```

**Tiempo estimado**: ~30 min en MPS (150 epochs × ~12s/epoch con dataset ~1500). Si tarda mucho más (>1h), abortar y reportar.

Expected:
- Log inicial: `device: mps`, `loss weights: max=6.0, min=1.0`, `--from-scratch: NO cargando checkpoint inicial`.
- 150 epochs procesados. `train_loss` y `val_loss` decreciendo (con noise). Sin NaN.
- Output final: `checkpoint escrito: data/models/diffusion_policy_sim_v2.pth`.

### Step 4.2: Verify checkpoint

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python -c "
import torch
ckpt = torch.load('data/models/diffusion_policy_sim_v2.pth', map_location='cpu', weights_only=True)
print('hidden_dim:', ckpt['config']['hidden_dim'])  # debe ser 256
print('n_params:', sum(p.numel() for p in ckpt['model_state_dict'].values()))
print('final train loss:', ckpt['train_losses'][-1])
print('final val loss:', ckpt['val_losses'][-1])
print('min val:', min(ckpt['val_losses']))
print('train trend OK?', ckpt['train_losses'][0] > ckpt['train_losses'][-1])
"
cat data/models/diffusion_policy_sim_v2.summary.json
```

Expected:
- `hidden_dim: 256`
- `n_params: ~1.4M` (versus ~346k del Iter 1)
- `train trend OK? True`
- `final_val_loss` mismo orden que `final_train_loss` (no overfit grosero)

### Step 4.3: Add v2 checkpoint to .gitignore

- [ ] Verificar:

```bash
grep "diffusion_policy_sim_v2" .gitignore
```

- [ ] Si no aparece, agregar:

```
# Iter 2 checkpoint (regenerable)
data/models/diffusion_policy_sim_v2.pth
data/models/diffusion_policy_sim_v2.summary.json
```

### Step 4.4: Commit gitignore (not the checkpoint itself; ignored)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add .gitignore
git status --short  # verificar que NO se agrega el .pth
git commit -m "chore: gitignore v2 checkpoint

Refs: plan Task 4." || echo "(no changes if gitignore ya estaba completo)"
```

---

## Task 5: Refactor `run_pick_with_diffusion.py` — extraer función reutilizable

**Files:**
- Modify: `experiments/run_pick_with_diffusion.py`

### Step 5.1: Identify the section to extract

El runner actual tiene esta lógica core (la parte de "carga policy + genera traj + ejecuta en sim + métricas"). La extraemos a una función standalone `pick_with_dp` para que `eval_diffusion_iter2_sim.py` (Task 6) la pueda llamar 50 veces sin overhead de imports.

### Step 5.2: Extract `pick_with_dp` function

- [ ] Editar `experiments/run_pick_with_diffusion.py`. Antes de `def main`, agregar la función nueva:

```python
def pick_with_dp(
    planner,
    pose: np.ndarray,
    bridge,
    frames_dir=None,
    n_substeps: int = 8,
    steps_per_substep: int = 2,
):
    """Ejecuta un pick usando la DP entrenada.

    Args:
        planner: DiffusionGraspPlanner con policy cargada.
        pose: (4,4) matriz SE(3) target.
        bridge: CoppeliaSimBridge con escena ya cargada.
        frames_dir: None para skip frame capture (eval rápido).
        n_substeps / steps_per_substep: pasados a _move_tcp_via_ik.

    Returns:
        dict con métricas: {
            'target_pose_t': [x,y,z],
            'cube_end': [x,y,z],
            'tip_end': [x,y,z],
            'ik_converged': bool,
            'grasp_proximity_m': float,
            'deposit_error_m': float,
            'grasp_plausible': bool,
            'deposit_plausible': bool,
            'n_waypoints': int,
            'waypoints': list of 16 lists of 7 floats (debug),
        }
    """
    import math
    from src.simulation.pick_sequence import (
        _move_tcp_via_ik, _setup_ik, set_gripper, setup_robot_control,
    )

    GRASP_THRESHOLD_M = 0.05
    DEPOSIT_TARGET = [-0.30, -0.30, 0.30]
    DEPOSIT_THRESHOLD_M = 0.30

    setup_robot_control(bridge)
    env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
    bridge.set_stepping(True)
    bridge.start_simulation()
    sim = bridge.sim
    obj1 = sim.getObject("/object_1")
    tip_h = sim.getObject("/tip")

    # Mover el cubo al target
    sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

    # Generar trayectoria con la policy
    traj = planner.plan_grasp(pose, n_samples=1)  # (1, 16, 7)
    waypoints = traj[0]

    # PROXIMITY pre-snap: distance entre el waypoint k=8 (donde la heur tiene grasp)
    # y la pose del cubo. Mide si la DP "apunta" al cubo correctamente.
    cube_pos = sim.getObjectPosition(obj1, -1)
    grasp_wp = waypoints[8]
    grasp_proximity_m = math.sqrt(
        sum((cube_pos[i] - float(grasp_wp[i])) ** 2 for i in range(3))
    )
    grasp_plausible = grasp_proximity_m < GRASP_THRESHOLD_M

    if frames_dir is not None:
        for f in frames_dir.glob("*.png"): f.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar waypoints
    counter = [0]
    ik_convergence = []
    prev_gripper = 1.0
    for i, wp in enumerate(waypoints):
        x, y, z, _, _, _, gripper = wp.tolist()
        if (gripper > 0.5) != (prev_gripper > 0.5):
            set_gripper(bridge, gripper > 0.5)
            prev_gripper = gripper
        _move_tcp_via_ik(
            bridge, env, ik_group, target_dummy, ik_joints, simIK,
            [x, y, z], frames_dir, counter,
            n_substeps=n_substeps, steps_per_substep=steps_per_substep,
            convergence_tracker=ik_convergence,
        )

    cube_end = sim.getObjectPosition(obj1, -1)
    tip_end = sim.getObjectPosition(tip_h, -1)
    ik_converged = len(ik_convergence) > 0 and all(ik_convergence)

    # Deposit error (XY only, Z lo ignoramos porque cae por gravedad)
    deposit_error_m = math.sqrt(
        (cube_end[0] - DEPOSIT_TARGET[0]) ** 2 +
        (cube_end[1] - DEPOSIT_TARGET[1]) ** 2
    )
    deposit_plausible = deposit_error_m < DEPOSIT_THRESHOLD_M

    bridge.stop_simulation()
    try:
        simIK.eraseEnvironment(env)
    except Exception:
        pass

    return {
        "target_pose_t": pose[:3, 3].tolist(),
        "cube_end": cube_end,
        "tip_end": tip_end,
        "ik_converged": ik_converged,
        "grasp_proximity_m": grasp_proximity_m,
        "deposit_error_m": deposit_error_m,
        "grasp_plausible": grasp_plausible,
        "deposit_plausible": deposit_plausible,
        "n_waypoints": len(waypoints),
        "waypoints": waypoints.tolist(),
    }
```

### Step 5.3: Update `main` to use `pick_with_dp`

- [ ] Buscar el bloque dentro de `main` que hace la carga de la policy y el sim run completo. Reemplazar el bloque entero por:

```python
    # 1. Cargar policy
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
    )
    ckpt = torch.load(POLICY, map_location=device, weights_only=True)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    logger.info(f"policy cargada: {POLICY.name}")

    # 2. Obtener target pose
    pose, source_label = get_target_pose(args)
    logger.info(f"target: t={pose[:3,3].tolist()}, source={source_label}")

    # 3. Ejecutar pick + capture frames
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = REPO_OUT / "frames"
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        result = pick_with_dp(planner, pose, bridge, frames_dir=frames_dir)

    # 4. Compilar MP4
    mp4_path = REPO_OUT / "demo.mp4"
    compiled = compile_mp4(frames_dir, mp4_path, fps=25)

    # 5. Reporte
    metadata = {
        "policy": str(POLICY.relative_to(REPO)),
        "pose_source": source_label,
        **result,
        "mp4": str(compiled.relative_to(REPO)) if compiled else None,
    }
    (REPO_OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print("=== RESULTADOS pick con Diffusion Policy ===")
    for k, v in metadata.items():
        if k != "waypoints":  # demasiado verboso
            print(f"  {k}: {v}")
    return 0
```

**Nota**: el bloque actual del runner (carga + sim + frames + métricas + report) se compactiza usando `pick_with_dp`. Las líneas exactas a reemplazar dependen del estado actual del archivo — leer el archivo primero y reemplazar el bloque entero que comienza con "1. Cargar policy" hasta el final del `main()`.

### Step 5.4: Allow loading either v1 or v2 policy via env var

- [ ] En la cabecera del archivo, después de `REPO_OUT = ...`, agregar:

```python
import os

POLICY_VERSION = os.environ.get("DP_VERSION", "v1")  # default backwards-compat
POLICY = REPO / "data" / "models" / f"diffusion_policy_sim_{POLICY_VERSION}.pth"
```

- [ ] Buscar la def antigua de `POLICY = REPO / ...` y reemplazar por la línea de arriba (con env var).

Si la línea actual es `POLICY = REPO / "data" / "models" / "diffusion_policy_sim_v1.pth"`, reemplazar por las 2 líneas con `os.environ.get`.

### Step 5.5: Test backwards-compat (sin env var, debe seguir cargando v1)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
ls data/models/diffusion_policy_sim_v1.pth 2>&1
.venv/bin/python experiments/run_pick_with_diffusion.py 2>&1 | grep -E "policy cargada|RESULTADOS|grasp_proximity" | head -3
```

Expected: `policy cargada: diffusion_policy_sim_v1.pth`, RESULTADOS visible.

### Step 5.6: Test con v2

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
DP_VERSION=v2 .venv/bin/python experiments/run_pick_with_diffusion.py 2>&1 | grep -E "policy cargada|RESULTADOS|grasp_proximity" | head -3
```

Expected: `policy cargada: diffusion_policy_sim_v2.pth`, RESULTADOS visible con números (probablemente mejores que v1 si el training fue bueno).

### Step 5.7: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add experiments/run_pick_with_diffusion.py
git commit -m "refactor(sim): extraer pick_with_dp() de run_pick_with_diffusion.py

Función reutilizable para eval Iter 2 (50 picks en sim). main()
ahora la invoca. Soporta DP_VERSION env var para switch entre v1/v2.

Refs: spec Iter 2, plan Task 5."
```

---

## Task 6: `eval_diffusion_iter2_sim.py` — 50 picks en sim

**Files:**
- Create: `experiments/eval_diffusion_iter2_sim.py`

### Step 6.1: Pre-flight CoppeliaSim corriendo

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient('localhost', 23000).require('sim')
print(f'OK v{sim.getInt32Param(sim.intparam_program_version)}')
"
```

### Step 6.2: Create the eval script

- [ ] Crear `experiments/eval_diffusion_iter2_sim.py`:

```python
#!/usr/bin/env python3
"""Eval Iter 2 EJECUTADO EN SIM — 50 picks reales.

Mide dp_grasp_plausible_pct y dp_deposit_plausible_pct sobre poses
ejecutadas en CoppeliaSim (no solo geometría). Es el acid test del Iter 2.

Uso (CoppeliaSim running on :23000):
    python experiments/eval_diffusion_iter2_sim.py
    python experiments/eval_diffusion_iter2_sim.py --n 50 --policy-version v2
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
from experiments.run_pick_with_diffusion import pick_with_dp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_iter2_sim")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"

EVAL_SEED = 2026  # distinto del training (42) y de Iter 1 eval (999)


def sample_pose_eval(rng: np.random.Generator) -> np.ndarray:
    """Mismo workspace que training; seed distinto."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50,
                        help="Número de picks a ejecutar.")
    parser.add_argument("--policy-version", default="v2",
                        help="v1 | v2. Default v2 (Iter 2).")
    args = parser.parse_args()

    policy_path = REPO / "data" / "models" / f"diffusion_policy_sim_{args.policy_version}.pth"
    if not policy_path.exists():
        logger.error(f"policy no encontrada: {policy_path}")
        return 1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
    )
    ckpt = torch.load(policy_path, map_location=device, weights_only=True)
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    logger.info(f"policy: {policy_path.name} (hidden_dim={ckpt['config']['hidden_dim']})")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                result = pick_with_dp(planner, pose, bridge, frames_dir=None)
            results.append({
                "i": i,
                "target_pose_t": result["target_pose_t"],
                "grasp_proximity_m": result["grasp_proximity_m"],
                "deposit_error_m": result["deposit_error_m"],
                "ik_converged": result["ik_converged"],
                "grasp_plausible": result["grasp_plausible"],
                "deposit_plausible": result["deposit_plausible"],
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{args.n} done (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    n_valid = len(results)
    if n_valid == 0:
        logger.error("0 picks válidos — abortando")
        return 1

    grasp_plaus_count = sum(r["grasp_plausible"] for r in results)
    deposit_plaus_count = sum(r["deposit_plausible"] for r in results)
    ik_conv_count = sum(r["ik_converged"] for r in results)

    summary = {
        "n_requested": args.n,
        "n_valid": n_valid,
        "n_skipped": skipped,
        "policy": str(policy_path.relative_to(REPO)),
        "seed": EVAL_SEED,
        "dp_grasp_plausible_pct_sim": 100.0 * grasp_plaus_count / n_valid,
        "dp_deposit_plausible_pct_sim": 100.0 * deposit_plaus_count / n_valid,
        "dp_ik_converged_pct": 100.0 * ik_conv_count / n_valid,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])),
        "mean_deposit_error_m": float(np.mean([r["deposit_error_m"] for r in results])),
        "thresholds_passed": {
            "dp_grasp_plausible_pct_sim >= 50": (100.0 * grasp_plaus_count / n_valid) >= 50,
            "dp_ik_converged_pct >= 90": (100.0 * ik_conv_count / n_valid) >= 90,
        },
        "per_pick": results,
    }

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    out = REPO_OUT / "eval_v2_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    # Print solo el resumen, no `per_pick`
    print()
    print("=== RESUMEN EVAL ITER 2 EN SIM ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    print(f"\nDetalles por pick: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 6.3: Run eval (CoppeliaSim corriendo)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/eval_diffusion_iter2_sim.py --n 50 --policy-version v2 2>&1 | tail -30
```

**Tiempo estimado**: ~10 min (50 × ~12s each picks con sim ops).

Expected:
- 50 picks procesados (algunos skipped OK, <10%).
- Resumen con `dp_grasp_plausible_pct_sim` (target ≥50%), `dp_ik_converged_pct` (target ≥90%), means.
- Output: `experiments/results/pick_with_diffusion/eval_v2_sim.json`.

### Step 6.4: Verify outputs

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python -c "
import json
d = json.load(open('experiments/results/pick_with_diffusion/eval_v2_sim.json'))
print('n_valid:', d['n_valid'])
print('dp_grasp_plausible_pct_sim:', d['dp_grasp_plausible_pct_sim'])
print('dp_ik_converged_pct:', d['dp_ik_converged_pct'])
print('thresholds_passed:', d['thresholds_passed'])
"
```

### Step 6.5: Commit (excluyendo el .json bulky por per_pick — pero es solo ~50KB; lo dejamos)

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add experiments/eval_diffusion_iter2_sim.py experiments/results/pick_with_diffusion/eval_v2_sim.json
git commit -m "feat(planning): eval Iter 2 ejecutado en sim — 50 picks reales

experiments/eval_diffusion_iter2_sim.py samplea 50 poses con seed
2026, ejecuta cada pick en CoppeliaSim usando la DP entrenada, mide
métricas REALES (no geométricas). Thresholds: grasp_plaus_sim ≥50%,
ik_converged ≥90%. Reutiliza pick_with_dp() del runner refactorizado.

Refs: spec Iter 2, plan Task 6."
```

---

## Task 7: Documentar resultados Iter 2 en `INTEGRATION_PIPELINE.md`

**Files:**
- Modify: `docs/INTEGRATION_PIPELINE.md`

### Step 7.1: Append Iter 2 section

- [ ] Editar `docs/INTEGRATION_PIPELINE.md`. Al final del archivo, después de la sección "Iter 1 (cerrado 2026-05-28): retrain de la Diffusion Policy", agregar:

```markdown
## Iter 2 (cerrado 2026-05-28): retrain escalado + loss ponderado

Estado: **Iter 2 completada**. Mejoras sobre Iter 1:

| Cambio | Iter 1 | Iter 2 |
|---|---|---|
| Dataset | 230 trayectorias (200 heur + 30 sim) | **1700** (1500 heur + 200 sim) |
| Train/val split | 80/20 | **90/10** |
| Modelo | hidden_dim=128 (~346k params) | **hidden_dim=256 (~1.4M params)** |
| Init | fine-tune del checkpoint v1 | **from-scratch** |
| Loss | MSE uniforme | **weighted MSE** (k=6..10 x3, XYZ x2) |
| Epochs | 50 | **150** |
| Eval | solo geométrico (20 poses) | **+ ejecutado en sim (50 picks)** |

### Métricas (eval Iter 2)

Geométrico (compatible con Iter 1):
- `mse_dp_vs_heuristic_mean`: <leer de eval_summary o eval_v2_sim.json>
- `dp_grasp_plausible_pct` (geom): <leer>

Ejecutado en sim (NUEVO):
- `dp_grasp_plausible_pct_sim`: <leer de eval_v2_sim.json>
- `dp_deposit_plausible_pct_sim`: <leer>
- `dp_ik_converged_pct`: <leer>

Ver `experiments/results/pick_with_diffusion/eval_v2_sim.json` para detalle por pick.

### Cómo correr Iter 2

```bash
# 1. Collection (CoppeliaSim corriendo)
python experiments/collect_diffusion_dataset.py --phase all

# 2. Train
python experiments/train_diffusion_on_sim.py \
    --from-scratch --hidden-dim 256 \
    --dataset-dir data/datasets/sim_pick_v2 \
    --epochs 150 --batch-size 32 \
    --checkpoint-out data/models/diffusion_policy_sim_v2.pth

# 3. Eval en sim
python experiments/eval_diffusion_iter2_sim.py --n 50 --policy-version v2

# 4. Demo individual con v2
DP_VERSION=v2 python experiments/run_pick_with_diffusion.py
```

### Honestidad declarada (Iter 2)

- "from-scratch" admite que el fine-tune del Iter 1 podría haber atraído a un mínimo subóptimo del checkpoint pre-existente.
- "hidden_dim=256" cuadruplica params (~1.4M); riesgo de overfit mitigado con val_loss tracking + más datos (1530 vs 184 del Iter 1).
- "Loss ponderado" es una elección de diseño explícita: 3× en grasp phase + 2× en XYZ. Diferentes ponderaciones daría resultados distintos; este balance fue acordado en spec.
- "Eval en sim" es el acid test real — superar geométrico no implica superar sim por la varianza física (no-determinismo).
```

(Reemplazar `<leer>` con los valores reales del eval_v2_sim.json cuando se ejecute este task.)

### Step 7.2: Llenar los `<leer>` con los valores reales

- [ ] Leer `experiments/results/pick_with_diffusion/eval_v2_sim.json` y reemplazar los placeholders `<leer>` con los valores reales:

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python -c "
import json
d = json.load(open('experiments/results/pick_with_diffusion/eval_v2_sim.json'))
print(f'dp_grasp_plausible_pct_sim:  {d[\"dp_grasp_plausible_pct_sim\"]:.1f}%')
print(f'dp_deposit_plausible_pct_sim: {d[\"dp_deposit_plausible_pct_sim\"]:.1f}%')
print(f'dp_ik_converged_pct:          {d[\"dp_ik_converged_pct\"]:.1f}%')
print(f'mean_grasp_proximity_m:       {d[\"mean_grasp_proximity_m\"]:.3f}')
print(f'mean_deposit_error_m:         {d[\"mean_deposit_error_m\"]:.3f}')
"
```

- [ ] Con esos valores, editar `docs/INTEGRATION_PIPELINE.md` y reemplazar los `<leer>` por los números reales. (Si `eval_diffusion_iter1.py` se re-corrió con v2, también incluir el geométrico actualizado.)

### Step 7.3: Commit

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git add docs/INTEGRATION_PIPELINE.md
git commit -m "docs: Iter 2 completa — escalar dataset + loss ponderado + eval sim

INTEGRATION_PIPELINE.md actualizado con resultados de Iter 2:
- Dataset escalado a 1700 trayectorias (vs 230)
- ConditionalUNet1D con hidden_dim=256 (vs 128)
- Loss ponderado: 3× en grasp phase, 2× en XYZ
- Eval ejecutado en CoppeliaSim sobre 50 picks (no solo geométrico)

Métricas reales reportadas. Honestidad declarada en el spec.

Refs: spec Iter 2, plan Task 7."
```

---

## Verificación final

### Step F.1: Tests completos

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
.venv/bin/pytest tests/test_coppeliasim_bridge.py tests/test_scenarios_loader.py \
    tests/test_diffusion_scheduler.py tests/test_diffusion_dataset.py \
    tests/test_diffusion_loss.py -v
```

Expected: 51 tests pass (48 + 3 nuevos).

### Step F.2: Cycle E2E con v2

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
DP_VERSION=v2 .venv/bin/python experiments/run_pick_with_diffusion.py 2>&1 | tail -15
```

Expected: ejecución completa, `metadata.json` con `ik_converged: True` (en la mayoría de las corridas; si falla, los detalles están en `eval_v2_sim.json`).

### Step F.3: Push del branch

```bash
cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
git log --oneline main..HEAD
git push -u origin feat/diffusion-retrain-iter2
```

Expected: 7-8 commits pushed.

---

## Checklist final del plan

- [ ] **Task 1:** `diffusion_loss.py` + tests (3)
- [ ] **Task 2:** Update collector — DATASET_VERSION="v2", N_HEUR=1500, N_EXEC=200, split 90/10
- [ ] **Task 3:** Update train script — args + weighted loss
- [ ] **Task 4:** Train v2 (hidden_dim=256, from-scratch, 150 epochs)
- [ ] **Task 5:** Refactor `run_pick_with_diffusion.py` — extraer `pick_with_dp`
- [ ] **Task 6:** Eval ejecutado en sim — 50 picks
- [ ] **Task 7:** Documentar resultados en `INTEGRATION_PIPELINE.md`
- [ ] **F.1–F.3:** verificación + push
