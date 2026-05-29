# Re-entrenamiento de Diffusion Policy — Iter 2

**Fecha:** 2026-05-28
**Estado:** diseño aprobado, pendiente plan de implementación
**Autor:** ggodoy@mtechsol.com
**Branch sugerido:** `feat/diffusion-retrain-iter2`
**Spec previa:** `docs/superpowers/specs/2026-05-28-diffusion-retrain-sim-design.md` (Iter 1)

## Contexto

Iter 1 cerró la Brecha B del pipeline: la Diffusion Policy se conecta al sim
vía IK. Pero la métrica `dp_grasp_plausible_pct` quedó en **25%** (vs target
del spec de 70%). El waypoint medio (k=8) queda en promedio 7.3 cm del cubo
(máx 22.8 cm). La policy aprendió la **forma global** de la heurística
(MSE 0.0022 — 50× por debajo del threshold) pero no la **precisión** del
grasp point.

Hipótesis de las causas:
1. **Dataset chico**: 230 trayectorias × 50 epochs no son suficientes para una red
   de 346k params capturar la precisión del grasp.
2. **Loss uniforme**: el MSE pondera todos los waypoints/dims por igual.
   La fase de grasp (k=6..10) merece más peso porque ahí la precisión cuenta.
3. **Capacity insuficiente**: hidden_dim=128 puede ser demasiado chico para
   capturar la relación pose-trayectoria precisa.
4. **Fine-tune del checkpoint v1**: el checkpoint pre-existente (entrenado en otro task)
   puede haber atraído a la red a un mínimo subóptimo.

Iter 2 ataca las 4 causas simultáneamente.

## Objetivos

- Subir `dp_grasp_plausible_pct` (geométrico) de 25% a **≥70%**.
- Medir `dp_grasp_plausible_pct` **EJECUTADO EN SIM** (no solo geométrico) sobre 50 picks. Threshold ≥50%.
- Mantener `mse_dp_vs_heuristic_mean` < 0.05 (relajado vs Iter 1, porque más capacity puede overfit).
- Mantener convergencia: `ik_converged` en al menos 90% de los 50 picks.

## No-objetivos (Iter 2)

- Cambiar el pipeline base (sigue siendo IK + attach + snap).
- Reemplazar la heurística como ground truth (sigue siendo el oracle policy).
- Conectar FoundationPose en vivo (sigue siendo `fp_ckpt`).
- Re-entrenar con RL u oracle policies más sofisticadas (eso sería Iter 3+).
- Tocar `pipeline_e2e`, `e2e_live` runners.

## Arquitectura

```
┌────────────────────────────────────────────────────────────┐
│  PHASE A2 — Data collection ESCALADO (~25 min)             │
│  ┌──────────────────────┐  ┌──────────────────────────┐   │
│  │ 1500 heurísticas     │  │ 200 sim-executed         │   │
│  │ (~5s total)          │  │ (~20 min con frames=None)│   │
│  └──────────┬───────────┘  └────────┬─────────────────┘   │
│             ▼                       ▼                       │
│  experiments/collect_diffusion_dataset.py                  │
│  → data/datasets/sim_pick_v2/{train,val}.pt (1530/170)     │
└────────────────────────────┬───────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE B2 — Re-train DESDE CERO (~30 min en MPS)           │
│  experiments/train_diffusion_on_sim.py                     │
│  • ConditionalUNet1D con hidden_dim=256 (~1.4M params)     │
│  • from_scratch=True (NO carga checkpoint v1)              │
│  • Loss ponderado:                                          │
│    weights_k = [1,1,1,1,1,1, 3,3,3,3,3, 1,1,1,1,1]         │
│    weights_dim = [2,2,2, 1,1,1, 1]                          │
│    weights = weights_k * weights_dim → (16, 7) broadcast    │
│    loss = ((pred - target)² * weights).mean()              │
│  • 150 epochs, Adam lr=1e-4, batch=32                      │
│  → data/models/diffusion_policy_sim_v2.pth                  │
└────────────────────────────┬───────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE C2 — Eval EJECUTADO EN SIM (~10 min)                │
│  experiments/eval_diffusion_iter2_sim.py                   │
│  • 50 poses con seed=2026 (no vistas en training ni Iter1) │
│  • Por cada pose: ejecutar pick_with_diffusion en sim      │
│  • Capturar: grasp_proximity, deposit_error, ik_converged  │
│  • Métricas agregadas:                                      │
│    - dp_grasp_plausible_pct_sim                            │
│    - dp_deposit_plausible_pct_sim                          │
│    - dp_ik_converged_pct                                   │
│  → experiments/results/pick_with_diffusion/eval_v2_sim.json │
└────────────────────────────────────────────────────────────┘
```

## Diseño detallado

### Phase A2 — Data collection escalado

#### Cambios al collector existente

`experiments/collect_diffusion_dataset.py`:

- Nueva constante: `DATASET_VERSION = "v2"`, `DATASET_DIR = REPO / "data" / "datasets" / f"sim_pick_{DATASET_VERSION}"`.
- `N_HEURISTIC = 1500` (era 200).
- `N_EXECUTED = 200` (era 30).
- El resto de la lógica idéntica (sample_pose, phase_heuristic, phase_executed, phase_split).

**Time estimate**: heurísticas 1500 × 1ms = 1.5s. Sim-executed 200 × 6s = 20 min.

#### Output

- `data/datasets/sim_pick_v2/heuristic.pt` (1500 trayectorias)
- `data/datasets/sim_pick_v2/executed.pt` (~200, menos skipped)
- `data/datasets/sim_pick_v2/train.pt` (~1530, 90% del total)
- `data/datasets/sim_pick_v2/val.pt` (~170, 10%)

**Nota**: split cambia a 90/10 (vs 80/20 del Iter 1) para entrenar con más datos.

### Phase B2 — Re-train desde cero con loss ponderado

#### Cambios a `experiments/train_diffusion_on_sim.py`

1. **Nuevo argumento `--from-scratch`**: si está, NO carga el checkpoint inicial.
2. **Nuevo argumento `--hidden-dim`**: default 128 (backwards-compat), pero Iter 2 lo invoca con 256.
3. **Nuevo argumento `--dataset-dir`**: default `sim_pick_v1`, Iter 2 con `sim_pick_v2`.
4. **Loss ponderado**: importar `weighted_mse_loss` del nuevo módulo `src/planning/diffusion_loss.py`.

#### Nuevo módulo `src/planning/diffusion_loss.py`

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

    weights_k * weights_dim (outer product broadcast):
    - weights_k[grasp_phase_start:grasp_phase_end] = weight_grasp_phase
    - weights_dim[0:3] = weight_xyz (XYZ)
    - weights_dim[3:] = weight_rot_gripper (rotación + gripper)

    Returns: (horizon, action_dim) tensor.
    """
    weights_k = torch.ones(horizon, device=device)
    weights_k[grasp_phase_start:grasp_phase_end] = weight_grasp_phase
    weights_dim = torch.full((action_dim,), weight_rot_gripper, device=device)
    weights_dim[:3] = weight_xyz
    return weights_k.view(-1, 1) * weights_dim.view(1, -1)


def weighted_mse_loss(
    pred: torch.Tensor,    # (B, horizon, action_dim)
    target: torch.Tensor,  # same shape
    weights: torch.Tensor, # (horizon, action_dim) — broadcasts across batch
) -> torch.Tensor:
    """Mean squared error ponderado por (horizon, action_dim) matrix."""
    return ((pred - target) ** 2 * weights).mean()
```

#### Tests `tests/test_diffusion_loss.py`

```python
import torch
from src.planning.diffusion_loss import make_grasp_weights, weighted_mse_loss


def test_make_grasp_weights_shape_and_values():
    w = make_grasp_weights(horizon=16, action_dim=7)
    assert w.shape == (16, 7)
    # Verificar peso en grasp phase (k=6..10) para XYZ = 3 * 2 = 6
    assert w[8, 0].item() == 6.0
    # Verificar peso fuera del grasp phase (k=0) para XYZ = 1 * 2 = 2
    assert w[0, 0].item() == 2.0
    # Verificar peso en grasp phase para gripper = 3 * 1 = 3
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
    # Error exclusivo en k=8 (grasp phase) y XYZ
    pred = torch.zeros(1, 16, 7)
    target = torch.zeros(1, 16, 7)
    target[0, 8, 0] = 1.0  # error de 1 en k=8, dim X
    weights = make_grasp_weights()
    weighted = weighted_mse_loss(pred, target, weights).item()
    # weight[8, 0] = 6.0 vs uniform 1.0 → weighted should be 6× the unweighted
    unweighted = ((pred - target) ** 2).mean().item()
    assert abs(weighted - 6.0 * unweighted) < 1e-5
```

#### Cambio al train loop principal

```python
# ... después de cargar dataset y crear model ...
from src.planning.diffusion_loss import make_grasp_weights, weighted_mse_loss

weights = make_grasp_weights(
    horizon=config["horizon"],
    action_dim=config["action_dim"],
    device=device,
)

# en el loop:
loss = weighted_mse_loss(noise_pred, noise, weights)
```

### Phase C2 — Eval EJECUTADO EN SIM

#### Nuevo archivo `experiments/eval_diffusion_iter2_sim.py`

```python
"""Eval Iter 2 ejecutado en CoppeliaSim (no geométrico)."""
# Carga policy v2, samplea 50 poses con seed=2026.
# Por cada una: ejecuta run_pick_with_diffusion en sim, captura métricas reales.
# Reporta: dp_grasp_plausible_pct, dp_deposit_plausible_pct, dp_ik_converged_pct.
```

Cada pick toma ~5-8s sin captura de frames. 50 × 7s = ~6 min.

**Implementación**: reusa `run_pick_with_diffusion.py` extrayendo su lógica core a una función `pick_with_dp_inferences(planner, bridge, pose, capture_frames=False)` que retorna un dict de métricas.

#### Refactor mínimo necesario

`experiments/run_pick_with_diffusion.py` se refactoriza levemente: extraer la lógica de "carga policy + genera traj + ejecuta en sim + métricas" a una función reutilizable. El runner se vuelve un wrapper CLI.

### Métricas de éxito (Iter 2)

| Métrica | Threshold | Si NO se cumple |
|---|---|---|
| `mse_dp_vs_heuristic_mean` | < 0.05 | Documentar (puede ser overfit con más capacity) |
| `dp_grasp_plausible_pct` (geom) | ≥ 70% | Iter 3 (más datos / loss tuning) |
| `dp_grasp_plausible_pct` (SIM) | ≥ 50% | Iter 3 (debugging individual picks) |
| `dp_ik_converged_pct` | ≥ 90% | Investigar waypoints fuera workspace |
| Tests | 48 + 3 nuevos = 51 passing | Bloqueante |

## Estructura de archivos

**Crear:**
- `src/planning/diffusion_loss.py` (~40 líneas)
- `tests/test_diffusion_loss.py` (~50 líneas)
- `experiments/eval_diffusion_iter2_sim.py` (~150 líneas)
- `docs/superpowers/plans/2026-05-28-diffusion-retrain-iter2.md` (en writing-plans)

**Modificar:**
- `experiments/collect_diffusion_dataset.py` — `DATASET_VERSION`, `N_HEURISTIC`, `N_EXECUTED`, split 90/10
- `experiments/train_diffusion_on_sim.py` — args `--from-scratch`, `--hidden-dim`, `--dataset-dir`, uso de `weighted_mse_loss`
- `experiments/run_pick_with_diffusion.py` — extraer función reutilizable
- `.gitignore` — agregar `data/datasets/sim_pick_v2/` + `data/models/diffusion_policy_sim_v2.pth`

**Mantener (no tocar):**
- `src/planning/diffusion_policy.py` (la red existente, solo se invoca con hidden_dim=256)
- `src/planning/diffusion_dataset.py` (sin cambios)
- `experiments/eval_diffusion_iter1.py` (eval geométrico — sirve como baseline)

## Manejo de errores

| Punto | Falla | Comportamiento |
|---|---|---|
| Phase A2 collection | Bridge desconecta mid-collection | Skip pose, continuar; abortar si >10 fails |
| Phase B2 train | NaN loss | Abortar con error, no guardar checkpoint corrupto |
| Phase B2 train | OOM en MPS (improbable, modelo ~1.4M) | Fallback a CPU |
| Phase C2 eval | IK no converge en algún pick | Registrar `ik_converged=False`, continuar |
| Phase C2 eval | Bridge crash mid-pick | Skip ese pick, registrar como fallido |

## Plan de ejecución (alto nivel)

1. **B2.1: Update collector** (Phase A2): cambiar constantes + correr collection masivo (~25 min cómputo).
2. **B2.2: Loss module + tests** (TDD).
3. **B2.3: Train loop refactor**: args nuevos + uso del loss ponderado.
4. **B2.4: Re-train v2** (~30 min cómputo).
5. **B2.5: Refactor run_pick_with_diffusion** para extraer función reutilizable.
6. **B2.6: Eval en sim** (~10 min cómputo).
7. **B2.7: Documentar resultados Iter 2** en INTEGRATION_PIPELINE.md.

**Checkpoints de revisión**: después de B2.4 (training listo), B2.6 (eval listo).

## Estimación de superficie

- ~250 líneas nuevas (4 archivos)
- ~60 líneas modificadas (3 archivos)
- ~100 MB de datos persistidos (.pt + .pth)
- 3 tests nuevos

## Time total estimado

~1 h efectivo: 25 min collection + 30 min training + 10 min eval + buffers.
