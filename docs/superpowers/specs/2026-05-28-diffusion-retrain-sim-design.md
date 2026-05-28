# Re-entrenamiento de Diffusion Policy con trayectorias del sim — Iter 1

**Fecha:** 2026-05-28
**Estado:** diseño aprobado, pendiente plan de implementación
**Autor:** ggodoy@mtechsol.com
**Branch sugerido:** `feat/diffusion-retrain-iter1`

## Contexto

El repo tiene tres piezas del pipeline de bin-picking pero **desconectadas
entre sí**:

1. **FoundationPose** — checkpoints reales en `experiments/checkpoints/` (1098
   poses estimadas en Colab T4). Acceso de lectura ya implementado en
   `experiments/run_pick_with_fp_pose.py`.
2. **Diffusion Policy** — `ConditionalUNet1D` (346k params) con checkpoint
   `data/models/diffusion_policy_grasp.pth`. **El TFM declara que la red de
   denoising NO fue re-entrenada en este trabajo**; se usa la trayectoria
   heurística `plan_grasp_heuristic` para comparación cualitativa.
3. **Simulación CoppeliaSim** — `pick_sequence.py` con IK + attach técnica,
   genera videos MP4 plausibles (grasp_proximity 0.8 cm en los 3 escenarios).

El documento `docs/INTEGRATION_PIPELINE.md` enumera tres brechas (A, B, C).
La A se cerró parcialmente con `run_pick_with_fp_pose.py`. **Este spec ataca
la Brecha B** (Diffusion Policy → trayectoria → IK del UR5).

## Objetivos

- **Cerrar la Brecha B** con un re-entrenamiento mínimo viable (Iter 1).
- Producir un checkpoint nuevo `diffusion_policy_sim_v1.pth` entrenado sobre
  trayectorias generadas en/desde el sim.
- Generar un runner `run_pick_with_diffusion.py` que demuestra el flujo
  completo: pose → DP → waypoints → IK → sim.
- Documentar honestamente que en Iter 1 la DP **imita la heurística**
  (Iter 2, fuera de scope, mejoraría la diversidad).

## No-objetivos (Iter 1)

- Mejorar la métrica de grasp success sobre el demo actual (Iter 2).
- Re-entrenar desde cero con dataset masivo (>1000 trayectorias).
- Generar trayectorias novedosas con RL u oracle policies (Iter 2+).
- Integrar FP en vivo (sigue siendo Brecha A.v1 pendiente).
- Tocar `pipeline_e2e` o `e2e_live` runners (siguen con tiempo nominal).

## Arquitectura

```
┌───────────────────────────────────────────────────────────┐
│  PHASE A — Data collection (~25 min)                      │
│  ┌──────────────────────┐  ┌────────────────────────┐    │
│  │ Heuristic generator  │  │ Sim-executed generator │    │
│  │ (200 trayectorias)   │  │ (30 trayectorias)      │    │
│  └──────────┬───────────┘  └────────┬───────────────┘    │
│             ▼                       ▼                     │
│  experiments/collect_diffusion_dataset.py                 │
│  → data/datasets/sim_pick_v1/{train,val}.pt               │
└────────────────────────────┬──────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────┐
│  PHASE B — Fine-tune (~10 min)                            │
│  experiments/train_diffusion_on_sim.py                    │
│  • Carga diffusion_policy_grasp.pth                       │
│  • 80/20 train/val (184/46)                               │
│  • 50 epochs, Adam lr=1e-4, batch=16                      │
│  • Loss: MSE noise prediction (DDPM estándar)             │
│  → data/models/diffusion_policy_sim_v1.pth                │
└────────────────────────────┬──────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────┐
│  PHASE C — Connect to sim (~25 min)                       │
│  experiments/run_pick_with_diffusion.py                   │
│  • Toma pose (ground_truth | FP_ckpt)                     │
│  • policy.plan_grasp(pose) → (1, 16, 7) waypoints         │
│  • Para cada waypoint xyz: _move_tcp_via_ik               │
│  • Frames + MP4 + métricas (incluyendo grasp_plausible)   │
│  → experiments/results/pick_with_diffusion/               │
└───────────────────────────────────────────────────────────┘
```

## Diseño detallado

### Phase A — Data collection

#### A.1 Heuristic generator (200 trayectorias)

Variar la pose target uniformemente dentro del workspace alcanzable:

- X ∈ [0.40, 0.55] m (rango del bin en eje X)
- Y ∈ [-0.15, -0.05] m (rango del bin en eje Y)
- Z = 0.033 m (altura del cube center sobre la table; fija)
- Rotación R: tres opciones discretas — identity, π/4 alrededor de Z, π/2 alrededor de Z. Esto da variabilidad sin requerir SE(3) completo.

Para cada pose:
1. Construir matriz 4×4 `pose = [[R | t], [0 0 0 1]]`.
2. Llamar `planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)` → `(1, 16, 7)`.
3. `cond = planner.encode_observation(pose)` → `(1, 64)`.
4. Guardar tupla `(cond, traj)`.

**Output**: `data/datasets/sim_pick_v1/heuristic.pt` con dict `{conds: (200, 64), trajs: (200, 16, 7)}`.

#### A.2 Sim-executed generator (30 trayectorias)

Por cada una de 30 poses (subset random de las 200 heurísticas):

1. Cargar `bin_base.ttt`, mover `/object_1` a la pose target (deshabilitar respondable temporal para que no caiga).
2. Configurar robot con `setup_robot_control(bridge)`.
3. Ejecutar `pick_sequence` SIN frame capture (modificar para skip si `frames_dir=None`).
4. Durante la ejecución, capturar el TCP world position por step en una lista.
5. Submuestrear 870 steps → 16 waypoints usando `np.linspace(0, n-1, 16).astype(int)`.
6. Para cada waypoint: extraer xyz + rotación (de `get_tip_pose`) + gripper signal en ese step.
7. Acción 7-D: `[x, y, z, rx, ry, rz, gripper]` donde `(rx, ry, rz)` es el SO(3) log de la rotación del tip.
8. Guardar como `(cond, traj_sim)`.

**Esfuerzo**: cada pick_sequence sin frames toma ~50s → 30 × 50s = 25 min.

**Modificación requerida**: `run_pick_sequence` debe aceptar `frames_dir=None` y skipear `_capture_frame`. Una línea en `_move_tcp_via_ik` y otra en el loop principal.

**Output**: `data/datasets/sim_pick_v1/executed.pt` con `{conds: (30, 64), trajs: (30, 16, 7)}`.

#### A.3 Combine + split

`collect_diffusion_dataset.py` concatena ambos sources:
- `all_conds: (230, 64)`, `all_trajs: (230, 16, 7)`
- Split 80/20: train = 184, val = 46 (shuffled con seed=42)
- Guardado en `train.pt` y `val.pt`

### Phase B — Fine-tune

#### B.1 Dataset class

`src/planning/diffusion_dataset.py`:

```python
class SimPickDataset(Dataset):
    def __init__(self, pt_path):
        d = torch.load(pt_path, weights_only=True)
        self.conds = d['conds']    # (N, 64)
        self.trajs = d['trajs']    # (N, 16, 7)

    def __len__(self): return len(self.conds)

    def __getitem__(self, i):
        return self.conds[i], self.trajs[i]
```

#### B.2 Train loop

`experiments/train_diffusion_on_sim.py`:

1. Cargar checkpoint actual (`diffusion_policy_grasp.pth`) en `ConditionalUNet1D`.
2. Train + val `DataLoader` (batch=16, shuffle train).
3. Por epoch (50 total):
   - Para cada `(cond, traj_gt)` batch:
     - Sample `t ~ Uniform(0, T-1)` con `T = scheduler.num_timesteps = 100`
     - `noise = randn_like(traj_gt)`
     - `traj_noisy = scheduler.add_noise(traj_gt, t, noise)` (necesita una versión batch del `add_noise` existente — agregar método si no existe)
     - `noise_pred = model(traj_noisy, t, cond)`
     - `loss = MSE(noise_pred, noise)`
     - `optim.step()`
   - Eval en val: misma loss promedio.
   - Log `train_loss`, `val_loss`.
4. Guardar checkpoint con `{model_state_dict, optimizer_state_dict, train_losses, val_losses, config}` (mismo formato que el existente para compatibilidad).

**Device**: MPS si disponible, sino CPU. M1 Pro tiene MPS.

**Output**: `data/models/diffusion_policy_sim_v1.pth`

### Phase C — Connect to sim

#### C.1 `run_pick_with_diffusion.py`

Flujo:
1. Argumentos CLI: `--pose-source {groundtruth, fp_ckpt}`, `--fp-index N`.
2. Cargar policy entrenada (`diffusion_policy_sim_v1.pth`).
3. Obtener pose target:
   - groundtruth: leer `/object_1` pos del sim
   - fp_ckpt: usar `map_fp_pose_to_sim_workspace(t_pred)` (helper ya existe en `run_pick_with_fp_pose.py`, extraer a `src/simulation/utils.py`)
4. `traj = planner.plan_grasp(pose, n_samples=1)` → `(1, 16, 7)`
5. Para cada waypoint `(x, y, z, _, _, _, gripper)`:
   - Si gripper > 0.5: `set_gripper(bridge, True)`, sino False.
   - `_move_tcp_via_ik(bridge, ..., target_xyz=[x, y, z], ...)` con n_substeps reducido (10 en vez de 40, porque la DP ya da 16 puntos densos).
6. Captura frames + compile MP4.
7. Output: `experiments/results/pick_with_diffusion/{frames,demo.mp4,metadata.json}`
8. Métricas: ya cubiertas por `PickResult` (proximity, deposit_error, ik_converged). Adicionalmente `policy_path` y `dataset_version` en metadata.

#### C.2 Test set evaluation

Después del entrenamiento, ejecutar 20 picks con la DP entrenada sobre poses no vistas (sample diferente del seed). Reportar:
- `dp_grasp_plausible_pct`: % con grasp_proximity < 5 cm
- `dp_deposit_plausible_pct`: % con deposit_error < 30 cm
- `mse_dp_vs_heuristic`: distancia promedio entre trayectoria DP y la heurística sobre las mismas poses

Esto va en `experiments/results/pick_with_diffusion/eval_summary.json`.

## Estructura de archivos

**Crear:**
- `src/planning/diffusion_dataset.py` (~30 líneas)
- `experiments/collect_diffusion_dataset.py` (~150 líneas)
- `experiments/train_diffusion_on_sim.py` (~120 líneas)
- `experiments/run_pick_with_diffusion.py` (~100 líneas)
- `src/simulation/utils.py` (mover `map_fp_pose_to_sim_workspace` aquí, ~15 líneas)
- `data/datasets/sim_pick_v1/{heuristic,executed,train,val}.pt` (outputs Phase A)
- `data/models/diffusion_policy_sim_v1.pth` (output Phase B)
- `tests/test_diffusion_dataset.py` (smoke: dataset carga, shape correcto)
- `docs/superpowers/plans/2026-05-28-diffusion-retrain-iter1.md` (en writing-plans)

**Modificar:**
- `src/planning/diffusion_policy.py`: agregar `SimpleDDPMScheduler.add_noise_batch` si la versión actual no acepta batch.
- `src/simulation/pick_sequence.py`: permitir `frames_dir=None` para skip frame capture (sin output visual, más rápido).
- `experiments/run_pick_with_fp_pose.py`: importar el helper movido a `src/simulation/utils.py`.

## Manejo de errores

| Punto | Falla | Comportamiento |
|---|---|---|
| Pick ejecutado en data collection | Bridge desconectado / IK no converge | Skip la trayectoria, log warning, continuar con siguiente. Si >5 fallos consecutivos, abortar collection. |
| Train | MPS OOM (improbable, modelo pequeño) | Fallback a CPU (config flag). |
| Train | Loss diverge (NaN) | Abortar con error explícito. No guardar checkpoint corrupto. |
| Inferencia | `plan_grasp` devuelve waypoints fuera del workspace | Clamp a workspace bounds + log warning. |
| Inferencia | IK no converge en algún waypoint | Log warning, ejecutar el siguiente waypoint (best-effort). `ik_converged=False` en metadata. |

## Testing

- `tests/test_diffusion_dataset.py`: smoke test — el dataset carga, shapes correctos, len = expected.
- Tests existentes (43): deben seguir pasando.
- Validation manual:
  - `train_losses` decreciente monotónicamente (al menos los primeros 20 epochs).
  - `val_loss` no debe explotar (overfit guardrail).
  - `run_pick_with_diffusion.py` con seed fija debe producir trayectoria reproducible.

## Métricas de éxito (Iter 1)

| Métrica | Threshold |
|---|---|
| `val_loss` al final del training | < `train_loss` × 2 (no overfit grosero) |
| `train_loss` | Decreciente: epoch 50 < epoch 1 |
| `dp_grasp_plausible_pct` en eval | ≥ 70% (la DP debería imitar la heurística) |
| `mse_dp_vs_heuristic` | < 0.10 (DP produce trayectorias razonablemente cercanas a la heurística) |
| Pipeline end-to-end | `run_pick_with_diffusion.py` ejecuta sin errores en al menos 1 corrida |

## Plan de ejecución

1. **B1: Phase A — Heuristic data collector** (30 min). Generar 200 trayectorias heurísticas + guardar.
2. **B2: Phase A — Sim-executed data collector** (30 min). Modificar pick_sequence para skip frames, generar 30 ejecutadas, guardar.
3. **B3: Phase A — Combine + split** (15 min). Combinar, split 80/20, guardar train.pt / val.pt.
4. **B4: Phase B — Dataset class + tests** (20 min). `SimPickDataset` + tests/test_diffusion_dataset.py.
5. **B5: Phase B — Train loop** (40 min implementación + 10 min ejecución).
6. **B6: Phase C — Connect to sim** (40 min). `run_pick_with_diffusion.py` + helper extracted.
7. **B7: Eval test set + report** (30 min). 20 picks de test + `eval_summary.json`.
8. **B8: Documentar resultados honestos** en `docs/INTEGRATION_PIPELINE.md` (5 min).

**Checkpoints de revisión**: después de B3 (datos listos), B5 (training listo), B7 (eval listo).

## Estimación de superficie

- ~415 líneas nuevas (5 archivos)
- ~30 líneas modificadas
- ~50 MB de datos persistidos (.pt + .pth)
- 1 nuevo test
