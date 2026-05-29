# Integración Sim ↔ Training — estado y roadmap

## El pipeline conceptual del TFM

```
┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│  Captura RGB-D │──▶│  FoundationPose  │──▶│ Diffusion Policy│──▶ ┌──────────────┐
│  (cámara real  │   │  (pose 6-DoF)    │   │ (trayectoria)   │    │ Ejecución    │
│   o sim)       │   │                  │   │                 │    │ en robot     │
└────────────────┘   └──────────────────┘   └─────────────────┘    └──────────────┘
        ▲                     ▲                     ▲                      ▲
        │                     │                     │                      │
   bin_base.ttt        fp_*_ckpt.json     diffusion_policy.pth     pick_sequence.py
   (CoppeliaSim)       (1098 poses)        (entrenada parcial)     (CoppeliaSim)
```

## Estado real de cada bloque (HONESTO)

| Componente | Existe | Entrenado | Conectado al sim |
|---|---|---|---|
| **Captura RGB-D** | ✓ `bridge.capture_rgbd()` | n/a | ✓ Sí (vision sensor) |
| **FoundationPose** | ✓ checkpoints en disk | ✓ pre-trained (Wen et al. 2024) | **✗ NO** se ejecuta en el pick demo |
| **Diffusion Policy** | ✓ `diffusion_policy.py` + .pth | ⚠ parcial (la DDPM-net no fue re-entrenada en este TFM, se usa heurística) | **✗ NO** se llama desde el pick demo |
| **Ejecución en sim** | ✓ `pick_sequence.py` con IK | n/a | ✓ Sí — pero target hardcoded |
| **Métricas E2E** | ✓ `run_pipeline_e2e.py` | n/a | ⚠ usa tiempos NOMINALES de FP, no ejecuta FP real |

## Las brechas explícitas

### Brecha A: FoundationPose ↔ Pick target

**Hoy**: `run_pick_sequence` recibe `target_object="/object_1"` (str) y lee la pose
del cubo directamente del estado de CoppeliaSim. Eso ES "cheating" — usa
ground truth en vez de estimar la pose desde la imagen RGB-D.

**Cómo cerrarla**:
1. `bridge.capture_rgbd()` ya produce la imagen RGB-D.
2. Llamar FP (o cargar predicción de `fp_ycbv_checkpoint.json`) para estimar la pose.
3. Convertir la pose estimada al frame del workspace (multiplicar por matriz de la cámara).
4. Pasar esa pose XYZ como target al pick.

**Esfuerzo**: 4-6 h. Requiere convertir frame de cámara → mundo + handling de errores
si FP falla la detección.

**Valor**: el demo dejaría de ser "ground truth oracle" y pasaría a "perception + IK".

### Brecha B: Diffusion Policy ↔ Trayectoria del pick

**Hoy**: las trayectorias del pick son **keyframes hardcoded** (home, approach,
descend, lift, deposit). La diffusion policy NO se usa.

**Honestidad sobre la policy**: el TFM declara que la DDPM-net **no fue
re-entrenada en este trabajo** — se usa la trayectoria heurística generada
por `GraspSampler.sample()` + `generate_approach_trajectory()` para comparación
cualitativa. La policy "trained" en `diffusion_policy_grasp.pth` es el
checkpoint del autor del paper original (Chi et al. 2023).

**Cómo cerrarla** (si tuviera valor):
1. Pasar la pose estimada (R, t) como conditioning a `ConditionalUNet1D.forward()`.
2. Muestrear DDPM para generar 16 waypoints de 7-DOF.
3. Para cada waypoint, IK del UR5 → joint config → ejecutar.

**Esfuerzo**: 8-12 h. Pero el output de la policy random-init no es útil sin
re-entrenamiento contra trayectorias de pick reales.

**Valor**: dudoso sin re-entrenamiento dedicado. La heurística geométrica
existente es probablemente mejor para el demo.

### Brecha C: Métricas E2E ↔ Sim live

**Hoy**: `run_pipeline_e2e.py` usa `NOMINAL_FP_MS = 4154` (mediana medida en
GPU T4 en Colab el 27-04-2026) y NO ejecuta FP real. Para H3 (ciclo<10s) eso
es suficiente para reproducir el número del TFM.

**Cómo cerrarla** (si tuviera valor):
- Si hay GPU local (no es el caso aquí, MacBook M1 Pro), ejecutar FP por instancia.
- Sin GPU, mantener el tiempo nominal documentado.

**Estado**: ACEPTABLE — el TFM declara explícitamente este uso de tiempo nominal.

## Roadmap priorizado por valor/esfuerzo

| # | Iteración | Esfuerzo | Valor para defensa |
|---|---|---|---|
| 1 | **Brecha A** (FP → pick target real) | 4-6 h | ALTO — elimina el oracle, pipeline parece real |
| 2 | Adaptar pick para usar `fp_checkpoint.json` directamente (sin GPU) | 2-3 h | MEDIO — demuestra interop con outputs reales de FP |
| 3 | Re-entrenamiento de Diffusion Policy con trayectorias del sim | 1-2 días | BAJO — el demo ya cumple H3 sin esto |
| 4 | Brecha B (DP → joints) | 8-12 h | DUDOSO sin re-entrenamiento |
| 5 | Generar dataset de trayectorias desde el sim para re-entrenar la DP | 2-3 días | ALTO si se quiere re-entrenar |

## Iter 1 (cerrado 2026-05-28): retrain de la Diffusion Policy

Estado: **Brecha B cerrada en su versión mínima viable**.

Pipeline ahora ejecuta:
1. Cargar pose (groundtruth | FoundationPose checkpoint).
2. `DiffusionGraspPlanner` (re-entrenado en `diffusion_policy_sim_v1.pth`) genera 16 waypoints.
3. Cada waypoint XYZ se ejecuta en CoppeliaSim vía `_move_tcp_via_ik`.
4. Captura frames + MP4 + métricas (proximity, deposit_error, ik_converged).

Demo entry point: `experiments/run_pick_with_diffusion.py`.

### Métricas del re-entrenamiento (eval sobre 20 poses no vistas)

Ver `experiments/results/pick_with_diffusion/eval_summary.json` para datos crudos.

| Métrica | Valor | Threshold del spec | Pasó? |
|---|---|---|---|
| `mse_dp_vs_heuristic_mean` | **0.0022** | < 0.10 | ✅ |
| `dp_grasp_plausible_pct` | **25%** | ≥ 70% | ❌ |

### Lectura honesta del resultado

- ✅ La DP **imita la forma global** de la trayectoria heurística (MSE 50× por debajo del threshold). El pipeline FP → DP → IK → Sim funciona end-to-end.
- ❌ La DP **no clava el grasp point** con precisión: el waypoint medio (k=8, donde la heurística está en grasp pose) queda en promedio 7.3 cm del cubo (máx 22.8 cm). Solo 5/20 grasps serían físicamente plausibles.
- Esto es **esperado para Iter 1**: 230 trayectorias × 50 epochs sobre una red de 346k params. La policy aprende la forma pero no la precisión.

### Datos generados

- `data/datasets/sim_pick_v1/heuristic.pt` — 200 trayectorias del heurístico
- `data/datasets/sim_pick_v1/executed.pt` — 30 trayectorias ejecutadas en sim
- `data/datasets/sim_pick_v1/{train,val}.pt` — split 80/20 (184/46)
- `data/models/diffusion_policy_sim_v1.pth` — checkpoint Iter 1
- `experiments/results/pick_with_diffusion/{demo.mp4,metadata.json,eval_summary.json}`

### Roadmap Iter 2 (fuera de scope de Iter 1)

Para mejorar `dp_grasp_plausible_pct` de 25% a >70%:

1. **Más datos**: 1000+ trayectorias (vs 230 actuales), focalizadas en variación de pose target.
2. **Loss ponderado**: dar más peso a waypoints del grasp (k ∈ [6, 10]) en el MSE.
3. **Conditioning más rico**: actualmente solo 12/64 dims del cond están en uso (pose flat). Agregar features de la escena (e.g., posiciones de otros objetos para evitar colisión).
4. **Re-train desde cero** en vez de fine-tune (el checkpoint pre-existente puede estar atrayendo a un mínimo subóptimo).
5. **Eval ejecutado en sim** (no solo geométrico): 20 picks reales con `run_pick_with_diffusion.py` y medir `grasp_plausible` real.

### Recomendación para próxima iteración

Si la defensa del TFM ya está cubierta con Iter 1 (pipeline E2E funcionando), Iter 2 es opcional. Si se persigue mejorar performance: **roadmap punto 1 + 2** son los más alto-ROI (~4-8 h de trabajo).
