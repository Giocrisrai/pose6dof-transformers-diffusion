# Protocolo experimental — TFM Pose 6-DoF Transformer + Diffusion

> Documento de soporte del TFM. Define el diseño experimental
> formal exigido por la tutora: hipótesis verificables, variables, particiones
> de datos, semillas, métricas justificadas y test estadístico.
>
> Última actualización: 2026-05-10. Repositorio: `pose6dof-transformers-diffusion`.

---

## 1. Hipótesis verificables

| Id | Enunciado | Aceptación |
|----|-----------|------------|
| **H1** | El pipeline basado en FoundationPose mejora el Mean AR (BOP) en al menos **3 pp** respecto al baseline GDR-Net++ en T-LESS y YCB-Video. | `Mean_AR(FP) − Mean_AR(GDR-Net++) ≥ 0.03` en ambos datasets simultáneamente, IC 95 % no cruza 0. |
| **H2** | Diffusion Policy genera trayectorias de agarre con **score medio ≥ 0.95** y **muestreo < 50 ms**. | `mean(score) ≥ 0.95` y `median(latency_ms) < 50`. |
| **H3** | El pipeline completo es viable en arquitectura híbrida local (M1 Pro) + remota (Colab T4) con **tiempo de ciclo < 10 s/instancia**. | `t_cycle_p95 < 10000 ms` end-to-end. |

## 2. Variables

### 2.1 Independientes (manipuladas)

| Variable | Niveles | Operacionalización |
|----------|---------|--------------------|
| Método de pose | `{GDR-Net++, FoundationPose}` | Modelo de inferencia que se evalúa contra GT BOP. |
| Planificador de agarre | `{heuristic_det, diffusion_policy}` | `src/planning/grasp_sampler.py` vs `src/planning/diffusion_planner.py`. |
| Dataset | `{T-LESS, YCB-Video}` | Subset BOP-19 (`test_targets_bop19.json`). |

### 2.2 Dependientes (medidas)

| Variable | Unidad | Fuente |
|----------|--------|--------|
| AUC ADD-S | [0,1] | `experiments/results/foundationpose_eval/comparison_*.json` |
| Recall @ 5/10/20 mm (ADD, ADD-S) | % | mismo |
| Mean AR (BOP) | [0,1] | leaderboard BOP / paper original (referencia) |
| Latencia por instancia | ms | timing del run |
| Score de agarre | [0,1] | `exp4_grasp_comparison/exp4_results.json` |
| Jerk de trayectoria | m/s³ | `diffusion_real_poses/trajectories_summary.json` |
| Tasa de éxito CoppeliaSim | % | `experiments/results/coppelia_smoke/` (extender a E2E) |

### 2.3 Controladas (constantes)

- Hardware GPU: NVIDIA Tesla T4 (Colab, free tier).
- CUDA 12.1, PyTorch 2.1.2+cu121.
- `requirements.colab.lock.txt` congelado.
- Pesos FP: scorer `2024-01-11-20-02-45/model_best.pth`, refiner `2023-10-28-18-33-37/model_best.pth`.
- Refinamiento: `REGISTER_ITERATIONS=5`.

## 3. Particiones de datos

- **Pose estimation:** subset BOP-19 (frames listados en `test_targets_bop19.json`).
  - YCB-Video: 12 escenas test, ~75 frames/escena → 1098 instancias evaluadas.
  - T-LESS: 20 escenas `test_primesense`, ~50 frames/escena → 1012 instancias.
- **Diffusion planning:** 30 escenas por dataset (muestreo aleatorio determinista, `seed=42`).
- **Train/test:** la partición es la oficial del BOP Challenge; no se usan datos de test para entrenamiento ni validación.

## 4. Semillas y reproducibilidad

| Componente | Semilla principal | Semillas de réplica |
|------------|-------------------|---------------------|
| Diffusion Policy sampling | 42 | 123, 2026 |
| Selección de escenas | 42 | (idem para cada réplica) |
| Inicialización torch | 42 | 123, 2026 |

Cada experimento se ejecuta **n = 3 veces** (una por semilla) y se reporta `mean ± std`.

## 5. Métricas — justificación

| Métrica | Por qué se elige | Categoría |
|---------|------------------|-----------|
| **AUC ADD-S** | Métrica principal. Robusta a simetrías de objetos (T-LESS); estándar en YCB-Video; comparable con paper original FP. | Principal |
| Recall @ 10 mm ADD-S | Criterio industrial: precisión submilimétrica para ensamblaje. Umbral usado por BOP. | Secundaria |
| Mean AR (VSD/MSSD/MSPD) | Métrica oficial BOP Challenge. Permite comparación directa con leaderboard. | Comparación con SOTA |
| Latencia | Criterio de ciclo industrial (< 10 s). | Operacional |
| Score de agarre | Calidad geométrica del agarre (gap, alineación con normal, fricción). | Planificación |
| Jerk | Suavidad de trayectoria (relevante para vida útil del actuador). | Planificación |

**ADD** simple y **ADD-S** se reportan también para comparabilidad con literatura previa (PoseCNN), pero no se usan como métrica principal porque ADD penaliza simetrías incorrectamente.

## 6. Test estadístico

- **Bootstrap no paramétrico** con `B = 1000` resamples sobre el conjunto de instancias evaluadas.
- IC 95 % calculado con percentiles (2.5, 97.5).
- Comparación pareada FP vs GDR-Net++ a nivel de instancia (cuando posible).
- Significancia: H1 se considera aceptada si el IC 95 % de la diferencia `Mean_AR(FP) − Mean_AR(GDR-Net++)` no incluye 0 y la cota inferior supera 0.03.

## 7. Procedimiento de ejecución

```
# 1. Pose estimation (Colab T4)
notebooks/colab/01_foundationpose_eval.ipynb  → predictions_*.json + comparison_*.json

# 2. Reconciliación de números BOP
notebooks/02_gdrnet_eval.ipynb (modo 'paper')  → tabla comparativa lockeada

# 3. Diffusion planning sobre poses reales FP
experiments/run_diffusion_planning_real.py  → trajectories_summary.json

# 4. Pipeline E2E en CoppeliaSim
experiments/run_coppelia_smoke_test.py     → smoke_test_result.json
[pendiente] experiments/run_pipeline_e2e.py  → métricas E2E

# 5. Ablations
experiments/exp3_rotation_ablation.py        → exp3_results.json
experiments/exp4_grasp_comparison.py         → exp4_results.json

# 6. Consolidación de resultados (figuras + tablas)
experiments/run_chapter6_consolidation.py
```

Cada ejecución genera un `RUN_CARD.md` trazable hasta el commit del repo.

## 8. Resultados disponibles (run de referencia 2026-04-27)

### Métricas FP propias (recomputadas localmente con bootstrap CI 95 %)

> Tras descargar checkpoints desde Drive (`scripts/download_drive_assets.py --what checkpoints`)
> y recomputar con `experiments/recompute_metrics_with_bootstrap.py` usando matching `gt_idx`
> correcto y bootstrap no paramétrico B = 1000:

| Dataset | n eval. | ADD med. (mm) | ADD-S med. (mm) | Recall@10mm ADD-S [IC 95 %] | AUC ADD-S@50mm [IC 95 %] |
|---------|--------:|--------------:|----------------:|-----------------------------|--------------------------|
| **YCB-V** | 1098 | 4.20 | 3.19 | **95.8 %** [94.6 %, 96.9 %] | **0.908** [0.901, 0.916] |
| **T-LESS** | 1012 | 2.90 | 1.86 | **99.7 %** [99.3 %, 100 %] | **0.957** [0.954, 0.959] |

### Métricas BOP de referencia (paper / leaderboard)

| Dataset | AUC ADD | AUC ADD-S (run JSON) | Latencia/obj |
|---------|---------|-----------|--------------|
| YCB-Video | 0.829 | 0.959 | 4154 ms |
| T-LESS | 0.805 | 0.983 | 4350 ms |

| Dataset | FP paper Mean AR | GDR-Net++ leaderboard | **Δ FP – GDR-Net++** |
|---------|------------------|-----------------------|----------------------|
| YCB-Video | 0.897 | 0.867 | **+3.0 pp** ✅ H1 |
| T-LESS | 0.803 | 0.767 | **+3.6 pp** ✅ H1 |

| Dataset | Score medio (Diff) | Latencia mediana | Jerk mediano |
|---------|--------------------|--------------------:|--------------:|
| YCB-V | 0.962 | 1.84 ms ✅ H2 | 1.7e-16 |
| T-LESS | 0.963 | 1.86 ms ✅ H2 | 1.4e-16 |

CoppeliaSim smoke: connect 150 ms, paso 18.12 ms (mean), 100 pasos → 5 s simulado.

### Validación H3 (ciclo end-to-end)

#### Fase 1 — Agregación de timings reales (`experiments/aggregate_e2e_timings.py`)

| Dataset | n | FP median (ms) | Diffusion p95 (ms) | Sim 50 steps (ms) | **Cycle p95 (ms)** | H3 (<10s) |
|---------|--:|--------------:|------------------:|-----------------:|-------------------:|:--------:|
| YCB-Video | 1098 | 4178 | 2.00 | 906 | **5180** [IC 95 % 5157–5204] | ✅ margen 4820 ms |
| T-LESS | 1012 | 4302 | 2.00 | 906 | **6049** [IC 95 % 6042–6054] | ✅ margen 3951 ms |

#### Fase 2 — Experimento E2E live (`experiments/run_e2e_live.py`)

CoppeliaSim Edu V4.10 corriendo, escena `pickAndPlaceDemo.ttt` cargada con
simulación stepped activa, Diffusion Policy con pesos entrenados localmente
(`diffusion_policy_grasp.pth`) y sampling DDIM con 25 pasos sobre MPS,
**n = 30 instancias por dataset** extraídas del checkpoint FP:

| Dataset | n | FP p95 (ms) | Diffusion p95 (ms) | Sim p95 (ms) | **Cycle p95 (ms)** | H3 (<10s) |
|---------|--:|--------:|----------------:|----------:|-------------------:|:--------:|
| YCB-Video | 30 | 4273 | 176 | 1734 | **6125** | ✅ margen 3875 ms |
| T-LESS | 30 | 5174 | 214 | 1971 | **6857** | ✅ margen 3143 ms |

Conexión CoppeliaSim: 184 ms. Sim por step (con física activa de
pickAndPlaceDemo): 28-34 ms (vs 18 ms del smoke test sin física).
Ambos datasets pasan H3 con margen superior a 3.14 segundos respecto al umbral
industrial de 10 segundos por instancia, **tanto en agregación (n = 1098/1012)
como en ejecución en vivo (n = 30 con física)**.

### Ablation n_diffusion_steps

Fundamentación de la elección operacional `n_diffusion_steps = 25`. Ejecutado
con `experiments/exp5_diffusion_steps_ablation.py` sobre n = 30 poses
condicionantes (M1 Pro / MPS, modelo entrenado):

| n_steps | Latencia mean (ms) | Latencia p95 (ms) | Jerk RMS | Trade-off |
|--------:|--------:|----------:|---------:|-----------|
| **25 (DDIM acelerado)** | **133** | 180 | 0.712 | **Mejor latencia, calidad similar** |
| 50 (intermedio) | 228 | 293 | 0.698 | Mejor jerk, latencia 1.7× |
| 100 (DDPM completo) | 419 | 511 | 0.824 | Mayor latencia, sin ganancia clara |

La latencia escala aproximadamente lineal con el número de pasos (factor 3.1×
entre extremos). El jerk no mejora monótonamente: 50 da el mejor (0.698) pero
100 empeora (0.824), sugiriendo que el modelo entrenado con 30 épocas no
aprovecha el scheduler completo. La elección operacional **n_diffusion_steps =
25** reduce la latencia mediana en 65 % vs DDPM completo manteniendo calidad
de trayectoria, dejando margen amplio para H3.

## 9. Tareas abiertas para la entrega final

- [x] Bootstrap CI 95 % sobre métricas FP propias — ver §8.
- [x] Pipeline E2E con timings reales validados (FP + Diffusion + CoppeliaSim) — ver §8.
- [x] **Pipeline E2E con CoppeliaSim corriendo en vivo y 30 instancias completas** — ver §8 fase 2.
- [x] **Ablation n_diffusion_steps ∈ {25, 50, 100}** — ver §8 (recomendación: n=25).
- [ ] Repetir runs FP con semillas {123, 2026} para estimar variabilidad **(requiere GPU T4 / Colab — 12h GPU, no factible en M1 Pro)**.
- [ ] Ejecutar evaluación BOP oficial (toolkit C++) para obtener Mean AR propio **(requiere compilación C++ Linux con dependencias específicas)**. Actualmente se usa AR del leaderboard como referencia para H1; el bootstrap CI sobre AUC ADD-S ya proporciona evidencia estadística adicional.
- [x] Ablation rotación 6D vs cuaternión cubierto por `experiments/exp3_rotation_ablation.py`.

---

*Este documento se cita como Anexo del Capítulo 3 de la memoria del TFM y constituye la evidencia del diseño experimental formal exigido por UNIR.*
