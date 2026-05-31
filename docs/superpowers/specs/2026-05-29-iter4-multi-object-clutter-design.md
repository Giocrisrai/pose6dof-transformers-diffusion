# Iter 4: multi-object clutter — Diffusion Policy en escenas con distractores

## Contexto y motivación

Iter 3 alcanzó 78 % `grasp_plausible_pct_sim` con un cubo aislado. Frente al heurístico geométrico (que tiene acceso explícito a la pose, 100 %), la DP queda 22 pp atrás. La pregunta del jurado: *"si el heurístico funciona perfecto, ¿para qué entrenar una policy?"*

**Respuesta**: el heurístico necesita la pose del target Y asume bin vacío. En presencia de distractores (otros objetos en el bin), el heurístico:
1. Aún apunta correctamente al target (sigue 100 % `grasp_plausible`).
2. Pero NO ve los distractores → puede colisionar con ellos durante el approach o descent.

La DP con conditioning visual ve TODA la escena en RGB-D. En principio puede aprender a evitar distractores. Esta iter mide eso cuantitativamente.

## Goal

Demostrar que en escena con 3–8 cubos (1 target rojo, 2–7 distractores azul/verde), la DP v4 logra **`distractor_collision_pct < heurístico` con margen ≥ 20 pp**, manteniendo `grasp_plausible_pct_sim ≥ 60 %` (algo de degradación frente a v3 es esperable por la mayor complejidad).

## Escena y dataset

**Aprovechamos lo existente**: la escena `bin_base.ttt` ya tiene `/object_1` ... `/object_5` (5 cubos pre-construidos en posiciones repartidas). Para escenas con 6–8 cubos, clonamos `/object_1` vía `sim.copyPasteObjects` al inicio de cada trayectoria.

**Por trayectoria**:
- Sample `n_cubes ∈ [3, 8]` uniforme.
- Si `n_cubes > 5`, clonar `(n_cubes − 5)` cubos extra.
- Sample `n_cubes` posiciones random en el bin (`x ∈ [0.38, 0.55]`, `y ∈ [-0.17, -0.02]`), con distancia mínima de 4 cm entre centros (rechazo + reintentos, máx 50 antes de descartar la escena).
- **Pintar** el primero (`/object_1`) en rojo `(0.85, 0.15, 0.15)`; resto en azul o verde random.
- `/object_1` es siempre el **target**.

**Dataset v4** (`data/datasets/sim_pick_v4/`):
- Mismo formato que v3 pero RGB-D contiene la escena multi-objeto.
- Por trayectoria: `pose` (de `/object_1`), `rgbd` (4×224×224), `waypoints` (16×7 desde el heurístico sobre `/object_1`), `n_distractors` (escalar para análisis).
- Volumen: 2000 heurísticas + 200 ejecutadas (más data por mayor variabilidad visual). Estimado: ~2 GB.

## Arquitectura y training

**Misma arquitectura que v3**:
- ResNet-18 frozen + Linear(512, 52) trainable.
- Diffusion U-Net hidden_dim=256.
- Weighted MSE loss (peso 3× en k∈[6,10], 2× en XYZ).

**Cambios**:
- Encoder fresh (random head re-inicializado y persistido como `visual_encoder_iter4.pth`).
- 150 epochs from-scratch.
- 90/10 split.

Tiempo estimado en M1 MPS:
- Collector heuristic 2000: ~25 min (bridge persistente, igual que v3).
- Collector executed 200: ~50 min.
- Precompute embeddings: ~1 min.
- Training v4: ~10 min.
- Eval n=50 multi-obj: ~50 min.

## Métricas (las nuevas son críticas)

### Métricas existentes (reportadas)

- `grasp_plausible_pct_sim`: target ≥ 60 % (degradación leve aceptable vs 78 % v3).
- `ik_converged_pct`: target ≥ 90 %.
- `mean_grasp_proximity_m`: informativo.

### Métricas nuevas (las que justifican Iter 4)

- **`distractor_collision_pct`**: % de picks donde algún distractor se movió > 1 cm de su posición inicial al final del pick. Mide colisiones físicas.
  - Threshold: DP v4 < heurístico − 20 pp.
- **`mean_min_distractor_dist_m`**: distancia mínima en cualquier waypoint a cualquier distractor. Mide cuánto "esquiva" la policy.
  - Informativo.
- **`grasp_success_with_no_collision_pct`**: combinación: grasp_plausible AND no collision. Métrica conjunta más estricta.
  - Target: DP v4 > heurístico.

## Eval setup

`experiments/eval_diffusion_iter4_multi_sim.py`:
- 50 escenas seed=2026 (mismo seed que iter3 para comparabilidad de poses target).
- Para cada escena: sample n_cubes ∈ [3,8] con seed derivado de i.
- Spawn cubos, paint, capture RGB-D.
- Para DP v4: planner.plan_grasp con cond visual.
- Para heurístico baseline (Iter 4): plan_grasp_heuristic sobre `/object_1`.
- Medir las métricas arriba.

`experiments/eval_heuristic_baseline_multi_sim.py` (paralelo): mismas 50 escenas con heurístico.

## Comparativa esperada (predicción honesta)

| Métrica | Heurístico | DP v4 estimado | Δ |
|---|---|---|---|
| `grasp_plausible_pct_sim` | ~100 % | ~65 % | −35 pp |
| `ik_converged_pct` | ~95 % | ~85 % | −10 pp |
| `distractor_collision_pct` | 40-60 % | 15-25 % | **−25 pp** ⬅ clave |
| `grasp_success_with_no_collision_pct` | ~50 % | ~50 % | empate o ventaja DP |

La hipótesis: la DP v4 cambia ~13 pp de `grasp_plausible` por ~25 pp menos de colisiones, resultando en métrica conjunta similar o superior. Si esto se confirma, es el primer escenario donde la DP supera al heurístico.

## Riesgos

1. **El heurístico nunca colisiona**: si los distractores están suficientemente lejos del path approach→descend→lift en las 50 escenas, ninguno colisiona. Mitigación: forzar densidad de spawn (distancia mínima 4 cm pero target en zona "central"). Si aún no colisiona, la hipótesis del valor del DP es falsa para este setup — documentar honestamente y proponer denser clutter en Iter 5.
2. **DP v4 colisiona MÁS que heurístico**: el conditioning visual no le alcanza para evitar. Posibles causas: encoder demasiado frozen, dataset insuficiente, target no distinguible visualmente. Mitigación documentada con failure analysis.
3. **Spawn rechazado**: con 8 cubos en 15×10 cm, encontrar posiciones no-overlapping es difícil. Mitigación: reducir a 7 max si rechazo persiste >50 iters.
4. **copyPasteObjects no preserva propiedades**: los cubos clonados pueden no ser respondable, no ser graspable, etc. Mitigación: smoke test al inicio confirmando que un clone tiene mismas propiedades que el original.

## File structure

**Nuevos**:
- `src/simulation/multi_object_scene.py` — helpers para spawn/paint/place de cubos.
- `experiments/eval_diffusion_iter4_multi_sim.py`.
- `experiments/eval_heuristic_baseline_multi_sim.py`.

**Modificados**:
- `experiments/collect_diffusion_dataset.py` — `DATASET_VERSION = "v4"`, multi-object spawn en `phase_heuristic` y `phase_executed`.
- `experiments/precompute_visual_cond.py` — añadir override del path al v4.
- `experiments/train_diffusion_on_sim.py` — ningún cambio (lee `--dataset-dir` arbitrario).
- `experiments/run_pick_with_diffusion.py` — opcionalmente, soporte para escenarios multi-obj (DP_VERSION=v4 + spawn).
- `docs/INTEGRATION_PIPELINE.md` — nueva sección Iter 4.
- `README.md` — actualizar tabla resumen.

## Success criteria

**Iter 4 = éxito si**:
- `distractor_collision_pct` (DP) < `distractor_collision_pct` (heurístico) − 20 pp **Y**
- `grasp_plausible_pct_sim` (DP) ≥ 60 %.

**Iter 4 = fracaso parcial si**:
- DP no reduce colisiones significativamente.
- Documentar honestamente, proponer Iter 5 (closed-loop, segmentación explícita, dataset cleanup).

## Out of scope

- Objetos no-cubicos (esferas, cilindros, YCB-V) → Iter 5+.
- Closed-loop con re-captura RGB-D → Iter 5.
- Multi-target (agarrar varios) → Iter 5+.
- Stacking físico (cubos apilados) → Iter 5.
