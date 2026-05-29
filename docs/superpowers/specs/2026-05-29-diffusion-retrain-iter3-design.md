# Iter 3: Diffusion Policy con conditioning visual (RGB-D) — diseño

## Contexto

Iter 2 (2026-05-28) cerró Brecha B pero quedó por debajo del threshold ≥50 % en `dp_grasp_plausible_pct_sim` (36 %). Diagnóstico documentado en `docs/INTEGRATION_PIPELINE.md`: la causa más probable es el conditioning, que sólo usa 12/64 dims del vector (pose flattened zero-padded). La red de 1.35 M params satura el dataset porque tiene poca señal discriminativa entre poses cercanas.

Iter 3 ataca ese bottleneck.

## Goal

Subir `dp_grasp_plausible_pct_sim` de 36 % → **≥55 %** en 50 picks en sim, manteniendo `dp_ik_converged_pct ≥ 90 %`, reemplazando `encode_observation` por un encoder visual ResNet-18 sobre la RGB-D del sim.

## Architecture

```
encode_observation(pose 4×4, rgbd (224,224,4)):
    rgbd -> conv1 (4-canales, conv1.weight[:, :3, :, :] = imagenet,
                   conv1.weight[:, 3, :, :] = 0)
         -> resnet18 backbone (frozen completo, incluye conv1)
         -> 512d features
         -> Linear(512, 52)            [trainable]
         -> 52d visual_emb
    pose -> flatten 12d (canal residual)
    cond = concat[visual_emb(52), pose(12)] -> 64d
```

- **ResNet-18 pretrained**: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`. Todo frozen.
- **Patch a conv1**: conv1 original es `Conv2d(3, 64, kernel=7)`. Reemplazamos por `Conv2d(4, 64, kernel=7)`. Los pesos RGB (3 canales) se copian del checkpoint y el 4° canal D se inicializa en cero. Conv1 también frozen.
- **Head trainable**: `Linear(512, 52)`. ~27 k params nuevos vs 1.35 M de la diffusion net.
- **Pose como señal residual**: 12 dims flat (R 3×3 + t 3×1 = primeras 12 entradas de la matriz). Si el encoder visual falla, la red aún tiene la pose explícita.

## Pipeline

### Phase A: dataset v3

`data/datasets/sim_pick_v3/` con la misma estructura de v2 pero cada trayectoria incluye además `rgbd_obs` capturado **al inicio del pick** (open-loop conditioning, como Chi et al. 2023).

- `heuristic.pt` — 1500 trayectorias con `(pose_4x4, rgbd, waypoints)`. Para cada pose target: colocar cubo en la pose, capturar RGB-D, generar trayectoria heurística sin ejecutar.
- `executed.pt` — 200 trayectorias ejecutadas en CoppeliaSim. Capturar RGB-D antes del descenso (que es cuando el cubo está visible y estático).
- Split 90/10 mismo que v2.
- RGB-D guardado como `uint8` (RGB) + `uint16` (depth × 1000 → mm). Decodificado a float32 al cargar.
- Resolución captura: 640×480 (default del bridge). Resize a 224×224 en el preprocess.

### Phase B: precompute visual embeddings

`experiments/precompute_visual_cond.py` corre ResNet-18 sobre las 1700 RGB-D una vez y guarda `visual_emb` (52d float32) por trayectoria. Esto desacopla el encoder pesado del training loop.

- Tiempo estimado M1 MPS: 1700 × ~10 ms = ~30 s.
- Output: `data/datasets/sim_pick_v3/{train,val}_with_emb.pt` con campo nuevo `visual_emb`.

### Phase C: training v3

Modificar `train_diffusion_on_sim.py` para que el cond venga de `visual_emb + pose_flat` (concat, total 64d), sin re-correr ResNet en cada batch.

- Hiperparams: hidden_dim=256 (igual que v2), 150 epochs from-scratch, weighted loss igual que v2.
- Tiempo estimado: ~3 min (similar a v2; el encoder ya está precomputado).
- Output: `data/models/diffusion_policy_sim_v3.pth` (gitignored).

### Phase D: eval en sim

`experiments/eval_diffusion_iter3_sim.py` corre 50 picks (seed=2026) usando `pick_with_dp` modificado:
- Captura RGB-D del bridge **antes** del primer waypoint.
- Pasa por ResNet-18 en vivo → `visual_emb` 52d.
- Concat con pose 12d → cond 64d.
- Genera trayectoria, ejecuta, mide métricas.

Output: `experiments/results/pick_with_diffusion/eval_v3_sim.json`.

## Métricas y thresholds

| Métrica | Iter 2 | Iter 3 threshold | Acción si no pasa |
|---|---|---|---|
| `dp_grasp_plausible_pct_sim` | 36 % | **≥55 %** | documentar honestamente; Iter 4 sería closed-loop |
| `dp_ik_converged_pct` | 90 % | ≥90 % | regresión: detener, debuggear |
| `dp_deposit_plausible_pct_sim` | 0 % | sin threshold (informativo) | — |
| `mean_grasp_proximity_m` | 0.056 | **<0.05** | documentar |
| `final_val_loss` (weighted MSE) | 0.051 | <0.05 | overfit check |

## File structure

**Nuevos:**
- `src/planning/visual_encoder.py` — `ResNet18RGBDEncoder` class + tests.
- `experiments/precompute_visual_cond.py` — Phase B.
- `experiments/eval_diffusion_iter3_sim.py` — Phase D.
- `tests/test_visual_encoder.py` — unit tests (shape, frozen status, conv1 init).

**Modificados:**
- `experiments/collect_diffusion_dataset.py` — agregar `rgbd_obs` por trayectoria; `DATASET_VERSION = "v3"`.
- `src/planning/diffusion_policy.py` — `DiffusionGraspPlanner.encode_observation()` acepta `visual_emb` opcional.
- `experiments/train_diffusion_on_sim.py` — `--dataset-dir` apunta a v3 con `*_with_emb.pt`.
- `experiments/run_pick_with_diffusion.py` — `pick_with_dp` captura RGB-D + corre encoder en vivo.
- `docs/INTEGRATION_PIPELINE.md` — sección Iter 3 con lectura honesta.

## Riesgos y mitigación

1. **Canal D pretrained-zero aporta poco al inicio.** Mitigación: las features RGB pretrained ya dan señal fuerte. Si el resultado es peor que v2, fallback: descongelar conv1 y entrenar 50 epochs adicionales.
2. **Vision sensor del sim mal posicionado.** Mitigación: smoke test (1 pose, capture, visualize) antes de coleccionar 1700.
3. **Dataset v3 pesa ~100 MB.** Mitigación: uint8 RGB + uint16 depth, gitignored.
4. **MPS no soporta resnet ops específicas.** Mitigación: probar 5 forwards antes del precompute. Fallback a CPU si es necesario.
5. **El encoder visual añade ~100 ms al eval por pick.** Mitigación: aceptable (eval n=50 sigue corriendo en <12 min).

## Success criteria

Iter 3 se considera **éxito** si:
- ≥55 % grasp plausible en sim **O**
- mejora ≥+15 pp sobre Iter 2 (36 % + 15 = 51 %) con análisis honesto de por qué no se llegó.

Iter 3 se considera **fracaso parcial** si:
- <50 % y mejora <+10 pp. Caso: la hipótesis del conditioning era incorrecta o secundaria.
- Acción: documentar y proponer Iter 4 (closed-loop o dataset cleanup).

## Out of scope

- Closed-loop policy (re-captura RGB-D en cada waypoint). Iter 4 si Iter 3 falla.
- Multi-objeto / multi-bin scenes. Iter 5+.
- Fine-tuning ResNet conv1+layer4. Solo si Iter 3 base falla y se quiere reintento rápido.
- Reemplazo del scheduler DDPM. Está fuera del alcance de esta brecha.
