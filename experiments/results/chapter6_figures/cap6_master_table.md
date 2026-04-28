# Tabla maestra del cap. 6 — TFM

_Generada 2026-04-27 20:24 por `experiments/run_chapter6_consolidation.py`_


## 1. FoundationPose — métricas reales (subset BOP-19)

| Métrica | YCB-V | T-LESS |
|---------|-------|--------|
| Objetos evaluados | 1098 | 1012 |
| ADD mediana (mm) | 4.17 | 2.90 |
| ADD-S mediana (mm) | 2.09 | 1.36 |
| AUC ADD | 0.829 | 0.805 |
| AUC ADD-S | 0.959 | 0.983 |
| Recall@10mm ADD-S | 96.5% | 99.7% |

_Fuente: `experiments/results/foundationpose_eval/comparison_20260427_084807.json`_


## 2. Comparación con baselines oficiales (Mean AR)

| Dataset | FP propio (ADD med) | FP paper (Mean AR) | GDR-Net++ BOP22 (Mean AR) |
|---------|---------------------|--------------------|---------------------------|
| YCBV | 4.17 mm | 0.884 | 0.845 |
| TLESS | 2.90 mm | 0.777 | 0.731 |

_Nota: ADD/ADD-S y Mean AR son métricas distintas; comparación cualitativa documentada en cap. 6._


## 3. Diffusion + Grasp Sampler con poses reales

| Métrica | YCB-V | T-LESS |
|---------|-------|--------|
| Poses con éxito | 30/30 | 30/30 |
| Score top-1 (mediana) | 0.964 | 0.964 |
| Approach length (mediana) | 10.0 cm | 10.0 cm |
| DDPM-style length (mediana) | 25.0 cm | 25.0 cm |
| Sampler latency p95 | 2.0 ms | 2.0 ms |
| Pinza open-at-start / closed-at-end | 100% / 100% | 100% / 100% |

_Fuente: `experiments/results/diffusion_real_poses/trajectories_summary.json`_


## 4. Infraestructura CoppeliaSim (smoke test)

| Métrica | Valor |
|---------|-------|
| Conexión ZMQ | 149.83 ms |
| Servidor sim | v41000 |
| Escena cargada | `pickAndPlaceDemo.ttt` |
| Latencia por step | 18.117 ms (mean) |
| Sim time avanzado (100 pasos) | 5.0 s |
| Render del sensor | 640×480, intensidad media 152.9 |

_Fuente: `experiments/results/coppelia_smoke/smoke_test_result.json`_

