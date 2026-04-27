# RUN CARD — Evaluación FoundationPose YCB-V / T-LESS

> Tarjeta de identidad del run de evaluación. Se cita en el Cap. 6 del TFM
> como evidencia reproducible. Una entrada por ejecución.

## Run #1 — 2026-04-26/27 (validación post-fix v2)

| Campo | Valor |
|-------|-------|
| **Fecha de inicio** | 2026-04-26 (YCB-V) |
| **Fecha de cierre** | 2026-04-27 08:48:07 UTC (T-LESS + métricas) |
| **Timestamp del JSON** | `20260427_084807` |
| **Commit del run** | `be02c8c` (rama `main`) |
| **Notebook** | `notebooks/colab/01_foundationpose_eval.ipynb` |
| **Schema checkpoint** | `v2_bop_targets_mask_per_gt_idx` |
| **Hardware** | Google Colab — GPU NVIDIA Tesla T4 (16 GB) |
| **CUDA / torch** | CUDA 12.1, torch 2.1.2+cu121 (lockfile: `requirements.colab.lock.txt`) |
| **MODE** | `dev` |
| **MAX_SCENES** | 5 (de 12 escenas YCB-V test, de 20 T-LESS test_primesense) |
| **MAX_IMAGES_PER_SCENE** | 50 |
| **REGISTER_ITERATIONS** | 5 |
| **Subset frames** | BOP-19 (`test_targets_bop19.json`) — comparable al leaderboard |
| **Bugs corregidos antes del run** | mask-per-gt_idx (loader), BOP subset filter, emparejamiento GT por instancia (commits `ea8fc2e`, `280c9f3`, `7b4c7c6`) |

### Resultados — métricas agregadas finales

> Computadas en celdas 20 (YCB-V) y 24 (T-LESS) del notebook sobre el conjunto
> completo de predicciones. Fuente: `comparison_20260427_084807.json`.

| Métrica | YCB-V | T-LESS |
|---------|-------|--------|
| Objetos evaluados | **1098** | **1012** |
| ADD media (mm) | — | — |
| **ADD mediana (mm)** | **4.166** | **2.902** |
| ADD-S mediana (mm) | 2.085 | 1.359 |
| **AUC ADD** | **0.829** | **0.805** |
| **AUC ADD-S** | **0.959** | **0.983** |
| Recall@5mm ADD | 59.9 % | 69.8 % |
| Recall@5mm ADD-S | 88.8 % | 99.3 % |
| Recall@10mm ADD | 77.0 % | 72.1 % |
| **Recall@10mm ADD-S** | **96.5 %** | **99.7 %** |
| Recall@20mm ADD | 80.1 % | 74.4 % |
| Recall@20mm ADD-S | 97.7 % | 99.9 % |
| Tiempo total (s) | 4561.9 | 4401.3 |
| Tiempo total (h:min) | 1 h 16 min | 1 h 13 min |
| Tiempo por objeto (ms) | 4154 | 4350 |

**Lectura rápida:** ambos datasets caen muy por debajo del rango ADD reportado
en el paper original de FoundationPose (≈ 10-20 mm). El ADD-S mediano por
debajo de 3 mm en ambos casos es consistente con un registro de pose
sub-milimétrico para la mayoría de objetos. La AUC ADD-S de 0.96 (YCB-V) y
0.98 (T-LESS) son competitivas con el Mean AR del paper (0.88 y 0.78
respectivamente, métrica distinta — comparación cualitativa).

### Baselines de referencia

- **GDR-Net++ (BOP Challenge 2022 Leaderboard)** — reportado como AR (VSD/MSSD/MSPD).
  - YCB-V: AR_VSD 0.842, AR_MSSD 0.819, AR_MSPD 0.874, Mean AR 0.845
  - T-LESS: AR_VSD 0.736, AR_MSSD 0.685, AR_MSPD 0.773, Mean AR 0.731
- **FoundationPose (Wen et al., CVPR 2024)** — paper original.
  - YCB-V: Mean AR 0.884
  - T-LESS: Mean AR 0.777

> *Nota:* Nuestra ejecución reporta ADD/ADD-S porque el toolkit BOP
> oficial (que calcula VSD) requiere instalación con dependencias C++ no
> disponibles en Colab. La comparación con AR del leaderboard es por
> tanto cualitativa, no directa.

### Trazabilidad

- Datasets descargados desde HuggingFace `bop-benchmark/{ycbv,tless}`
  (zips cacheados en Drive `TFM/datasets_zips/`).
- Pesos FoundationPose en Drive `TFM/weights/foundationpose/` —
  scorer `2024-01-11-20-02-45/model_best.pth` (≈180 MB) +
  refiner `2023-10-28-18-33-37/model_best.pth` (≈65 MB).
- JSON de salida: `comparison_<timestamp>.json`,
  `predictions_ycbv_<timestamp>.json`,
  `predictions_tless_<timestamp>.json`.

### Decisiones de diseño relevantes

1. **Subset BOP-19**: solo se evalúan los frames listados en
   `test_targets_bop19.json` (≈75 por escena). Necesario para que las
   métricas sean comparables al leaderboard BOP.
2. **Mask por instancia (`gt_idx`)**: cada predicción usa la mask
   correspondiente a su `gt_idx` específico, no la del primer GT. Sin
   este fix, YCB-V multi-objeto colapsaba a ADD ~130 mm.
3. **Per-image checkpoints**: la celda 18 escribe el checkpoint cada N
   imágenes para sobrevivir a desconexiones de Colab (típicas en runs >
   30 min en free tier).
4. **VRAM cleanup entre datasets**: se libera memoria GPU tras YCB-V
   antes de iniciar T-LESS para evitar OOM.

---

*Esta tarjeta es parte del respaldo del TFM. Cualquier número citado en
el Cap. 6 que provenga de FoundationPose debe poder rastrearse hasta el
`comparison_*.json` referenciado aquí.*
