# RUN CARD — Evaluación FoundationPose YCB-V / T-LESS

> Tarjeta de identidad del run de evaluación. Se cita en el Cap. 6 del TFM
> como evidencia reproducible. Una entrada por ejecución.

## Run #1 — 2026-04-26 (validación post-fix)

| Campo | Valor |
|-------|-------|
| **Fecha** | 2026-04-26 |
| **Commit** | `be02c8c` (rama `main`) |
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

### Resultados YCB-V

| Métrica | Valor | Nota |
|---------|-------|------|
| Imágenes evaluadas | 250 | 5 escenas × 50 imgs (subset BOP-19) |
| Objetos evaluados | 1098 | Multi-objeto por imagen, GT por `obj_id+gt_idx` |
| **ADD mediano** | **3.5 mm** | Excelente (BOP paper FoundationPose ≈ 10-20 mm) |
| Tiempo por objeto | 4125 ms | T4, 5 register iters |
| Throughput | 0.2 obj/s | |
| Tiempo total | 1 h 18 min | |
| Checkpoint | `fp_ycbv_checkpoint.json` (Drive) | Reanudable |

### Resultados T-LESS

> *Pendiente: el run de T-LESS estaba en curso al cerrar la sesión. Llenar
> tras completar celdas 22 y 24 del notebook.*

| Métrica | Valor |
|---------|-------|
| Imágenes evaluadas | _TBD_ |
| Objetos evaluados | _TBD_ |
| ADD mediano | _TBD_ |
| AUC ADD-S | _TBD_ |
| Tiempo total | _TBD_ |

### Métricas finales agregadas (celdas 20 y 24)

> *Pendiente: rellenar con los valores de `comparison_<timestamp>.json`
> tras la ejecución completa. Las claves esperadas son: `auc_add`,
> `auc_adds`, `recall_add_5mm`, `recall_add_10mm`, `recall_add_20mm`,
> `recall_adds_5mm`, `recall_adds_10mm`, `recall_adds_20mm`.*

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
