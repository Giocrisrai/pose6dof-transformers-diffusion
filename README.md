# Estimacion de Pose 6-DoF mediante Transformers y Modelos de Difusion para Bin Picking Robotico

**Trabajo Fin de Master** | Master en Ingenieria Matematica y Computacion | UNIR 2026

## Autores

- **Giocrisrai Godoy Bonillo** -- giocrisrai@gmail.com
- **Jose Miguel Carrasco** -- jmcarrascoc@gmail.com

**Directora:** Profesora Benitez

---

## Notebooks en Google Colab

| Notebook | Descripcion |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/pose6dof-transformers-diffusion/blob/main/notebooks/colab/00_colab_setup.ipynb) | **Setup Colab** -- Entorno, datasets BOP, Google Drive |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/pose6dof-transformers-diffusion/blob/main/notebooks/colab/01_foundationpose_eval.ipynb) | **FoundationPose Eval** -- Inferencia GPU en YCB-V y T-LESS |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/pose6dof-transformers-diffusion/blob/main/notebooks/colab/02_gdrnet_eval.ipynb) | **GDR-Net++ Eval** -- Baseline comparativo + tabla FP vs GDR |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/pose6dof-transformers-diffusion/blob/main/notebooks/colab/03_results_analysis.ipynb) | **Results Analysis** -- Agregación de métricas y figuras finales |

---

## Descripcion

Pipeline de percepcion y planificacion robotica para bin picking industrial que integra:

- **FoundationPose** (Wen et al., CVPR 2024) -- Estimacion de pose 6-DoF mediante atencion cruzada 2D-3D en SE(3)
- **GDR-Net++** (Wang et al., CVPR 2021) -- Baseline comparativo, regression directa de geometria
- **Diffusion Policy** (Chi et al., RSS 2023) -- Generacion de trayectorias de agarre multimodales mediante SDEs

### Fundamentos Matematicos
- Grupos de Lie: SE(3), SO(3) -- exponential/logarithmic maps
- Representaciones de rotacion: cuaterniones unitarios, representacion 6D continua (Zhou et al.)
- Mecanismos de atencion multi-cabeza (Transformers)
- Ecuaciones diferenciales estocasticas (SDEs), score matching, dinamica de Langevin

## Estructura del Repositorio

```
pose6dof-transformers-diffusion/
├── src/
│   ├── perception/          # FoundationPose, GDR-Net++ (baseline)
│   │   ├── foundation_pose.py
│   │   ├── gdrnet.py
│   │   └── evaluator.py     # PoseEvaluator (metricas BOP)
│   ├── planning/            # Diffusion Policy, trayectorias de agarre
│   │   └── diffusion_policy.py  # DDPM + ConditionalUNet1D
│   ├── simulation/          # CoppeliaSim + ROS 2 + MoveIt 2
│   └── utils/               # Utilidades comunes
│       ├── lie_groups.py    # SE(3)/SO(3) exp/log/adjoint
│       ├── rotations.py     # Quat, 6D, Euler, axis-angle
│       ├── metrics.py       # ADD, ADD-S, VSD, MSSD, MSPD
│       ├── visualization.py # Overlay poses, nubes de puntos
│       └── dataset_loader.py# BOP format loader (T-LESS, YCB-V)
├── notebooks/
│   ├── colab/               # Notebooks para Google Colab (GPU)
│   │   ├── 00_colab_setup.ipynb
│   │   └── 01_foundationpose_eval.ipynb
│   └── 04_math_foundations.ipynb  # Demos matematicas
├── tests/                   # pytest (77/77 passing)
├── docker/                  # ROS 2 Humble + MoveIt 2
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/datasets/           # BOP datasets (T-LESS, YCB-V)
├── experiments/             # Resultados de evaluacion
├── PLANNING.md              # Planificacion 12 semanas
└── pyproject.toml           # Dependencias (uv)
```

## Quick Start

```bash
# Clonar
git clone https://github.com/Giocrisrai/pose6dof-transformers-diffusion.git
cd pose6dof-transformers-diffusion

# Instalar con uv (recomendado)
uv sync

# O con pip
pip install -e ".[dev,colab]"

# Tests
pytest tests/ -v  # 77/77 passing (sin GPU; los módulos que requieren cv2 lo importan perezosamente)

# Descargar datasets BOP (requiere ~30 GB)
bash scripts/download_datasets.sh
```

### Google Colab (recomendado para GPU)

1. Abrir `00_colab_setup.ipynb` con el badge de arriba
2. Ejecutar todas las celdas (descarga datasets + configura entorno)
3. Abrir `01_foundationpose_eval.ipynb` para evaluacion

### Reproducibilidad (cualquier recurso)

Ver [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) para las alternativas soportadas:

| Escenario | Opcion | Costo | Ver |
|-----------|--------|-------|-----|
| Sin presupuesto | Colab Free (actual) | $0 | Notebook 01 |
| Colab agotado | Kaggle Notebooks T4x2, 30h/sem | $0 | REPRODUCIBILITY Opcion B |
| Con credito cloud | Docker GPU en Vast.ai / RunPod | ~$1-3/run | `docker/README-GPU.md` |
| GPU NVIDIA local | `docker compose run inference-gpu` | $0 | `docker/README-GPU.md` |
| Mac M1 sin GPU | Analisis local + Docker ROS 2 simulacion | $0 | REPRODUCIBILITY Opcion E |

El `docker/inference-gpu.Dockerfile` congela torch 2.1.2+cu121 + pytorch3d v0.7.8 +
nvdiffrast v0.3.3 + FoundationPose commit fijo. El `requirements.colab.lock.txt`
documenta las versiones validadas en Colab Free.

## Evidencia experimental (respaldo del TFM)

Los resultados de la evaluación de FoundationPose / GDR-Net++ que se citan en
el Cap. 6 del TFM están versionados en este repositorio para auditoría:

| Artefacto | Ruta | Contenido |
|-----------|------|-----------|
| Tarjeta del run | `experiments/results/foundationpose_eval/RUN_CARD.md` | Commit, fecha, GPU, MODE, fixes aplicados, métricas validadas |
| Política de carpeta | `experiments/results/foundationpose_eval/README.md` | Schema de los JSON, qué se versiona y qué no |
| Resultados crudos | `experiments/results/foundationpose_eval/comparison_*.json` | Métricas agregadas (ADD/ADD-S/AUC/recalls) |
| Predicciones por frame | `experiments/results/foundationpose_eval/predictions_*.json` | Pose estimada por imagen para reproducibilidad |
| Patch Cap. 6 | `docs/cap6_seccion_foundationpose.md` | Sección redactada lista para integrar al `.docx` |
| Figuras Cap. 6 | `experiments/results/chapter6_figures/` | PNG y `fp_results_table.tex` generados por `experiments/generate_chapter6_figures.py` |
| Lockfile Colab | `requirements.colab.lock.txt` | `pip freeze` del entorno Colab que produjo los resultados |
| Contenedor GPU equivalente | `docker/inference-gpu.Dockerfile` | torch 2.1.2+cu121, pytorch3d v0.7.8, FP commit fijo |

Resumen de un golpe de vista (run del 2026-04-26/27, JSON
`comparison_20260427_084807.json`, schema `v2_bop_targets_mask_per_gt_idx`):

| Dataset | N obj | ADD med (mm) | ADD-S med (mm) | AUC ADD-S | Recall@10mm ADD-S |
|---------|------:|-------------:|---------------:|----------:|------------------:|
| YCB-V   | 1098  | **4.17**     | 2.09           | **0.959** | **96.5 %**        |
| T-LESS  | 1012  | **2.90**     | 1.36           | **0.983** | **99.7 %**        |

Subset BOP-19, 5 escenas × 50 imágenes por dataset, GPU Colab T4. Detalles
completos en `experiments/results/foundationpose_eval/RUN_CARD.md`.

## Datasets de Evaluacion

| Dataset | Objetos | Tipo | Uso en TFM |
|---------|---------|------|------------|
| **T-LESS** | 30 industriales sin textura | RGB-D | Benchmark principal (sim. industrial) |
| **YCB-Video** | 21 domesticos | RGB-D | Comparativa con literatura |

## Metricas BOP

- **VSD** -- Visible Surface Discrepancy
- **MSSD** -- Maximum Symmetry-Aware Surface Distance
- **MSPD** -- Maximum Symmetry-Aware Projection Distance
- **ADD / ADD-S** -- Average Distance (simetrico)

## Entorno de Desarrollo

| Tarea | Local (M1 Pro) | Colab (T4 GPU) |
|-------|---------------|----------------|
| Desarrollo codigo | VSCode + git | -- |
| Simulacion | CoppeliaSim + ROS 2 | -- |
| Datasets BOP (30+ GB) | Limitado | ~80 GB libres |
| FoundationPose inferencia | -- | T4/A100 CUDA |
| GDR-Net++ inferencia | -- | T4/A100 CUDA |
| Diffusion Policy training | -- | GPU acelerado |
| Visualizacion / figuras | matplotlib local | Inline |

## Licencia y citación

El **código de este repositorio** se distribuye bajo licencia **MIT** (ver
[`LICENSE`](LICENSE)). Las dependencias y modelos pre-entrenados de terceros
mantienen sus licencias propias, listadas en `LICENSE` y resumidas a
continuación:

- FoundationPose: NVIDIA Source Code License (uso académico/no comercial)
- Diffusion Policy: MIT
- CoppeliaSim: Educational
- ROS 2: Apache 2.0
- Datasets BOP: CC BY 4.0 / CC BY-NC-SA 4.0
- pytorch3d, nvdiffrast: BSD 3-Clause

Si usas este código o los resultados experimentales, cita el TFM siguiendo
el formato definido en [`CITATION.cff`](CITATION.cff).

## Bibliografia Principal

1. Liu et al. (2025). *Deep Learning-Based Object Pose Estimation: A Comprehensive Survey*. IJCV.
2. Cordeiro et al. (2025). *A Review of Visual Perception for Robotic Bin-Picking*. R&AS.
3. Wen et al. (2024). *FoundationPose: Unified 6D Pose Estimation and Tracking*. CVPR.
4. Wang et al. (2021). *GDR-Net: Geometry-Guided Direct Regression Network*. CVPR.
5. Hodan et al. (2025). *BOP Challenge 2024*. CVPRW.
6. Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS.

---

*Master Universitario en Ingenieria Matematica y Computacion -- Universidad Internacional de La Rioja (UNIR) -- 2026*
