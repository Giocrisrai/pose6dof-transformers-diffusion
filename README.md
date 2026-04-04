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
├── tests/                   # pytest (24/24 passing)
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
pytest tests/ -v  # 24/24 passing

# Descargar datasets BOP (requiere ~30 GB)
bash scripts/download_datasets.sh
```

### Google Colab (recomendado para GPU)

1. Abrir `00_colab_setup.ipynb` con el badge de arriba
2. Ejecutar todas las celdas (descarga datasets + configura entorno)
3. Abrir `01_foundationpose_eval.ipynb` para evaluacion

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

## Licencias de Software

- FoundationPose: NVIDIA Non-Commercial (uso academico)
- Diffusion Policy: MIT
- CoppeliaSim: Educational
- ROS 2: Apache 2.0
- Datasets BOP: CC BY 4.0 / CC BY-NC-SA 4.0

## Bibliografia Principal

1. Liu et al. (2025). *Deep Learning-Based Object Pose Estimation: A Comprehensive Survey*. IJCV.
2. Cordeiro et al. (2025). *A Review of Visual Perception for Robotic Bin-Picking*. R&AS.
3. Wen et al. (2024). *FoundationPose: Unified 6D Pose Estimation and Tracking*. CVPR.
4. Wang et al. (2021). *GDR-Net: Geometry-Guided Direct Regression Network*. CVPR.
5. Hodan et al. (2025). *BOP Challenge 2024*. CVPRW.
6. Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS.

---

*Master Universitario en Ingenieria Matematica y Computacion -- Universidad Internacional de La Rioja (UNIR) -- 2026*
