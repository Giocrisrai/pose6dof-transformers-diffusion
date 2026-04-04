# Estimación de Pose 6-DoF mediante Transformers y Modelos de Difusión para Bin Picking Robótico

**Trabajo Fin de Máster** | Máster en Ingeniería Matemática y Computación | UNIR 2026

## Autores

- **Giocrisrai Godoy Bonillo** — giocrisrai@gmail.com
- **José Miguel Carrasco** — jmcarrascoc@gmail.com

**Directora:** Profesora Benítez

---

## Descripción

Pipeline de percepción y planificación robótica para bin picking industrial que integra:

- **FoundationPose** (Wen et al., CVPR 2024) — Estimación de pose 6-DoF mediante atención cruzada 2D-3D en SE(3)
- **Diffusion Policy** (Chi et al., RSS 2023) — Generación de trayectorias de agarre multimodales mediante SDEs

### Fundamentos Matemáticos
- Grupos de Lie: SE(3), SO(3)
- Representaciones de rotación: cuaterniones unitarios, representación 6D continua
- Mecanismos de atención multi-cabeza (Transformers)
- Ecuaciones diferenciales estocásticas (SDEs), score matching, dinámica de Langevin

## Estructura del Repositorio

```
pose6dof-transformers-diffusion/
├── docs/                    # Memoria del TFM y documentos
├── notebooks/               # Jupyter/Colab notebooks de experimentación
├── src/
│   ├── perception/          # FoundationPose, GDR-Net (baseline)
│   ├── planning/            # Diffusion Policy, trayectorias de agarre
│   ├── simulation/          # CoppeliaSim + ROS 2 + MoveIt 2
│   └── utils/               # Utilidades comunes
├── data/
│   ├── models/              # Modelos CAD y pesos pre-entrenados
│   └── configs/             # Configuraciones de experimentos
├── experiments/             # Scripts y resultados de evaluación
├── papers/                  # PDFs de los 6 papers clave (no subir a GitHub)
└── requirements.txt         # Dependencias Python
```

## Datasets de Evaluación

| Dataset | Objetos | Tipo | Licencia |
|---------|---------|------|----------|
| T-LESS | 30 industriales sin textura | RGB-D | CC BY 4.0 |
| YCB-Video | 21 domésticos | RGB-D | MIT |
| XYZ-IBD | 15 industriales reales | RGB-D | CC BY-NC-SA 4.0 |

## Métricas BOP

- **VSD** — Visible Surface Discrepancy
- **MSSD** — Maximum Symmetry-Aware Surface Distance
- **MSPD** — Maximum Symmetry-Aware Projection Distance

## Entorno de Desarrollo

| Componente | Local (M1 Pro) | Remoto (Colab) |
|-----------|----------------|----------------|
| Simulación | CoppeliaSim ARM64 + ROS 2 Humble | - |
| DL Training | - | GPU T4 (gratuita) |
| Inferencia | Open3D, OpenCV | FoundationPose, Diffusion Policy |
| Planificación | MoveIt 2 | - |

## Setup

```bash
# Clonar repositorio
git clone git@github.com:TU_USUARIO/pose6dof-transformers-diffusion.git
cd pose6dof-transformers-diffusion

# Instalar dependencias
pip install -r requirements.txt
```

## Licencias de Software

- FoundationPose: NVIDIA Non-Commercial (uso académico)
- Diffusion Policy: MIT
- CoppeliaSim: Educational
- ROS 2: Apache 2.0
- Datasets BOP: CC BY 4.0 / CC BY-NC-SA 4.0

## Bibliografía Principal (6 Papers Clave)

1. Liu et al. (2025). *Deep Learning-Based Object Pose Estimation: A Comprehensive Survey*. IJCV.
2. Cordeiro et al. (2025). *A Review of Visual Perception for Robotic Bin-Picking*. R&AS.
3. Wen et al. (2024). *FoundationPose: Unified 6D Pose Estimation and Tracking*. CVPR.
4. Wang et al. (2021). *GDR-Net: Geometry-Guided Direct Regression Network*. CVPR.
5. Hodaň et al. (2025). *BOP Challenge 2024*. CVPRW.
6. Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS.

---

*Máster Universitario en Ingeniería Matemática y Computación — Universidad Internacional de La Rioja (UNIR) — 2026*
