# Reproducibilidad — Pose 6-DoF TFM

Este documento describe cómo reproducir los experimentos del TFM **según los recursos
disponibles**. El diseño prioriza servicios gratuitos, pero ofrece alternativas pagadas
para quienes las necesiten (tribunal, futuros investigadores, cluster universitario).

---

## Decisión de diseño

La evaluación de FoundationPose y GDR-Net requiere GPU NVIDIA con CUDA (torch+cu12x,
pytorch3d, nvdiffrast). Se adoptó un enfoque **multi-plataforma** para garantizar que
cualquier persona pueda reproducir los resultados independientemente de su presupuesto:

| Escenario | Recomendación | Costo | Esfuerzo setup |
|-----------|---------------|-------|----------------|
| **Estudiante sin presupuesto** (este TFM) | Colab Free + Kaggle como backup | $0 | Bajo |
| Lab universitario con cluster HPC | Dockerfile GPU en HPC | $0 | Medio |
| Investigador con crédito cloud | Dockerfile GPU en Vast.ai/RunPod/Lambda | ~$1-3/run | Medio |
| Empresa / laboratorio con GPU local | Dockerfile GPU nativo | $0 después del hardware | Bajo |
| Desarrollo local sin GPU (Mac M1, etc.) | Tests CPU + Docker ROS 2 para simulación | $0 | Bajo |

La reproducibilidad de las **versiones exactas** está garantizada por:

1. **`docker/inference-gpu.Dockerfile`** — versiones pinneadas de torch 2.1.2+cu121,
   pytorch3d v0.7.8, nvdiffrast v0.3.3, FoundationPose commit fijo.
2. **`requirements.colab.lock.txt`** — snapshot de versiones validadas en Colab Free,
   regenerable automáticamente desde el notebook (última celda).
3. **`pyproject.toml` + `uv.lock`** — gestor determinista de dependencias Python del proyecto.

---

## Opción A: Colab Free (usado en este TFM)

### Ventajas
- **Costo cero**, acceso inmediato con cuenta Google.
- T4 GPU (15 GB VRAM) suficiente para FoundationPose.
- Mount nativo de Google Drive para persistencia de datasets y pesos.

### Limitaciones conocidas
- Session timeout 12 h, idle 90 min.
- Quota diaria de GPU variable (~12 h al día).
- Versiones de la imagen base de Colab cambian sin aviso.
- Setup completo toma 25-35 min por sesión (build de pytorch3d + nvdiffrast).

### Cómo ejecutar

1. Abrir el notebook:
   `https://colab.research.google.com/github/Giocrisrai/pose6dof-transformers-diffusion/blob/main/notebooks/colab/01_foundationpose_eval.ipynb`
2. `Runtime → Change runtime type → T4 GPU`.
3. Ejecutar todas las celdas en orden. La celda 14 define `MODE` (`smoke` / `dev` / `full`).
4. Resultados JSON se guardan en `/content/drive/MyDrive/TFM/experiments/foundationpose_eval/`.

### Mitigación de las limitaciones

- **Checkpoints por imagen** (cada 25 imgs en modo `dev`) en Drive → al cortarse la
  sesión, la próxima retoma exactamente donde quedó.
- **Cache de datasets y pesos en Drive** → no se re-descarga nada después de la primera vez.
- **Patches idempotentes** (`docker/patch_foundationpose.py`) → re-ejecutar celdas no
  duplica fixes.

---

## Opción B: Kaggle Notebooks (backup cuando Colab se agota)

### Ventajas sobre Colab
- **30 h/semana de GPU T4×2 fresco**, independiente de Colab.
- Kaggle Datasets (hasta 20 GB) como alternativa a Drive.
- Notebooks persisten versiones ejecutadas (histórico automático).

### Cómo adaptarlo (diferencias con Colab)

1. Crear Kaggle Notebook, habilitar GPU (T4×2 o P100).
2. Subir los zips de BOP (`ycbv_test_all.zip`, etc.) como **Kaggle Dataset privado**
   (o usar un Dataset público si existe — mirar "bop-challenge" en Kaggle).
3. Cambiar en el notebook:
   - `drive.mount('/content/drive')` → eliminar (no aplica en Kaggle).
   - `DRIVE_ZIPS = '/content/drive/.../datasets_zips'` → `DRIVE_ZIPS = '/kaggle/input/bop-datasets-tfm'`.
   - `/content/drive/MyDrive/TFM/` → `/kaggle/working/` (espacio de trabajo).
4. Los pesos de FoundationPose se suben como otro Kaggle Dataset privado.
5. Ejecutar igual que en Colab.

**Al final**: los JSON se descargan como artefactos del notebook de Kaggle.

---

## Opción C: Servicios GPU de pago (Vast.ai, RunPod, Lambda Cloud)

Para quienes tengan presupuesto (~$5-10 USD para un TFM completo) o quieran garantizar
máxima reproducibilidad con versiones exactas.

### Cómo ejecutar

Ver [`docker/README-GPU.md`](docker/README-GPU.md) — documentación completa.

Resumen:
```bash
docker run --gpus all --rm \
  -v ./datasets:/datasets \
  -v ./weights:/weights \
  -v ./results:/workspace/results \
  ghcr.io/giocrisrai/pose6dof-inference:cu121 \
  run-inference dev
```

Un run completo (todas escenas YCB-V + T-LESS) en Vast.ai T4 = **3-4 horas × $0.30/h ≈ $1-2 USD**.

---

## Opción D: Local con GPU NVIDIA (Linux/WSL2)

### Requisitos
- GPU NVIDIA con driver ≥ 525.60 (para CUDA 12.1)
- nvidia-container-toolkit instalado
- Docker + docker-compose

### Ejecutar

```bash
cd docker
DATASETS_HOST=/path/to/bop_datasets \
WEIGHTS_HOST=/path/to/fp_weights \
docker compose run --rm inference-gpu
run-inference dev
```

---

## Opción E: Local Mac M1/M2 (SIN GPU NVIDIA)

**Mac Apple Silicon NO tiene CUDA**, no puede correr FoundationPose / GDR-Net.
Pero sí sirve para:

### Qué SÍ se puede hacer en M1/M2

1. **Análisis de resultados post-inferencia**:
   ```bash
   # Descargar los JSON desde Colab/Kaggle a experiments/results/foundationpose_eval/
   uv sync
   python experiments/generate_chapter6_figures.py
   ```
   Genera figuras y tablas LaTeX para el Capítulo 6 de la memoria.

2. **Tests unitarios del proyecto**:
   ```bash
   uv run pytest tests/ -v
   ```
   Cubre módulos matemáticos (lie_groups, rotations, metrics), pipeline, diffusion policy.

3. **Simulación ROS 2 + MoveIt + CoppeliaSim** (Dockerfile dedicado, arm64 compatible):
   ```bash
   cd docker
   docker compose up ros2-sim
   ```

4. **Desarrollo y edición del código** — todo el pipeline es Python puro, solo la
   inferencia necesita GPU NVIDIA.

---

## Cómo el tribunal del TFM puede verificar los resultados

Según los recursos del tribunal:

### Si el tribunal tiene acceso a GPU NVIDIA
```bash
git clone https://github.com/Giocrisrai/pose6dof-transformers-diffusion.git
cd pose6dof-transformers-diffusion
docker compose run --rm inference-gpu run-inference smoke
```

### Si el tribunal solo tiene Colab
Abrir el notebook `notebooks/colab/01_foundationpose_eval.ipynb` — mismo flujo que
ejecutamos nosotros.

### Si el tribunal solo quiere revisar el código
- `src/` + `tests/` + `docs/` cubren la totalidad del trabajo sin necesidad de ejecutar.
- Los JSON de resultados (tras descargarse desde Drive) están en
  `experiments/results/foundationpose_eval/` del repo.

---

## Tabla de fallbacks de FoundationPose

Cuando no se dispone de NVIDIA Warp (Colab Free / contenedores sin Warp CUDA 11), se
activan los siguientes fallbacks Python, todos idempotentes y documentados en
[`docker/patch_foundationpose.py`](docker/patch_foundationpose.py):

| Componente original | Fallback | Impacto esperado en métricas |
|--------------------|----------|------------------------------|
| `Utils.erode_depth` (Warp kernel) | NumPy vectorizado con vecindario NxN | < 2% en ADD/ADD-S |
| `Utils.bilateral_filter_depth` (Warp kernel) | `cv2.bilateralFilter` | < 1% |
| `mycpp.cluster_poses` (C++ pybind11) | Python puro, greedy | < 3% por simetrías |
| `open3d` completo | Mock selectivo (solo `voxel_down_sample`, `transform`) | 0% (solo usados esos) |

Estos fallbacks se declaran explícitamente en el Capítulo 6 de la memoria como
limitaciones conocidas del entorno Colab.

---

## Estrategia recomendada para retomar el trabajo

Si otra persona retoma este TFM (o el propio autor tras una pausa):

1. **Clonar el repo**: `git clone ...`.
2. **Leer**: `README.md` → este `REPRODUCIBILITY.md` → `PLANNING.md` → `docs/`.
3. **Setup local Mac M1**: `uv sync --all-extras`, `pytest tests/`.
4. **Primer run GPU**: Opción A (Colab Free) con el notebook, `MODE='smoke'`.
5. **Análisis local**: descargar JSON, `python experiments/generate_chapter6_figures.py`.
6. **Escalar**: si quiere runs completos → Opción B (Kaggle) o C (Dockerfile en Vast.ai).

---

*Última actualización: 2026-04-21. Mantenido como parte del repositorio.*
