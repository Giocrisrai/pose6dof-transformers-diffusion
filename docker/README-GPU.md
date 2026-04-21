# Contenedor GPU para Inferencia — TFM Pose 6-DoF

Este documento describe cómo usar el `inference-gpu.Dockerfile` para ejecutar
FoundationPose / GDR-Net con reproducibilidad garantizada (versiones pinneadas
de torch, pytorch3d, nvdiffrast, FoundationPose).

## ¿Por qué un contenedor?

El notebook de Colab funciona, pero tiene tres dolores crónicos:

1. **Re-instalación completa en cada sesión** (~25-35 min de pytorch3d + nvdiffrast + mmcv).
2. **Versiones no reproducibles** — Colab cambia la imagen base periódicamente.
3. **Sin control sobre torch/cuda** — a veces PyPI nvdiffrast no tiene wheel compatible.

El contenedor resuelve los tres: build una vez, push a GHCR, pull en 2-3 min en cualquier máquina GPU NVIDIA.

## Qué versiones queda fijadas

| Componente | Versión | Fuente |
|------------|---------|--------|
| Base OS | Ubuntu 22.04 | `nvidia/cuda:12.1.0-devel-ubuntu22.04` |
| CUDA | 12.1.0 | NVIDIA base image |
| Python | 3.10 | apt |
| torch | 2.1.2+cu121 | PyTorch index |
| torchvision | 0.16.2+cu121 | PyTorch index |
| pytorch3d | v0.7.8 | GitHub tag |
| nvdiffrast | v0.3.3 | GitHub tag |
| FoundationPose | commit `main` (build-arg configurable) | GitHub |
| uv | latest | astral.sh installer |

Los fallbacks Python para `erode_depth`, `bilateral_filter_depth`, `mycpp.cluster_poses` y el mock de `open3d` se aplican automáticamente vía `docker/patch_foundationpose.py`.

## Opción 1: Ejecutar en Vast.ai (recomendado para TFM)

[Vast.ai](https://vast.ai) es un marketplace de GPUs. Un T4 cuesta ~$0.20-0.50/hora, un run completo del TFM (todas escenas YCB-V + T-LESS) cuesta ~$1-3 USD.

### Setup inicial (una vez)

1. Crear cuenta en Vast.ai, cargar $5-10 de crédito.
2. **Opcional pero recomendado**: build + push la imagen a GHCR (GitHub Container Registry, gratis para repos públicos):

   ```bash
   # Build local (requiere GPU NVIDIA — hacerlo en una máquina con GPU, o skip y dejar que Vast lo haga)
   docker build -f docker/inference-gpu.Dockerfile \
       -t ghcr.io/giocrisrai/pose6dof-inference:cu121 .

   # Login a GHCR (necesita Personal Access Token con scope write:packages)
   echo $GITHUB_TOKEN | docker login ghcr.io -u Giocrisrai --password-stdin

   # Push
   docker push ghcr.io/giocrisrai/pose6dof-inference:cu121
   ```

### Ejecutar un run

1. En Vast.ai, buscar instancia con:
   - GPU: **RTX 3090** o **T4**
   - Disk: ≥ 60 GB
   - CUDA: 12.1+
   - "Launch mode": **Docker image**
   - Docker image: `ghcr.io/giocrisrai/pose6dof-inference:cu121`
   - On-start script:
     ```bash
     cd /workspace
     git clone https://github.com/Giocrisrai/pose6dof-transformers-diffusion.git repo_tfm
     cd repo_tfm
     # Los datasets y pesos se pueden sincronizar desde Google Drive
     # usando rclone, gdown, o subidos vía SSH
     ```

2. SSH a la instancia, ejecutar:
   ```bash
   cd /workspace/repo_tfm
   run-inference smoke       # validación rápida
   run-inference dev         # corrida de desarrollo
   ```

3. Los resultados JSON quedan en `/workspace/results/` — descargar vía SCP antes de destruir la instancia.

## Opción 2: Ejecutar en RunPod

Similar a Vast.ai, interfaz más pulida. Precios algo más altos (~$0.40/h T4).

1. Template "Docker Image" → `ghcr.io/giocrisrai/pose6dof-inference:cu121`
2. Volume ≥ 60 GB
3. Exponer puerto 8888 si quieres JupyterLab

## Opción 3: Ejecutar local (solo si tienes GPU NVIDIA)

```bash
# Requisitos: Linux/WSL2 + nvidia-container-toolkit + docker-compose
cd docker

# Build
docker compose --profile gpu build inference-gpu

# Correr con volúmenes apuntando a tus datasets y pesos locales
DATASETS_HOST=/path/to/bop_datasets \
WEIGHTS_HOST=/path/to/fp_weights \
docker compose run --rm inference-gpu

# Dentro del contenedor:
run-inference smoke
```

**Nota Mac M1/M2:** NO funciona. Apple Silicon no tiene CUDA. Usa Opción 1 o 2.

## Estructura de volúmenes esperada

```
/datasets/
├── ycbv/
│   ├── models/obj_*.ply
│   ├── test/000048/rgb/000001.png
│   └── ...
└── tless/
    ├── models/obj_*.ply
    ├── test_primesense/...
    └── ...

/weights/foundationpose/
├── 2024-01-11-20-02-45/          # Scorer
│   ├── config.yml
│   └── model_best.pth            # ~180 MB
└── 2023-10-28-18-33-37/          # Refiner
    ├── config.yml
    └── model_best.pth            # ~65 MB
```

## Comparativa Colab vs Contenedor

| Aspecto | Colab Free | Contenedor GPU |
|---------|-----------|----------------|
| Costo | $0 | ~$0.30/hora |
| Setup por sesión | 25-35 min | 2-3 min (pull) |
| Reproducibilidad | Baja (imagen base cambia) | **Alta (versiones pinneadas)** |
| Session timeout | 12 h | No timeout |
| Background exec | No | Sí |
| Versionado | Manual | **Tagged images en GHCR** |
| Compartir con tribunal | "Abrí este notebook" | "Pull esta imagen" |

## Reproducibilidad en el TFM

En el Capítulo 5 (Pipeline) o 8 (Reproducibilidad) del TFM, se puede incluir:

> La reproducibilidad de los experimentos se garantiza mediante un contenedor
> Docker público (`ghcr.io/giocrisrai/pose6dof-inference:cu121`) con todas las
> dependencias de inferencia congeladas en versiones específicas: torch 2.1.2+cu121,
> pytorch3d v0.7.8, nvdiffrast v0.3.3, y el repositorio FoundationPose anclado a un
> commit Git fijo. El Dockerfile fuente se encuentra en `docker/inference-gpu.Dockerfile`
> del repositorio. Cualquier verificador puede reproducir los resultados ejecutando:
>
> ```bash
> docker run --gpus all --rm \
>   -v ./datasets:/datasets -v ./weights:/weights \
>   ghcr.io/giocrisrai/pose6dof-inference:cu121 \
>   run-inference dev
> ```

## Troubleshooting

**`CUDA driver version is insufficient for CUDA runtime`**: el host tiene driver NVIDIA antiguo. Necesitas driver ≥ 525.60 para CUDA 12.1.

**`nvidia-container-runtime not found`**: instalar [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**pytorch3d / nvdiffrast fallan en build**: revisa que `TORCH_CUDA_ARCH_LIST` en el Dockerfile incluya la arquitectura de tu GPU. Defaults: 7.5 (T4), 8.0 (A100), 8.6 (RTX 3090), 8.9 (RTX 4090), 9.0 (H100).

**Pesos FoundationPose no encontrados**: el contenedor NO incluye los pesos (~250 MB, licencia NVIDIA). Montarlos desde `/weights` como volumen, o descargarlos dentro del contenedor con `gdown`.
