# =============================================================
# TFM Pose 6-DoF — Inference GPU Container
# =============================================================
# Base: NVIDIA CUDA 12.1 devel (incluye headers para compilar pytorch3d/nvdiffrast)
# Python 3.10 (mejor compatibilidad con wheels pre-built)
# uv como installer de dependencias (determinista + rápido)
# torch 2.1.2 + cu121 pinneado
# FoundationPose con commit fijo + patches idempotentes
#
# Build:
#   docker build -f docker/inference-gpu.Dockerfile -t pose6dof-inference:cu121 .
#
# Push a GHCR (opcional):
#   docker tag pose6dof-inference:cu121 ghcr.io/giocrisrai/pose6dof-inference:cu121
#   docker push ghcr.io/giocrisrai/pose6dof-inference:cu121
# =============================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL maintainer="Giocrisrai Godoy <giocrisrai@gmail.com>"
LABEL description="GPU inference container for FoundationPose / GDR-Net TFM evaluation"
LABEL org.opencontainers.image.source="https://github.com/Giocrisrai/pose6dof-transformers-diffusion"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.cargo/bin:/root/.local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- Dependencias del sistema ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git wget curl unzip ca-certificates \
    libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglvnd-dev \
    libboost-all-dev libeigen3-dev \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Symlinks python3/pip3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# --- uv (gestor de paquetes) ---------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- Instalar torch 2.1.2 + cu121 FIRST (pytorch3d/nvdiffrast lo necesitan) ---
RUN uv pip install --system --no-cache \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- Dependencias Python del proyecto -------------------------
WORKDIR /workspace
COPY pyproject.toml uv.lock* ./
COPY requirements.txt ./

# Instalar deps del proyecto (core + inference-gpu extras)
# Usamos requirements.txt porque uv sync requiere el layout completo del proyecto
# y queremos mantener la imagen ligera (código se monta por volumen)
RUN uv pip install --system --no-cache -r requirements.txt && \
    uv pip install --system --no-cache \
    accelerate>=0.25 transformers>=4.36 kornia>=0.7 einops>=0.7 \
    ruamel.yaml>=0.18 transformations ninja pybind11 gdown scikit-learn

# --- pytorch3d (compile from source, pinned) ------------------
RUN uv pip install --system --no-cache --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

# --- nvdiffrast (compile from source, pinned) -----------------
RUN uv pip install --system --no-cache --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"

# --- FoundationPose (commit fijo) -----------------------------
# Commit actualmente verificado (actualiza cuando cambie tu run exitoso).
ARG FP_COMMIT=main
ENV FP_DIR=/opt/FoundationPose
RUN git clone https://github.com/NVlabs/FoundationPose.git ${FP_DIR} && \
    cd ${FP_DIR} && git checkout ${FP_COMMIT} && \
    cd ${FP_DIR}/mycpp && mkdir -p build && cd build && \
    (cmake .. && make -j$(nproc)) || echo "[WARN] mycpp build failed, Python fallback will be used"

# --- Aplicar patches del proyecto al FoundationPose ------------
COPY docker/patch_foundationpose.py /opt/patch_foundationpose.py
RUN FP_DIR=${FP_DIR} python3 /opt/patch_foundationpose.py

# --- Permitir que Python encuentre FoundationPose -------------
ENV PYTHONPATH="${FP_DIR}:${PYTHONPATH}"

# --- Workspace final ------------------------------------------
WORKDIR /workspace
# El código del proyecto se monta por volumen en runtime (ver docker-compose.yml)
# Los pesos FP se montan desde /weights, los datasets desde /datasets

# Script de arranque
COPY scripts/run_inference.sh /usr/local/bin/run-inference
RUN chmod +x /usr/local/bin/run-inference

# Verificación rápida en build-time
RUN python3 -c "import torch; assert torch.__version__.startswith('2.1.2'), torch.__version__; print(f'torch {torch.__version__} OK')" && \
    python3 -c "import pytorch3d; print(f'pytorch3d {pytorch3d.__version__} OK')" && \
    python3 -c "import nvdiffrast; print('nvdiffrast OK')" && \
    python3 -c "import kornia, trimesh, transformations; print('core deps OK')"

CMD ["bash"]
