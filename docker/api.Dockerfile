# =============================================================
# TFM Pose 6-DoF — API REST + Gradio demo (ligero, sin ROS)
# Multi-arch: ARM64 (Apple Silicon) y AMD64 (servidor x86)
# =============================================================
# Build:
#   docker build -f docker/api.Dockerfile -t tfm-pose6dof-api:latest .
# Run (API en 8000, Gradio en 7860):
#   docker run --rm -p 8000:8000 -p 7860:7860 \
#       -v $(pwd)/data/models:/app/data/models:ro \
#       -v $(pwd)/experiments:/app/experiments:ro \
#       tfm-pose6dof-api:latest
# =============================================================

FROM python:3.12-slim AS base

LABEL maintainer="Giocrisrai Godoy <giocrisrai@gmail.com>"
LABEL description="API REST + Gradio demo del pipeline TFM Pose 6-DoF"
LABEL org.opencontainers.image.source="https://github.com/Giocrisrai/pose6dof-transformers-diffusion"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema mínimas (libGL para OpenCV/matplotlib backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python deps (capa cacheable) ────────────────────────────
# Instalamos torch CPU para reducir tamaño (~200MB vs ~2GB CUDA)
RUN pip install --no-cache-dir \
        torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Resto de deps (FastAPI + Gradio + pipeline)
RUN pip install --no-cache-dir \
        numpy scipy matplotlib \
        opencv-python-headless Pillow \
        pytransform3d trimesh \
        fastapi "uvicorn[standard]" pydantic \
        gradio>=4.0 \
        diffusers tqdm pyyaml

# ─── Código del proyecto ─────────────────────────────────────
# Copiamos sólo lo necesario para inferencia (no notebooks, no docs)
COPY src/ /app/src/
COPY scripts/api_server.py /app/scripts/api_server.py
COPY scripts/gradio_demo.py /app/scripts/gradio_demo.py
COPY scripts/launch_api_and_gradio.sh /app/scripts/launch_api_and_gradio.sh
COPY README.md /app/README.md

# Los pesos y checkpoints se montan por volumen (no se incluyen en la imagen)
RUN mkdir -p /app/data/models /app/experiments/results /app/experiments/checkpoints

RUN chmod +x /app/scripts/launch_api_and_gradio.sh

# ─── Healthcheck ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000 7860

# Por defecto lanza API + Gradio en paralelo (script bash)
CMD ["/app/scripts/launch_api_and_gradio.sh"]
