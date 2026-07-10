# =============================================================
# TFM Pose 6-DoF — Aplicación: API REST + dashboard ejecutivo (ligero, sin ROS)
# Multi-arch: ARM64 (Apple Silicon) y AMD64 (servidor x86)
# =============================================================
# Build:
#   docker build -f docker/api.Dockerfile -t tfm-pose6dof-api:latest .
# Run (dashboard en / y API en /docs, puerto 8000):
#   docker run --rm -p 8000:8000 \
#       -v $(pwd)/data/models:/app/data/models:ro \
#       -v $(pwd)/experiments:/app/experiments:ro \
#       tfm-pose6dof-api:latest
# =============================================================

FROM python:3.12-slim AS base

LABEL maintainer="Giocrisrai Godoy <giocrisrai@gmail.com>"
LABEL description="API REST + dashboard ejecutivo del pipeline TFM Pose 6-DoF"
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

# Resto de deps (FastAPI + pipeline). Sin Gradio: la API sirve el dashboard en /.
RUN pip install --no-cache-dir \
        numpy scipy matplotlib \
        opencv-python-headless Pillow \
        pytransform3d trimesh \
        fastapi "uvicorn[standard]" pydantic \
        diffusers tqdm pyyaml

# ─── Código del proyecto ─────────────────────────────────────
# Copiamos sólo lo necesario para inferencia (no notebooks, no docs)
COPY src/ /app/src/
COPY scripts/api_server.py /app/scripts/api_server.py
COPY README.md /app/README.md
# Dashboard ejecutivo (servido en / por la API) + assets (vídeo demo + poster,
# servidos en /assets como estáticos — HTML ligero, el vídeo carga bajo demanda).
COPY docs/dashboard_ejecutivo.html /app/docs/dashboard_ejecutivo.html
COPY docs/assets/ /app/docs/assets/

# Los pesos y checkpoints se montan por volumen (no se incluyen en la imagen)
RUN mkdir -p /app/data/models /app/experiments/results /app/experiments/checkpoints

# ─── Healthcheck ─────────────────────────────────────────────
# Respeta $PORT (Render/Fly asignan uno propio). La forma shell del CMD
# expande la variable en runtime; con exec-form quedaría literal y fallaría.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

EXPOSE 8000

# API (sirve el dashboard en /, docs en /docs). PORT override para PaaS (Render/Fly).
CMD ["sh", "-c", "uvicorn --app-dir scripts api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
