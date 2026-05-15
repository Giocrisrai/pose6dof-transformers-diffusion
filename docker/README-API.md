# Contenedor API REST + Gradio — TFM Pose 6-DoF

Imagen ligera (~1.5 GB) que empaqueta el pipeline de inferencia con dos servicios:

- **FastAPI** en `:8000` con endpoints `/plan-grasp`, `/e2e`, `/models`, `/metrics`, `/docs`.
- **Gradio** en `:7860` con UI interactiva para probar los **10 modelos Diffusion** (original/extended/ultra/ultra_fast distillado + 6 VLA-lite: clip, clip_shapes, clip_multi, clip_size, clip_image, clip_spatial).

A diferencia de `inference-gpu.Dockerfile` (CUDA + FoundationPose + nvdiffrast, ~6 GB),
este contenedor sólo necesita CPU/MPS y los pesos ya entrenados. Apto para deployment
en cualquier host: Mac, Linux x86 o Linux ARM (Jetson).

## Build

```bash
docker build -f docker/api.Dockerfile -t tfm-pose6dof-api:latest .
```

## Run

```bash
docker run --rm \
    -p 8000:8000 -p 7860:7860 \
    -v $(pwd)/data/models:/app/data/models:ro \
    -v $(pwd)/experiments/checkpoints:/app/experiments/checkpoints:ro \
    -v $(pwd)/experiments/results:/app/experiments/results:ro \
    tfm-pose6dof-api:latest
```

O con docker-compose:

```bash
docker compose -f docker/docker-compose.yml up api
```

## Verificación

```bash
# Health check
curl http://localhost:8000/health

# Listar modelos
curl http://localhost:8000/models | jq

# Generar trayectoria con el modelo ultra
curl -X POST http://localhost:8000/plan-grasp \
     -H "Content-Type: application/json" \
     -d '{"object_position": [0.0, 0.0, 0.8], "model": "ultra", "n_samples": 3}' | jq

# Gradio en navegador
open http://localhost:7860
```

## Notas

- Los pesos (`data/models/*.pth`) **no** se incluyen en la imagen; se montan por volumen.
- El contenedor usa `torch==2.4.1+cpu`; en host con GPU se puede sustituir por la build CUDA.
- El healthcheck verifica `/health` cada 30 s.
- El script `scripts/launch_api_and_gradio.sh` lanza ambos servicios y propaga señales para cierre limpio.
