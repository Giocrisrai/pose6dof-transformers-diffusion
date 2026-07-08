# Contenedor API REST + Dashboard — TFM Pose 6-DoF

Imagen ligera (~1.5 GB) que empaqueta el pipeline de inferencia como una app unificada:

- **Dashboard ejecutivo** en `/` (HTML autocontenido: métricas AUC, garantías de consistencia y demo en vídeo).
- **API REST (FastAPI)** con endpoints `/plan-grasp`, `/e2e`, `/models`, `/metrics`, `/health` y documentación interactiva en `/docs` (`/api` devuelve la info del servicio en JSON).

Sirve los **10 modelos Diffusion** (original/extended/ultra/ultra_fast distillado + 6 VLA-lite:
clip, clip_shapes, clip_multi, clip_size, clip_image, clip_spatial).

A diferencia de `inference-gpu.Dockerfile` (CUDA + FoundationPose + nvdiffrast, ~6 GB),
este contenedor sólo necesita CPU/MPS y los pesos ya entrenados. Apto para deployment
en cualquier host: Mac, Linux x86 o Linux ARM (Jetson), y PaaS como Render/Fly
(respeta la variable `$PORT`).

## Build

```bash
docker build -f docker/api.Dockerfile -t tfm-pose6dof-api:latest .
```

## Run

```bash
docker run --rm \
    -p 8000:8000 \
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
# Dashboard ejecutivo en el navegador
open http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Listar modelos
curl http://localhost:8000/models | jq

# Generar trayectoria con el modelo ultra (posición en la zona diestra del UR5e)
curl -X POST http://localhost:8000/plan-grasp \
     -H "Content-Type: application/json" \
     -d '{"object_position": [-0.05, -0.22, 0.05], "model": "ultra", "n_samples": 3}' | jq

# Documentación interactiva (Swagger UI)
open http://localhost:8000/docs
```

## Notas

- Los pesos (`data/models/*.pth`) **no** se incluyen en la imagen; se montan por volumen.
- El contenedor usa `torch` CPU; en host con GPU se puede sustituir por la build CUDA.
- El healthcheck verifica `/health` cada 30 s.
- El CMD arranca `uvicorn` en `${PORT:-8000}` (un solo proceso; el dashboard se sirve
  como estático desde `/`, sin servicios adicionales).
