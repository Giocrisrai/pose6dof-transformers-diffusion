#!/usr/bin/env python3
"""API REST FastAPI para el pipeline TFM Pose 6-DoF.

Endpoints:
- GET  /                   - Info del servicio
- GET  /health             - Health check
- GET  /models             - Lista de modelos Diffusion disponibles
- POST /plan-grasp         - Generar trayectoria de agarre desde pose 6-DoF
- POST /e2e                - Pipeline completo (con poses sintéticas demo)
- GET  /metrics            - Métricas del pipeline (cargadas de JSONs)
- GET  /docs               - OpenAPI Swagger UI (automático)

Uso:
    .venv/bin/uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
    # Probar:
    curl http://localhost:8000/docs
    curl -X POST http://localhost:8000/plan-grasp \
         -H "Content-Type: application/json" \
         -d '{"object_position": [0.0, 0.0, 0.8], "model": "ultra"}'
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "docs/dashboard_executive.html"  # inglés (landing)
DASHBOARD_ES = REPO / "docs/dashboard_ejecutivo.html"
ASSETS = REPO / "docs/assets"  # vídeo demo + poster (servidos como estáticos, no inline)
sys.path.insert(0, str(REPO))

app = FastAPI(
    title="TFM Pose 6-DoF Pipeline API",
    description="API REST para integración Transformer + Diffusion Policy en bin picking robótico",
    version="1.0.0",
    contact={
        "name": "Giocrisrai Godoy Bonillo, José Miguel Carrasco",
        "url": "https://github.com/Giocrisrai/pose6dof-transformers-diffusion",
    },
    license_info={"name": "MIT (TFM components)"},
)

# Sirve el vídeo demo y el poster como archivos estáticos en /assets (el dashboard
# los referencia con rutas relativas), en vez de embeberlos como data-URI en el HTML.
if ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")

# Cache de modelos
_models_cache: dict = {}


# ============================================================================
# SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: list[str]


class ModelInfo(BaseModel):
    name: str
    description: str
    path: str
    exists: bool
    mse_val: Optional[float] = None
    size_mb: Optional[float] = None


class GraspRequest(BaseModel):
    object_position: list[float] = Field(..., description="[x, y, z] en metros", min_length=3, max_length=3)
    object_rotation_axis_angle: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="axis-angle representation, 3 floats",
        min_length=3, max_length=3,
    )
    model: str = Field(
        default="ultra_fast",
        description="Modelo a usar: ultra_fast (destilado, por defecto), ultra, extended, original",
    )
    n_diffusion_steps: int = Field(default=25, ge=1, le=100)
    n_samples: int = Field(default=1, ge=1, le=20, description="Número de trayectorias a muestrear")


class GraspResponse(BaseModel):
    trajectory: list[list[list[float]]]  # [N, horizon, action_dim]
    horizon: int
    action_dim: int
    n_samples: int
    model_used: str
    latency_ms: float
    metadata: dict


class E2ERequest(BaseModel):
    dataset: str = Field(default="ycbv", description="ycbv o tless")
    n_instances: int = Field(default=1, ge=1, le=30)
    model: Optional[str] = Field(
        default=None,
        description="Modelo del planificador. Por defecto ultra_fast (destilado, 1 evaluación).",
    )
    use_ultra_model: bool = Field(
        default=True,
        description="Obsoleto: usar `model`. Solo se respeta si se envía y `model` no.",
    )


class E2EResponse(BaseModel):
    n_instances: int
    cycle_times_ms: list[float]
    cycle_p95_ms: float
    h3_passed: bool
    h3_margin_ms: float
    dataset: str
    model_used: str = "ultra_fast"


# ============================================================================
# HELPERS
# ============================================================================

MODELS_INFO: dict = {
    "original": {
        "path": REPO / "data/models/diffusion_policy_grasp.pth",
        "hidden_dim": 128,
        "mse_val": 0.020,
        "description": "Modelo original: 30 epochs, 2K trayectorias",
        "nfe": 25,
        "is_distilled": False,
    },
    "extended": {
        "path": REPO / "data/models/diffusion_policy_extended_mps.pth",
        "hidden_dim": 192,
        "mse_val": 0.01288,
        "description": "Modelo extendido: 50 epochs, 5K trayectorias",
        "nfe": 25,
        "is_distilled": False,
    },
    "ultra": {
        "path": REPO / "data/models/diffusion_policy_ultra.pth",
        "hidden_dim": 256,
        "mse_val": 0.00221,
        "description": "Modelo ultra: 100 epochs, 10K trayectorias (DDIM-25)",
        "nfe": 25,
        "is_distilled": False,
    },
    "ultra_fast": {
        "path": REPO / "data/models/diffusion_policy_ultra_fast.pth",
        "hidden_dim": 256,
        "mse_val": 0.01235,  # MSE student vs GT heuristic (exp14)
        "description": "Modelo ultra destilado a 1 evaluación (exp14): más preciso en el punto de agarre y ~20x más rápido que DDIM-25 (exp28, latency.json)",
        "nfe": 1,
        "is_distilled": True,
    },
    # === Modelos VLA-lite (exploraciones post-TFM 4-13) ===
    # Estos modelos NO son drop-in replacements del DP estandar (requieren
    # CLIP text/image encoder + gate especifico). Aparecen en /models para
    # transparencia pero /plan-grasp solo soporta los modelos estandar
    # (original/extended/ultra/ultra_fast). Para usar VLA-lite usar Gradio o
    # los scripts experiments/exp{16,18,20,22,24,26}.
    "clip": {
        "path": REPO / "data/models/diffusion_policy_clip.pth",
        "hidden_dim": 256,
        "mse_val": None,
        "description": "VLA-lite color (exp16): 'pick the red object' - 98.6% acc",
        "nfe": 25,
        "is_distilled": False,
        "is_vla_lite": True,
    },
    "clip_shapes": {
        "path": REPO / "data/models/diffusion_policy_clip_shapes.pth",
        "hidden_dim": 256,
        "mse_val": None,
        "description": "VLA-lite color+forma (exp18): 'pick the red sphere' - 99.9% acc",
        "nfe": 25,
        "is_distilled": False,
        "is_vla_lite": True,
    },
    "clip_multi": {
        "path": REPO / "data/models/diffusion_policy_clip_multi.pth",
        "hidden_dim": 256,
        "mse_val": None,
        "description": "VLA-lite multi-objeto N=2..5 (exp20): 100% acc",
        "nfe": 25,
        "is_distilled": False,
        "is_vla_lite": True,
    },
    "clip_size": {
        "path": REPO / "data/models/diffusion_policy_clip_size.pth",
        "hidden_dim": 256,
        "mse_val": None,
        "description": "VLA-lite con atributo TAMANO (exp22): 'pick the large red box' - 99.9%",
        "nfe": 25,
        "is_distilled": False,
        "is_vla_lite": True,
    },
    "clip_image": {
        "path": REPO / "data/models/diffusion_policy_clip_image.pth",
        "hidden_dim": 256,
        "mse_val": None,
        "description": "VLA-lite CLIP-image visual grounding (exp24): 100% sin atributos declarados",
        "nfe": 25,
        "is_distilled": False,
        "is_vla_lite": True,
    },
    "clip_spatial": {
        "path": REPO / "data/models/diffusion_policy_clip_spatial.pth",
        "hidden_dim": 256,
        "mse_val": None,
        "description": "VLA-lite razonamiento espacial (exp26): 'the leftmost', 'closest' - 98.4%",
        "nfe": 25,
        "is_distilled": False,
        "is_vla_lite": True,
    },
}


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str):
    """Lazy load del modelo solicitado."""
    if name not in MODELS_INFO:
        raise HTTPException(status_code=400, detail=f"Modelo desconocido: {name}")
    info = MODELS_INFO[name]
    if info.get("is_vla_lite"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modelo VLA-lite '{name}' requiere CLIP encoder + gate especifico. "
                "Usa el demo Gradio (python scripts/demo_charla.py) o el script "
                f"experiments/exp{16 if 'shapes' not in name and name == 'clip' else 18}.py."
            ),
        )
    if name in _models_cache:
        return _models_cache[name]

    if not info["path"].exists():
        raise HTTPException(status_code=404, detail=f"Pesos no encontrados: {info['path'].name}")

    import torch

    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

    device = get_device()
    model = ConditionalUNet1D(
        action_dim=7, horizon=16, cond_dim=64, hidden_dim=info["hidden_dim"]
    ).to(device)
    ckpt = torch.load(info["path"], map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)  # type: ignore[arg-type]  # sd es el state_dict del checkpoint
    model.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=100)
    _models_cache[name] = {"model": model, "scheduler": scheduler, "device": device, "info": info}
    return _models_cache[name]


def ddim_sample(model, scheduler, cond, device, n_steps=25):
    import torch
    horizon, action_dim = 16, 7
    x = torch.randn(1, horizon, action_dim, device=device)
    full_t = scheduler.num_timesteps
    step_indices = np.linspace(0, full_t - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.tensor([step], dtype=torch.long, device=device)
            noise_pred = model(x, t_tensor, cond)
            ab_t = alpha_bar[step]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                next_step = step_indices[i + 1]
                ab_next = alpha_bar[next_step]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0
    return x.cpu().numpy()[0]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
def root():
    """Landing: el dashboard ejecutivo (o el JSON de servicio si no está)."""
    if DASHBOARD.exists():
        return FileResponse(str(DASHBOARD), media_type="text/html")
    return JSONResponse(_service_info())


@app.get("/dashboard_executive.html", include_in_schema=False)
@app.get("/dashboard_ejecutivo.html", include_in_schema=False)
def dashboard_por_idioma(request: Request):
    """Las dos versiones del dashboard se enlazan entre sí por nombre de archivo."""
    archivo = DASHBOARD_ES if request.url.path.endswith("ejecutivo.html") else DASHBOARD
    if not archivo.exists():
        raise HTTPException(status_code=404, detail="dashboard no disponible")
    return FileResponse(str(archivo), media_type="text/html")


def _service_info() -> dict:
    return {
        "service": "TFM Pose 6-DoF Pipeline API",
        "version": "1.0.0",
        "description": "Integración Transformer + Diffusion Policy para bin picking robótico",
        "docs": "/docs",
        "dashboard": "/",
        "github": "https://github.com/Giocrisrai/pose6dof-transformers-diffusion",
    }


@app.get("/api", tags=["info"])
def api_info():
    """Información general del servicio (contrato JSON)."""
    return _service_info()


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    """Health check con info del entorno."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=get_device(),
        models_loaded=list(_models_cache.keys()),
    )


@app.get("/models", response_model=list[ModelInfo], tags=["info"])
def list_models():
    """Lista los modelos Diffusion Policy disponibles (incluido ultra_fast distillado)."""
    out = []
    for name, info in MODELS_INFO.items():
        path = info["path"]
        out.append(ModelInfo(
            name=name,
            description=info["description"],
            path=str(path.relative_to(REPO)),
            exists=path.exists(),
            mse_val=info["mse_val"],
            size_mb=round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
        ))
    return out


def muestrear_trayectoria(cached: dict, cond, n_pasos: int) -> np.ndarray:
    """Muestrea una trayectoria [16, 7] con el muestreador propio del checkpoint.

    Los modelos destilados se evalúan una sola vez; el resto usa DDIM con
    `n_pasos` pasos.
    """
    import torch
    model = cached["model"]
    device = cached["device"]
    if cached["info"].get("is_distilled", False):
        with torch.no_grad():
            x_ruido = torch.randn(1, 16, 7, device=device)
            t_cero = torch.zeros(1, dtype=torch.long, device=device)
            return model(x_ruido, t_cero, cond).cpu().numpy()[0]
    return ddim_sample(model, cached["scheduler"], cond, device, n_pasos)


def modelo_e2e(req: "E2ERequest") -> str:
    """Modelo del ciclo E2E: `model` manda; `use_ultra_model` solo si se envió."""
    if req.model is not None:
        return req.model
    if "use_ultra_model" in req.model_fields_set:
        return "ultra" if req.use_ultra_model else "original"
    return "ultra_fast"


@app.post("/plan-grasp", response_model=GraspResponse, tags=["pipeline"])
def plan_grasp(req: GraspRequest):
    """Genera trayectoria(s) de agarre desde una pose 6-DoF objetivo.

    Ejemplo:
        curl -X POST http://localhost:8000/plan-grasp \\
             -H "Content-Type: application/json" \\
             -d '{"object_position": [0.0, 0.0, 0.8], "n_samples": 3}'
    """
    import torch
    cached = load_model(req.model)
    device = cached["device"]

    # Cada checkpoint espera la condición con la disposición de su entrenamiento.
    from src.planning.conditioning import LAYOUT_BY_MODEL, cond_pos3, cond_pose12
    from src.utils.lie_groups import so3_exp

    posicion = np.asarray(req.object_position, dtype=np.float64)
    eje_angulo = np.asarray(req.object_rotation_axis_angle, dtype=np.float64)
    if LAYOUT_BY_MODEL.get(req.model) == "pose12":
        cond_vec = cond_pose12(so3_exp(eje_angulo), posicion)
    else:
        cond_vec = cond_pos3(posicion, extra=eje_angulo)
    cond = torch.tensor(cond_vec, device=device).unsqueeze(0)

    t0 = time.time()
    trajectories = []
    for _ in range(req.n_samples):
        traj = muestrear_trayectoria(cached, cond, req.n_diffusion_steps)
        trajectories.append(traj.tolist())
    latency_ms = (time.time() - t0) * 1000

    return GraspResponse(
        trajectory=trajectories,
        horizon=16,
        action_dim=7,
        n_samples=req.n_samples,
        model_used=req.model,
        latency_ms=latency_ms,
        metadata={
            "device": device,
            "n_diffusion_steps": req.n_diffusion_steps,
            "mse_val": cached["info"]["mse_val"],
        },
    )


@app.post("/e2e", response_model=E2EResponse, tags=["pipeline"])
def e2e_pipeline(req: E2ERequest):
    """Ejecuta el pipeline E2E sobre poses reales de los checkpoints (simulación temporal).

    Usa los tiempos reales del checkpoint FP + sampling Diffusion + tiempo nominal sim.
    No requiere CoppeliaSim corriendo (usa nominal).
    """
    import torch
    ckpt_path = REPO / f"experiments/checkpoints/fp_{req.dataset}_checkpoint.json"
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint no disponible: {req.dataset}")

    with open(ckpt_path) as f:
        ckpt = json.load(f)
    preds = ckpt["results"][:req.n_instances]

    model_name = modelo_e2e(req)
    cached = load_model(model_name)
    device = cached["device"]

    NOMINAL_SIM_MS = 906  # 50 steps × 18 ms del smoke test

    from src.planning.conditioning import condition_for_model

    cycle_times = []
    for pred in preds:
        fp_ms = pred["time_s"] * 1000.0
        # Sampling Diffusion
        R = np.array(pred["R_pred"])
        t = np.array(pred["t_pred"])
        # El checkpoint FP guarda t en metros o en milímetros según el dataset;
        # la condición va siempre en metros y con la disposición de entrenamiento.
        t_m = t.flatten() / 1000.0 if np.linalg.norm(t) > 5 else t.flatten()
        cond = torch.tensor(condition_for_model(model_name, R, t_m), device=device).unsqueeze(0)
        t0 = time.time()
        _ = muestrear_trayectoria(cached, cond, 25)
        if device == "mps":
            torch.mps.synchronize()
        diff_ms = (time.time() - t0) * 1000.0
        cycle_times.append(fp_ms + diff_ms + NOMINAL_SIM_MS)

    p95 = float(np.percentile(cycle_times, 95))
    return E2EResponse(
        n_instances=len(cycle_times),
        cycle_times_ms=[float(x) for x in cycle_times],
        cycle_p95_ms=p95,
        h3_passed=p95 < 10000,
        h3_margin_ms=10000 - p95,
        dataset=req.dataset,
        model_used=model_name,
    )


@app.get("/metrics", tags=["info"])
def get_metrics():
    """Devuelve las métricas consolidadas de los experimentos commiteados."""
    metrics = {}
    files = {
        "fp_pose_estimation": "experiments/results/local_metrics_with_bootstrap.json",
        "e2e_live_n30": "experiments/results/pipeline_e2e/e2e_live_metrics.json",
        "e2e_ultra_n30": "experiments/results/pipeline_e2e/e2e_live_ultra_metrics.json",
        "diffusion_comparison": "experiments/results/exp13_model_comparison/exp13_results.json",
        "robustness": "experiments/results/exp6_robustness/exp6_results.json",
        "profiling": "experiments/results/exp10_profiling/exp10_results.json",
    }
    for key, rel in files.items():
        p = REPO / rel
        if p.exists():
            metrics[key] = json.loads(p.read_text())
    return metrics


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scripts.api_server:app", host="0.0.0.0", port=8000, reload=False)
