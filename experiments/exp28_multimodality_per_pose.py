#!/usr/bin/env python3
"""Multimodalidad por pose de Diffusion Policy, con la condición de entrenamiento.

Por qué existe este experimento
-------------------------------
`exp8_diffusion_diversity.py` reportó K = 2 modos. Tiene dos problemas que este
script corrige:

1. **Condición distinta de la de entrenamiento.** exp8 arma la condición como
   `[R.flatten(), t]`. Ningún modelo se entrenó así:
     - `diffusion_policy_grasp.pth` (notebook 06) usa `pose[:3, :].flatten()`,
       es decir las tres filas de [R | t], con poses sobre la mesa.
     - `ultra` y `ultra_fast` usan la posición en `cond[:3]`.
   Con la condición cambiada el modelo se evalúa fuera de su distribución.
2. **Modos entre escenas, no por pose.** exp8 agrupa los puntos finales de cinco
   escenas distintas. Multimodalidad significa varias soluciones para *una misma*
   pose, así que aquí el análisis es por pose.

Qué mide
--------
Para cada modelo, P poses tomadas de su distribución de entrenamiento y N
trayectorias por pose:

- **Error en el punto de agarre**: distancia entre los waypoints de la fase de
  agarre y el objeto. Dice si la política llega al objetivo.
- **Dispersión**: distancia media entre los puntos finales de una misma pose.
- **Calidad de los grupos**: error de agarre dentro del grupo mayoritario y en el
  resto. Si el resto es peor, los "modos" son muestras fallidas, no estrategias
  de agarre alternativas.
- **Modos**: mejor silhouette de K-means (K = 2..5) sobre los puntos finales,
  comparada con una referencia gaussiana de igual media y covarianza, que por
  construcción no tiene modos. El p-valor es la fracción de referencias con
  silhouette mayor o igual que la observada. Una pose "muestra modos" si p < 0.05;
  por azar se espera un 5 % de las poses.

Como contraste se repite la medición del modelo `grasp` con la condición de exp8.

    python experiments/exp28_multimodality_per_pose.py            # ~2 min en M1

Salida: experiments/results/exp28_multimodality/exp28_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "experiments/results/exp28_multimodality"

HORIZON, ACTION_DIM, COND_DIM = 16, 7, 64
# Índices de la fase de agarre del planificador heurístico (0.3 <= progreso < 0.5),
# donde la posición de referencia es exactamente la del objeto.
FASE_AGARRE = [i for i in range(HORIZON) if 0.3 <= i / (HORIZON - 1) < 0.5]
ALFA = 0.05


# ── Condición ───────────────────────────────────────────────────────────────
# Las disposiciones de entrenamiento viven en src/planning/conditioning.py.

from src.planning.conditioning import cond_pos3, cond_pose12  # noqa: E402


def cond_exp8(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """La que usa exp8: [R.flatten(), t]. No coincide con ningún entrenamiento."""
    cond = np.zeros(COND_DIM, dtype=np.float32)
    cond[:12] = np.concatenate([R.flatten(), t])
    return cond


# ── Poses de la distribución de entrenamiento de cada modelo ────────────────

def poses_grasp(n: int, rng: np.random.Generator):
    """Mismos rangos que `generate_training_data` del notebook 06."""
    from src.utils.lie_groups import so3_exp

    for _ in range(n):
        t = np.array([rng.uniform(0.1, 0.5), rng.uniform(-0.2, 0.2), rng.uniform(0.02, 0.08)])
        yield so3_exp(rng.standard_normal(3) * 0.3), t


def poses_ultra(n: int, rng: np.random.Generator):
    """Mismos rangos que `train_diffusion_ultra.py`."""
    for _ in range(n):
        t = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.0)])
        yield rng.standard_normal(3) * 0.1, t


# ── Muestreo por lotes ──────────────────────────────────────────────────────

def muestrear_ddim(modelo, scheduler, cond: np.ndarray, n: int, device: str, pasos: int = 25):
    import torch

    c = torch.tensor(cond, device=device).unsqueeze(0).repeat(n, 1)
    x = torch.randn(n, HORIZON, ACTION_DIM, device=device)
    indices = np.linspace(0, scheduler.num_timesteps - 1, pasos).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, paso in enumerate(indices):
            t = torch.full((n,), int(paso), dtype=torch.long, device=device)
            eps = modelo(x, t, c)
            ab = alpha_bar[paso]
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            if i < len(indices) - 1:
                ab_sig = alpha_bar[indices[i + 1]]
                x = torch.sqrt(ab_sig) * x0 + torch.sqrt(1 - ab_sig) * eps
            else:
                x = x0
    return x.cpu().numpy()


def muestrear_un_paso(modelo, cond: np.ndarray, n: int, device: str):
    """Modelo destilado: una pasada directa, como en la API."""
    import torch

    c = torch.tensor(cond, device=device).unsqueeze(0).repeat(n, 1)
    with torch.no_grad():
        x = torch.randn(n, HORIZON, ACTION_DIM, device=device)
        return modelo(x, torch.zeros(n, dtype=torch.long, device=device), c).cpu().numpy()


# ── Métricas ────────────────────────────────────────────────────────────────

def mejor_agrupacion(puntos: np.ndarray) -> tuple[float, int, np.ndarray]:
    """Mejor silhouette de K-means para K = 2..5, con sus etiquetas."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    mejor, mejor_k, mejores = -1.0, 0, np.zeros(len(puntos), dtype=int)
    for k in (2, 3, 4, 5):
        etiquetas = KMeans(n_clusters=k, random_state=0, n_init=5).fit_predict(puntos)
        s = float(silhouette_score(puntos, etiquetas))
        if s > mejor:
            mejor, mejor_k, mejores = s, k, etiquetas
    return mejor, mejor_k, mejores


def mejor_silhouette(puntos: np.ndarray) -> tuple[float, int]:
    s, k, _ = mejor_agrupacion(puntos)
    return s, k


def prueba_de_modos(puntos: np.ndarray, rng: np.random.Generator, n_ref: int = 100) -> dict:
    """Silhouette observada frente a una referencia gaussiana sin modos."""
    observada, k, etiquetas = mejor_agrupacion(puntos)
    principal = int(np.argmax(np.bincount(etiquetas)))
    media = puntos.mean(axis=0)
    cov = np.cov(puntos.T) + 1e-12 * np.eye(puntos.shape[1])
    referencia = np.array([
        mejor_silhouette(rng.multivariate_normal(media, cov, len(puntos)))[0] for _ in range(n_ref)
    ])
    return {
        "silhouette": observada,
        "best_k": k,
        "null_mean": float(referencia.mean()),
        "null_p95": float(np.percentile(referencia, 95)),
        "p_value": float((np.sum(referencia >= observada) + 1) / (n_ref + 1)),
        "main_cluster_share": float(np.mean(etiquetas == principal)),
        "_in_main_cluster": etiquetas == principal,
    }


def medir_pose(trayectorias: np.ndarray, t_obj: np.ndarray, rng, n_ref: int) -> dict:
    finales = trayectorias[:, -1, :3]
    pares = np.linalg.norm(finales[:, None] - finales[None], axis=-1)
    dispersion = pares[np.triu_indices(len(finales), 1)].mean()
    agarre = np.linalg.norm(trayectorias[:, FASE_AGARRE, :3] - t_obj, axis=-1).mean(axis=1)
    modos = prueba_de_modos(finales, rng, n_ref)
    en_principal = modos.pop("_in_main_cluster")
    # Si hay grupos, ¿son soluciones alternativas igual de buenas o muestras peores?
    resto = float(agarre[~en_principal].mean() * 100) if (~en_principal).any() else None
    return {
        "grasp_point_error_cm": float(agarre.mean() * 100),
        "grasp_point_error_main_cluster_cm": float(agarre[en_principal].mean() * 100),
        "grasp_point_error_other_clusters_cm": resto,
        "endpoint_dispersion_cm": float(dispersion * 100),
        **modos,
    }


def resumir(poses: list[dict]) -> dict:
    con_modos = sum(p["p_value"] < ALFA for p in poses)
    return {
        "n_poses": len(poses),
        "grasp_point_error_cm_median": float(np.median([p["grasp_point_error_cm"] for p in poses])),
        "endpoint_dispersion_cm_median": float(np.median([p["endpoint_dispersion_cm"] for p in poses])),
        "silhouette_median": float(np.median([p["silhouette"] for p in poses])),
        "null_silhouette_median": float(np.median([p["null_mean"] for p in poses])),
        "main_cluster_share_median": float(np.median([p["main_cluster_share"] for p in poses])),
        "grasp_point_error_main_cluster_cm_median": float(
            np.median([p["grasp_point_error_main_cluster_cm"] for p in poses])),
        "grasp_point_error_other_clusters_cm_median": float(
            np.median([p["grasp_point_error_other_clusters_cm"] for p in poses])),
        "poses_with_modes": int(con_modos),
        "poses_with_modes_expected_by_chance": round(ALFA * len(poses), 1),
    }


# ── Modelos ─────────────────────────────────────────────────────────────────

def cargar(nombre: str, hidden_dim: int, device: str):
    import torch

    from src.planning.diffusion_policy import ConditionalUNet1D

    modelo = ConditionalUNet1D(
        action_dim=ACTION_DIM, horizon=HORIZON, cond_dim=COND_DIM, hidden_dim=hidden_dim
    ).to(device)
    ckpt = torch.load(REPO / f"data/models/diffusion_policy_{nombre}.pth",
                      map_location=device, weights_only=True)
    estado = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poses", type=int, default=20)
    parser.add_argument("--muestras", type=int, default=100, help="trayectorias por pose")
    parser.add_argument("--referencias", type=int, default=100, help="réplicas de la referencia sin modos")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    import torch

    from src.planning.diffusion_policy import SimpleDDPMScheduler

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scheduler = SimpleDDPMScheduler(num_timesteps=100)
    torch.manual_seed(args.seed)

    grasp = cargar("grasp", 128, device)
    ultra = cargar("ultra", 256, device)
    ultra_fast = cargar("ultra_fast", 256, device)

    # (nombre, descripción, generador de poses, condición, muestreador)
    configuraciones = [
        ("grasp", "DDIM-25 · condición de entrenamiento (filas de [R|t])", poses_grasp,
         lambda a, t: cond_pose12(a, t), lambda c, n: muestrear_ddim(grasp, scheduler, c, n, device)),
        ("grasp_cond_exp8", "DDIM-25 · condición usada en exp8 ([R.flatten(), t])", poses_grasp,
         lambda a, t: cond_exp8(a, t), lambda c, n: muestrear_ddim(grasp, scheduler, c, n, device)),
        ("ultra", "DDIM-25 · condición de entrenamiento (posición)", poses_ultra,
         lambda a, t: cond_pos3(t, a), lambda c, n: muestrear_ddim(ultra, scheduler, c, n, device)),
        ("ultra_fast", "destilado 1 paso · condición de entrenamiento (posición)", poses_ultra,
         lambda a, t: cond_pos3(t, a), lambda c, n: muestrear_un_paso(ultra_fast, c, n, device)),
    ]

    resultado: dict = {
        "config": {"poses": args.poses, "samples_per_pose": args.muestras,
                   "null_replicates": args.referencias, "seed": args.seed, "alpha": ALFA,
                   "device": device},
        "models": {},
    }
    print(f"[exp28] {args.poses} poses × {args.muestras} trayectorias · device {device}\n")
    print(f"{'modelo':16s} {'err. agarre':>12s} {'dispersión':>11s} {'silhouette':>11s} "
          f"{'ref. nula':>10s} {'poses con modos':>16s}")
    for nombre, descripcion, generar, condicion, muestrear in configuraciones:
        rng = np.random.default_rng(args.seed)  # mismas poses para las variantes de un modelo
        por_pose = []
        for extra, t in generar(args.poses, rng):
            trayectorias = muestrear(condicion(extra, t), args.muestras)
            por_pose.append(medir_pose(trayectorias, t, rng, args.referencias))
        resumen = resumir(por_pose)
        resultado["models"][nombre] = {"description": descripcion, "summary": resumen,
                                       "per_pose": por_pose}
        print(f"{nombre:16s} {resumen['grasp_point_error_cm_median']:9.2f} cm "
              f"{resumen['endpoint_dispersion_cm_median']:8.2f} cm "
              f"{resumen['silhouette_median']:11.3f} {resumen['null_silhouette_median']:10.3f} "
              f"{resumen['poses_with_modes']:>9d} / {resumen['n_poses']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp28_results.json").write_text(json.dumps(resultado, indent=2) + "\n")
    print(f"\n→ {(OUT / 'exp28_results.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
