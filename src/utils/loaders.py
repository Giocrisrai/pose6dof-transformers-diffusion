"""Loaders compartidos para experimentos.

Centraliza la carga de checkpoints FoundationPose, ground-truth de BOP datasets
y meshes CAD. Elimina duplicación entre experimentos exp1, exp6, exp8, exp11, exp12.

Ejemplo:
    from src.utils.loaders import load_predictions_with_gt
    samples = load_predictions_with_gt("ycbv", n_max=300)
    for s in samples:
        e = add_s_metric(s["R_pred"], s["t_pred"], s["R_gt"], s["t_gt"], s["points"])
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_REPO = Path(__file__).resolve().parents[2]


DATASET_INFO = {
    "ycbv": {
        "split": "test",
        "checkpoint": "experiments/checkpoints/fp_ycbv_checkpoint.json",
        "data_path": "data/datasets/ycbv",
    },
    "tless": {
        "split": "test_primesense",
        "checkpoint": "experiments/checkpoints/fp_tless_checkpoint.json",
        "data_path": "data/datasets/tless",
    },
}


@dataclass
class PoseSample:
    """Una instancia con prediccion FP y ground-truth listos para metricas."""
    scene_id: str
    img_id: int
    obj_id: int
    R_pred: np.ndarray
    t_pred: np.ndarray
    R_gt: np.ndarray
    t_gt: np.ndarray
    points: np.ndarray
    fp_time_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _load_checkpoint(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _normalize_translation(t: np.ndarray) -> np.ndarray:
    """Normaliza traslacion FP de metros a milimetros si es necesario."""
    if np.linalg.norm(t) < 5.0:  # heuristica: <5 → metros
        return t * 1000.0
    return t


def _load_mesh_points(model_path: Path, n_pts: int = 1000, seed: int = 42) -> np.ndarray | None:
    """Carga vértices del CAD y submuestra a n_pts."""
    import trimesh
    if not Path(model_path).exists():
        return None
    try:
        mesh = trimesh.load(str(model_path), process=False)
        pts = np.array(mesh.vertices, dtype=np.float64)
        if len(pts) > n_pts:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pts), n_pts, replace=False)
            pts = pts[idx]
        return pts
    except Exception:
        return None


def load_predictions_with_gt(
    ds_name: str,
    repo_root: Path = DEFAULT_REPO,
    n_max: int | None = None,
    n_model_points: int = 1000,
    seed: int = 42,
) -> list[PoseSample]:
    """Carga predicciones FP + ground-truth + mesh CAD por instancia.

    Args:
        ds_name: 'ycbv' o 'tless'
        repo_root: ruta al root del repositorio.
        n_max: limite de instancias a cargar (None = todas).
        n_model_points: puntos a submuestrear del CAD.
        seed: semilla para muestreo determinista.

    Returns:
        Lista de PoseSample listos para calcular ADD/ADD-S.
    """
    from src.utils.dataset_loader import BOPDataset

    if ds_name not in DATASET_INFO:
        raise ValueError(f"Dataset desconocido: {ds_name}. Use uno de: {list(DATASET_INFO)}")

    info = DATASET_INFO[ds_name]
    ckpt_path = repo_root / info["checkpoint"]
    data_path = repo_root / info["data_path"]
    split = info["split"]

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {ckpt_path}\n"
            f"Ejecuta: python scripts/download_drive_assets.py --what checkpoints"
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {data_path}\n"
            f"Descarga BOP: https://bop.felk.cvut.cz/datasets/"
        )

    ckpt = _load_checkpoint(ckpt_path)
    ds = BOPDataset(str(data_path), split=split)

    gt_cache: dict[str, dict] = {}
    mesh_cache: dict[int, np.ndarray | None] = {}
    samples: list[PoseSample] = []

    preds = ckpt["results"][:n_max] if n_max else ckpt["results"]

    for pred in preds:
        scene_id = pred["scene_id"]
        if isinstance(scene_id, int):
            scene_id = f"{scene_id:06d}"
        img_id = int(pred["img_id"])
        obj_id = int(pred["obj_id"])
        gt_idx = pred.get("gt_idx", -1)

        if scene_id not in gt_cache:
            gt_cache[scene_id] = ds.load_scene_gt(scene_id)
        gt_list = gt_cache[scene_id].get(str(img_id), [])
        if gt_idx < 0 or gt_idx >= len(gt_list):
            continue
        gt = gt_list[gt_idx]
        if gt["obj_id"] != obj_id:
            continue

        if obj_id not in mesh_cache:
            mesh_cache[obj_id] = _load_mesh_points(
                ds.get_model_path(obj_id), n_pts=n_model_points, seed=seed
            )
        if mesh_cache[obj_id] is None:
            continue

        R_pred = np.array(pred["R_pred"], dtype=np.float64)
        t_pred = _normalize_translation(np.array(pred["t_pred"], dtype=np.float64))

        samples.append(PoseSample(
            scene_id=scene_id,
            img_id=img_id,
            obj_id=obj_id,
            R_pred=R_pred,
            t_pred=t_pred,
            R_gt=np.array(gt["cam_R_m2c"], dtype=np.float64),
            t_gt=np.array(gt["cam_t_m2c"], dtype=np.float64),
            points=mesh_cache[obj_id],
            fp_time_ms=float(pred.get("time_s", 0.0)) * 1000.0,
            extra={"gt_idx": gt_idx},
        ))

    return samples


def load_diffusion_planner(
    weights_path: Path | None = None,
    device: str | None = None,
    cond_dim: int = 64,
    horizon: int = 16,
    action_dim: int = 7,
    hidden_dim: int = 128,
    n_timesteps: int = 100,
):
    """Carga el modelo Diffusion Policy entrenado y devuelve (planner, scheduler, device)."""
    import torch
    from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if weights_path is None:
        weights_path = DEFAULT_REPO / "data/models/diffusion_policy_grasp.pth"

    planner = ConditionalUNet1D(
        action_dim=action_dim, horizon=horizon,
        cond_dim=cond_dim, hidden_dim=hidden_dim,
    ).to(device)

    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                if key in ckpt:
                    planner.load_state_dict(ckpt[key])
                    break
            else:
                planner.load_state_dict(ckpt)
        else:
            planner.load_state_dict(ckpt)

    planner.eval()
    scheduler = SimpleDDPMScheduler(num_timesteps=n_timesteps)
    return planner, scheduler, device


__all__ = [
    "PoseSample",
    "DATASET_INFO",
    "load_predictions_with_gt",
    "load_diffusion_planner",
]
