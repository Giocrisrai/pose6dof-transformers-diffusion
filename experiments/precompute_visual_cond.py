#!/usr/bin/env python3
"""Precompute visual embeddings (ResNet-18) sobre el dataset v3.

Lee {train,val}.pt, corre el encoder sobre el campo `rgbds`, agrega
campo `visual_emb` (N, 52) y sobrescribe el archivo. El field `rgbds`
se mantiene por si se quiere re-entrenar el encoder en el futuro.

Uso:
    python experiments/precompute_visual_cond.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.visual_encoder import ResNet18RGBDEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("precompute")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v3"
EMBED_DIM = 52
BATCH_SIZE = 32


def precompute(split: str, encoder: ResNet18RGBDEncoder, device: str) -> None:
    in_path = DATASET_DIR / f"{split}.pt"
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    data = torch.load(in_path, weights_only=True)
    rgbds = data["rgbds"]
    n = len(rgbds)
    logger.info(f"split={split} n={n}")

    embs = torch.zeros(n, EMBED_DIM, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            batch = rgbds[start : start + BATCH_SIZE].to(device)
            embs[start : start + BATCH_SIZE] = encoder(batch).cpu()
            if (start // BATCH_SIZE) % 5 == 0:
                logger.info(f"  {start}/{n}")

    data["visual_emb"] = embs
    torch.save(data, in_path)
    logger.info(f"escrito (overwrite): {in_path}")


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"device: {device}")
    encoder = ResNet18RGBDEncoder(out_dim=EMBED_DIM).to(device).eval()
    for split in ("train", "val"):
        precompute(split, encoder, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
