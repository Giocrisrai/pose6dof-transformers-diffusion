#!/usr/bin/env python3
"""Precompute visual embeddings (ResNet-18) sobre el dataset.

Lee {train,val}.pt, corre el encoder sobre el campo `rgbds`, agrega
campo `visual_emb` (N, 52) y sobrescribe el archivo. El field `rgbds`
se mantiene por si se quiere re-entrenar el encoder en el futuro.

Uso:
    python experiments/precompute_visual_cond.py                # v3 (default)
    python experiments/precompute_visual_cond.py --version v4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.visual_encoder import ResNet18RGBDEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("precompute")

EMBED_DIM = 52
BATCH_SIZE = 32


def precompute(
    split: str, encoder: ResNet18RGBDEncoder, device: str, dataset_dir: Path
) -> None:
    in_path = dataset_dir / f"{split}.pt"
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default="v3",
        help="dataset version: v3 | v4 | ... (default v3)",
    )
    args = parser.parse_args()

    dataset_dir = REPO / "data" / "datasets" / f"sim_pick_{args.version}"
    iter_num = args.version.lstrip("v")
    encoder_ckpt = REPO / "data" / "models" / f"visual_encoder_iter{iter_num}.pth"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"version={args.version} dataset_dir={dataset_dir} device={device}")
    encoder = ResNet18RGBDEncoder(out_dim=EMBED_DIM).to(device).eval()
    encoder_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder.state_dict(), "out_dim": EMBED_DIM}, encoder_ckpt)
    logger.info(f"encoder ckpt: {encoder_ckpt}")
    for split in ("train", "val"):
        precompute(split, encoder, device, dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
