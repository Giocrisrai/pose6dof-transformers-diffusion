#!/usr/bin/env python3
"""Fine-tune de la Diffusion Policy sobre dataset del sim (Iter 1).

Carga el checkpoint existente, entrena 50 epochs con MSE noise loss,
guarda nuevo checkpoint y curvas train/val.

Uso (CoppeliaSim NO requerido para esta phase):
    python experiments/train_diffusion_on_sim.py
    python experiments/train_diffusion_on_sim.py --epochs 50 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_dataset import SimPickDataset
from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_dp")

REPO_OUT_MODELS = REPO / "data" / "models"
DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-in", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_grasp.pth")
    parser.add_argument("--checkpoint-out", type=Path,
                        default=REPO_OUT_MODELS / "diffusion_policy_sim_v1.pth")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"device: {device}")

    # Dataset
    train_ds = SimPickDataset(DATASET_DIR / "train.pt")
    val_ds = SimPickDataset(DATASET_DIR / "val.pt")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f"train={len(train_ds)}, val={len(val_ds)}")

    # Model + scheduler
    config = {"action_dim": 7, "horizon": 16, "cond_dim": 64,
              "hidden_dim": 128, "n_timesteps": 100, "n_epochs": args.epochs}
    model = ConditionalUNet1D(
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        cond_dim=config["cond_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
    scheduler = SimpleDDPMScheduler(num_timesteps=config["n_timesteps"])

    # Cargar checkpoint existente si esta
    if args.checkpoint_in.exists():
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"checkpoint cargado: {args.checkpoint_in.name}")
            except RuntimeError as e:
                logger.warning(f"checkpoint incompatible ({e}); entreno desde init aleatorio")
        else:
            logger.warning("checkpoint sin 'model_state_dict'; entreno desde init aleatorio")
    else:
        logger.warning(f"checkpoint no existe: {args.checkpoint_in}; entreno desde init aleatorio")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    t0 = time.time()
    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_train = 0.0
        n_batches = 0
        for cond, traj in train_dl:
            cond, traj = cond.to(device), traj.to(device)
            B = cond.shape[0]
            t = torch.randint(0, config["n_timesteps"], (B,), device=device, dtype=torch.long)
            traj_noisy, noise = scheduler.add_noise_batch(traj, t)
            noise_pred = model(traj_noisy, t, cond)
            loss = loss_fn(noise_pred, noise)
            if torch.isnan(loss):
                logger.error(f"NaN loss en epoch {epoch}, abortando")
                return 1
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_train += loss.item()
            n_batches += 1
        epoch_train /= max(n_batches, 1)

        # Val
        model.eval()
        epoch_val = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for cond, traj in val_dl:
                cond, traj = cond.to(device), traj.to(device)
                B = cond.shape[0]
                t = torch.randint(0, config["n_timesteps"], (B,), device=device, dtype=torch.long)
                traj_noisy, noise = scheduler.add_noise_batch(traj, t)
                noise_pred = model(traj_noisy, t, cond)
                loss = loss_fn(noise_pred, noise)
                epoch_val += loss.item()
                n_val_batches += 1
        epoch_val /= max(n_val_batches, 1)

        train_losses.append(epoch_train)
        val_losses.append(epoch_val)
        logger.info(f"epoch {epoch+1:3d}/{args.epochs}  "
                    f"train_loss={epoch_train:.4f}  val_loss={epoch_val:.4f}")

    elapsed = time.time() - t0
    logger.info(f"training total: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Guardar checkpoint
    REPO_OUT_MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config,
    }, args.checkpoint_out)
    logger.info(f"checkpoint escrito: {args.checkpoint_out}")

    # Resumen final
    summary = {
        "epochs": args.epochs,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "min_val_loss": min(val_losses),
        "min_val_epoch": val_losses.index(min(val_losses)) + 1,
    }
    summary_path = args.checkpoint_out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
