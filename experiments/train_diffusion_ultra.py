#!/usr/bin/env python3
"""Entrenamiento ULTRA-extendido de Diffusion Policy en MPS.

Configuracion mas agresiva que el extended:
- 100 epochs (vs 50)
- 10000 trayectorias (vs 5000)
- hidden_dim 256 (vs 192)
- Schedule cosine annealing con warmup
- Gradient clipping
- Early stopping

Objetivo: MSE final < 0.005 (vs 0.013 extended, 0.020 original).

Salida: data/models/diffusion_policy_ultra.pth
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "data/models"
OUTPUT.mkdir(parents=True, exist_ok=True)

from src.planning.diffusion_policy import (
    ConditionalUNet1D,
    DiffusionGraspPlanner,
    SimpleDDPMScheduler,
)

# CONFIG ULTRA
N_SAMPLES = 10000
N_VAL = 2000
N_EPOCHS = 100
WARMUP_EPOCHS = 5
BATCH = 128
LR = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
HORIZON = 16
ACTION_DIM = 7
COND_DIM = 64
HIDDEN_DIM = 256
TIMESTEPS = 100
SEED = 42
EARLY_STOP_PATIENCE = 15


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ultra] device: {device}, torch: {torch.__version__}")
    print(f"[ultra] config: {N_EPOCHS} ep x {N_SAMPLES} trajs, hidden={HIDDEN_DIM}, batch={BATCH}, lr={LR}")

    # === Generar datos ===
    print("\n[1/3] Generando datos...")
    planner_ref = DiffusionGraspPlanner(action_dim=ACTION_DIM, horizon=HORIZON)
    def gen(n, seed):
        rng = np.random.default_rng(seed)
        trajs = np.empty((n, HORIZON, ACTION_DIM), dtype=np.float32)
        conds = np.zeros((n, COND_DIM), dtype=np.float32)
        for i in tqdm(range(n), desc="gen", leave=False):
            t_obj = np.array([
                rng.uniform(-0.4, 0.4),
                rng.uniform(-0.4, 0.4),
                rng.uniform(0.7, 1.0),
            ])
            conds[i, :3] = t_obj
            conds[i, 3:6] = rng.standard_normal(3) * 0.1
            T_obj = np.eye(4)
            T_obj[:3, 3] = t_obj
            traj_batch = planner_ref.plan_grasp_heuristic(T_obj)
            trajs[i] = traj_batch[0]
        return trajs, conds

    X_train, c_train = gen(N_SAMPLES, SEED)
    X_val, c_val = gen(N_VAL, SEED + 1)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    class GraspDataset(Dataset):
        def __init__(self, x, c):
            self.x = torch.tensor(x)
            self.c = torch.tensor(c)
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return self.x[i], self.c[i]

    train_loader = DataLoader(GraspDataset(X_train, c_train), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(GraspDataset(X_val, c_val), batch_size=BATCH, num_workers=0)

    # === Modelo ===
    print("\n[2/3] Configurando modelo...")
    model = ConditionalUNet1D(
        action_dim=ACTION_DIM, horizon=HORIZON,
        cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros: {n_params/1e6:.2f} M")

    scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup linear + cosine annealing
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(N_EPOCHS - WARMUP_EPOCHS, 1)
        return max(LR_MIN/LR, 0.5 * (1 + np.cos(np.pi * progress)))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # === Training loop ===
    print("\n[3/3] Entrenamiento...")
    train_losses, val_losses, lrs = [], [], []
    best_val = float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        model.train()
        sum_loss, n = 0, 0
        for xb, cb in train_loader:
            xb, cb = xb.to(device), cb.to(device)
            B = xb.shape[0]
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bar[t].view(-1, 1, 1)
            x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
            eps_pred = model(x_noisy, t, cb)
            loss = F.mse_loss(eps_pred, eps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            sum_loss += loss.item() * B
            n += B
        train_loss = sum_loss / n
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            sum_loss, n = 0, 0
            for xb, cb in val_loader:
                xb, cb = xb.to(device), cb.to(device)
                B = xb.shape[0]
                t = torch.randint(0, TIMESTEPS, (B,), device=device)
                eps = torch.randn_like(xb)
                ab_t = alpha_bar[t].view(-1, 1, 1)
                x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
                eps_pred = model(x_noisy, t, cb)
                sum_loss += F.mse_loss(eps_pred, eps).item() * B
                n += B
            val_loss = sum_loss / n
        val_losses.append(val_loss)
        lrs.append(optimizer.param_groups[0]["lr"])

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "horizon": HORIZON, "action_dim": ACTION_DIM, "cond_dim": COND_DIM,
                    "hidden_dim": HIDDEN_DIM, "n_samples": N_SAMPLES, "n_epochs": N_EPOCHS,
                    "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH,
                },
            }, OUTPUT / "diffusion_policy_ultra.pth")
        else:
            no_improve += 1

        lr_sched.step()

        if (epoch + 1) % 10 == 0 or improved:
            marker = "★" if improved else " "
            elapsed = (time.time() - t0) / 60
            print(f"  Ep {epoch+1:3d}/{N_EPOCHS} {marker} | train={train_loss:.5f} | val={val_loss:.5f} | best={best_val:.5f} | lr={lrs[-1]:.2e} | {elapsed:.1f}m")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stop @ epoch {epoch+1} (sin mejora en {EARLY_STOP_PATIENCE} epochs)")
            break

    elapsed_total = (time.time() - t0) / 60
    print(f"\n[OK] FINAL: best_val={best_val:.5f} en {elapsed_total:.1f} min")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(train_losses, label="Train", color="#0098CD", linewidth=1.5)
        ax1.plot(val_losses, label="Validation", color="#FF6B35", linewidth=1.5)
        ax1.axhline(best_val, color='gray', linestyle='--', alpha=0.5, label=f"Best val: {best_val:.5f}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE")
        ax1.set_title(f"Diffusion Policy ULTRA — {N_SAMPLES} trajs × {len(train_losses)} ep, hidden={HIDDEN_DIM}")
        ax1.legend()
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        ax2.plot(lrs, color="#35876B", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning rate")
        ax2.set_title("LR schedule (warmup + cosine)")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT / "ultra_training_curves.png", dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[OK] {OUTPUT / 'ultra_training_curves.png'}")
    except Exception as e:
        print(f"[warn] plot: {e}")

    # Guardar metadata
    summary = {
        "config": {
            "n_samples": N_SAMPLES, "n_val": N_VAL, "n_epochs": N_EPOCHS,
            "epochs_completed": len(train_losses), "batch_size": BATCH,
            "lr_max": LR, "lr_min": LR_MIN, "warmup_epochs": WARMUP_EPOCHS,
            "hidden_dim": HIDDEN_DIM, "horizon": HORIZON, "action_dim": ACTION_DIM,
            "n_params_M": n_params / 1e6, "device": device, "seed": SEED,
        },
        "results": {
            "best_val_loss": best_val,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "time_min": elapsed_total,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        "comparison": {
            "original (30ep, 2K, h=128)": 0.020,
            "extended (50ep, 5K, h=192)": 0.01288,
            "ultra (current)": best_val,
        },
    }
    with open(OUTPUT / "diffusion_policy_ultra_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Resumen: {OUTPUT / 'diffusion_policy_ultra_summary.json'}")


if __name__ == "__main__":
    main()
