import os, sys
from pathlib import Path
if Path('.').resolve().name == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')

import torch
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | torch: {torch.__version__}')
OUTPUT_DIR = Path('data/models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.planning.diffusion_policy import (
    ConditionalUNet1D,
    DiffusionGraspPlanner,
    SimpleDDPMScheduler,
)

# CONFIG EXTENDIDA (vs original local: 30 ep, 2K, 128)
N_SAMPLES = 5000
N_EPOCHS = 50
BATCH = 96
LR = 2e-4
HORIZON = 16
ACTION_DIM = 7
COND_DIM = 64
HIDDEN_DIM = 192
TIMESTEPS = 100
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def generate_data(n_samples, seed):
    np.random.seed(seed)
    planner = DiffusionGraspPlanner(action_dim=ACTION_DIM, horizon=HORIZON)
    trajs, conds = [], []
    for _ in tqdm(range(n_samples), desc='Gen'):
        t_obj = np.array([np.random.uniform(-0.4, 0.4),
                          np.random.uniform(-0.4, 0.4),
                          np.random.uniform(0.7, 1.0)])
        cond_vec = np.zeros(64)
        cond_vec[:3] = t_obj
        cond_vec[3:6] = np.random.randn(3) * 0.1
        T_obj = np.eye(4)
        T_obj[:3, 3] = t_obj
        traj_batch = planner.plan_grasp_heuristic(T_obj)
        traj = traj_batch[0]  # (horizon, 7)
        trajs.append(traj)
        conds.append(cond_vec)
    return np.array(trajs, dtype=np.float32), np.array(conds, dtype=np.float32)

X_train, c_train = generate_data(N_SAMPLES, SEED)
X_val, c_val = generate_data(N_SAMPLES // 5, SEED + 1)
print(f'Train: {X_train.shape}, Val: {X_val.shape}')

class GraspDataset(Dataset):
    def __init__(self, x, c):
        self.x = torch.tensor(x); self.c = torch.tensor(c)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.c[i]

train_loader = DataLoader(GraspDataset(X_train, c_train), batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(GraspDataset(X_val, c_val), batch_size=BATCH, num_workers=0)

model = ConditionalUNet1D(action_dim=ACTION_DIM, horizon=HORIZON,
                          cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM).to(device)
scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
print(f'Modelo: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params')

train_losses, val_losses = [], []
best_val = float('inf')
t0 = time.time()
for epoch in range(N_EPOCHS):
    model.train()
    sum_loss = 0; n = 0
    for xb, cb in train_loader:
        xb, cb = xb.to(device), cb.to(device)
        B = xb.shape[0]
        t = torch.randint(0, TIMESTEPS, (B,), device=device)
        eps = torch.randn_like(xb)
        ab_t = alpha_bar[t].view(-1, 1, 1)
        x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
        eps_pred = model(x_noisy, t, cb)
        loss = F.mse_loss(eps_pred, eps)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        sum_loss += loss.item() * B; n += B
    train_loss = sum_loss / n
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        sum_loss = 0; n = 0
        for xb, cb in val_loader:
            xb, cb = xb.to(device), cb.to(device)
            B = xb.shape[0]
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bar[t].view(-1, 1, 1)
            x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
            eps_pred = model(x_noisy, t, cb)
            sum_loss += F.mse_loss(eps_pred, eps).item() * B; n += B
        val_loss = sum_loss / n
    val_losses.append(val_loss)
    lr_sched.step()

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss,
                    'config': dict(horizon=HORIZON, action_dim=ACTION_DIM, cond_dim=COND_DIM,
                                   hidden_dim=HIDDEN_DIM, n_samples=N_SAMPLES, n_epochs=N_EPOCHS)},
                   OUTPUT_DIR / 'diffusion_policy_extended_mps.pth')

    if (epoch + 1) % 5 == 0:
        elapsed = (time.time() - t0) / 60
        print(f'Epoch {epoch+1:2d}/{N_EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | Best: {best_val:.5f} | {elapsed:.1f} min')

print(f'\nFINAL: best_val_loss = {best_val:.5f}')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label='Train', color='#0098CD')
ax.plot(val_losses, label='Validation', color='#FF6B35')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
ax.set_title(f'Diffusion Policy extendido en MPS — {N_SAMPLES} samples, {N_EPOCHS} ep, hidden={HIDDEN_DIM}')
ax.legend(); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'extended_mps_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Modelo guardado: {OUTPUT_DIR / "diffusion_policy_extended_mps.pth"}')
