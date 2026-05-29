# Iter 3 Implementation Plan — Conditioning visual RGB-D

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar el zero-padded conditioning de la Diffusion Policy por un encoder visual ResNet-18 sobre RGB-D del sim, apuntando a `dp_grasp_plausible_pct_sim ≥ 55%` (vs 36% en Iter 2).

**Architecture:** ResNet-18 pretrained (frozen, conv1 patched a 4 canales) + Linear(512,52) trainable + pose 12d → cond 64d. Embeddings precomputados para training rápido.

**Tech Stack:** PyTorch 2.x + torchvision (ResNet-18 weights), CoppeliaSim ZMQ + simIK, MPS backend M1 Pro.

---

## File Structure

**Nuevos:**
- `src/planning/visual_encoder.py` — `ResNet18RGBDEncoder`
- `tests/test_visual_encoder.py`
- `experiments/precompute_visual_cond.py`
- `experiments/eval_diffusion_iter3_sim.py`

**Modificados:**
- `experiments/collect_diffusion_dataset.py` — DATASET_VERSION=v3, agrega rgbd
- `src/planning/diffusion_policy.py` — `encode_observation` acepta `visual_emb` opcional
- `experiments/train_diffusion_on_sim.py` — usa `visual_emb` precomputado
- `experiments/run_pick_with_diffusion.py` — `pick_with_dp` captura RGB-D en vivo
- `docs/INTEGRATION_PIPELINE.md` — sección Iter 3
- `.gitignore` — `sim_pick_v3/` + `diffusion_policy_sim_v3.pth`

---

## Pre-flight checks

- [ ] CoppeliaSim corriendo en :23000:
  ```bash
  cd /Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm
  .venv/bin/python -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient; sim = RemoteAPIClient('localhost', 23000).require('sim'); print(f'OK v{sim.getInt32Param(sim.intparam_program_version)}')"
  ```
- [ ] torchvision instalado: `.venv/bin/python -c "import torchvision; print(torchvision.__version__)"` (>=0.15)
- [ ] Branch: `feat/diffusion-retrain-iter3` (ya creada).

---

## Task 1: `visual_encoder.py` + tests

**Files:**
- Create: `src/planning/visual_encoder.py`
- Test: `tests/test_visual_encoder.py`

### Step 1.1: Write failing test

```python
# tests/test_visual_encoder.py
import torch
from src.planning.visual_encoder import ResNet18RGBDEncoder


def test_output_shape():
    enc = ResNet18RGBDEncoder(out_dim=52)
    rgbd = torch.zeros(2, 4, 224, 224)  # batch=2
    emb = enc(rgbd)
    assert emb.shape == (2, 52)


def test_resnet_frozen():
    enc = ResNet18RGBDEncoder(out_dim=52)
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    total = sum(p.numel() for p in enc.parameters())
    assert trainable < total * 0.05, f"too many trainable: {trainable}/{total}"
    # head trainable
    assert enc.head.weight.requires_grad


def test_conv1_4channels_init():
    enc = ResNet18RGBDEncoder(out_dim=52)
    # Conv1 must accept 4 channels
    assert enc.backbone.conv1.in_channels == 4
    # D-channel weights initialized to zero
    w = enc.backbone.conv1.weight.data
    assert torch.allclose(w[:, 3, :, :], torch.zeros_like(w[:, 3, :, :]))
    # RGB weights non-zero (from pretrained)
    assert not torch.allclose(w[:, :3, :, :], torch.zeros_like(w[:, :3, :, :]))
```

### Step 1.2: Run test — should fail

```bash
.venv/bin/python -m pytest tests/test_visual_encoder.py -v 2>&1 | tail -5
```
Expected: ModuleNotFoundError.

### Step 1.3: Implement encoder

```python
# src/planning/visual_encoder.py
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18RGBDEncoder(nn.Module):
    """ResNet-18 pretrained on ImageNet, patched to accept 4-channel RGB-D.

    Conv1 RGB weights are copied from the ImageNet checkpoint and the depth
    channel is initialized to zero. All backbone params are frozen; only the
    final Linear head is trainable.
    """

    def __init__(self, out_dim: int = 52):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT
        backbone = tvm.resnet18(weights=weights)

        # Patch conv1: 3-channel -> 4-channel
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=4, out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            new_conv1.weight[:, 3, :, :] = 0.0
        backbone.conv1 = new_conv1

        # Replace fc with identity → we want the 512d features
        backbone.fc = nn.Identity()

        # Freeze the backbone entirely (incl. patched conv1)
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.head = nn.Linear(512, out_dim)  # trainable

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        """rgbd: (B, 4, 224, 224) → (B, out_dim)."""
        feats = self.backbone(rgbd)  # (B, 512)
        return self.head(feats)
```

### Step 1.4: Run tests — should pass

```bash
.venv/bin/python -m pytest tests/test_visual_encoder.py -v 2>&1 | tail -10
```
Expected: 3 passed.

### Step 1.5: Commit

```bash
git add src/planning/visual_encoder.py tests/test_visual_encoder.py
git commit -m "feat(planning): ResNet18RGBDEncoder — backbone frozen + 4-canales conv1

Conv1 RGB inicializado desde ImageNet, canal D en cero. Solo el head
Linear(512, 52) es trainable. 3 tests unitarios.

Refs: spec Iter 3, plan Task 1."
```

---

## Task 2: Modificar collector — agregar `rgbd_obs` por trayectoria

**Files:**
- Modify: `experiments/collect_diffusion_dataset.py`

### Step 2.1: Update header constants

Reemplazar:
```python
DATASET_VERSION = "v2"
```
por:
```python
DATASET_VERSION = "v3"
```

(Mantener N_HEURISTIC=1500 y N_EXECUTED=200.)

### Step 2.2: Add RGB-D capture helper

Antes de `phase_heuristic`, agregar:

```python
def _capture_rgbd_for_pose(bridge, pose: np.ndarray) -> np.ndarray:
    """Coloca el cubo en la pose, captura RGB-D, devuelve tensor (4, 224, 224) float32."""
    import torch.nn.functional as F
    sim = bridge.sim
    obj1 = sim.getObject("/object_1")
    sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))
    for _ in range(3):  # settle
        bridge.step()

    rgb, depth = bridge.capture_rgbd()  # rgb (H,W,3) uint8, depth (H,W) float32
    # depth → mm uint16 está fuera; aquí trabajamos en float32 normalizado.
    # Normalize RGB to [0,1] float32
    rgb_f = rgb.astype(np.float32) / 255.0  # (H, W, 3)
    # Depth: clip + normalize a [0,1] (asumiendo near=0.05, far=2.0 m del bridge default)
    depth_clip = np.clip(depth, 0.05, 2.0)
    depth_norm = (depth_clip - 0.05) / (2.0 - 0.05)  # [0,1]
    depth_norm = depth_norm[..., None]  # (H, W, 1)
    rgbd = np.concatenate([rgb_f, depth_norm], axis=-1)  # (H, W, 4)
    # To (4, H, W) and resize to (4, 224, 224)
    rgbd_t = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
    rgbd_resized = F.interpolate(rgbd_t, size=(224, 224), mode="bilinear", align_corners=False)
    return rgbd_resized.squeeze(0).numpy().astype(np.float32)  # (4, 224, 224)
```

(Importar `import torch` y `import torch.nn.functional` si falta — ya está `torch`.)

### Step 2.3: Rewrite `phase_heuristic` con bridge persistente + rgbd capture

Reemplazar la función entera por:

```python
def phase_heuristic(n: int = N_HEURISTIC) -> None:
    """Genera n trayectorias heurísticas con RGB-D capturado del sim."""
    from src.simulation.coppeliasim_bridge import CoppeliaSimBridge

    logger.info(f"Phase A.1 (v3): generando {n} trayectorias heurísticas + RGB-D")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device="cpu",
    )

    rng = np.random.default_rng(SEED)
    poses = np.zeros((n, 16), dtype=np.float32)  # pose 4x4 flattened (16)
    rgbds = np.zeros((n, 4, 224, 224), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)

    SCENE = REPO / "data" / "scenes" / "bin_base.ttt"
    with CoppeliaSimBridge() as bridge:
        bridge.load_scene(SCENE)
        bridge.set_stepping(True)
        bridge.start_simulation()
        try:
            for i in range(n):
                pose = sample_pose(rng)
                rgbd = _capture_rgbd_for_pose(bridge, pose)
                traj = planner.plan_grasp_heuristic(pose, approach_distance=0.15, lift_height=0.10)
                trajs[i] = traj[0]
                rgbds[i] = rgbd
                poses[i] = pose.flatten().astype(np.float32)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{n}")
        finally:
            bridge.stop_simulation()

    out = DATASET_DIR / "heuristic.pt"
    torch.save({
        "poses": torch.from_numpy(poses),
        "rgbds": torch.from_numpy(rgbds),
        "trajs": torch.from_numpy(trajs),
        "source": "heuristic",
        "seed": SEED,
    }, out)
    logger.info(f"escrito: {out} ({n} trayectorias, ~{out.stat().st_size / 1e6:.0f} MB)")
```

### Step 2.4: Update `phase_executed` para capturar rgbd al inicio

Buscar el bloque `with CoppeliaSimBridge() as bridge:` y, después de `bridge.start_simulation()` pero **antes de** `set_gripper(bridge, True)`, agregar:

```python
                # Capture RGB-D al inicio (open-loop conditioning)
                rgbd_obs = _capture_rgbd_for_pose(bridge, pose)
```

Reemplazar la línea `conds[i] = planner_aux.encode_observation(pose).cpu().numpy()[0]` por:
```python
        # Pose flat (16) + rgbd serán arrays paralelos
```

Cambiar las declaraciones:
```python
    conds = np.zeros((n, 64), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)
```
por:
```python
    poses = np.zeros((n, 16), dtype=np.float32)
    rgbds = np.zeros((n, 4, 224, 224), dtype=np.float32)
    trajs = np.zeros((n, 16, 7), dtype=np.float32)
```

Después de `rgbd_obs = _capture_rgbd_for_pose(bridge, pose)`:
```python
                poses[i] = pose.flatten().astype(np.float32)
                rgbds[i] = rgbd_obs
```
(quitar la línea de `conds[i] = ...`)

Y al final, el `torch.save` debe ser:
```python
    out = DATASET_DIR / "executed.pt"
    torch.save({
        "poses": torch.from_numpy(poses[:n_valid]),
        "rgbds": torch.from_numpy(rgbds[:n_valid]),
        "trajs": torch.from_numpy(trajs[:n_valid]),
        "source": "executed",
        "seed": SEED + 1,
    }, out)
```

### Step 2.5: Update `phase_split` para los nuevos campos

Reemplazar la función entera:

```python
def phase_split() -> None:
    """Combina heuristic.pt + executed.pt y hace 90/10 split."""
    logger.info("Phase A.3 (v3): combinando + split 90/10")
    heur_path = DATASET_DIR / "heuristic.pt"
    exec_path = DATASET_DIR / "executed.pt"
    if not heur_path.exists() or not exec_path.exists():
        raise FileNotFoundError("Faltan datasets parciales.")

    h = torch.load(heur_path, weights_only=True)
    e = torch.load(exec_path, weights_only=True)
    all_poses = torch.cat([h["poses"], e["poses"]], dim=0)
    all_rgbds = torch.cat([h["rgbds"], e["rgbds"]], dim=0)
    all_trajs = torch.cat([h["trajs"], e["trajs"]], dim=0)
    n = len(all_poses)

    rng = np.random.default_rng(SEED + 2)
    indices = rng.permutation(n)
    train_n = int(0.9 * n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    torch.save({
        "poses": all_poses[train_idx],
        "rgbds": all_rgbds[train_idx],
        "trajs": all_trajs[train_idx],
        "split": "train",
    }, DATASET_DIR / "train.pt")
    torch.save({
        "poses": all_poses[val_idx],
        "rgbds": all_rgbds[val_idx],
        "trajs": all_trajs[val_idx],
        "split": "val",
    }, DATASET_DIR / "val.pt")
    logger.info(f"escrito: train.pt ({train_n}), val.pt ({n - train_n})")
```

### Step 2.6: Smoke test — 1 pose, verify RGB-D capture

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from experiments.collect_diffusion_dataset import _capture_rgbd_for_pose, sample_pose
from pathlib import Path
SCENE = Path('data/scenes/bin_base.ttt')
rng = np.random.default_rng(0)
pose = sample_pose(rng)
with CoppeliaSimBridge() as br:
    br.load_scene(SCENE)
    br.set_stepping(True); br.start_simulation()
    rgbd = _capture_rgbd_for_pose(br, pose)
    br.stop_simulation()
print('rgbd shape:', rgbd.shape, 'min:', rgbd.min(), 'max:', rgbd.max(), 'mean:', rgbd.mean())
assert rgbd.shape == (4, 224, 224)
assert 0 <= rgbd.min() and rgbd.max() <= 1.0
print('SMOKE TEST OK')
"
```
Expected: SMOKE TEST OK con valores razonables (rgb means ~0.3-0.7, depth no all-zero ni all-one).

### Step 2.7: Add v3 to .gitignore

Verificar (y agregar si falta) en `.gitignore`:
```
data/datasets/sim_pick_v3/
data/models/diffusion_policy_sim_v3.pth
data/models/diffusion_policy_sim_v3.pth.summary.json
```

### Step 2.8: Commit collector changes

```bash
git add experiments/collect_diffusion_dataset.py .gitignore
git commit -m "feat(planning): collector v3 — capturar RGB-D por trayectoria

DATASET_VERSION=v3. Cada trayectoria guarda pose 4x4 + rgbd (4,224,224)
+ waypoints. Bridge persistente para heuristic (1500 capturas seguidas).
Mantiene 90/10 split. Smoke test del capture OK.

Refs: spec Iter 3, plan Task 2."
```

---

## Task 3: Coleccionar dataset v3

### Step 3.1: Phase A.1 — heuristic (1500 con RGB-D, ~15 min)

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase heuristic 2>&1 | tail -10
```

Expected:
- `data/datasets/sim_pick_v3/heuristic.pt` (~280 MB float32).
- 1500 progresando en pasos de 50.

### Step 3.2: Phase A.2 — executed (200 picks, ~2.5-3 h)

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase executed > /tmp/collect_v3_exec.log 2>&1 &
```
**No matar.** Tarda ~2.5–3 h. Monitor con `tail -f /tmp/collect_v3_exec.log`.

Expected al final:
- `data/datasets/sim_pick_v3/executed.pt` (~37 MB).
- `Phase A.2: 200/200 (skipped <10)`.

### Step 3.3: Phase A.3 — split

```bash
.venv/bin/python experiments/collect_diffusion_dataset.py --phase split 2>&1 | tail -5
```
Expected: `train.pt (1530), val.pt (170)`.

### Step 3.4: Verify dataset

```bash
.venv/bin/python -c "
import torch
d = torch.load('data/datasets/sim_pick_v3/train.pt', weights_only=True)
print('keys:', list(d.keys()))
print('poses:', d['poses'].shape)
print('rgbds:', d['rgbds'].shape, 'dtype:', d['rgbds'].dtype)
print('trajs:', d['trajs'].shape)
print('rgbd[0] stats: min=', d['rgbds'][0].min().item(), 'max=', d['rgbds'][0].max().item())
"
```
Expected: `poses (1530, 16)`, `rgbds (1530, 4, 224, 224)`, `trajs (1530, 16, 7)`.

(No commit aquí — dataset gitignored.)

---

## Task 4: Precompute visual embeddings

**Files:**
- Create: `experiments/precompute_visual_cond.py`

### Step 4.1: Create script

```python
#!/usr/bin/env python3
"""Precompute visual embeddings (ResNet-18) sobre el dataset v3.

Lee {train,val}.pt, corre el encoder sobre el campo `rgbds`, agrega
campo `visual_emb` (N, 52) y guarda como {train,val}_with_emb.pt.

Uso:
    python experiments/precompute_visual_cond.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.visual_encoder import ResNet18RGBDEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("precompute")

DATASET_DIR = REPO / "data" / "datasets" / "sim_pick_v3"
EMBED_DIM = 52
BATCH_SIZE = 32


def precompute(split: str, device: str) -> None:
    in_path = DATASET_DIR / f"{split}.pt"
    out_path = DATASET_DIR / f"{split}_with_emb.pt"
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    d = torch.load(in_path, weights_only=True)
    rgbds = d["rgbds"]  # (N, 4, 224, 224)
    n = len(rgbds)
    logger.info(f"split={split} n={n} dev={device}")

    encoder = ResNet18RGBDEncoder(out_dim=EMBED_DIM).to(device).eval()
    embs = torch.zeros(n, EMBED_DIM, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            batch = rgbds[start : start + BATCH_SIZE].to(device)
            embs[start : start + BATCH_SIZE] = encoder(batch).cpu()
            if (start // BATCH_SIZE) % 5 == 0:
                logger.info(f"  {start}/{n}")

    d["visual_emb"] = embs
    torch.save(d, out_path)
    logger.info(f"escrito: {out_path}")


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for split in ("train", "val"):
        precompute(split, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 4.2: Run

```bash
.venv/bin/python experiments/precompute_visual_cond.py 2>&1 | tail -10
```
Expected (~30 s total): logs `0/1530 ... 1530/1530` y dos archivos `*_with_emb.pt`.

### Step 4.3: Verify

```bash
.venv/bin/python -c "
import torch
d = torch.load('data/datasets/sim_pick_v3/train_with_emb.pt', weights_only=True)
print('visual_emb:', d['visual_emb'].shape, d['visual_emb'].dtype)
print('mean:', d['visual_emb'].mean().item(), 'std:', d['visual_emb'].std().item())
"
```
Expected: `visual_emb (1530, 52)`, std > 0 (no constante).

### Step 4.4: Commit

```bash
git add experiments/precompute_visual_cond.py
git commit -m "feat(planning): precompute_visual_cond.py — embeddings RGB-D one-shot

Corre ResNet-18 frozen sobre dataset v3 rgbds, guarda visual_emb (52d)
en {train,val}_with_emb.pt. ~30 s en M1 MPS.

Refs: spec Iter 3, plan Task 4."
```

---

## Task 5: `encode_observation` + training script para usar `visual_emb`

**Files:**
- Modify: `src/planning/diffusion_policy.py`
- Modify: `experiments/train_diffusion_on_sim.py`

### Step 5.1: Update `encode_observation` para aceptar visual_emb

En `src/planning/diffusion_policy.py`, buscar:
```python
    def encode_observation(self, object_pose: np.ndarray) -> torch.Tensor:
```
Reemplazar la firma + cuerpo por:

```python
    def encode_observation(
        self,
        object_pose: np.ndarray,
        visual_emb: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Encode pose + (optional) visual embedding into 64d cond vector.

        Layout:
            cond[:52] = visual_emb (or zeros if None)
            cond[52:64] = pose[:3, :].flatten()[:12]
        """
        cond = np.zeros(64, dtype=np.float32)
        if visual_emb is not None:
            v = np.asarray(visual_emb, dtype=np.float32).reshape(-1)
            cond[: min(52, len(v))] = v[:52]
        pose_flat = object_pose[:3, :].flatten().astype(np.float32)  # 12
        cond[52:64] = pose_flat[:12]
        return torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(self.device)
```

(Mantiene backwards-compat: si `visual_emb` es None, los primeros 52 dims quedan en cero — comportamiento similar a v1/v2, sólo cambia el layout.)

### Step 5.2: Actualizar training para usar `*_with_emb.pt`

En `experiments/train_diffusion_on_sim.py`, ubicar el carga del dataset (típicamente lee `train.pt` o `val.pt`). Modificar para:

- Si `--dataset-dir` apunta a v3, leer `train_with_emb.pt` y `val_with_emb.pt`.
- El cond pasado al modelo se construye batch a batch:
  ```python
  # B = batch size
  visual_emb_b = batch["visual_emb"]   # (B, 52)
  poses_flat_b = batch["poses"]         # (B, 16) — flat 4x4
  pose12 = poses_flat_b[:, :12]         # (B, 12)
  cond = torch.cat([visual_emb_b, pose12], dim=1)  # (B, 64)
  ```

Cambios concretos: localizar donde el dataset retorna `conds` y reemplazar con la construcción anterior. Usar fallback al esquema antiguo si el archivo no tiene `visual_emb` (para retro-compat con v1/v2).

Pseudo-código del fix:
```python
# en el dataset loader
data = torch.load(path, weights_only=True)
if "visual_emb" in data:
    poses_flat = data["poses"]  # (N, 16)
    pose12 = poses_flat[:, :12]
    conds = torch.cat([data["visual_emb"], pose12], dim=1)  # (N, 64)
else:
    conds = data["conds"]  # v1/v2 compat
```

(Leer el archivo actual antes de editar — la implementación específica depende del shape actual del loader.)

### Step 5.3: Verify training script reads v3

```bash
.venv/bin/python experiments/train_diffusion_on_sim.py \
    --dataset-dir data/datasets/sim_pick_v3 \
    --hidden-dim 256 --from-scratch --epochs 2 2>&1 | tail -10
```
Expected: epoch 0/2 loss visible, sin errores de shape. (No commitear el ckpt.)

### Step 5.4: Commit

```bash
git add src/planning/diffusion_policy.py experiments/train_diffusion_on_sim.py
git commit -m "feat(planning): encode_observation acepta visual_emb + train loader v3

cond[:52] = visual_emb (ResNet-18 RGB-D), cond[52:64] = pose[:3,:].flatten()[:12].
Backwards-compat con v1/v2 cuando visual_emb no está presente.

Refs: spec Iter 3, plan Task 5."
```

---

## Task 6: Train v3

### Step 6.1: Train run

```bash
.venv/bin/python experiments/train_diffusion_on_sim.py \
    --dataset-dir data/datasets/sim_pick_v3 \
    --hidden-dim 256 --from-scratch --epochs 150 \
    --output data/models/diffusion_policy_sim_v3.pth \
    2>&1 | tail -30
```
Expected: 150 epochs, ~3 min, `final_val_loss < 0.06`, ckpt guardado.

### Step 6.2: Verify ckpt

```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('data/models/diffusion_policy_sim_v3.pth', map_location='cpu', weights_only=True)
print('config:', ckpt['config'])
print('final_train_loss:', ckpt.get('final_train_loss'))
print('final_val_loss:', ckpt.get('final_val_loss'))
"
```

(No commit del ckpt — gitignored.)

---

## Task 7: `pick_with_dp` + eval Iter 3 — encoder en vivo

**Files:**
- Modify: `experiments/run_pick_with_diffusion.py`
- Create: `experiments/eval_diffusion_iter3_sim.py`

### Step 7.1: Update `pick_with_dp` para capturar RGB-D + correr encoder en vivo

En `experiments/run_pick_with_diffusion.py`, modificar la signatura de `pick_with_dp`:

```python
def pick_with_dp(
    planner,
    pose: np.ndarray,
    bridge,
    frames_dir=None,
    n_substeps: int = 8,
    steps_per_substep: int = 2,
    visual_encoder=None,
):
```

Después de:
```python
    sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))
```
agregar:

```python
    # Capture RGB-D y correr encoder en vivo si está disponible
    visual_emb = None
    if visual_encoder is not None:
        from experiments.collect_diffusion_dataset import _capture_rgbd_for_pose
        rgbd = _capture_rgbd_for_pose(bridge, pose)  # (4,224,224) float32
        rgbd_t = torch.from_numpy(rgbd).unsqueeze(0).to(planner.device)  # (1,4,224,224)
        with torch.no_grad():
            visual_emb = visual_encoder(rgbd_t).cpu().numpy()[0]  # (52,)
```

Donde el código actual hace `planner.plan_grasp(pose, n_samples=1)`, ese método llama internamente a `encode_observation(pose)`. Para pasar visual_emb, necesitamos un override.

Cambiar a:
```python
    cond = planner.encode_observation(pose, visual_emb=visual_emb)
    # plan_grasp con cond explícito → expand a plan_grasp directly:
    traj = planner.plan_grasp(pose, n_samples=1, cond=cond) if visual_emb is not None \
           else planner.plan_grasp(pose, n_samples=1)
```

**Importante**: hay que verificar/agregar el kwarg `cond` a `DiffusionGraspPlanner.plan_grasp`. Si no existe, agregarlo: si `cond is None`, hacer `cond = self.encode_observation(object_pose)`; en cualquier otro caso usar el cond pasado. Cambio mínimo en `src/planning/diffusion_policy.py`.

### Step 7.2: Crear `eval_diffusion_iter3_sim.py`

```python
#!/usr/bin/env python3
"""Eval Iter 3 — 50 picks en sim con conditioning visual ResNet-18."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from experiments.run_pick_with_diffusion import pick_with_dp
from experiments.eval_diffusion_iter2_sim import sample_pose_eval, EVAL_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_iter3_sim")

REPO_OUT = REPO / "experiments" / "results" / "pick_with_diffusion"
SCENE = REPO / "data" / "scenes" / "bin_base.ttt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy_path = REPO / "data" / "models" / "diffusion_policy_sim_v3.pth"
    ckpt = torch.load(policy_path, map_location=device, weights_only=True)
    hidden_dim = ckpt.get("config", {}).get("hidden_dim", 256)
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device,
        hidden_dim=hidden_dim,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    planner.model.eval()
    encoder = ResNet18RGBDEncoder(out_dim=52).to(device).eval()
    logger.info(f"policy: v3 (hidden_dim={hidden_dim}) + ResNet-18")

    rng = np.random.default_rng(EVAL_SEED)
    results = []
    skipped = 0
    for i in range(args.n):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                r = pick_with_dp(planner, pose, bridge, frames_dir=None, visual_encoder=encoder)
            results.append({
                "i": i,
                "target_pose_t": r["target_pose_t"],
                "grasp_proximity_m": r["grasp_proximity_m"],
                "deposit_error_m": r["deposit_error_m"],
                "ik_converged": r["ik_converged"],
                "grasp_plausible": r["grasp_plausible"],
                "deposit_plausible": r["deposit_plausible"],
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{args.n} done (skipped {skipped})")
        except Exception as e:
            logger.warning(f"  [{i}] falló: {e}")
            skipped += 1

    n_valid = len(results)
    if n_valid == 0:
        logger.error("0 picks válidos")
        return 1

    gp = sum(r["grasp_plausible"] for r in results)
    dp = sum(r["deposit_plausible"] for r in results)
    ik = sum(r["ik_converged"] for r in results)

    summary = {
        "n_requested": args.n, "n_valid": n_valid, "n_skipped": skipped,
        "policy": str(policy_path.relative_to(REPO)), "seed": EVAL_SEED,
        "dp_grasp_plausible_pct_sim": 100.0 * gp / n_valid,
        "dp_deposit_plausible_pct_sim": 100.0 * dp / n_valid,
        "dp_ik_converged_pct": 100.0 * ik / n_valid,
        "mean_grasp_proximity_m": float(np.mean([r["grasp_proximity_m"] for r in results])),
        "mean_deposit_error_m": float(np.mean([r["deposit_error_m"] for r in results])),
        "thresholds_passed": {
            "dp_grasp_plausible_pct_sim >= 55": 100.0 * gp / n_valid >= 55,
            "dp_ik_converged_pct >= 90": 100.0 * ik / n_valid >= 90,
        },
        "per_pick": results,
    }
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    out = REPO_OUT / "eval_v3_sim.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== RESUMEN EVAL ITER 3 EN SIM ===")
    for k, v in summary.items():
        if k != "per_pick":
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 7.3: Smoke test n=2

```bash
.venv/bin/python experiments/eval_diffusion_iter3_sim.py --n 2 2>&1 | tail -20
```
Expected: 2/2 valid, números no-todo-cero, sin crashes.

### Step 7.4: Commit

```bash
git add experiments/run_pick_with_diffusion.py experiments/eval_diffusion_iter3_sim.py src/planning/diffusion_policy.py
git commit -m "feat(planning): pick_with_dp acepta visual_encoder + eval Iter 3

pick_with_dp captura RGB-D del bridge y corre ResNet-18 en vivo,
construye cond via encode_observation(pose, visual_emb).
eval_diffusion_iter3_sim.py corre 50 picks con seed=2026.

Refs: spec Iter 3, plan Task 7."
```

---

## Task 8: Run eval n=50 + documentar

### Step 8.1: Eval completo

```bash
.venv/bin/python experiments/eval_diffusion_iter3_sim.py --n 50 > /tmp/eval_v3_sim.log 2>&1 &
```
Expected ~10–12 min. Cuando termine, leer `eval_v3_sim.json`.

### Step 8.2: Update `docs/INTEGRATION_PIPELINE.md`

Agregar sección `## Iter 3 (cerrado 2026-05-29): conditioning visual con ResNet-18` después de la sección Iter 2. Estructura paralela a Iter 2: cambios, tabla de métricas, lectura honesta, datos generados, diagnóstico, conclusión.

Reemplazar placeholders en la tabla con los valores reales de `eval_v3_sim.json`:
- `dp_grasp_plausible_pct_sim`: usar valor real
- `dp_ik_converged_pct`: usar valor real
- `mean_grasp_proximity_m`: usar valor real
- `final_val_loss`: leer de `data/models/diffusion_policy_sim_v3.pth.summary.json` o del log

### Step 8.3: Commit

```bash
git add docs/INTEGRATION_PIPELINE.md experiments/results/pick_with_diffusion/eval_v3_sim.json
git commit -m "docs(integration): cerrar Iter 3 con resultados del conditioning visual

Refs: spec Iter 3, plan Task 8."
```

---

## Final verification

### F.1: Run all tests

```bash
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -5
```
Expected: todos pasan (incluyendo los 3 nuevos de `test_visual_encoder.py`).

### F.2: Push branch + PR

```bash
git push -u origin feat/diffusion-retrain-iter3
gh pr create --base main --head feat/diffusion-retrain-iter3 \
    --title "Iter 3: conditioning visual RGB-D con ResNet-18" \
    --body "$(cat docs/INTEGRATION_PIPELINE.md | sed -n '/## Iter 3/,/## /p' | head -n -1)"
```

### F.3: Confirmación

Mostrar URL del PR.
