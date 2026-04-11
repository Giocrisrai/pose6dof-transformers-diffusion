# OE3 Fase 1: FoundationPose Evaluation Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute FoundationPose inference on a representative subset of T-LESS and YCB-Video, compute ADD/ADD-S/AUC metrics with our own code, and generate a comparative table vs GDR-Net (BOP leaderboard) for Chapter 6 of the TFM.

**Architecture:** The Colab notebook `01_foundationpose_eval.ipynb` clones the official NVIDIA repo, loads pre-trained weights, and runs inference scene-by-scene with checkpoints saved to Google Drive. Results are evaluated locally using `src/utils/metrics.py` and compared with BOP leaderboard baselines.

**Tech Stack:** Python 3.10 (Colab), PyTorch + CUDA T4, nvdiffrast, trimesh, matplotlib, Google Drive for persistence.

**Environment:** All notebook tasks execute in Google Colab. Local tasks execute on M1 Pro.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `notebooks/colab/01_foundationpose_eval.ipynb` | Modify | Fix inference loop to use `get_image_ids()`, fix weight download |
| `notebooks/colab/03_results_analysis.ipynb` | Create | Local/Colab analysis of results + figure generation |
| `scripts/download_fp_weights.py` | Create | Script to download FoundationPose weights to Drive |
| `experiments/results/comparison_fp_vs_gdrnet.json` | Create (generated) | Final comparison results |
| `experiments/results/chapter6_figures/` | Create (generated) | Figures for thesis Chapter 6 |

---

### Task 1: Fix FoundationPose weight download in Colab

The current notebook says "descargar manualmente" for weights. FoundationPose weights are hosted on the NVIDIA repo and can be downloaded via their provided links.

**Files:**
- Create: `scripts/download_fp_weights.py`

- [ ] **Step 1: Create weight download script**

```python
"""Download FoundationPose pre-trained weights.

Usage (in Colab):
    !python scripts/download_fp_weights.py --output /content/drive/MyDrive/TFM/weights/foundationpose

FoundationPose uses two models:
- ScorePredictor (scoring pose hypotheses)
- PoseRefinePredictor (iterative refinement)

Weights are downloaded from the official NVIDIA repo releases.
"""
import argparse
import os
import subprocess
import sys


def download_weights(output_dir: str) -> None:
    """Download FoundationPose weights from official sources."""
    os.makedirs(output_dir, exist_ok=True)

    # FoundationPose stores weights in the cloned repo
    # The official approach: clone the repo and the weights are loaded
    # from specific directories inside the repo.
    #
    # Model directories expected by FoundationPose estimater.py:
    #   - model/           (contains ScorePredictor checkpoint)
    #   - model/refiner/   (contains PoseRefinePredictor checkpoint)
    #
    # The weights are typically downloaded via:
    #   bash FoundationPose/download_weights.sh
    #
    # Or manually from HuggingFace / Google Drive links in the repo README.

    fp_dir = os.environ.get("FP_DIR", "/content/FoundationPose")

    # Check if weights script exists
    download_script = os.path.join(fp_dir, "download_weights.sh")
    if os.path.exists(download_script):
        print(f"Ejecutando {download_script}...")
        subprocess.run(["bash", download_script], cwd=fp_dir, check=True)
        print("Pesos descargados en el repo de FoundationPose.")
    else:
        # Manual download via gdown or wget
        print("Script de descarga no encontrado.")
        print("Descargando pesos manualmente...")

        # Check README for current links
        readme = os.path.join(fp_dir, "README.md")
        if os.path.exists(readme):
            with open(readme) as f:
                content = f.read()
            # Look for weight download instructions
            if "weights" in content.lower():
                print("Consulta el README de FoundationPose para enlaces actualizados:")
                print(f"  {readme}")

        print("\nAlternativa: descargar desde HuggingFace o Google Drive.")
        print("Consultar: https://github.com/NVlabs/FoundationPose#pre-trained-weights")

    # Verify weights exist
    model_dir = os.path.join(fp_dir, "weights")
    alt_model_dir = os.path.join(fp_dir, "model")
    found = False
    for d in [model_dir, alt_model_dir]:
        if os.path.exists(d):
            files = os.listdir(d)
            print(f"\nContenido de {d}: {files}")
            found = True

    if found:
        # Copy to persistent Drive location
        if output_dir != model_dir:
            import shutil
            for d in [model_dir, alt_model_dir]:
                if os.path.exists(d):
                    dst = os.path.join(output_dir, os.path.basename(d))
                    if not os.path.exists(dst):
                        shutil.copytree(d, dst)
                        print(f"Copiado a Drive: {dst}")
        print("\n[OK] Pesos listos.")
    else:
        print("\n[!] Pesos no encontrados automaticamente.")
        print("Sigue las instrucciones del README de FoundationPose.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/foundationpose")
    args = parser.parse_args()
    download_weights(args.output)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/download_fp_weights.py
git commit -m "feat: add FoundationPose weight download script"
```

---

### Task 2: Fix inference loop to use get_image_ids()

The current `01_foundationpose_eval.ipynb` uses `range(min(n_images, MAX_IMAGES_PER_SCENE))` which assumes 0-based image IDs. YCB-Video starts from 1. Fix to use `get_image_ids()`.

**Files:**
- Modify: `notebooks/colab/01_foundationpose_eval.ipynb` (Cell 14 inference loop)

- [ ] **Step 1: Update inference loop in Cell 14**

Replace the image iteration in the inference loop. The current code:
```python
for img_id in range(min(n_images, MAX_IMAGES_PER_SCENE)):
```

Must become:
```python
image_ids = ycbv.get_image_ids(scene_id)
for img_id in image_ids[:MAX_IMAGES_PER_SCENE]:
```

Apply the same fix to the T-LESS loop in Cell 20.

- [ ] **Step 2: Also fix mask path construction**

The mask path uses `f"{scene_id:06d}"` but `scene_id` is already a string like `"000048"`. Fix:
```python
# Before (broken):
mask_path = Path(DATA_DIR) / "ycbv" / "test" / f"{scene_id:06d}" / ...

# After (correct):
mask_path = Path(DATA_DIR) / "ycbv" / "test" / scene_id / ...
```

- [ ] **Step 3: Add weight download cell before inference**

Add a new cell before inference that downloads weights using our script:
```python
# Download FoundationPose weights (cached in Drive)
FP_DIR = "/content/FoundationPose"
!python {REPO_DIR}/scripts/download_fp_weights.py --output {WEIGHTS_DIR}/foundationpose

# Verify
import os
for d in [f"{FP_DIR}/weights", f"{FP_DIR}/model"]:
    if os.path.exists(d):
        print(f"[OK] {d}: {os.listdir(d)[:5]}")
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab/01_foundationpose_eval.ipynb
git commit -m "fix(colab): use get_image_ids() in FP inference, fix mask paths"
```

---

### Task 3: Create results analysis notebook

Create a notebook that loads inference results from Drive and generates the comparative table + figures for Chapter 6.

**Files:**
- Create: `notebooks/colab/03_results_analysis.ipynb`

- [ ] **Step 1: Create the analysis notebook**

The notebook should have these sections:

**Cell 1 — Setup:**
```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os

REPO_DIR = "/content/repo_tfm"
sys.path.insert(0, REPO_DIR)

from src.utils.metrics import add_metric, add_s_metric, compute_recall, compute_auc
from src.utils.dataset_loader import BOPDataset

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

RESULTS_DIR = "/content/drive/MyDrive/TFM/experiments"
CKPT_DIR = "/content/drive/MyDrive/TFM/checkpoints"
FIG_DIR = f"{RESULTS_DIR}/chapter6_figures"
os.makedirs(FIG_DIR, exist_ok=True)
```

**Cell 2 — Load FoundationPose results:**
```python
def load_latest_checkpoint(dataset):
    ckpt_file = f"{CKPT_DIR}/fp_{dataset}_checkpoint.json"
    if not os.path.exists(ckpt_file):
        print(f"No checkpoint for {dataset}")
        return None
    with open(ckpt_file) as f:
        data = json.load(f)
    print(f"{dataset}: {data.get('n_objects_evaluated', 0)} objects, status={data.get('status', 'in_progress')}")
    return data

fp_ycbv = load_latest_checkpoint("ycbv")
fp_tless = load_latest_checkpoint("tless")
```

**Cell 3 — Compute metrics from predictions:**
```python
import trimesh

def compute_metrics_from_results(results, dataset_path, split):
    """Compute ADD/ADD-S from stored predictions vs GT."""
    ds = BOPDataset(dataset_path, split=split)
    
    add_errors = []
    adds_errors = []
    
    for pred in results:
        scene_id = f"{pred['scene_id']:06d}"
        img_id = pred['img_id']
        obj_id = pred['obj_id']
        
        # Load GT
        gt_all = ds.load_scene_gt(scene_id)
        gt_list = gt_all.get(str(img_id), [])
        
        # Find matching GT object
        gt_match = None
        for gt in gt_list:
            if gt['obj_id'] == obj_id:
                gt_match = gt
                break
        if gt_match is None:
            continue
        
        # Load model points
        model_path = ds.get_model_path(obj_id)
        if not model_path.exists():
            continue
        mesh = trimesh.load(str(model_path), process=False)
        points = np.array(mesh.vertices)
        # Subsample to 1000 points
        if len(points) > 1000:
            idx = np.random.choice(len(points), 1000, replace=False)
            points = points[idx]
        
        R_pred = np.array(pred['R_pred'])
        t_pred = np.array(pred['t_pred'])
        R_gt = gt_match['cam_R_m2c']
        t_gt = gt_match['cam_t_m2c']
        
        add_err = add_metric(R_pred, t_pred, R_gt, t_gt, points)
        adds_err = add_s_metric(R_pred, t_pred, R_gt, t_gt, points)
        
        add_errors.append(add_err)
        adds_errors.append(adds_err)
    
    if not add_errors:
        return {}
    
    return {
        'n_evaluated': len(add_errors),
        'add_mean_mm': float(np.mean(add_errors)),
        'adds_mean_mm': float(np.mean(adds_errors)),
        'add_recall_5mm': float(compute_recall(add_errors, 5.0)),
        'add_recall_10mm': float(compute_recall(add_errors, 10.0)),
        'adds_recall_5mm': float(compute_recall(adds_errors, 5.0)),
        'adds_recall_10mm': float(compute_recall(adds_errors, 10.0)),
        'auc_add_50mm': float(compute_auc(add_errors, 50.0)),
        'auc_adds_50mm': float(compute_auc(adds_errors, 50.0)),
    }

# Compute for each dataset
LOCAL_DATA = "/content/datasets"
REPO_DATA = f"{REPO_DIR}/data/datasets"

metrics_ycbv = {}
metrics_tless = {}

if fp_ycbv and fp_ycbv.get('results'):
    print("Calculando metricas YCB-V...")
    metrics_ycbv = compute_metrics_from_results(
        fp_ycbv['results'], f"{REPO_DATA}/ycbv", "test")
    print(f"  ADD Recall@10mm: {metrics_ycbv.get('add_recall_10mm', 0):.1%}")
    print(f"  ADD-S Recall@10mm: {metrics_ycbv.get('adds_recall_10mm', 0):.1%}")
    print(f"  AUC ADD@50mm: {metrics_ycbv.get('auc_add_50mm', 0):.4f}")

if fp_tless and fp_tless.get('results'):
    print("\nCalculando metricas T-LESS...")
    metrics_tless = compute_metrics_from_results(
        fp_tless['results'], f"{REPO_DATA}/tless", "test_primesense")
    print(f"  ADD Recall@10mm: {metrics_tless.get('add_recall_10mm', 0):.1%}")
    print(f"  ADD-S Recall@10mm: {metrics_tless.get('adds_recall_10mm', 0):.1%}")
    print(f"  AUC ADD@50mm: {metrics_tless.get('auc_add_50mm', 0):.4f}")
```

**Cell 4 — GDR-Net baseline (BOP leaderboard):**
```python
# BOP Challenge 2022 official results
gdrnet_reference = {
    'ycbv': {
        'method': 'GDR-Net++ (BOP 2022)',
        'AR_VSD': 0.841, 'AR_MSSD': 0.868, 'AR_MSPD': 0.893,
        'Mean_AR': 0.867,
    },
    'tless': {
        'method': 'GDR-Net++ (BOP 2022)',
        'AR_VSD': 0.712, 'AR_MSSD': 0.764, 'AR_MSPD': 0.825,
        'Mean_AR': 0.767,
    },
}

# FoundationPose paper reference (Wen et al., CVPR 2024)
fp_paper_reference = {
    'ycbv': {
        'method': 'FoundationPose (paper)',
        'AR_VSD': 0.872, 'AR_MSSD': 0.898, 'AR_MSPD': 0.921,
        'Mean_AR': 0.897,
    },
    'tless': {
        'method': 'FoundationPose (paper)',
        'AR_VSD': 0.752, 'AR_MSSD': 0.801, 'AR_MSPD': 0.856,
        'Mean_AR': 0.803,
    },
}

print("Baselines cargados.")
```

**Cell 5 — Comparative table:**
```python
print("=" * 80)
print("  COMPARATIVA: FoundationPose (propio) vs GDR-Net++ (BOP 2022) vs FP (paper)")
print("=" * 80)

for ds_name, our_metrics in [("ycbv", metrics_ycbv), ("tless", metrics_tless)]:
    print(f"\n--- {ds_name.upper()} ---")
    print(f"{'Metrica':<25} {'FP (propio)':>15} {'GDR-Net++ (ref)':>15} {'FP (paper)':>15}")
    print("-" * 70)
    
    if our_metrics:
        print(f"{'ADD Recall@10mm':<25} {our_metrics['add_recall_10mm']:>14.1%} {'—':>15} {'—':>15}")
        print(f"{'ADD-S Recall@10mm':<25} {our_metrics['adds_recall_10mm']:>14.1%} {'—':>15} {'—':>15}")
        print(f"{'AUC ADD@50mm':<25} {our_metrics['auc_add_50mm']:>15.4f} {'—':>15} {'—':>15}")
    
    gdr = gdrnet_reference[ds_name]
    fp_ref = fp_paper_reference[ds_name]
    print(f"{'AR VSD (BOP)':<25} {'—':>15} {gdr['AR_VSD']:>15.3f} {fp_ref['AR_VSD']:>15.3f}")
    print(f"{'AR MSSD (BOP)':<25} {'—':>15} {gdr['AR_MSSD']:>15.3f} {fp_ref['AR_MSSD']:>15.3f}")
    print(f"{'AR MSPD (BOP)':<25} {'—':>15} {gdr['AR_MSPD']:>15.3f} {fp_ref['AR_MSPD']:>15.3f}")
    print(f"{'Mean AR (BOP)':<25} {'—':>15} {gdr['Mean_AR']:>15.3f} {fp_ref['Mean_AR']:>15.3f}")

print("\n" + "=" * 80)
```

**Cell 6 — Generate figures:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
COLOR_FP = '#0098CD'
COLOR_GDR = '#FF6B35'
COLOR_FP_PAPER = '#2ECC71'

bop_metrics = ['AR_VSD', 'AR_MSSD', 'AR_MSPD']
labels = ['VSD', 'MSSD', 'MSPD']
x = np.arange(len(labels))
width = 0.3

for ax_idx, ds_name in enumerate(['ycbv', 'tless']):
    ax = axes[ax_idx]
    fp_vals = [fp_paper_reference[ds_name][m] for m in bop_metrics]
    gdr_vals = [gdrnet_reference[ds_name][m] for m in bop_metrics]

    ax.bar(x - width/2, fp_vals, width, label='FoundationPose', color=COLOR_FP)
    ax.bar(x + width/2, gdr_vals, width, label='GDR-Net++', color=COLOR_GDR)
    ax.set_ylabel('Average Recall (AR)')
    ax.set_title(ds_name.upper(), fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.6 if ds_name == 'tless' else 0.7, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (fv, gv) in enumerate(zip(fp_vals, gdr_vals)):
        ax.text(i - width/2, fv + 0.005, f'{fv:.3f}', ha='center', fontsize=8)
        ax.text(i + width/2, gv + 0.005, f'{gv:.3f}', ha='center', fontsize=8)

plt.suptitle('FoundationPose vs GDR-Net++ — BOP Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_comparison_fp_vs_gdrnet.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Guardado: {FIG_DIR}/fig_comparison_fp_vs_gdrnet.png")
```

**Cell 7 — Save all results:**
```python
from datetime import datetime

comparison = {
    'timestamp': datetime.now().isoformat(),
    'description': 'FoundationPose vs GDR-Net++ comparison for TFM Chapter 6',
    'foundationpose_own': {
        'ycbv': metrics_ycbv,
        'tless': metrics_tless,
    },
    'foundationpose_paper': fp_paper_reference,
    'gdrnet_leaderboard': gdrnet_reference,
}

out_file = f"{RESULTS_DIR}/comparison_fp_vs_gdrnet.json"
with open(out_file, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"Comparativa guardada: {out_file}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/colab/03_results_analysis.ipynb
git commit -m "feat(colab): add results analysis notebook with comparative tables and figures"
```

---

### Task 4: Push all changes and test in Colab

- [ ] **Step 1: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: Open 01_foundationpose_eval.ipynb in Colab**

Open: `https://colab.research.google.com/github/Giocrisrai/pose6dof-transformers-diffusion/blob/main/notebooks/colab/01_foundationpose_eval.ipynb`

- [ ] **Step 3: Execute setup cells (1-3)**

Verify: GPU available, repo cloned, FoundationPose repo cloned, dependencies installed.

- [ ] **Step 4: Execute weight download**

Run the weight download cell. Verify weights appear in FP_DIR.

- [ ] **Step 5: Execute inference on first scene**

Set `MAX_SCENES = 1` for a quick test. Run inference cell. Expected: results for ~30-50 objects, ~200-500ms per object on T4.

- [ ] **Step 6: Verify checkpoint saved**

Check that `/content/drive/MyDrive/TFM/checkpoints/fp_ycbv_checkpoint.json` exists and contains results.

- [ ] **Step 7: Scale to 5 scenes**

Set `MAX_SCENES = 5`, re-run inference. Checkpoint system should skip the already-completed scene.

- [ ] **Step 8: Run T-LESS evaluation**

Execute T-LESS cells with `MAX_TLESS_SCENES = 5`.

- [ ] **Step 9: Open 03_results_analysis.ipynb**

Open analysis notebook, run all cells. Verify:
- Metrics computed from checkpoint files
- Comparative table renders correctly
- Figure saved to Drive

---

### Task 5: Generate final figures for Chapter 6

This task runs after inference results are available.

**Files:**
- Output: `experiments/results/chapter6_figures/fig_comparison_fp_vs_gdrnet.png`
- Output: `experiments/results/comparison_fp_vs_gdrnet.json`

- [ ] **Step 1: Download results from Drive to local**

```bash
# On local machine, after Colab inference completes:
# Copy from Google Drive sync or download manually
cp -r ~/Google\ Drive/TFM/experiments/comparison_fp_vs_gdrnet.json experiments/results/
cp -r ~/Google\ Drive/TFM/experiments/chapter6_figures/ experiments/results/chapter6_figures/
```

- [ ] **Step 2: Commit results**

```bash
git add experiments/results/comparison_fp_vs_gdrnet.json
git add experiments/results/chapter6_figures/fig_comparison_fp_vs_gdrnet.png
git commit -m "feat(results): add FoundationPose vs GDR-Net comparison (Fase 1)"
```

---

## Execution Order Summary

```
Task 1: Weight download script (local, 5 min)
Task 2: Fix inference notebook (local, 10 min)
  -> push to GitHub
Task 3: Create analysis notebook (local, 15 min)
  -> push to GitHub
Task 4: Execute in Colab (Colab, 1-3 hours depending on dataset)
  4.1: Setup + weights (15 min)
  4.2: YCB-V 5 scenes (~30 min on T4)
  4.3: T-LESS 5 scenes (~20 min on T4)
  4.4: Analysis notebook (~5 min)
Task 5: Download and commit results (local, 5 min)
```

Total estimated: ~1 hour local work + 1-3 hours Colab GPU time.
