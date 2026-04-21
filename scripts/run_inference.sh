#!/bin/bash
# =============================================================
# TFM Pose 6-DoF — Inference Runner
# =============================================================
# Ejecuta el pipeline de evaluación FoundationPose dentro del contenedor GPU.
#
# Uso:
#   run-inference smoke           # 1 escena x 3 imgs por dataset (~2 min)
#   run-inference dev             # 5 escenas x 50 imgs (~1-2 h)
#   run-inference full            # todas las escenas (horas)
#   run-inference test-imports    # solo valida que todo carga
#
# Variables de entorno esperadas (se montan por volumen en docker-compose):
#   REPO_DIR      (default /workspace/repo_tfm)   código del proyecto
#   DATASETS_DIR  (default /datasets)             BOP datasets (ycbv, tless)
#   WEIGHTS_DIR   (default /weights/foundationpose) pesos FP
#   RESULTS_DIR   (default /workspace/results)    output JSON + figuras
# =============================================================

set -euo pipefail

MODE="${1:-smoke}"

REPO_DIR="${REPO_DIR:-/workspace/repo_tfm}"
DATASETS_DIR="${DATASETS_DIR:-/datasets}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/weights/foundationpose}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
FP_DIR="${FP_DIR:-/opt/FoundationPose}"

echo "=============================================="
echo "  TFM Inference Runner"
echo "=============================================="
echo "  MODE:         $MODE"
echo "  REPO_DIR:     $REPO_DIR"
echo "  DATASETS_DIR: $DATASETS_DIR"
echo "  WEIGHTS_DIR:  $WEIGHTS_DIR"
echo "  RESULTS_DIR:  $RESULTS_DIR"
echo "  FP_DIR:       $FP_DIR"
echo "=============================================="

# Validar que existe todo lo que necesitamos
[[ -d "$REPO_DIR" ]] || { echo "[ERROR] REPO_DIR no existe: $REPO_DIR"; exit 1; }
[[ -d "$FP_DIR"   ]] || { echo "[ERROR] FP_DIR no existe: $FP_DIR"; exit 1; }

# Copiar pesos a la ubicación que FoundationPose espera
mkdir -p "$FP_DIR/weights"
if [[ -d "$WEIGHTS_DIR/2024-01-11-20-02-45" ]]; then
    cp -rn "$WEIGHTS_DIR/2024-01-11-20-02-45" "$FP_DIR/weights/" || true
    echo "[OK] Scorer pesos listos"
fi
if [[ -d "$WEIGHTS_DIR/2023-10-28-18-33-37" ]]; then
    cp -rn "$WEIGHTS_DIR/2023-10-28-18-33-37" "$FP_DIR/weights/" || true
    mkdir -p "$FP_DIR/bundlesdf/ckpts"
    cp -rn "$WEIGHTS_DIR/2023-10-28-18-33-37" "$FP_DIR/bundlesdf/ckpts/" || true
    echo "[OK] Refiner pesos listos"
fi

mkdir -p "$RESULTS_DIR"

case "$MODE" in
    test-imports)
        python3 <<'EOF'
import torch, pytorch3d, nvdiffrast, trimesh, kornia, transformations
import sys, os
sys.path.insert(0, os.environ.get('FP_DIR', '/opt/FoundationPose'))
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
print("[OK] Todos los imports funcionan")
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
EOF
        ;;

    smoke|dev|full)
        cd "$REPO_DIR"
        export PYTHONPATH="$REPO_DIR:$FP_DIR:${PYTHONPATH:-}"
        python3 - <<EOF
import os, sys, json, time
sys.path.insert(0, '$REPO_DIR')
sys.path.insert(0, '$FP_DIR')

from pathlib import Path
from src.utils.dataset_loader import BOPDataset

MODE = '$MODE'
PRESETS = {
    'smoke': dict(MAX_SCENES=1, MAX_IMAGES=3),
    'dev':   dict(MAX_SCENES=5, MAX_IMAGES=50),
    'full':  dict(MAX_SCENES=None, MAX_IMAGES=None),
}
p = PRESETS[MODE]
print(f"Running {MODE}: MAX_SCENES={p['MAX_SCENES']} MAX_IMAGES={p['MAX_IMAGES']}")
print("Este runner shell es un placeholder: la lógica completa está en")
print("notebooks/colab/01_foundationpose_eval.ipynb — ejecutarla adaptada")
print("para el contenedor (sin celdas Colab-específicas) requiere extraer")
print("las cells 13-26 a un script Python. Se hará en commit siguiente.")
EOF
        ;;

    *)
        echo "[ERROR] MODE desconocido: $MODE"
        echo "  Opciones: smoke | dev | full | test-imports"
        exit 1
        ;;
esac

echo "[OK] run-inference $MODE completado"
