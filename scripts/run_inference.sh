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
        cat <<EOF
[NOT_IMPLEMENTED] 'run-inference $MODE' todavia no ejecuta inferencia real.
  El pipeline de evaluacion vive en notebooks/colab/01_foundationpose_eval.ipynb.
  Para validar el contenedor usa:
      run-inference test-imports
  Para reproducir los resultados BOP:
      Abre el notebook en Colab y corre celdas 1-26 (ver docker/README-GPU.md).
EOF
        exit 2
        ;;

    *)
        echo "[ERROR] MODE desconocido: $MODE"
        echo "  Opciones: smoke | dev | full | test-imports"
        exit 1
        ;;
esac

echo "[OK] run-inference $MODE completado"
