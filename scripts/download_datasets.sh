#!/bin/bash
# =============================================================
# Download BOP datasets for TFM evaluation
# Usage: bash scripts/download_datasets.sh [tless|ycbv|all]
# =============================================================

set -e

DATA_DIR="data/datasets"
mkdir -p "${DATA_DIR}"

HF_BASE="https://huggingface.co/datasets/bop-benchmark"

download_tless() {
    echo "=== Downloading T-LESS ==="
    local DIR="${DATA_DIR}/tless"
    mkdir -p "${DIR}"

    echo "  [1/3] Base (camera, GT annotations)..."
    curl -L -o "${DIR}/tless_base.zip" \
        "${HF_BASE}/tless/resolve/main/tless_base.zip"

    echo "  [2/3] 3D Models..."
    curl -L -o "${DIR}/tless_models.zip" \
        "${HF_BASE}/tless/resolve/main/tless_models.zip"

    echo "  [3/3] Test images (Primesense)..."
    curl -L -o "${DIR}/tless_test_primesense_all.zip" \
        "${HF_BASE}/tless/resolve/main/tless_test_primesense_all.zip"

    echo "  Extracting..."
    cd "${DIR}"
    unzip -qo tless_base.zip
    unzip -qo tless_models.zip
    unzip -qo tless_test_primesense_all.zip
    cd -

    echo "  ✓ T-LESS ready"
}

download_ycbv() {
    echo "=== Downloading YCB-Video ==="
    local DIR="${DATA_DIR}/ycbv"
    mkdir -p "${DIR}"

    echo "  [1/3] Base (camera, GT annotations)..."
    curl -L -o "${DIR}/ycbv_base.zip" \
        "${HF_BASE}/ycbv/resolve/main/ycbv_base.zip"

    echo "  [2/3] 3D Models..."
    curl -L -o "${DIR}/ycbv_models.zip" \
        "${HF_BASE}/ycbv/resolve/main/ycbv_models.zip"

    echo "  [3/3] Test images..."
    curl -L -o "${DIR}/ycbv_test_all.zip" \
        "${HF_BASE}/ycbv/resolve/main/ycbv_test_all.zip"

    echo "  Extracting..."
    cd "${DIR}"
    unzip -qo ycbv_base.zip
    unzip -qo ycbv_models.zip
    unzip -qo ycbv_test_all.zip
    cd -

    echo "  ✓ YCB-Video ready"
}

case "${1:-all}" in
    tless) download_tless ;;
    ycbv)  download_ycbv ;;
    all)
        download_tless
        download_ycbv
        ;;
    *)
        echo "Usage: $0 [tless|ycbv|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Verifying datasets ==="
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
python -m src.utils.dataset_loader "${DATA_DIR}/tless" test
python -m src.utils.dataset_loader "${DATA_DIR}/ycbv" test
echo "Done!"
