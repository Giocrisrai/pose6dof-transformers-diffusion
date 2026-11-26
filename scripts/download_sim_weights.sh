#!/usr/bin/env bash
# Descarga los pesos de las políticas visuales entrenadas en simulación (Iter 8 e Iter 9)
# y de sus encoders RGB-D desde la release `weights-sim-v9`, y verifica su huella.
# Son necesarios para experiments/eval_diffusion_iter8_sim.py y eval_diffusion_iter9_sim.py.
#
# Uso: scripts/download_sim_weights.sh [directorio de destino, por defecto data/models]
set -euo pipefail

BASE="https://github.com/Giocrisrai/pose6dof-transformers-diffusion/releases/download/weights-sim-v9"
DESTINO="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/models}"
mkdir -p "$DESTINO"
cd "$DESTINO"

while read -r huella archivo; do
  if [ -f "$archivo" ] && echo "$huella  $archivo" | shasum -a 256 -c --status; then
    echo "ya está: $archivo"
    continue
  fi
  echo "descargando $archivo"
  curl -fL --retry 3 -o "$archivo" "$BASE/$archivo"
  echo "$huella  $archivo" | shasum -a 256 -c
done <<'SUMAS'
ee130a9ddf50e2b630e04094af8e2f9b924112fb45aefda3915dfd9985d1e3c8 diffusion_policy_v8_randomized.pth
cfb8e1bacd09d3112e372bd336c163e92b11bb0a618201ffc8b75f747c5cf6b9 diffusion_policy_v9_clutter.pth
178091df25638a043face8ed4a5ba8620cd39c806900dd175e375f77321f1085 visual_encoder_iter8rand.pth
b4c8856c6047e47e566411ea79fb639fa22c2627713b89e3bcd233003cb1a4b3 visual_encoder_iter9clut.pth
SUMAS
echo "Pesos listos en $DESTINO"
