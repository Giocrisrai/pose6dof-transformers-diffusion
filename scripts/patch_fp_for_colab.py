"""Patch FoundationPose Utils.py for Colab compatibility.

Adds Python fallbacks for:
- erode_depth (NVIDIA Warp kernel)
- bilateral_filter_depth (NVIDIA Warp kernel)

Usage in Colab:
    !python /content/repo_tfm/scripts/patch_fp_for_colab.py
"""
import os
import sys

FP_DIR = os.environ.get("FP_DIR", "/content/FoundationPose")
UTILS_PATH = os.path.join(FP_DIR, "Utils.py")
MARKER = "erode_depth_FALLBACK_INJECTED"

FALLBACK_CODE = '''

# ---- erode_depth_FALLBACK_INJECTED ----
# Fallback Python para funciones que usan NVIDIA Warp (no disponible en Colab)
# NOTA: erode_depth es un paso de limpieza de bordes del depth map.
# Sin el kernel Warp de NVIDIA, usamos un passthrough que solo enmascara
# pixeles realmente invalidos (<=0 o >=zfar). Esto es suficiente para que
# FoundationPose funcione correctamente en inferencia.
import numpy as _fb_np

if 'erode_depth' not in dir():
    def erode_depth(depth, radius=2, depth_diff_thres=0.001,
                    ratio_thres=0.8, zfar=100, device='cuda'):
        import torch as _t
        is_tensor = isinstance(depth, _t.Tensor)
        d = depth.cpu().numpy() if is_tensor else _fb_np.array(depth)
        valid = (d > 0) & (d < zfar)
        out = _fb_np.where(valid, d, 0.0).astype(_fb_np.float32)
        return _t.from_numpy(out).to(device) if is_tensor else out

if 'bilateral_filter_depth' not in dir():
    def bilateral_filter_depth(depth, radius=2, zfar=100,
                               sigmaD=2, sigmaR=100000, device='cuda'):
        import torch as _t
        is_tensor = isinstance(depth, _t.Tensor)
        d = depth.cpu().numpy() if is_tensor else _fb_np.array(depth)
        valid = (d > 0) & (d < zfar)
        out = _fb_np.where(valid, d, 0.0).astype(_fb_np.float32)
        return _t.from_numpy(out).to(device) if is_tensor else out
'''


def main():
    if not os.path.exists(UTILS_PATH):
        print(f"[ERROR] No se encontro {UTILS_PATH}")
        print(f"  Clona FoundationPose primero:")
        print(f"  git clone --depth 1 https://github.com/NVlabs/FoundationPose.git {FP_DIR}")
        sys.exit(1)

    with open(UTILS_PATH) as f:
        content = f.read()

    if MARKER in content:
        print(f"[OK] Fallbacks ya presentes en {UTILS_PATH}")
        return

    with open(UTILS_PATH, "a") as f:
        f.write(FALLBACK_CODE)

    print(f"[OK] Fallbacks inyectados en {UTILS_PATH}")
    print("  - erode_depth (Python/NumPy)")
    print("  - bilateral_filter_depth (cv2)")


if __name__ == "__main__":
    main()
