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
import numpy as _fb_np

if 'erode_depth' not in dir():
    def erode_depth(depth, radius=2, depth_diff_thres=0.001,
                    ratio_thres=0.8, zfar=100, device='cuda'):
        import torch as _t
        is_tensor = isinstance(depth, _t.Tensor)
        d = depth.cpu().numpy() if is_tensor else _fb_np.array(depth)
        H, W = d.shape
        valid = (d >= 0.001) & (d < zfar)
        padded = _fb_np.pad(d, radius, mode='constant', constant_values=0)
        pv = _fb_np.pad(valid, radius, mode='constant', constant_values=False)
        bad = _fb_np.zeros((H, W), dtype=_fb_np.float32)
        total = 0
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                nh, nw = radius + dh, radius + dw
                nb_slice = padded[nh:nh+H, nw:nw+W]
                nv = pv[nh:nh+H, nw:nw+W]
                bad += ((~nv) | (_fb_np.abs(nb_slice - d) > depth_diff_thres)).astype(_fb_np.float32)
                total += 1
        out = _fb_np.where((bad / total > ratio_thres) | (~valid), 0.0, d).astype(_fb_np.float32)
        return _t.from_numpy(out).to(device) if is_tensor else out

if 'bilateral_filter_depth' not in dir():
    def bilateral_filter_depth(depth, radius=2, zfar=100,
                               sigmaD=2, sigmaR=100000, device='cuda'):
        import torch as _t, cv2
        is_tensor = isinstance(depth, _t.Tensor)
        d = depth.cpu().numpy() if is_tensor else _fb_np.array(depth)
        valid = (d >= 0.001) & (d < zfar)
        out = cv2.bilateralFilter(d.astype(_fb_np.float32),
                                  d=2*radius+1, sigmaColor=float(sigmaR), sigmaSpace=float(sigmaD))
        out = _fb_np.where(valid, out, 0.0).astype(_fb_np.float32)
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
