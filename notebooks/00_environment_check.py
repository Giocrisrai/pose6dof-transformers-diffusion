"""
Notebook 00: Environment Check
===============================
Verifica que todo el entorno está correctamente configurado:
- Python, PyTorch, MPS/CUDA
- Módulos del TFM (lie_groups, rotations, metrics)
- Datasets BOP descargados
- Dependencias instaladas

Ejecutar: python notebooks/00_environment_check.py
"""

import sys
import importlib
from pathlib import Path


def check(name: str, test_fn):
    """Run a check and print result."""
    try:
        result = test_fn()
        print(f"  ✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("  TFM Pose 6-DoF — Environment Check")
    print("=" * 60)
    passed = 0
    total = 0

    # ── Python ──
    print("\n[Python]")
    total += 1
    if check("Version", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"):
        passed += 1

    # ── Core packages ──
    print("\n[Core Packages]")
    for pkg in ["numpy", "scipy", "matplotlib", "cv2", "PIL", "tqdm", "yaml"]:
        total += 1
        if check(pkg, lambda p=pkg: importlib.import_module(p).__version__
                 if hasattr(importlib.import_module(p), '__version__')
                 else "OK"):
            passed += 1

    # ── PyTorch ──
    print("\n[PyTorch]")
    total += 1
    if check("torch", lambda: f"{__import__('torch').__version__}"):
        passed += 1

    total += 1
    try:
        import torch
        if torch.backends.mps.is_available():
            check("MPS (Apple Silicon)", lambda: "Available")
            passed += 1
        elif torch.cuda.is_available():
            check("CUDA", lambda: f"GPU: {torch.cuda.get_device_name(0)}")
            passed += 1
        else:
            check("GPU", lambda: "CPU only (MPS and CUDA unavailable)")
            passed += 1
    except:
        print("  ✗ GPU check failed")

    # ── 3D Libraries ──
    print("\n[3D Libraries]")
    for pkg in ["open3d", "trimesh", "pytransform3d"]:
        total += 1
        mod = importlib.import_module(pkg)
        if check(pkg, lambda m=mod: getattr(m, '__version__', 'OK')):
            passed += 1

    # ── TFM Modules ──
    print("\n[TFM Modules]")
    modules = [
        ("src.utils.lie_groups", "SO(3)/SE(3) operations"),
        ("src.utils.rotations", "Rotation representations"),
        ("src.utils.metrics", "BOP metrics (VSD, MSSD, MSPD)"),
        ("src.utils.visualization", "Pose visualization"),
        ("src.utils.dataset_loader", "BOP dataset loader"),
        ("src.perception.foundation_pose", "FoundationPose wrapper"),
        ("src.perception.gdrnet", "GDR-Net++ wrapper"),
    ]
    for mod_name, desc in modules:
        total += 1
        if check(desc, lambda m=mod_name: importlib.import_module(m).__name__):
            passed += 1

    # ── Quick Math Test ──
    print("\n[Quick Math Test]")
    total += 1
    try:
        import numpy as np
        from src.utils.lie_groups import so3_exp, so3_log
        omega = np.array([0.1, 0.2, 0.3])
        R = so3_exp(omega)
        omega_rec = so3_log(R)
        err = np.linalg.norm(omega - omega_rec)
        if check("SO(3) exp/log roundtrip", lambda: f"error = {err:.2e}"):
            passed += 1
    except Exception as e:
        print(f"  ✗ SO(3) roundtrip: {e}")

    total += 1
    try:
        from src.utils.rotations import matrix_to_6d, sixd_to_matrix
        rep = matrix_to_6d(R)
        R_rec = sixd_to_matrix(rep)
        err = np.linalg.norm(R - R_rec)
        if check("6D representation roundtrip", lambda: f"error = {err:.2e}"):
            passed += 1
    except Exception as e:
        print(f"  ✗ 6D roundtrip: {e}")

    # ── Datasets ──
    print("\n[BOP Datasets]")
    data_dir = Path("data/datasets")
    for ds_name in ["tless", "ycbv"]:
        total += 1
        ds_path = data_dir / ds_name
        if ds_path.exists():
            models = list((ds_path / "models").glob("*.ply")) if (ds_path / "models").exists() else []
            test_scenes = list((ds_path / "test").iterdir()) if (ds_path / "test").exists() else []
            if check(ds_name.upper(), lambda m=models, s=test_scenes:
                     f"{len(m)} models, {len(s)} test scenes"):
                passed += 1
        else:
            print(f"  ⏳ {ds_name.upper()}: Not downloaded yet")
            print(f"     Run: bash scripts/download_datasets.sh {ds_name}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  🎯 Environment fully configured!")
    else:
        print(f"  ⚠  {total - passed} checks need attention")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
