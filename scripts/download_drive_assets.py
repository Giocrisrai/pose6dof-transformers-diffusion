#!/usr/bin/env python3
"""Descarga reproducible de pesos y assets desde Google Drive.

Uso:
    python scripts/download_drive_assets.py [--what foundationpose|checkpoints|all]

Requiere `gdown`:
    pip install gdown

Nota: los IDs apuntan al Drive del autor del TFM (giocrisrai@gmail.com).
Para reproducibilidad publica, los archivos deben tener acceso "cualquiera con
el enlace" o estar mirroreados en un release de GitHub.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ─── Manifest de assets en Drive ──────────────────────────────────
ASSETS: dict = {
    "foundationpose_scorer": {
        "drive_id": "1DJlDdd4rnqiPjOpemxfn1Y7bawjYoGIY",
        "dest": "data/models/foundationpose/2024-01-11-20-02-45/model_best.pth",
        "size_mb": 181.4,
        "description": "FoundationPose scorer (Wen et al., 2024)",
    },
    "foundationpose_refiner": {
        "drive_id": "1Ozomqy6t0-dt9ixDCxeaa9Q0P_AIZiBu",
        "dest": "data/models/foundationpose/2023-10-28-18-33-37/model_best.pth",
        "size_mb": 65.1,
        "description": "FoundationPose refiner (Wen et al., 2024)",
    },
    "fp_checkpoint_ycbv": {
        "drive_id": "1iTKcleHqU5QTyR9Y69FmzDiiDIDMyBXx",
        "dest": "experiments/checkpoints/fp_ycbv_checkpoint.json",
        "size_mb": 0.4,
        "description": "Predicciones FP YCB-V (run 2026-04-27, 1098 instancias)",
    },
    "fp_checkpoint_tless": {
        "drive_id": "1ppPRkiB1ERIqsireVVyyQFy_sAfetwo7",
        "dest": "experiments/checkpoints/fp_tless_checkpoint.json",
        "size_mb": 0.4,
        "description": "Predicciones FP T-LESS (run 2026-04-27, 1012 instancias)",
    },
}

GROUPS = {
    "foundationpose": ["foundationpose_scorer", "foundationpose_refiner"],
    "checkpoints": ["fp_checkpoint_ycbv", "fp_checkpoint_tless"],
    "all": list(ASSETS.keys()),
}


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[setup] instalando gdown ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown>=5.2"])


def download_asset(name: str) -> bool:
    asset = ASSETS[name]
    dest = REPO_ROOT / asset["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [skip] {name}: ya existe ({size_mb:.1f} MB) en {dest.relative_to(REPO_ROOT)}")
        return True

    import gdown
    url = f"https://drive.google.com/uc?id={asset['drive_id']}"
    print(f"  [download] {name} ({asset['size_mb']} MB) ...")
    print(f"    desde: {url}")
    print(f"    hacia: {dest.relative_to(REPO_ROOT)}")
    try:
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
        actual_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    OK: {actual_mb:.1f} MB descargados")
        return True
    except Exception as e:
        print(f"    FALLO: {e}")
        print(f"    Sugerencia: descarga manual desde Drive con ID {asset['drive_id']}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        choices=list(GROUPS.keys()),
        default="all",
        help="Qué descargar (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar assets disponibles sin descargar",
    )
    args = parser.parse_args()

    if args.list:
        print(f"Manifest de assets ({len(ASSETS)} entradas):")
        for name, a in ASSETS.items():
            print(f"  - {name}: {a['size_mb']} MB — {a['description']}")
            print(f"      → {a['dest']}")
        return

    targets = GROUPS[args.what]
    print(f"[download_drive_assets] descargando grupo '{args.what}' ({len(targets)} assets)...")
    ensure_gdown()

    failed = []
    for name in targets:
        if not download_asset(name):
            failed.append(name)

    print()
    if failed:
        print(f"[!] {len(failed)} fallo(s): {failed}")
        sys.exit(1)
    print(f"[OK] grupo '{args.what}' descargado completo.")


if __name__ == "__main__":
    main()
