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


def download_weights(output_dir: str) -> None:
    """Download FoundationPose weights from official sources."""
    os.makedirs(output_dir, exist_ok=True)

    fp_dir = os.environ.get("FP_DIR", "/content/FoundationPose")

    if not os.path.isdir(fp_dir):
        print(f"[!] Repo FoundationPose no encontrado en: {fp_dir}")
        print("    Clonalo primero: git clone https://github.com/NVlabs/FoundationPose.git")
        print("    O establece FP_DIR al directorio correcto.")
        return

    # Check if weights script exists
    download_script = os.path.join(fp_dir, "download_weights.sh")
    if os.path.exists(download_script):
        print(f"Ejecutando {download_script}...")
        subprocess.run(["bash", download_script], cwd=fp_dir, check=True)
        print("Pesos descargados en el repo de FoundationPose.")
    else:
        print("Script de descarga no encontrado.")
        print("Instrucciones para descarga manual:")

        readme = os.path.join(fp_dir, "README.md")
        if os.path.exists(readme):
            with open(readme) as f:
                content = f.read()
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
