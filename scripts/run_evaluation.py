#!/usr/bin/env python3
"""
Run BOP evaluation for one or more methods.

Usage:
    python scripts/run_evaluation.py --method foundationpose --dataset tless
    python scripts/run_evaluation.py --method gdrnet_pp --dataset ycbv
    python scripts/run_evaluation.py --compare  # run all methods and compare

Requires predictions to be generated first (via Colab notebooks).
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dataset_loader import BOPDataset, verify_dataset
from src.perception.evaluator import compare_methods, load_predictions, evaluate_method
from src.utils.visualization import plot_metrics_comparison
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_single_evaluation(method: str, dataset_name: str, data_root: str = "data/datasets"):
    """Evaluate a single method on a single dataset."""
    print(f"\n{'='*60}")
    print(f"  Evaluating {method} on {dataset_name.upper()}")
    print(f"{'='*60}")

    dataset_root = f"{data_root}/{dataset_name}"

    # Verify dataset
    result = verify_dataset(dataset_root, "test")
    if result["errors"]:
        print(f"WARNING: Dataset has {len(result['errors'])} errors")

    # Load predictions
    pred_path = f"experiments/results/{method}_{dataset_name}.json"
    if not Path(pred_path).exists():
        print(f"\n⚠️  Predictions not found: {pred_path}")
        print(f"   Run the Colab notebook first to generate predictions.")
        print(f"   Then save results to: {pred_path}")
        return None

    predictions = load_predictions(pred_path)
    print(f"Loaded {len(predictions)} predictions")

    # Evaluate
    dataset = BOPDataset(dataset_root, split="test")
    results = evaluate_method(dataset, predictions, method)

    # Save results
    output_path = f"experiments/results/{method}_{dataset_name}_metrics.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    return results


def run_comparison(datasets=None, data_root: str = "data/datasets"):
    """Compare all methods across datasets."""
    if datasets is None:
        datasets = ["tless", "ycbv"]

    methods = ["foundationpose", "gdrnet_pp"]

    for dataset_name in datasets:
        pred_files = {}
        for method in methods:
            pred_path = f"experiments/results/{method}_{dataset_name}.json"
            if Path(pred_path).exists():
                pred_files[method] = pred_path
            else:
                print(f"⚠️  Missing: {pred_path}")

        if len(pred_files) >= 2:
            results = compare_methods(
                dataset_name=dataset_name,
                dataset_root=f"{data_root}/{dataset_name}",
                prediction_files=pred_files,
                output_dir="experiments/results",
            )

            # Generate comparison plot
            method_names = list(results.keys())
            vsd_scores = [results[m].get("MSSD", {}).get("auc", 0) * 100 for m in method_names]
            mssd_scores = [results[m].get("MSSD", {}).get("auc", 0) * 100 for m in method_names]
            mspd_scores = [results[m].get("MSPD", {}).get("auc", 0) * 100 for m in method_names]

            fig = plot_metrics_comparison(
                method_names, vsd_scores, mssd_scores, mspd_scores,
                title=f"BOP Metrics — {dataset_name.upper()}"
            )
            fig_path = f"experiments/results/comparison_{dataset_name}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {fig_path}")
            plt.close()
        else:
            print(f"\n⚠️  Not enough predictions for {dataset_name}")
            print(f"   Need at least 2 methods. Found: {list(pred_files.keys())}")


def main():
    parser = argparse.ArgumentParser(description="BOP Evaluation Runner")
    parser.add_argument("--method", type=str, help="Method name (foundationpose, gdrnet_pp)")
    parser.add_argument("--dataset", type=str, help="Dataset name (tless, ycbv)")
    parser.add_argument("--compare", action="store_true", help="Run comparative evaluation")
    parser.add_argument("--data-root", type=str, default="data/datasets")
    args = parser.parse_args()

    print(f"Device: {get_device()}")

    if args.compare:
        datasets = [args.dataset] if args.dataset else None
        run_comparison(datasets=datasets, data_root=args.data_root)
    elif args.method and args.dataset:
        run_single_evaluation(args.method, args.dataset, args.data_root)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/run_evaluation.py --method foundationpose --dataset tless")
        print("  python scripts/run_evaluation.py --compare")


if __name__ == "__main__":
    main()
