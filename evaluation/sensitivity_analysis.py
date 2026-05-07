#!/usr/bin/env python3
"""
Sensitivity Analysis for AgenticSZZ

Analyzes the impact of candidate_limit (K) on accuracy.
Generates figure for Discussion section.

Usage:
    # First, run evaluations with different K values:
    python evaluation/run_evaluation.py --candidate-limit 5 --output-dir results/sensitivity_k5 --sample-file results/pilot/pilot_cases_20pct.json --config full --workers 32 --dataset all
    python evaluation/run_evaluation.py --candidate-limit 10 --output-dir results/sensitivity_k10 ...
    python evaluation/run_evaluation.py --candidate-limit 15 --output-dir results/sensitivity_k15 ...
    python evaluation/run_evaluation.py --candidate-limit 20 --output-dir results/sensitivity_k20 ...
    python evaluation/run_evaluation.py --candidate-limit 30 --output-dir results/sensitivity_k30 ...
    python evaluation/run_evaluation.py --candidate-limit 50 --output-dir results/sensitivity_k50 ...

    # Then generate the figure:
    python evaluation/sensitivity_analysis.py --results-dir results --output figures/sensitivity_k.pdf
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def extract_k_from_dirname(dirname: str) -> int:
    """Extract K value from directory name like 'sensitivity_k20'."""
    match = re.search(r"sensitivity_k(\d+)", dirname)
    if match:
        return int(match.group(1))
    return None


def load_results(results_dir: Path) -> Dict[int, Dict[str, float]]:
    """
    Load results from multiple K-value experiment directories.

    Searches for directories matching pattern 'sensitivity_k*' or 'pilot_k*'.

    Args:
        results_dir: Parent directory containing K-experiment subdirs

    Returns:
        Dict mapping K -> {dataset: precision} or K -> {'overall': precision}
    """
    results = {}

    # Find all sensitivity directories
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        k = extract_k_from_dirname(subdir.name)
        if k is None:
            continue

        # Look for CSV files with results
        csv_files = list(subdir.glob("*_results_*.csv"))
        if not csv_files:
            # Try JSON files
            json_files = list(subdir.glob("*_full_*.json"))
            if json_files:
                results[k] = load_from_json_files(json_files)
            continue

        # Load from CSV
        results[k] = load_from_csv_files(csv_files)

    return results


def load_from_csv_files(csv_files: List[Path]) -> Dict[str, Dict[str, float]]:
    """Load precision, recall, and F1-score from CSV result files."""
    metrics = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Filter for 'full' config only
        if "config" in df.columns:
            df = df[df["config"] == "full"]

        for _, row in df.iterrows():
            dataset = row.get("dataset", "unknown")
            precision = row.get("precision", 0)
            recall = row.get("recall", 0)
            f1 = row.get("f1", row.get("F1", 0))
            metrics[dataset] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    return metrics


def load_from_json_files(json_files: List[Path]) -> Dict[str, Dict[str, float]]:
    """Load precision, recall, and F1 from JSON result files."""
    metrics = {}
    total_correct = 0
    total_cases = 0

    for json_file in json_files:
        # Extract dataset from filename
        name = json_file.stem
        if "linux" in name.lower():
            dataset = "linux"
        elif "apache" in name.lower():
            dataset = "apache"
        elif "github" in name.lower():
            dataset = "github"
        else:
            continue

        with open(json_file) as f:
            data = json.load(f)

        # Count correct predictions
        cases = data if isinstance(data, list) else data.get("results", [])
        correct = sum(1 for c in cases if c.get("is_correct", False))
        total = len(cases)

        if total > 0:
            # For JSON files, precision == recall since single prediction
            prec_recall = correct / total
            metrics[dataset] = {
                "precision": prec_recall,
                "recall": prec_recall,
                "f1": prec_recall,
            }
            total_correct += correct
            total_cases += total

    if total_cases > 0:
        overall = total_correct / total_cases
        metrics["overall"] = {
            "precision": overall,
            "recall": overall,
            "f1": overall,
        }

    return metrics


def plot_sensitivity(
    results: Dict[int, Dict[str, Dict[str, float]]],
    output_path: Path,
    show_datasets: bool = False,
):
    """
    Generate sensitivity analysis figures - precision, recall, F1-score for all datasets.

    Generates three separate PDF files for horizontal layout in LaTeX.

    Args:
        results: Dict mapping K -> {dataset: {precision, recall, f1}}
        output_path: Base path to save figures (will add _precision, _recall, _f1 suffixes)
        show_datasets: Unused, kept for API compatibility
    """
    # Sort by K value
    k_values = sorted(results.keys())

    # Prepare data for each metric
    metrics_data = {
        "precision": {"linux": [], "apache": [], "github": []},
        "recall": {"linux": [], "apache": [], "github": []},
        "f1": {"linux": [], "apache": [], "github": []},
    }

    for k in k_values:
        k_metrics = results[k]
        for ds in ["linux", "apache", "github"]:
            if ds in k_metrics:
                metrics_data["precision"][ds].append(k_metrics[ds].get("precision", None))
                metrics_data["recall"][ds].append(k_metrics[ds].get("recall", None))
                metrics_data["f1"][ds].append(k_metrics[ds].get("f1", None))
            else:
                metrics_data["precision"][ds].append(None)
                metrics_data["recall"][ds].append(None)
                metrics_data["f1"][ds].append(None)

    # Softer publication-quality colors (muted palette)
    colors = {
        "linux": "#7FB069",  # Muted green
        "apache": "#D88C8C",  # Muted red/coral
        "github": "#9B8DC2",  # Muted purple
    }

    labels = {
        "linux": "DS_LINUX",
        "apache": "DS_APACHE",
        "github": "DS_GITHUB",
    }

    metric_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
    }

    # Generate three separate figures
    # Legend is placed inside the first (precision) figure
    base_path = output_path.parent / output_path.stem
    metric_order = ["precision", "recall", "f1"]  # Define order to ensure precision is first
    
    for idx, metric_name in enumerate(metric_order):
        dataset_values = metrics_data[metric_name]
        fig, ax = plt.subplots(figsize=(4.5, 3.5))

        for ds, values in dataset_values.items():
            # Filter None values
            valid_k = [k for k, v in zip(k_values, values) if v is not None]
            valid_v = [v for v in values if v is not None]

            if valid_k:
                ax.plot(
                    valid_k,
                    valid_v,
                    "o-",
                    linewidth=2.5,
                    markersize=8,
                    color=colors[ds],
                    label=labels[ds],
                )

        # Formatting with larger fonts for 3-panel layout
        ax.set_xlabel("Candidate Limit (K)", fontsize=14)
        ax.set_ylabel(metric_labels[metric_name], fontsize=14)
        ax.set_xticks(k_values)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylim(0, 0.8)  # Start from 0 per reviewer request
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Add legend to first figure (precision) only
        if idx == 0:
            ax.legend(
                loc='lower right',
                frameon=True,
                fancybox=False,
                edgecolor="gray",
                fontsize=11,
            )
        
        plt.tight_layout()

        # Save individual figure
        metric_output = base_path.parent / f"{base_path.name}_{metric_name}.pdf"
        plt.savefig(metric_output, dpi=300, bbox_inches="tight")
        print(f"Saved {metric_name} figure to {metric_output}")
        plt.close()

    # Also save the original combined F1 figure for backward compatibility
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds, f1_values in metrics_data["f1"].items():
        valid_k = [k for k, f in zip(k_values, f1_values) if f is not None]
        valid_f = [f for f in f1_values if f is not None]
        if valid_k:
            ax.plot(
                valid_k,
                valid_f,
                "o-",
                linewidth=2,
                markersize=7,
                color=colors[ds],
                label=labels[ds],
            )
    ax.set_xlabel("Candidate Limit (K)", fontsize=12)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_xticks(k_values)
    ax.set_ylim(0.4, 0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        edgecolor="gray",
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined F1 figure to {output_path}")
    plt.close()


def print_summary_table(results: Dict[int, Dict[str, Dict[str, float]]]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)

    k_values = sorted(results.keys())

    # Header
    print(f"{'K':>6} | {'Linux P/R/F1':>18} | {'Apache P/R/F1':>18} | {'GitHub P/R/F1':>18}")
    print("-" * 80)

    for k in k_values:
        metrics = results[k]
        linux = metrics.get("linux", {})
        apache = metrics.get("apache", {})
        github = metrics.get("github", {})
        
        linux_str = f"{linux.get('precision', 0):.3f}/{linux.get('recall', 0):.3f}/{linux.get('f1', 0):.3f}"
        apache_str = f"{apache.get('precision', 0):.3f}/{apache.get('recall', 0):.3f}/{apache.get('f1', 0):.3f}"
        github_str = f"{github.get('precision', 0):.3f}/{github.get('recall', 0):.3f}/{github.get('f1', 0):.3f}"
        
        print(f"{k:>6} | {linux_str:>18} | {apache_str:>18} | {github_str:>18}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis figure for candidate_limit (K)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing sensitivity_k* subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sensitivity_analysis/sensitivity_k.pdf",
        help="Output path for figure",
    )
    parser.add_argument(
        "--show-datasets",
        action="store_true",
        help="Show per-dataset lines in addition to overall",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Scanning {results_dir} for sensitivity experiment results...")
    results = load_results(results_dir)

    if not results:
        print("ERROR: No sensitivity results found!")
        print("Run experiments first with --candidate-limit option:")
        print(
            "  python evaluation/run_evaluation.py --candidate-limit 5 --output-dir results/sensitivity_k5 ..."
        )
        return

    print(f"Found results for K values: {sorted(results.keys())}")

    # Print summary table
    print_summary_table(results)

    # Generate figure
    plot_sensitivity(results, output_path, show_datasets=args.show_datasets)


if __name__ == "__main__":
    main()
