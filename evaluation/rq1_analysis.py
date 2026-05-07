#!/usr/bin/env python3
"""
RQ1 Analysis: Analyze evaluation results and generate paper-ready tables.

This script processes the per-case JSON results from run_evaluation.py and:
1. Recalculates metrics with proper BIC deduplication
2. Splits GitHub into GitHub-C and GitHub-J (following LLM4SZZ paper)
3. Generates LaTeX tables for the paper

Usage:
    python evaluation/rq1_analysis.py --results-dir results/rq1
    python evaluation/rq1_analysis.py --results-dir results/rq1 --output-latex results/rq1/rq1_table.tex
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetMetrics:
    """Metrics for a dataset/config combination."""
    dataset: str
    config: str
    total_cases: int
    correct: int
    total_bics: int
    precision: float
    recall: float
    f1: float


def load_ground_truth_bics(dataset_dir: Path) -> Dict[str, set]:
    """Load ground truth BICs with proper deduplication.
    
    Returns:
        Dict mapping BFC -> set of unique BICs
    """
    all_info_path = dataset_dir / "final_all_dataset.json"
    bfc_to_bics = {}
    
    with open(all_info_path) as f:
        all_data = json.load(f)
    
    for item in all_data:
        bfc = item.get("bug_fixing_commit")
        bic = item.get("bug_inducing_commit")
        if bfc and bic:
            if isinstance(bic, list):
                bics = {b for b in bic if b and isinstance(b, str)}
            elif isinstance(bic, str):
                bics = {bic}
            else:
                continue
            if bics:
                if bfc not in bfc_to_bics:
                    bfc_to_bics[bfc] = bics
                else:
                    bfc_to_bics[bfc].update(bics)
    
    return bfc_to_bics


def load_github_language_mapping(dataset_dir: Path) -> Dict[str, str]:
    """Load BFC -> language mapping for GitHub dataset.
    
    Returns:
        Dict mapping BFC -> 'c' or 'java'
    """
    github_path = dataset_dir / "DS_GITHUB.json"
    bfc_to_lang = {}
    
    with open(github_path) as f:
        data = json.load(f)
    
    for item in data:
        bfc = item['fix_commit_hash']
        lang = item.get('language', ['unknown'])
        if isinstance(lang, list):
            lang = lang[0] if lang else 'unknown'
        bfc_to_lang[bfc] = lang
    
    return bfc_to_lang


def calculate_metrics(
    results: List[dict],
    bfc_to_bics: Dict[str, set],
    dataset: str,
    config: str
) -> DatasetMetrics:
    """Calculate precision, recall, F1 using LLM4SZZ micro-average formula.
    
    - Precision = correct / total_cases
    - Recall = correct / total_BICs
    - F1 = harmonic mean
    """
    total = len(results)
    correct = sum(1 for r in results if r['success'])
    
    # Count total BICs
    total_bics = 0
    for r in results:
        bfc = r['bfc']
        if bfc in bfc_to_bics:
            total_bics += len(bfc_to_bics[bfc])
    
    precision = correct / total if total > 0 else 0.0
    recall = correct / total_bics if total_bics > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return DatasetMetrics(
        dataset=dataset,
        config=config,
        total_cases=total,
        correct=correct,
        total_bics=total_bics,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def find_result_file(results_dir: Path, dataset: str, config: str, timestamp: Optional[str] = None) -> Optional[Path]:
    """Find the result file for a given dataset/config, optionally filtered by timestamp.

    Each dataset may have a different timestamp (e.g., datasets were run sequentially).
    Returns the most recent matching file if no timestamp specified.
    """
    if timestamp:
        candidate = results_dir / f"{dataset}_{config}_{timestamp}.json"
        return candidate if candidate.exists() else None
    # Glob for any matching file and pick the most recent by timestamp in filename
    matches = sorted(results_dir.glob(f"{dataset}_{config}_*.json"))
    return matches[-1] if matches else None


def analyze_results(
    results_dir: Path,
    dataset_dir: Path,
    timestamp: Optional[str] = None
) -> Dict[str, Dict[str, DatasetMetrics]]:
    """Analyze all results and return metrics by dataset and config.

    Returns:
        Dict[dataset][config] -> DatasetMetrics
        Datasets: linux, apache, github, github-c, github-j
    """
    # Load ground truth
    bfc_to_bics = load_ground_truth_bics(dataset_dir)
    bfc_to_lang = load_github_language_mapping(dataset_dir)

    if timestamp:
        print(f"Using timestamp: {timestamp}")
    else:
        print("Using most recent file per dataset/config")

    configs = ['blame_only', 'blame_fallback', 'tkg_only', 'agent_only', 'full']
    datasets = ['linux', 'apache', 'github']

    all_metrics = {}

    for dataset in datasets:
        all_metrics[dataset] = {}

        for config in configs:
            result_file = find_result_file(results_dir, dataset, config, timestamp)
            if result_file is None:
                print(f"  Warning: {dataset}_{config} not found, skipping")
                continue
            
            with open(result_file) as f:
                results = json.load(f)
            
            # Calculate metrics for main dataset
            metrics = calculate_metrics(results, bfc_to_bics, dataset, config)
            all_metrics[dataset][config] = metrics
            
            # For GitHub, also split by language
            if dataset == 'github':
                c_results = [r for r in results if bfc_to_lang.get(r['bfc'], 'c') == 'c']
                j_results = [r for r in results if bfc_to_lang.get(r['bfc'], 'c') == 'java']
                
                if 'github-c' not in all_metrics:
                    all_metrics['github-c'] = {}
                if 'github-j' not in all_metrics:
                    all_metrics['github-j'] = {}
                
                all_metrics['github-c'][config] = calculate_metrics(
                    c_results, bfc_to_bics, 'github-c', config
                )
                all_metrics['github-j'][config] = calculate_metrics(
                    j_results, bfc_to_bics, 'github-j', config
                )
    
    return all_metrics


def print_metrics_table(all_metrics: Dict[str, Dict[str, DatasetMetrics]]):
    """Print metrics as a formatted table."""
    configs = ['blame_only', 'blame_fallback', 'tkg_only', 'agent_only', 'full']
    datasets = ['linux', 'apache', 'github-c', 'github-j']
    
    print("\n" + "=" * 100)
    print("RQ1: Evaluation Results (LLM4SZZ micro-average formula)")
    print("=" * 100)
    
    # Header
    print(f"\n{'Dataset':<12} {'Config':<15} {'Cases':>6} {'Correct':>8} {'BICs':>6} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 100)
    
    for dataset in datasets:
        if dataset not in all_metrics:
            continue
        for config in configs:
            if config not in all_metrics[dataset]:
                continue
            m = all_metrics[dataset][config]
            print(f"{m.dataset:<12} {m.config:<15} {m.total_cases:>6} {m.correct:>8} "
                  f"{m.total_bics:>6} {m.precision:>10.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
        print()


def generate_latex_tables(
    all_metrics: Dict[str, Dict[str, DatasetMetrics]],
    output_dir: Path
) -> Tuple[str, str, str]:
    """Generate LaTeX tables for paper.
    
    Creates 3 tables:
    - RQ1 Table 1: C language projects (Linux + GitHub-C)
    - RQ1 Table 2: Java language projects (GitHub-J + Apache)
    - RQ2 Table: Ablation study (Linux + GitHub + Apache)
    
    Returns:
        Tuple of (rq1_c_latex, rq1_java_latex, rq2_latex)
    """
    configs = ['blame_only', 'blame_fallback', 'tkg_only', 'agent_only', 'full']
    config_labels = {
        'blame_only': 'Blame-only',
        'blame_fallback': 'Blame-fallback', 
        'tkg_only': 'TKG-only',
        'agent_only': 'Agent-only',
        'full': 'Full Pipeline',
    }
    
    def make_table(datasets: List[str], dataset_labels: Dict[str, str], 
                   caption: str, label: str) -> str:
        """Helper to generate a single table."""
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append("\\small")
        
        # Column spec: l for config + 3 columns per dataset
        col_spec = "l" + " rrr" * len(datasets)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        # Header row 1: Dataset names
        header_parts = [""]
        cmidrule_parts = []
        col_idx = 2
        for ds in datasets:
            header_parts.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{dataset_labels[ds]}}}}}")
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+2}}}")
            col_idx += 3
        latex.append(" & ".join(header_parts) + " \\\\")
        latex.append(" ".join(cmidrule_parts))
        
        # Header row 2: Metric names
        header2_parts = ["\\textbf{Configuration}"]
        for _ in datasets:
            header2_parts.extend(["Precision", "Recall", "F1-score"])
        latex.append(" & ".join(header2_parts) + " \\\\")
        latex.append("\\midrule")
        
        # Data rows
        for config in configs:
            row_parts = [config_labels[config]]
            for dataset in datasets:
                if dataset in all_metrics and config in all_metrics[dataset]:
                    m = all_metrics[dataset][config]
                    row_parts.extend([
                        f"{m.precision:.3f}",
                        f"{m.recall:.3f}",
                        f"{m.f1:.3f}",
                    ])
                else:
                    row_parts.extend(["--", "--", "--"])
            
            # Bold the full pipeline row
            if config == 'full':
                row_parts = [f"\\textbf{{{config_labels[config]}}}"] + \
                           [f"\\textbf{{{p}}}" for p in row_parts[1:]]
            
            latex.append(" & ".join(row_parts) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    # RQ1 Table 1: C language (Linux + GitHub-C)
    rq1_c = make_table(
        datasets=['linux', 'github-c'],
        dataset_labels={'linux': 'Linux', 'github-c': 'GitHub-C'},
        caption="RQ1: AgenticSZZ effectiveness on C language projects.",
        label="tab:rq1-c-results"
    )
    
    # RQ1 Table 2: Java language (GitHub-J + Apache)
    rq1_java = make_table(
        datasets=['github-j', 'apache'],
        dataset_labels={'github-j': 'GitHub-J', 'apache': 'Apache'},
        caption="RQ1: AgenticSZZ effectiveness on Java language projects.",
        label="tab:rq1-java-results"
    )
    
    # RQ2 Table: Ablation study (Linux + GitHub + Apache)
    rq2 = make_table(
        datasets=['linux', 'github', 'apache'],
        dataset_labels={'linux': 'Linux', 'github': 'GitHub', 'apache': 'Apache'},
        caption="RQ2: Ablation study of AgenticSZZ components.",
        label="tab:rq2-ablation"
    )
    
    # Save tables
    rq1_path = output_dir / "rq1_table.tex"
    with open(rq1_path, 'w') as f:
        f.write("% RQ1: AgenticSZZ Evaluation Results\n")
        f.write("% Use with: \\usepackage{booktabs}\n\n")
        f.write("% Table 1: C Language Projects (Linux + GitHub-C)\n")
        f.write(rq1_c)
        f.write("\n\n% Table 2: Java Language Projects (GitHub-J + Apache)\n")
        f.write(rq1_java)
    print(f"RQ1 LaTeX tables saved to: {rq1_path}")
    
    rq2_path = output_dir / "rq2_table.tex"
    with open(rq2_path, 'w') as f:
        f.write("% RQ2: Ablation Study Results\n")
        f.write("% Use with: \\usepackage{booktabs}\n\n")
        f.write(rq2)
    print(f"RQ2 LaTeX table saved to: {rq2_path}")
    
    return rq1_c, rq1_java, rq2


def generate_summary_csv(
    all_metrics: Dict[str, Dict[str, DatasetMetrics]],
    output_path: Path
):
    """Generate summary CSV with all metrics."""
    import csv
    
    configs = ['blame_only', 'blame_fallback', 'tkg_only', 'agent_only', 'full']
    datasets = ['linux', 'apache', 'github', 'github-c', 'github-j']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'config', 'total_cases', 'correct', 'total_bics',
            'precision', 'recall', 'f1'
        ])
        
        for dataset in datasets:
            if dataset not in all_metrics:
                continue
            for config in configs:
                if config not in all_metrics[dataset]:
                    continue
                m = all_metrics[dataset][config]
                writer.writerow([
                    m.dataset, m.config, m.total_cases, m.correct, m.total_bics,
                    f"{m.precision:.3f}", f"{m.recall:.3f}", f"{m.f1:.3f}"
                ])
    
    print(f"Summary CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RQ1 evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/rq1",
        help="Directory containing per-case JSON results"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory containing dataset JSON files"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp of results to analyze (default: most recent)"
    )
    parser.add_argument(
        "--output-latex",
        type=str,
        default=None,
        help="Output path for LaTeX table"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output path for summary CSV"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Analyze results
    all_metrics = analyze_results(results_dir, dataset_dir, args.timestamp)
    
    # Print table
    print_metrics_table(all_metrics)
    
    # Generate LaTeX tables (RQ1 C/Java + RQ2 ablation)
    rq1_c, rq1_java, rq2 = generate_latex_tables(all_metrics, results_dir)
    print("\n" + "=" * 60)
    print("RQ1 Table (C Language):")
    print("=" * 60)
    print(rq1_c)
    print("\n" + "=" * 60)
    print("RQ1 Table (Java Language):")
    print("=" * 60)
    print(rq1_java)
    print("\n" + "=" * 60)
    print("RQ2 Table (Ablation):")
    print("=" * 60)
    print(rq2)
    
    # Generate summary CSV
    csv_path = Path(args.output_csv) if args.output_csv else results_dir / "rq1_summary.csv"
    generate_summary_csv(all_metrics, csv_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
