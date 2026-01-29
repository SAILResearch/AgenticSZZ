#!/usr/bin/env python3
"""
RQ0: Preliminary Study - Blameable vs Blameless Analysis

This script characterizes the SZZ problem across all datasets:
1. Analyzes blameable (has deleted lines) vs blameless (add-only) cases
2. For each ground truth BIC, determines if it's:
   - Direct in blame commits
   - Ancestor of blame commit (with depth)
   - Others (blameable but no relationship to blame)
3. Generates a single CSV output with all cases from all datasets

Output:
- rq0_all_cases.csv: All cases from all datasets (with dataset column)
- rq0_summary.csv: Aggregate statistics by dataset
"""

import json
import os
import csv
from multiprocessing import Pool, cpu_count
from collections import Counter

from agentic_szz.szz.blame import BlameAnalyzer
from agentic_szz.config.settings import DATASET_DIR, REPOS_DIR


def get_all_cases(dataset_name):
    """
    Load all test cases from a dataset

    Args:
        dataset_name: "linux", "apache", or "github"

    Returns:
        Tuple of (cases list, original_dataset_size)
        - cases: List of case dictionaries with dataset name, BFC, ground truth BICs, and repo info
        - original_dataset_size: The original size of the dataset (for LLM4SZZ consistency)
    """
    dataset_file = f"DS_{dataset_name.upper()}.json"
    dataset_path = os.path.join(DATASET_DIR, dataset_file)
    all_info_path = os.path.join(DATASET_DIR, "final_all_dataset.json")

    if not os.path.exists(dataset_path):
        print(f"Warning: {dataset_file} not found, skipping {dataset_name}")
        return [], 0

    with open(dataset_path) as f:
        test_info = json.load(f)
    with open(all_info_path) as f:
        all_info = json.load(f)

    # Original dataset size (for LLM4SZZ consistency)
    original_size = len(test_info)

    # Get set of test commit IDs (handle different field names)
    # Linux uses "fix_commit_hash", Apache uses "bug_fixing_commit"
    # GitHub also uses "fix_commit_hash"
    bfc_to_full_repo = {}
    if test_info and "fix_commit_hash" in test_info[0]:
        test_cids = {info["fix_commit_hash"] for info in test_info}
        # For GitHub, build BFC -> owner/repo mapping (repo_name in DS_GITHUB is owner/repo)
        if dataset_name.lower() == "github":
            for info in test_info:
                bfc = info["fix_commit_hash"]
                bfc_to_full_repo[bfc] = info["repo_name"]  # owner/repo format
    else:
        test_cids = {info["bug_fixing_commit"] for info in test_info}

    cases = []
    missing_repos = 0
    for info in all_info:
        repo_name = info["repo_name"]
        bfc = info["bug_fixing_commit"]

        if bfc not in test_cids:
            continue

        gt_bics = info["bug_inducing_commit"]

        # For GitHub, use full owner/repo path from DS_GITHUB.json
        if dataset_name.lower() == "github" and bfc in bfc_to_full_repo:
            full_repo_name = bfc_to_full_repo[bfc]
            repo_path = os.path.join(REPOS_DIR, full_repo_name)
        else:
            repo_path = os.path.join(REPOS_DIR, repo_name)

        # Verify repo exists
        if not os.path.exists(repo_path):
            print(f"Warning: Repo not found: {repo_path}")
            missing_repos += 1
            continue

        cases.append(
            {
                "dataset": dataset_name.upper(),
                "repo_name": repo_name,
                "repo_path": repo_path,
                "bfc": bfc,
                "ground_truth_bics": gt_bics,
            }
        )

    return cases, original_size


def analyze_single_case(case):
    """
    Analyze a single case (worker function for multiprocessing)

    Returns:
        List of dictionaries, one per ground truth BIC with detailed metrics
    """
    dataset = case["dataset"]
    repo_path = case["repo_path"]
    repo_name = case["repo_name"]
    bfc = case["bfc"]
    gt_bics = case["ground_truth_bics"]

    results = []

    try:
        analyzer = BlameAnalyzer(repo_path)
        blame_result = analyzer.analyze(bfc)

        # Case-level metrics from BlameResult
        total_deleted_lines = blame_result.total_deleted_lines
        total_added_lines = blame_result.total_added_lines
        is_blameable = blame_result.is_blameable
        num_blame_commits = len(blame_result.blame_commits)

        # Get files modified in BFC for file-history depth calculation
        modified_files = list(blame_result.files.keys())

        # Analyze each ground truth BIC
        for gt_bic in gt_bics:
            # Initialize depth fields
            ancestor_file_depth = -1  # blame → BIC (for ancestor cases)
            descendant_file_depth = -1  # BIC → BFC (for descendant cases)

            # For blameless cases, assign "blameless" category directly
            # Also compute context coverage using ±2 context lines
            if not is_blameable:
                category = "blameless"
                details = {}

                # Context fallback analysis for blameless cases
                context_category = "not_found"
                context_blame_depth = -1
                context_bfc_depth = -1

                try:
                    # Re-analyze with context fallback enabled
                    context_analyzer = BlameAnalyzer(
                        repo_path, use_context_fallback=True
                    )
                    context_result = context_analyzer.analyze(bfc)
                    pseudo_blames = context_result.blame_commits
                    context_files = (
                        list(context_result.files.keys())
                        if context_result.files
                        else []
                    )

                    if pseudo_blames:
                        # Check Direct: BIC is in pseudo-blames
                        if any(
                            gt_bic.startswith(pb[:8]) or pb.startswith(gt_bic[:8])
                            for pb in pseudo_blames
                        ):
                            context_category = "direct"
                            context_blame_depth = 0
                            context_bfc_depth = 0
                        else:
                            # Check Ancestor: BIC is ancestor of pseudo-blame
                            for pb in pseudo_blames:
                                if context_analyzer.is_ancestor(gt_bic, pb):
                                    depth = context_analyzer.get_file_history_depth(
                                        gt_bic, pb, context_files
                                    )
                                    context_category = "ancestor"
                                    context_blame_depth = depth if depth >= 0 else 100
                                    break
                            else:
                                # Check Descendant: BIC is between pseudo-blame and BFC
                                for pb in pseudo_blames:
                                    if context_analyzer.is_ancestor(pb, gt_bic):
                                        depth = context_analyzer.get_file_history_depth(
                                            pb, gt_bic, context_files
                                        )
                                        context_category = "descendant"
                                        context_bfc_depth = depth if depth >= 0 else 100
                                        break
                except Exception:
                    pass  # Keep defaults
            else:
                category, details = analyzer.compare_with_ground_truth(
                    blame_result, [gt_bic]
                )
                # Context fields not applicable for blameable cases
                context_category = ""
                context_blame_depth = -1
                context_bfc_depth = -1

                # Compute file-history depth based on category
                if category == "ancestor" and modified_files:
                    # Ancestor: blame → BIC (backward from blame)
                    ancestor_file_depth, _ = analyzer.get_min_file_history_depth(
                        blame_result.blame_commits, gt_bic, modified_files
                    )
                elif category == "descendant" and modified_files:
                    # Descendant: BIC → BFC (backward from BFC)
                    descendant_file_depth = analyzer.get_file_history_depth(
                        gt_bic, bfc, modified_files
                    )
                elif category == "direct":
                    ancestor_file_depth = 0
                    descendant_file_depth = 0

            case_result = {
                "dataset": dataset,
                "repo_name": repo_name,
                "bfc": bfc,
                "gt_bic": gt_bic,
                "is_blameable": is_blameable,
                "total_deleted_lines": total_deleted_lines,
                "total_added_lines": total_added_lines,
                "num_blame_commits": num_blame_commits,
                "category": category,
                "ancestor_depth": details.get("ancestor_depth", 0)
                if category == "ancestor"
                else 0,
                "descendant_depth": details.get("descendant_depth", 0)
                if category == "descendant"
                else 0,
                "ancestor_file_depth": ancestor_file_depth,
                "descendant_file_depth": descendant_file_depth,
                # New blameless context coverage fields
                "context_category": context_category,
                "context_blame_depth": context_blame_depth,
                "context_bfc_depth": context_bfc_depth,
                "success": True,
            }
            results.append(case_result)

    except Exception as e:
        # On error, create error result for each BIC
        for gt_bic in gt_bics:
            results.append(
                {
                    "dataset": dataset,
                    "repo_name": repo_name,
                    "bfc": bfc,
                    "gt_bic": gt_bic,
                    "is_blameable": None,
                    "total_deleted_lines": 0,
                    "total_added_lines": 0,
                    "num_blame_commits": 0,
                    "category": "error",
                    "ancestor_depth": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    return results


def analyze_all_cases(cases, workers=1):
    """
    Analyze all cases using multiprocessing

    Args:
        cases: List of case dictionaries
        workers: Number of parallel workers

    Returns:
        List of result dictionaries (one per BIC)
    """
    print(f"\nAnalyzing {len(cases)} test cases...")
    if workers > 1:
        print(f"Using {workers} parallel workers")
    print("=" * 60)

    all_results = []

    if workers > 1:
        with Pool(processes=workers) as pool:
            for idx, case_results in enumerate(
                pool.imap_unordered(analyze_single_case, cases)
            ):
                all_results.extend(case_results)
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(cases)} cases...")
    else:
        for idx, case in enumerate(cases):
            case_results = analyze_single_case(case)
            all_results.extend(case_results)
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(cases)} cases...")

    print(f"  Processed {len(cases)}/{len(cases)} cases")
    return all_results


def save_all_cases_csv(results, output_path):
    """Save all cases to a single CSV file"""
    fieldnames = [
        "dataset",
        "repo_name",
        "bfc",
        "gt_bic",
        "is_blameable",
        "total_deleted_lines",
        "total_added_lines",
        "num_blame_commits",
        "category",
        "ancestor_depth",
        "descendant_depth",
        "ancestor_file_depth",
        "descendant_file_depth",
        # Blameless context coverage fields
        "context_category",
        "context_blame_depth",
        "context_bfc_depth",
        "success",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ All cases saved to: {output_path}")


def generate_summary_stats(results, dataset_sizes=None):
    """Generate aggregate summary statistics per dataset

    Args:
        results: List of result dictionaries
        dataset_sizes: Dict mapping dataset name to original size (for LLM4SZZ consistency)
    """
    if dataset_sizes is None:
        dataset_sizes = {}

    summaries = []

    # Group by dataset - include ALL results (even errors) for total case count
    datasets = set(r["dataset"] for r in results)

    for dataset_name in sorted(datasets):
        # All results for this dataset (for total_cases count)
        all_dataset_results = [r for r in results if r["dataset"] == dataset_name]
        # Only successful results for category analysis
        dataset_results = [r for r in all_dataset_results if r["success"]]

        if not dataset_results:
            continue

        # Separate by blameable/blameless
        blameable_results = [r for r in dataset_results if r["is_blameable"]]
        blameless_results = [r for r in dataset_results if not r["is_blameable"]]

        # Count by category (now includes: direct, ancestor, descendant, others, blameless)
        category_counts = Counter(r["category"] for r in dataset_results)

        # Ancestor depth statistics
        ancestor_depths = [
            r["ancestor_depth"] for r in dataset_results if r["category"] == "ancestor"
        ]

        # NEW: Separate file-history depth stats for ancestor and descendant
        ancestor_file_depths = [
            r["ancestor_file_depth"]
            for r in dataset_results
            if r["category"] == "ancestor" and r.get("ancestor_file_depth", -1) >= 0
        ]
        descendant_file_depths = [
            r["descendant_file_depth"]
            for r in dataset_results
            if r["category"] == "descendant" and r.get("descendant_file_depth", -1) >= 0
        ]

        # Count error cases
        len(
            set(
                (r["repo_name"], r["bfc"])
                for r in all_dataset_results
                if not r["success"]
            )
        )

        # Use original dataset size if available (for LLM4SZZ consistency)
        analyzed_cases = len(
            set((r["repo_name"], r["bfc"]) for r in all_dataset_results)
        )
        original_size = dataset_sizes.get(dataset_name, analyzed_cases)

        summary = {
            "dataset": dataset_name,
            # original_dataset_size: The original LLM4SZZ dataset size for consistency
            "original_dataset_size": original_size,
            # analyzed_cases: Number of cases we could actually analyze (repos available)
            "analyzed_cases": analyzed_cases,
            "total_bics": len(dataset_results),  # Only successful BIC analyses
            "blameable_cases": len(
                set((r["repo_name"], r["bfc"]) for r in blameable_results)
            ),
            "blameless_cases": len(
                set((r["repo_name"], r["bfc"]) for r in blameless_results)
            ),
            "blameable_bics": len(blameable_results),
            "blameless_bics": len(blameless_results),
            # Category counts - now 5 categories
            "direct_bics": category_counts.get("direct", 0),
            "ancestor_bics": category_counts.get("ancestor", 0),
            "descendant_bics": category_counts.get("descendant", 0),
            "others_bics": category_counts.get("others", 0),  # renamed from unrelated
            # Statistics
            "avg_deleted_lines": sum(
                r["total_deleted_lines"] for r in blameable_results
            )
            / len(blameable_results)
            if blameable_results
            else 0,
            "avg_added_lines": sum(r["total_added_lines"] for r in blameless_results)
            / len(blameless_results)
            if blameless_results
            else 0,
            "avg_blame_commits": sum(r["num_blame_commits"] for r in blameable_results)
            / len(blameable_results)
            if blameable_results
            else 0,
            "avg_ancestor_depth": sum(ancestor_depths) / len(ancestor_depths)
            if ancestor_depths
            else 0,
            "median_ancestor_depth": sorted(ancestor_depths)[len(ancestor_depths) // 2]
            if ancestor_depths
            else 0,
            "max_ancestor_depth": max(ancestor_depths) if ancestor_depths else 0,
            # NEW: Ancestor file-history depth (blame → BIC)
            "avg_ancestor_file_depth": sum(ancestor_file_depths)
            / len(ancestor_file_depths)
            if ancestor_file_depths
            else 0,
            "median_ancestor_file_depth": sorted(ancestor_file_depths)[
                len(ancestor_file_depths) // 2
            ]
            if ancestor_file_depths
            else 0,
            "max_ancestor_file_depth": max(ancestor_file_depths)
            if ancestor_file_depths
            else 0,
            # NEW: Descendant file-history depth (BIC → BFC)
            "avg_descendant_file_depth": sum(descendant_file_depths)
            / len(descendant_file_depths)
            if descendant_file_depths
            else 0,
            "median_descendant_file_depth": sorted(descendant_file_depths)[
                len(descendant_file_depths) // 2
            ]
            if descendant_file_depths
            else 0,
            "max_descendant_file_depth": max(descendant_file_depths)
            if descendant_file_depths
            else 0,
        }
        summaries.append(summary)

    return summaries


def save_summary_csv(summaries, output_path):
    """Save summary statistics to CSV"""
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"✓ Summary statistics saved to: {output_path}")


def print_summary(summaries):
    """Print summary statistics"""
    print(f"\n{'=' * 60}")
    print("RQ0 SUMMARY - ALL DATASETS")
    print(f"{'=' * 60}\n")

    for summary in summaries:
        dataset = summary["dataset"]
        original_size = summary["original_dataset_size"]
        analyzed = summary["analyzed_cases"]
        total_bics = summary["total_bics"]

        print(f"{dataset} Dataset:")
        print(f"  Total test cases: {original_size} (analyzed: {analyzed})")
        print(
            f"    Blameable: {summary['blameable_cases']} ({100 * summary['blameable_cases'] / analyzed:.1f}%)"
        )
        print(
            f"    Blameless: {summary['blameless_cases']} ({100 * summary['blameless_cases'] / analyzed:.1f}%)"
        )

        print(f"\n  Ground truth BIC distribution ({total_bics} BICs):")
        print(
            f"    Direct:     {summary['direct_bics']:4d} ({100 * summary['direct_bics'] / total_bics:5.1f}%)"
        )
        print(
            f"    Ancestor:   {summary['ancestor_bics']:4d} ({100 * summary['ancestor_bics'] / total_bics:5.1f}%)"
        )
        print(
            f"    Descendant: {summary['descendant_bics']:4d} ({100 * summary['descendant_bics'] / total_bics:5.1f}%)"
        )
        print(
            f"    Others:     {summary['others_bics']:4d} ({100 * summary['others_bics'] / total_bics:5.1f}%)"
        )
        print(
            f"    Blameless:  {summary['blameless_bics']:4d} ({100 * summary['blameless_bics'] / total_bics:5.1f}%)"
        )

        if summary["ancestor_bics"] > 0 or summary["descendant_bics"] > 0:
            print(
                f"\n  DAG depth (ancestor): avg={summary['avg_ancestor_depth']:.1f}, median={summary['median_ancestor_depth']}, max={summary['max_ancestor_depth']}"
            )
            if summary["ancestor_bics"] > 0:
                print(
                    f"  Ancestor file-depth (blame→BIC): avg={summary['avg_ancestor_file_depth']:.1f}, median={summary['median_ancestor_file_depth']}, max={summary['max_ancestor_file_depth']}"
                )
            if summary["descendant_bics"] > 0:
                print(
                    f"  Descendant file-depth (BIC→BFC): avg={summary['avg_descendant_file_depth']:.1f}, median={summary['median_descendant_file_depth']}, max={summary['max_descendant_file_depth']}"
                )

        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RQ0: Blameable vs Blameless Analysis across all datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["linux", "apache", "github"],
        choices=["linux", "apache", "github"],
        help="Datasets to analyze (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases per dataset (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = all CPUs, default)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rq0",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine workers
    if args.workers <= 0:
        workers = cpu_count()
        print(f"Using all {workers} CPU cores")
    else:
        workers = args.workers
        print(f"Using {workers} worker(s)")

    # Load all datasets
    all_cases = []
    dataset_sizes = {}  # Track original sizes for LLM4SZZ consistency
    for dataset_name in args.datasets:
        print(f"Loading {dataset_name.upper()} dataset...")
        cases, original_size = get_all_cases(dataset_name)
        dataset_sizes[dataset_name.upper()] = original_size

        if args.limit:
            cases = cases[: args.limit]
            print(f"  Limiting to first {args.limit} cases")

        print(f"  Found {len(cases)} test cases (original dataset: {original_size})")
        all_cases.extend(cases)

    if not all_cases:
        print("No cases found! Check repo paths.")
        return

    print(f"\nTotal cases across all datasets: {len(all_cases)}")

    # Analyze all cases
    results = analyze_all_cases(all_cases, workers=workers)

    # Generate outputs
    print("\nGenerating CSV outputs...")

    # Single combined CSV with all cases
    all_cases_path = os.path.join(args.output_dir, "rq0_all_cases.csv")
    save_all_cases_csv(results, all_cases_path)

    # Summary statistics by dataset
    summaries = generate_summary_stats(results, dataset_sizes)
    summary_path = os.path.join(args.output_dir, "rq0_summary.csv")
    save_summary_csv(summaries, summary_path)

    # Print summary
    print_summary(summaries)


if __name__ == "__main__":
    main()
