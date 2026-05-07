#!/usr/bin/env python3
"""
Run LLM4SZZ baseline on pilot cases with configurable LLM backend.

This script validates the "LLM Selection" threat to validity by running LLM4SZZ
(our baseline) with DeepSeek instead of GPT-4o-mini on the same pilot cases.

Supports parallel execution by splitting cases into worker batches.

Usage:
    # Run with DeepSeek, 4 workers (default)
    python evaluation/run_llm4szz_baseline.py --model deepseek-chat --workers 4

    # Run with GPT-4o-mini, single worker
    python evaluation/run_llm4szz_baseline.py --model gpt-4o-mini --workers 1

    # Skip run, only compute metrics from existing logs
    python evaluation/run_llm4szz_baseline.py --skip-run

Results saved to: results/llm4szz_deepseek_pilot/
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LLM4SZZ_DIR = PROJECT_ROOT / "vendor" / "llmszz-replication"
DATASET_DIR = PROJECT_ROOT / "dataset"
REPOS_DIR = DATASET_DIR / "subject_repos"

# Load .env file for API keys
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # dotenv not installed, assume env vars are set externally
    pass


# =============================================================================
# Data Loading (reuses same datasets as run_evaluation.py)
# =============================================================================


def load_all_cases() -> List[dict]:
    """Load all cases from final_all_dataset.json (LLM4SZZ format).
    
    This uses the same dataset format as LLM4SZZ expects, with short repo names.
    The repos should be cloned with git_clone_subject_repos.py which creates
    appropriate symlinks for path compatibility.
    """
    dataset_path = DATASET_DIR / "final_all_dataset.json"
    with open(dataset_path) as f:
        all_data = json.load(f)
    
    # Build repo-to-dataset mappings from individual dataset files
    linux_repos = set()
    apache_repos = set()
    github_repos = set()
    
    linux_path = DATASET_DIR / "DS_LINUX.json"
    apache_path = DATASET_DIR / "DS_APACHE.json"
    github_path = DATASET_DIR / "DS_GITHUB.json"
    
    if linux_path.exists():
        with open(linux_path) as f:
            for item in json.load(f):
                # Linux uses "torvalds/linux" but final_all_dataset uses "linux"
                repo = item["repo_name"]
                if "/" in repo:
                    repo = repo.split("/")[-1]
                linux_repos.add(repo)
    
    if apache_path.exists():
        with open(apache_path) as f:
            for item in json.load(f):
                repo = item["repo_name"]
                if "/" in repo:
                    repo = repo.split("/")[-1]
                apache_repos.add(repo)
    
    if github_path.exists():
        with open(github_path) as f:
            for item in json.load(f):
                repo = item["repo_name"]
                if "/" in repo:
                    repo = repo.split("/")[-1]
                github_repos.add(repo)
    
    all_cases = []
    for item in all_data:
        bfc = item["bug_fixing_commit"]
        bic = item["bug_inducing_commit"]
        if isinstance(bic, list):
            bic = bic[0] if bic else ""
        
        # Determine dataset based on repo name
        repo = item["repo_name"]
        repo_short = repo.split("/")[-1] if "/" in repo else repo
        
        if repo_short in linux_repos:
            dataset = "LINUX"
        elif repo_short in apache_repos:
            dataset = "APACHE"
        else:
            dataset = "GITHUB"  # Default to GITHUB for all others (including github_repos)
        
        all_cases.append({
            "repo_name": repo,
            "bfc": bfc,
            "gt_bic": bic,
            "dataset": dataset,
        })
    
    print(f"Loaded {len(all_cases)} total cases from final_all_dataset.json")
    return all_cases


def load_pilot_cases(sample_file: str) -> List[dict]:
    """Load pilot cases from stratified sample file."""
    sample_path = PROJECT_ROOT / sample_file
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    with open(sample_path) as f:
        data = json.load(f)
    return data.get("cases", [])


def load_ground_truth_dataset() -> Dict[str, List[str]]:
    """
    Load ground truth BICs from final_all_dataset.json.

    Returns:
        Dict mapping BFC -> list of ground truth BICs
    """
    dataset_path = DATASET_DIR / "final_all_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        all_data = json.load(f)

    bfc_to_bics: Dict[str, List[str]] = {}
    for item in all_data:
        bfc = item.get("bug_fixing_commit")
        bic = item.get("bug_inducing_commit")
        if not bfc or not bic:
            continue

        if isinstance(bic, list):
            bics = [b for b in bic if b and isinstance(b, str)]
        elif isinstance(bic, str):
            bics = [bic]
        else:
            continue

        if bics:
            if bfc not in bfc_to_bics:
                bfc_to_bics[bfc] = bics
            else:
                bfc_to_bics[bfc].extend(b for b in bics if b not in bfc_to_bics[bfc])

    return bfc_to_bics


def load_full_llm4szz_dataset() -> List[dict]:
    """Load full LLM4SZZ dataset for format reference."""
    dataset_path = LLM4SZZ_DIR / "dataset" / "final_all_dataset.json"
    with open(dataset_path) as f:
        return json.load(f)


# =============================================================================
# Dataset Filtering for Pilot Cases
# =============================================================================


def create_filtered_dataset(pilot_cases: List[dict], output_path: Path) -> Tuple[List[dict], int]:
    """
    Create filtered dataset matching LLM4SZZ format for pilot cases only.
    """
    full_dataset = load_full_llm4szz_dataset()

    pilot_lookup: Dict[Tuple[str, str], dict] = {}
    for case in pilot_cases:
        repo = case["repo_name"]
        bfc = case["bfc"]
        key = (repo, bfc)
        pilot_lookup[key] = case

    filtered = []
    matched_bfcs = set()

    for item in full_dataset:
        repo = item["repo_name"]
        bfc = item.get("bug_fixing_commit") or item.get("fix_commit_hash", "")

        for (pilot_repo, pilot_bfc), _ in pilot_lookup.items():
            if repo == pilot_repo and _sha_prefix_match(bfc, pilot_bfc):
                if bfc not in matched_bfcs:
                    filtered.append(item)
                    matched_bfcs.add(bfc)
                break

    print(f"Matched {len(filtered)}/{len(pilot_cases)} pilot cases in LLM4SZZ dataset")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    return filtered, len(filtered)


def _sha_prefix_match(sha1: str, sha2: str) -> bool:
    """Check if two SHAs match via prefix comparison."""
    if not sha1 or not sha2:
        return False
    return sha1.startswith(sha2) or sha2.startswith(sha1)


# =============================================================================
# Parallel LLM4SZZ Execution
# =============================================================================


def run_worker(worker_id: int, worker_dir: Path, model: str, num_runs: int = 3) -> Tuple[int, bool, str]:
    """
    Run LLM4SZZ for a single worker's batch.

    Each worker has its own dataset directory to avoid conflicts.
    """
    env = os.environ.copy()

    # LLM configuration
    env["LLM_MODEL"] = model
    if "deepseek" in model.lower():
        env["LLM_BASE_URL"] = "https://api.deepseek.com"
    else:
        env["LLM_BASE_URL"] = "https://api.openai.com/v1"

    # Path configuration - each worker gets its own directories
    env["LLM4SZZ_CWD"] = str(LLM4SZZ_DIR)
    env["LLM4SZZ_REPOS_DIR"] = str(REPOS_DIR)
    env["LLM4SZZ_DATASET_DIR"] = str(worker_dir)
    env["LLM4SZZ_SAVE_LOG_DIR"] = str(worker_dir / "save_logs")
    
    # Number of runs per case
    env["LLM4SZZ_NUM_RUNS"] = str(num_runs)

    # Create save_logs directory
    (worker_dir / "save_logs").mkdir(parents=True, exist_ok=True)

    # Run llm4szz.py
    log_file = worker_dir / "worker.log"
    with open(log_file, "w") as f:
        result = subprocess.run(
            [sys.executable, "llm4szz.py"],
            cwd=LLM4SZZ_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    return worker_id, result.returncode == 0, str(log_file)


def count_completed_cases(output_dir: Path) -> int:
    """Count completed cases by checking save_logs directories."""
    total = 0
    for worker_dir in output_dir.iterdir():
        if worker_dir.is_dir() and worker_dir.name.startswith("worker_"):
            save_logs = worker_dir / "save_logs"
            if save_logs.exists():
                for repo_dir in save_logs.iterdir():
                    if repo_dir.is_dir():
                        total += len(list(repo_dir.iterdir()))
    return total


def run_llm4szz_parallel(
    model: str, filtered_dataset: List[dict], output_dir: Path, num_workers: int, num_runs: int = 3
) -> bool:
    """
    Run LLM4SZZ in parallel by splitting cases across workers.

    Each worker gets a subset of cases in its own directory.
    """
    print(f"\n{'=' * 60}")
    print(f"Running LLM4SZZ with {model} ({num_workers} workers, {num_runs} run(s) per case)")
    print(f"Total cases: {len(filtered_dataset)}")
    print(f"{'=' * 60}\n")

    # Split cases into worker batches
    batch_size = (len(filtered_dataset) + num_workers - 1) // num_workers
    batches = [
        filtered_dataset[i : i + batch_size] for i in range(0, len(filtered_dataset), batch_size)
    ]

    # Create worker directories and datasets
    worker_dirs = []
    for i, batch in enumerate(batches):
        worker_dir = output_dir / f"worker_{i}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        # Write worker's dataset
        dataset_file = worker_dir / "final_all_dataset.json"
        with open(dataset_file, "w") as f:
            json.dump(batch, f, indent=2)

        worker_dirs.append(worker_dir)
        print(f"  Worker {i}: {len(batch)} cases -> {worker_dir}")

    # Run workers in parallel with progress monitoring
    print(f"\nStarting {len(worker_dirs)} workers...")
    print(f"Progress will be shown every 30 seconds...\n")
    all_success = True
    total_cases = len(filtered_dataset)
    start_time = datetime.now()
    workers_done = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_worker, i, worker_dir, model, num_runs): i
            for i, worker_dir in enumerate(worker_dirs)
        }
        
        # Monitor progress while waiting for workers
        import time
        while workers_done < len(futures):
            # Check for completed futures
            done_futures = [f for f in futures if f.done()]
            
            for future in done_futures:
                if future not in [None]:
                    try:
                        worker_id, success, log_file = future.result()
                        status = "✓" if success else "✗"
                        print(f"  Worker {worker_id} finished {status} (log: {log_file})")
                        if not success:
                            all_success = False
                        workers_done += 1
                        futures[future] = None  # Mark as processed
                    except Exception as e:
                        print(f"  Worker error: {e}")
                        workers_done += 1
                        all_success = False
            
            # Show progress every 30 seconds if still running
            if workers_done < len(futures):
                completed = count_completed_cases(output_dir)
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{elapsed/60:.1f}min] Progress: {completed}/{total_cases} cases ({completed*100/total_cases:.1f}%) - {rate:.1f} cases/min")
                time.sleep(30)

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"\nAll workers finished in {elapsed_total/60:.1f} minutes")

    # Merge save_logs from all workers into main output directory
    print("\nMerging worker logs...")
    main_save_logs = output_dir / "save_logs"
    main_save_logs.mkdir(parents=True, exist_ok=True)

    for worker_dir in worker_dirs:
        worker_logs = worker_dir / "save_logs"
        if worker_logs.exists():
            for repo_dir in worker_logs.iterdir():
                if repo_dir.is_dir():
                    dest_repo = main_save_logs / repo_dir.name
                    if not dest_repo.exists():
                        shutil.copytree(repo_dir, dest_repo)
                    else:
                        # Merge BFC directories
                        for bfc_dir in repo_dir.iterdir():
                            if bfc_dir.is_dir():
                                dest_bfc = dest_repo / bfc_dir.name
                                if not dest_bfc.exists():
                                    shutil.copytree(bfc_dir, dest_bfc)

    print(f"Merged logs to {main_save_logs}")
    return all_success


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_metrics(
    pilot_cases: List[dict], ground_truth: Dict[str, List[str]], output_dir: Path
) -> dict:
    """
    Compute metrics from LLM4SZZ results by parsing save_logs.
    """
    save_logs_dir = output_dir / "save_logs"

    results = []
    dataset_stats = {
        "LINUX": {"find_cnt": 0, "true_cnt": 0, "correct_cnt": 0, "cases": 0},
        "APACHE": {"find_cnt": 0, "true_cnt": 0, "correct_cnt": 0, "cases": 0},
        "GITHUB": {"find_cnt": 0, "true_cnt": 0, "correct_cnt": 0, "cases": 0},
    }

    for case in pilot_cases:
        repo = case["repo_name"]
        bfc = case["bfc"]
        dataset = case["dataset"]

        gt_bics = _lookup_ground_truth_bics(bfc, ground_truth, case.get("gt_bic"))
        predicted, log_found, error = _find_prediction_in_logs(save_logs_dir, repo, bfc)

        dataset_stats[dataset]["cases"] += 1

        if not log_found:
            results.append(
                {
                    "dataset": dataset,
                    "repo": repo,
                    "bfc": bfc,
                    "gt_bics": gt_bics,
                    "predicted": None,
                    "correct": False,
                    "error": error,
                }
            )
            # Even if log not found, we count the missed TP (true_cnt increases)
            # effectively recall decreases, precision undefined/0 for this case
            dataset_stats[dataset]["true_cnt"] += len(gt_bics)
            continue

        # Count matches
        find_cnt = len(predicted)
        true_cnt = len(gt_bics)
        correct_cnt = 0

        if predicted and gt_bics:
            # To avoid double counting, we use sets for matching
            matched_predicted_indices = set()
            matched_gt_indices = set()

            for i, pred in enumerate(predicted):
                for j, gt in enumerate(gt_bics):
                    if _sha_prefix_match(pred, gt) and j not in matched_gt_indices:
                        correct_cnt += 1
                        matched_predicted_indices.add(i)
                        matched_gt_indices.add(j)
                        break  # Move to the next predicted BIC

        # Update stats
        dataset_stats[dataset]["find_cnt"] += find_cnt
        dataset_stats[dataset]["true_cnt"] += true_cnt
        dataset_stats[dataset]["correct_cnt"] += correct_cnt

        results.append(
            {
                "dataset": dataset,
                "repo": repo,
                "bfc": bfc,
                "gt_bics": gt_bics,
                "predicted": predicted,
                "correct_cnt": correct_cnt,
                "find_cnt": find_cnt,
                "true_cnt": true_cnt,
            }
        )

    # Print Report
    print(f"\n{'=' * 60}")
    print(f"LLM4SZZ Results ({os.path.basename(output_dir)})")
    print(f"{'=' * 60}")

    total_find_cnt = sum(s["find_cnt"] for s in dataset_stats.values())
    total_true_cnt = sum(s["true_cnt"] for s in dataset_stats.values())
    total_correct_cnt = sum(s["correct_cnt"] for s in dataset_stats.values())

    # Global Metrics
    precision = total_correct_cnt / total_find_cnt if total_find_cnt > 0 else 0
    recall = total_correct_cnt / total_true_cnt if total_true_cnt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Overall (Micro-Average):")
    print(f"  Precision: {precision:.4f} ({total_correct_cnt}/{total_find_cnt})")
    print(f"  Recall:    {recall:.4f} ({total_correct_cnt}/{total_true_cnt})")
    print(f"  F1 Score:  {f1:.4f}")

    print("\nBy Dataset:")
    for ds_name, stats in dataset_stats.items():
        p = stats["correct_cnt"] / stats["find_cnt"] if stats["find_cnt"] > 0 else 0
        r = stats["correct_cnt"] / stats["true_cnt"] if stats["true_cnt"] > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {ds_name} (n={stats['cases']}): P={p:.4f}, R={r:.4f}, F1={f:.4f}")

    # Saves results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"llm4szz_metrics_{timestamp}.json"

    output_data = {
        "summary": {
            "overall": {"precision": precision, "recall": recall, "f1": f1},
            "by_dataset": dataset_stats,
        },
        "details": results,
    }

    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {out_file}")
    return output_data


def _lookup_ground_truth_bics(
    bfc: str, ground_truth: Dict[str, List[str]], fallback_bic: str
) -> List[str]:
    """Lookup ALL ground truth BICs for a BFC."""
    if bfc in ground_truth:
        return ground_truth[bfc]

    for dataset_bfc, bics in ground_truth.items():
        if _sha_prefix_match(bfc, dataset_bfc):
            return bics

    return [fallback_bic] if fallback_bic else []


def _find_prediction_in_logs(
    save_logs_dir: Path, repo: str, bfc: str
) -> Tuple[Optional[List[str]], bool, Optional[str]]:
    """Find LLM4SZZ prediction from save_logs directory."""
    if not save_logs_dir.exists():
        return None, False, "save_logs directory not found"

    repo_dir = save_logs_dir / repo
    if not repo_dir.exists():
        for d in save_logs_dir.iterdir():
            if d.is_dir() and d.name.lower() == repo.lower():
                repo_dir = d
                break
        else:
            return None, False, f"repo directory not found: {repo}"

    for bfc_dir in repo_dir.iterdir():
        if not bfc_dir.is_dir():
            continue
        if _sha_prefix_match(bfc_dir.name, bfc):
            log_file = bfc_dir / "llm4szz0.json"
            if log_file.exists():
                predicted = _extract_prediction_from_log(log_file)
                return predicted, True, None
            else:
                return None, False, f"log file not found in {bfc_dir}"

    return None, False, f"BFC directory not found for {bfc}"


def _extract_prediction_from_log(log_file: Path) -> List[str]:
    """Extract predicted BIC(s) from LLM4SZZ log file."""
    with open(log_file) as f:
        logs = json.load(f)

    s2_final_cids = []
    s1_final_cids = []
    can_determine = True

    for log in logs:
        if isinstance(log, dict):
            if "can_determine" in log:
                can_determine = can_determine and log["can_determine"]
            if "s2_final_cid" in log:
                s2_final_cids.extend(log["s2_final_cid"])
            if "s1_ranked_stmts_infos" in log:
                for info in log["s1_ranked_stmts_infos"][:5]:
                    if "induce_cid" in info:
                        s1_final_cids.append(info["induce_cid"])

    if can_determine and s2_final_cids:
        return list(set(s2_final_cids))
    return list(set(s1_final_cids))


def _check_match_any(predicted: Optional[List[str]], gt_bics: List[str]) -> bool:
    """Check if any predicted BIC matches any ground truth BIC."""
    if not predicted or not gt_bics:
        return False
    for pred in predicted:
        for gt in gt_bics:
            if pred and gt and _sha_prefix_match(pred, gt):
                return True
    return False


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM4SZZ baseline with configurable LLM for threat validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        choices=["deepseek-chat", "deepseek-v4-pro", "gpt-4o-mini"],
        help="LLM model to use (default: deepseek-chat)",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default="results/pilot/pilot_cases_20pct.json",
        help="Path to pilot cases JSON (relative to project root)",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Run on full dataset (all 2083 cases) instead of sample file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/llm4szz_deepseek_pilot",
        help="Output directory for results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per case (default: 3, use 1 for faster execution)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running LLM4SZZ, only compute metrics from existing logs",
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cases (either full dataset or sample file)
    if args.full_dataset:
        print("Loading full dataset (all cases)...")
        pilot_cases = load_all_cases()
    else:
        print(f"Loading pilot cases from {args.sample_file}...")
        pilot_cases = load_pilot_cases(args.sample_file)
    print(f"Loaded {len(pilot_cases)} cases")

    # Load ground truth
    print("Loading ground truth dataset...")
    ground_truth = load_ground_truth_dataset()
    print(f"Loaded ground truth for {len(ground_truth)} BFCs")

    # Create filtered dataset
    filtered_path = output_dir / "pilot_dataset.json"
    filtered, matched = create_filtered_dataset(pilot_cases, filtered_path)

    if not args.skip_run:
        # Run LLM4SZZ in parallel
        success = run_llm4szz_parallel(args.model, filtered, output_dir, args.workers, args.runs)
        if not success:
            print("WARNING: Some workers may have had errors")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(pilot_cases, ground_truth, output_dir)

    # Print results
    print(f"\n{'=' * 60}")
    # Simple summary for exit
    summary = metrics["summary"]["overall"]
    print(
        f"\nFinal Summary: P={summary['precision']:.4f}, R={summary['recall']:.4f}, F1={summary['f1']:.4f}"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("-", "_")
    results_file = output_dir / f"llm4szz_{model_tag}_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "sample_file": args.sample_file,
                "workers": args.workers,
                "timestamp": timestamp,
                "metrics": metrics["summary"],
                "per_case_results": metrics["details"],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
