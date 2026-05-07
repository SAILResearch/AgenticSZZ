#!/usr/bin/env python3
"""
AgenticSZZ Evaluation Script

Evaluates AgenticSZZ on Linux, Apache, and GitHub datasets.
Supports configurations for RQ1 (effectiveness) and RQ2 (ablation):
- blame_only: Pure SZZ blame (no fallback for blameless cases)
- blame_fallback: Blame + context fallback for blameless cases
- tkg_only: Build TKG, return top candidate (RQ2)
- agent_only: Agent on blame candidates without TKG (RQ2)
- full: TKG + Agent (RQ1 effectiveness, RQ2)

Usage:
    # RQ1: Full pipeline effectiveness
    python evaluation/run_evaluation.py --dataset linux --config full --output-dir results/rq1
    
    # RQ2: Ablation study (run each separately)
    python evaluation/run_evaluation.py --dataset linux --config blame_only --output-dir results/rq2
    python evaluation/run_evaluation.py --dataset linux --config blame_fallback --output-dir results/rq2
    python evaluation/run_evaluation.py --dataset linux --config tkg_only --output-dir results/rq2
    python evaluation/run_evaluation.py --dataset linux --config agent_only --output-dir results/rq2
    python evaluation/run_evaluation.py --dataset linux --config full --output-dir results/rq2
"""

import argparse
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from agentic_szz import AgenticSZZ
from agentic_szz.kg import reset_neo4j_database
from agentic_szz.kg.tkg import SuspiciousCommitTKG
import asyncio

# Suppress noisy warnings from Graphiti, Neo4j driver, and asyncio
logging.getLogger("graphiti_core").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)  # Suppress "property key does not exist" warnings
logging.getLogger("neo4j.notifications").setLevel(logging.CRITICAL)  # Extra suppression
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

# Neo4j driver prints "Received notification from DBMS server" directly to stderr
# Filter these at the output level since they bypass logging
class StderrNoiseFilter:
    """Filter that suppresses noisy stderr messages from Neo4j and httpx.
    
    Suppresses:
    - Neo4j DBMS notification messages
    - httpx "Event loop is closed" errors from async client cleanup
    - "Task exception was never retrieved" from asyncio
    """
    # Patterns to suppress (checked with 'in' for efficiency)
    SUPPRESS_PATTERNS = [
        "Received notification from DBMS server",
        "GqlStatusObject",
        "property key does not exist",
        "Event loop is closed",
        "Task exception was never retrieved",
        "AsyncClient.aclose()",
        "RuntimeError: Event loop is closed",
    ]
    
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self._suppressing_traceback = False
    
    def write(self, data):
        # Buffer partial lines
        self.buffer += data
        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            
            # Check if this line starts a suppressed traceback
            if any(pattern in line for pattern in self.SUPPRESS_PATTERNS):
                self._suppressing_traceback = True
                continue
            
            # Continue suppressing if we're in a traceback block
            if self._suppressing_traceback:
                # Tracebacks have indented lines or start with "  File " or "Traceback"
                if line.startswith("  ") or line.startswith("Traceback") or not line.strip():
                    continue
                else:
                    # Non-indented, non-empty line - stop suppressing
                    self._suppressing_traceback = False
            
            self.stream.write(line + "\n")
    
    def flush(self):
        if self.buffer and not any(p in self.buffer for p in self.SUPPRESS_PATTERNS):
            self.stream.write(self.buffer)
        self.buffer = ""
        self.stream.flush()

import sys
sys.stderr = StderrNoiseFilter(sys.__stderr__)

# Project root for dataset paths
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Worker Database Management
# =============================================================================

def get_worker_db_config(worker_id: int) -> Dict[str, any]:
    """
    Get Neo4j configuration for a specific worker.
    
    Each worker gets its own ports to avoid conflicts with main Neo4j (7687/7474).
    Uses consecutive ports per worker to support up to 64 workers without overlap:
    - Worker 0: bolt=7700, http=7701
    - Worker 1: bolt=7702, http=7703
    - Worker 2: bolt=7704, http=7705
    ...
    - Worker 63: bolt=7826, http=7827
    
    Args:
        worker_id: Worker index (0, 1, 2, ...)
        
    Returns:
        Dict with uri, user, password, and container info
    """
    # Each worker uses 2 consecutive ports (bolt, http)
    # Starting at 7700 to avoid conflict with main Neo4j on 7687/7474
    base_port = 7700 + (worker_id * 2)
    bolt_port = base_port
    http_port = base_port + 1
    
    return {
        "uri": f"bolt://localhost:{bolt_port}",
        "user": "neo4j",
        "password": "password",
        "container_name": f"neo4j-worker-{worker_id}",
        "bolt_port": bolt_port,
        "http_port": http_port,
    }


def start_worker_database(worker_id: int) -> Dict[str, any]:
    """
    Start isolated Neo4j container for worker.
    
    Creates a fresh Neo4j instance in Docker with unique ports.
    Waits up to 30 seconds for the database to be ready.
    
    Args:
        worker_id: Worker index
        
    Returns:
        Dict with connection config (uri, user, password)
        
    Raises:
        RuntimeError: If container fails to start or become ready
    """
    import subprocess
    
    config = get_worker_db_config(worker_id)
    
    # Check if container already exists (cleanup from previous run)
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{config['container_name']}$", 
         "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    
    if config['container_name'] in result.stdout.strip():
        # Container exists, remove it and its volumes for clean state
        subprocess.run(
            ["docker", "rm", "-f", "-v", config['container_name']], 
            capture_output=True,
        )
    
    # Start new container
    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", config['container_name'],
                "-p", f"{config['bolt_port']}:7687",
                "-p", f"{config['http_port']}:7474",
                "-e", f"NEO4J_AUTH={config['user']}/{config['password']}",
                "neo4j:latest",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to start Neo4j container for worker {worker_id}: {e.stderr.decode()}"
        )
    
    # Wait for Neo4j to be ready using HTTP health check
    # Port check alone is insufficient - Neo4j opens port before it's ready
    print(f"  Worker {worker_id}: Starting Neo4j on ports bolt={config['bolt_port']}, http={config['http_port']}")
    
    # Give Neo4j time to start up before checking
    time.sleep(3)
    
    import urllib.request
    import urllib.error
    
    health_url = f"http://localhost:{config['http_port']}"
    
    for attempt in range(45):  # Up to 45 seconds total
        try:
            req = urllib.request.urlopen(health_url, timeout=2)
            if req.status == 200:
                print(f"  Worker {worker_id}: Neo4j ready (took {attempt + 4}s)")
                return config
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        
        time.sleep(1)
    
    # Failed to start within timeout
    stop_worker_database(worker_id)  # Cleanup
    raise RuntimeError(f"Worker {worker_id} Neo4j failed to become ready within 45s")


def stop_worker_database(worker_id: int):
    """
    Stop and remove worker's Neo4j container and its volumes.
    
    Forcefully removes the container and associated volumes.
    Safe to call even if container doesn't exist.
    
    Args:
        worker_id: Worker index
    """
    import subprocess
    
    config = get_worker_db_config(worker_id)
    
    # Force remove container AND its volumes (-v flag prevents orphaned volumes)
    subprocess.run(
        ["docker", "rm", "-f", "-v", config['container_name']],
        capture_output=True,
    )
    
    print(f"  Worker {worker_id}: Stopped Neo4j container")


# =============================================================================
# Evaluation Data Structures  
# =============================================================================


@dataclass
class EvalCase:
    """Single evaluation case"""

    bfc: str
    ground_truth_bics: List[str]  # All ground truth BICs (handles multi-BIC cases)
    repo_path: str
    dataset: str
    case_id: str = ""  # Format: DATASET_BFC (e.g., LINUX_abc123...) for deterministic tracking
    project: str = ""


@dataclass
class EvalResult:
    """Result for single case"""

    case: EvalCase
    predicted_bic: str
    is_correct: bool
    confidence: float
    time_ms: float
    config: str  # "blame_only", "blame_fallback", "tkg_only", "agent_only", "full"
    # Per-stage timing
    blame_time_ms: float = 0.0
    tkg_time_ms: float = 0.0
    agent_time_ms: float = 0.0
    # Token/Cost tracking (RQ3)
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    agent_steps: int = 0


@dataclass
class DatasetMetrics:
    """Aggregated metrics for a dataset"""

    dataset: str
    config: str
    total_cases: int
    correct: int
    precision: float
    recall: float
    f1: float
    avg_time_ms: float
    # Token/Cost aggregates (RQ3)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    avg_cost_per_case: float = 0.0


KNOWN_MISSING_REPOS = {
    "tmatsuya/linux-2.6",
    "Zeex/sampgdk",
    "mesa3d/mesa",
}

def load_sample_cases(sample_file: str) -> Dict[str, List[EvalCase]]:
    """Load cases from stratified sample file (from create_pilot_sample.py).
    
    Looks up ALL ground truth BICs from original dataset (not just the one in sample).
    
    Returns:
        Dict mapping dataset name to list of EvalCase objects
    """
    sample_path = PROJECT_ROOT / sample_file
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    with open(sample_path) as f:
        data = json.load(f)
    
    # Load ground truth BICs from final_all_dataset.json (same as RQ0)
    # Use set to handle: (1) duplicate BICs within a case, (2) duplicate BFC entries
    # This ensures counts match LLM4SZZ paper: Linux 1500/1562, Apache 241/351, GitHub 361/362
    all_info_path = PROJECT_ROOT / "dataset" / "final_all_dataset.json"
    bfc_to_bics = {}  # BFC -> set of unique BICs
    
    if all_info_path.exists():
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
                        bfc_to_bics[bfc].update(bics)  # Merge for duplicate BFC entries
    
    # Build BFC -> full repo path mappings from DS files (same as RQ0)
    # GitHub uses owner/repo format from DS_GITHUB.json
    github_bfc_to_repo = {}
    github_path = PROJECT_ROOT / "dataset" / "DS_GITHUB.json"
    if github_path.exists():
        with open(github_path) as f:
            github_ds = json.load(f)
        for item in github_ds:
            bfc = item.get("fix_commit_hash")
            repo = item.get("repo_name")  # owner/repo format
            if bfc and repo:
                github_bfc_to_repo[bfc] = repo
    
    cases_by_dataset = {}
    repos_dir = PROJECT_ROOT / "dataset" / "subject_repos"
    
    for item in data["cases"]:
        dataset = item["dataset"].lower()
        repo_name = item["repo_name"]
        bfc = item["bfc"]
        
        # Get ALL BICs from final_all_dataset (convert set to list), fallback to sample's gt_bic
        bics_set = bfc_to_bics.get(bfc)
        all_bics = list(bics_set) if bics_set else [item["gt_bic"]]
        
        # Determine repo path (same logic as RQ0)
        if dataset == "linux":
            repo_path = str(repos_dir / "linux")
        elif dataset == "github":
            # Use full owner/repo path from DS_GITHUB.json
            full_repo = github_bfc_to_repo.get(bfc, repo_name)
            repo_path = str(repos_dir / full_repo)
        else:
            # Apache uses short repo names
            repo_path = str(repos_dir / repo_name)
        
        # Verify repo exists (has .git)
        if not Path(repo_path, ".git").exists():
            # Check if it's a known missing repo to avoid noisy warnings
            is_known = any(known in repo_path for known in KNOWN_MISSING_REPOS)
            if is_known:
                continue
            else:
                print(f"WARNING: Repo missing unexpectedly: {repo_path}")
                continue
        
        case = EvalCase(
            bfc=bfc,
            ground_truth_bics=all_bics,
            repo_path=repo_path,
            dataset=dataset,
            case_id=f"{dataset.upper()}_{bfc}",
            project=repo_name,
        )
        
        if dataset not in cases_by_dataset:
            cases_by_dataset[dataset] = []
        cases_by_dataset[dataset].append(case)
    
    return cases_by_dataset


def load_dataset_cases(dataset: str, max_cases: int = None) -> List[EvalCase]:
    """Load evaluation cases from dataset JSON files.
    
    Uses final_all_dataset.json for ground truth BICs (consistent with RQ0).
    Uses DS_*.json files for BFC list and repo_name mapping.
    """
    cases = []
    repos_dir = PROJECT_ROOT / "dataset" / "subject_repos"
    
    # Dataset-specific config
    if dataset == "linux":
        ds_path = PROJECT_ROOT / "dataset" / "DS_LINUX.json"
        bfc_key = "fix_commit_hash"
    elif dataset == "apache":
        ds_path = PROJECT_ROOT / "dataset" / "DS_APACHE.json"
        bfc_key = "bug_fixing_commit"
    elif dataset == "github":
        ds_path = PROJECT_ROOT / "dataset" / "DS_GITHUB.json"
        bfc_key = "fix_commit_hash"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not ds_path.exists():
        print(f"Warning: Dataset file not found: {ds_path}")
        return cases

    # Load DS file for BFC list and repo mapping
    with open(ds_path, "r") as f:
        ds_data = json.load(f)
    
    # Build BFC -> repo_name mapping from DS file (GitHub uses owner/repo format)
    bfc_to_repo = {}
    for item in ds_data:
        bfc = item.get(bfc_key)
        repo_name = item.get("repo_name", "")
        if bfc:
            bfc_to_repo[bfc] = repo_name
    
    # Load ground truth BICs from final_all_dataset.json (same as RQ0)
    # Use set to handle: (1) duplicate BICs within a case, (2) duplicate BFC entries
    # This ensures counts match LLM4SZZ paper: Linux 1500/1562, Apache 241/351, GitHub 361/362
    all_info_path = PROJECT_ROOT / "dataset" / "final_all_dataset.json"
    bfc_to_bics = {}  # BFC -> set of unique BICs
    if all_info_path.exists():
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
                        bfc_to_bics[bfc].update(bics)  # Merge for duplicate BFC entries
    
    # Build cases using DS file BFCs and final_all_dataset BICs
    for bfc, repo_name in bfc_to_repo.items():
        # Get ground truth BICs (convert set to list for EvalCase)
        bics_set = bfc_to_bics.get(bfc)
        if not bics_set:
            continue  # Skip if no ground truth found
        bics = list(bics_set)
        
        # Determine repo path
        if dataset == "linux":
            case_repo = str(repos_dir / "linux")
        else:
            # GitHub uses owner/repo format, try nested path
            if "/" in repo_name:
                case_repo = str(repos_dir / repo_name)
            else:
                case_repo = str(repos_dir / repo_name)
        
        # Verify repo exists (has .git)
        if not Path(case_repo, ".git").exists():
            # Check if it's a known missing repo to avoid noisy warnings
            is_known = any(known in case_repo for known in KNOWN_MISSING_REPOS)
            if is_known:
                continue
            else:
                print(f"WARNING: Repo missing unexpectedly: {case_repo}")
                continue
        
        # Extract project name for display
        project = repo_name.split("/")[-1] if "/" in repo_name else repo_name

        case_id = f"{dataset.upper()}_{bfc}"
        cases.append(
            EvalCase(
                bfc=bfc,
                ground_truth_bics=bics,
                repo_path=case_repo,
                dataset=dataset,
                case_id=case_id,
                project=project if dataset != "linux" else "linux",
            )
        )

    if max_cases:
        cases = cases[:max_cases]

    print(f"Loaded {len(cases)} cases from {dataset}")
    return cases


def clean_case_data(repo_path: str, bfc: str):
    """
    Clean TKG data for a specific case (parallel-safe).
    
    Uses group_id=bfc to only delete this case's nodes/edges,
    allowing multiple workers to run in parallel without conflict.
    
    Note: reset_neo4j_database() is called once at evaluation start,
    so per-case cleanup failures are not critical.
    """
    async def _clean():
        async with SuspiciousCommitTKG(repo_path) as tkg:
            await tkg.clean_tkg(group_id=bfc)
    
    try:
        # Use new_event_loop + run_until_complete instead of asyncio.run
        # to avoid "Event loop is closed" errors from httpx cleanup.
        # Safe for multiprocessing: each worker process has its own event loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_clean())
        finally:
            # Clean up pending tasks before closing
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
    except Exception as e:
        # Just log - database was reset at start, so this is not critical
        print(f"  Warning: per-case cleanup failed for {bfc}: {e}")


def evaluate_case(
    case: EvalCase, config: str, szz: AgenticSZZ = None, trajectory_dir: str = None
) -> Optional[EvalResult]:
    """Evaluate a single case with given configuration."""
    start_time = time.time()

    try:
        # Use existing SZZ or create new
        if szz is None:
            szz = AgenticSZZ(case.repo_path)

        # Run prediction with trajectory naming (case_id format: DATASET_BFC)
        traj_kwargs = {"case_id": case.case_id, "trajectory_dir": trajectory_dir}
        if config == "blame_only":
            # Pure SZZ blame (no fallback for blameless cases)
            result = szz.predict(
                case.bfc, skip_tkg=True, skip_agent=True, **traj_kwargs
            )
        elif config == "blame_fallback":
            # Blame with context fallback (SZZ was created with use_context_fallback=True)
            result = szz.predict(
                case.bfc, skip_tkg=True, skip_agent=True, **traj_kwargs
            )
        elif config == "tkg_only":
            result = szz.predict(case.bfc, skip_tkg=False, skip_agent=True, **traj_kwargs)
        elif config == "agent_only":
            result = szz.predict(case.bfc, skip_tkg=True, skip_agent=False, **traj_kwargs)
        elif config == "full":
            result = szz.predict(case.bfc, **traj_kwargs)
        else:
            raise ValueError(f"Unknown config: {config}")

        # Check correctness against ALL ground truth BICs
        predicted = result.predicted_bic or ""  # Handle None
        is_correct = False
        if predicted:  # Only check if we have a prediction
            for gt in case.ground_truth_bics:
                # Match by prefix (handles short vs full SHA)
                if (
                    predicted.startswith(gt[:8])
                    or gt.startswith(predicted[:8])
                    or predicted == gt
                ):
                    is_correct = True
                    break

        elapsed_ms = (time.time() - start_time) * 1000

        eval_result = EvalResult(
            case=case,
            predicted_bic=predicted,
            is_correct=is_correct,
            confidence=result.confidence,
            time_ms=elapsed_ms,
            config=config,
            blame_time_ms=result.blame_time_ms,
            tkg_time_ms=result.tkg_time_ms,
            agent_time_ms=result.agent_time_ms,
            tokens_input=result.tokens_input,
            tokens_output=result.tokens_output,
            tokens_cached=result.tokens_cached,
            cost_usd=result.cost_usd,
            agent_steps=result.agent_steps,
        )
        
        # NO per-case cleanup - cases are isolated by unique group_id
        # Database will be torn down when worker finishes
        
        return eval_result

    except Exception as e:
        print(f"  Error on {case.bfc}: {e}")
        return None


def _evaluate_batch_worker(args: Tuple[List[EvalCase], str, int]) -> List[EvalResult]:
    """
    Worker function for batch evaluation with isolated database.
    
    Each worker:
    1. Starts its own Neo4j container (only for configs that need TKG)
    2. Processes entire batch of cases
    3. Tears down container when done (if started)
    
    Args:
        args: (cases, config, worker_id, trajectory_dir, context_expand_lines)
        
    Returns:
        List of EvalResult for processed cases
    """
    cases, config, worker_id, trajectory_dir, context_expand_lines = args
    
    # Apply stderr filter in worker process (workers don't inherit main process filter)
    import sys
    if not isinstance(sys.stderr, StderrNoiseFilter):
        sys.stderr = StderrNoiseFilter(sys.__stderr__)
    
    # Only start Neo4j container for configs that use TKG
    # Only start Neo4j container for configs that use TKG
    # blame_only, blame_fallback, agent_only don't need Neo4j
    needs_neo4j = config in ("full", "tkg_only")
    
    # Determine if context fallback should be enabled
    # blame_only = pure SZZ (no fallback), all others use fallback
    use_context_fallback = config != "blame_only"
    
    if needs_neo4j:
        db_config = start_worker_database(worker_id)
    else:
        db_config = None
        print(f"  Worker {worker_id}: Skipping Neo4j (not needed for {config})")
    
    try:
        results = []
        # Create SZZ with worker-specific database and fallback setting
        szz = AgenticSZZ(
            cases[0].repo_path, 
            db_config=db_config, 
            use_context_fallback=use_context_fallback,
            context_expand_lines=context_expand_lines,
        )
        
        # Use tqdm for progress if single worker (worker_id == 0 and called directly)
        case_iter = tqdm(cases, desc=f"Worker {worker_id}: {config}") if len(cases) > 1 else cases
        
        for case in case_iter:
            # Update repo path if different dataset
            if case.repo_path != szz.repo_path:
                szz = AgenticSZZ(
                    case.repo_path, 
                    db_config=db_config, 
                    use_context_fallback=use_context_fallback,
                    context_expand_lines=context_expand_lines,
                )
            
            result = evaluate_case(case, config, szz, trajectory_dir)
            if result:
                results.append(result)
        
        return results
    finally:
        # Tear down worker's database (only if we started one)
        if needs_neo4j:
            stop_worker_database(worker_id)


def compute_metrics(results: List[EvalResult]) -> DatasetMetrics:
    """Compute Precision, Recall, F1 from results.
    
    Uses LLM4SZZ micro-average formula for fair comparison:
    - Precision = correct_predictions / total_predictions (cases)
    - Recall = correct_predictions / total_ground_truth_BICs
    - F1 = harmonic mean of precision and recall
    
    Note: is_correct uses prefix matching (see evaluate_single_case).
    """
    if not results:
        return DatasetMetrics(
            dataset="",
            config="",
            total_cases=0,
            correct=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            avg_time_ms=0.0,
        )

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)

    # LLM4SZZ micro-average: Precision = correct / total_predictions
    # Since we make exactly 1 prediction per case: precision = correct / total_cases
    precision = correct / total if total > 0 else 0.0
    
    # LLM4SZZ micro-average: Recall = correct / total_ground_truth_BICs
    total_bics = sum(len(r.case.ground_truth_bics) for r in results)
    recall = correct / total_bics if total_bics > 0 else 0.0
    
    # F1 = harmonic mean
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    avg_time = sum(r.time_ms for r in results) / total if total > 0 else 0.0

    # Token/Cost aggregates (RQ3)
    total_tokens_input = sum(r.tokens_input for r in results)
    total_tokens_output = sum(r.tokens_output for r in results)
    total_cost = sum(r.cost_usd for r in results)
    avg_cost = total_cost / total if total > 0 else 0.0

    return DatasetMetrics(
        dataset=results[0].case.dataset,
        config=results[0].config,
        total_cases=total,
        correct=correct,
        precision=precision,
        recall=recall,
        f1=f1,
        avg_time_ms=avg_time,
        total_tokens_input=total_tokens_input,
        total_tokens_output=total_tokens_output,
        total_cost_usd=total_cost,
        avg_cost_per_case=avg_cost,
    )


def save_config_results(
    dataset: str,
    config: str,
    metrics: "DatasetMetrics",
    per_case_results: List[EvalResult],
    output_dir: Path,
    timestamp: str,
    summary_rows: List[dict],
    model: str = None,
):
    """
    Save results for a single config immediately after it completes.
    
    Saves per-case JSON and updates summary CSV incrementally.
    """
    # Add to summary rows
    summary_rows.append({
        "dataset": dataset,
        "config": config,
        "model": model or os.environ.get("LLM_MODEL", "deepseek-chat"),
        "total_cases": metrics.total_cases,
        "correct": metrics.correct,
        "precision": round(metrics.precision, 3),
        "recall": round(metrics.recall, 3),
        "f1": round(metrics.f1, 3),
        "avg_time_s": round(metrics.avg_time_ms / 1000, 1),
        "total_tokens_in": metrics.total_tokens_input,
        "total_tokens_out": metrics.total_tokens_output,
        "total_cost_usd": round(metrics.total_cost_usd, 4),
        "avg_cost_usd": round(metrics.avg_cost_per_case, 6),
    })
    
    # Save per-case JSON immediately
    per_case_data = []
    for r in per_case_results:
        per_case_data.append({
            "case_id": str(r.case.case_id),
            "dataset": r.case.dataset,
            "project": r.case.project,
            "bfc": r.case.bfc,
            "ground_truth": r.case.ground_truth_bics,  # List of all valid BICs
            "predicted": r.predicted_bic,
            "success": r.is_correct,
            "confidence": r.confidence,
            "time_ms": round(r.time_ms, 1),
            "blame_time_ms": round(r.blame_time_ms, 1),
            "tkg_time_ms": round(r.tkg_time_ms, 1),
            "agent_time_ms": round(r.agent_time_ms, 1),
            "tokens_input": r.tokens_input,
            "tokens_output": r.tokens_output,
            "tokens_cached": r.tokens_cached,
            "cost_usd": round(r.cost_usd, 6),
            "agent_steps": r.agent_steps,
            "model": model or os.environ.get("LLM_MODEL", "deepseek-chat"),
        })
    
    json_path = output_dir / f"{dataset}_{config}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(per_case_data, f, indent=2)
    print(f"Per-case results saved to {json_path}")

    # Update summary CSV (overwrite with current data)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = output_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)


def run_config_evaluation(
    cases: List[EvalCase],
    config: str,
    workers: int = 1,
    desc: str = None,
    trajectory_dir: str = None,
    context_expand_lines: int = 0,
) -> List[EvalResult]:
    """
    Evaluate a list of cases with a single config, using parallel or sequential execution.
    
    This is the core evaluation logic shared by both sample file and regular paths.
    
    Args:
        cases: List of evaluation cases
        config: Configuration to run (blame_only, blame_fallback, tkg_only, agent_only, full)
        workers: Number of parallel workers (1 = sequential)
        desc: Description for progress bar
        trajectory_dir: Output directory for trajectories (trajectories saved under trajectory_dir/trajectories/)
        context_expand_lines: Expand ±N lines around deleted lines for ancestor detection (0 = disabled)
        
    Returns:
        List of EvalResult objects
    """
    results = []
    desc = desc or config
    
    if workers > 1 and len(cases) > 1:
        # Batch processing with per-worker databases
        # Split cases into batches, one per worker
        batch_size = (len(cases) + workers - 1) // workers  # Round up
        batches = []
        for i in range(0, len(cases), batch_size):
            batches.append(cases[i:i + batch_size])
        
        work_items = [
            (batch, config, worker_id, trajectory_dir, context_expand_lines)
            for worker_id, batch in enumerate(batches)
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_evaluate_batch_worker, item): item
                for item in work_items
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"{desc} (workers)"
            ):
                try:
                    batch_results = future.result()  # Returns list
                    results.extend(batch_results)
                except Exception as e:
                    batch, _, worker_id, _, _ = futures[future]
                    print(f"  Worker {worker_id} error: {e}")
    else:
        # Single worker execution - use Docker container like multi-worker mode
        # This ensures no dependency on local Neo4j instance
        print(f"  Running {len(cases)} cases with single worker...")
        work_item = (cases, config, 0, trajectory_dir, context_expand_lines)  # worker_id = 0
        batch_results = _evaluate_batch_worker(work_item)
        results.extend(batch_results)
    
    return results


def run_evaluation(
    dataset: str,
    configs: List[str],
    max_cases: int = None,
    workers: int = 1,
    trajectory_dir: str = None,
    context_expand_lines: int = 0,
) -> Dict[str, DatasetMetrics]:
    """Run full evaluation on dataset with optional parallelization."""

    cases = load_dataset_cases(dataset, max_cases)
    if not cases:
        print(f"No cases found for {dataset}")
        return {}

    # Filter to cases with valid repo paths
    valid_cases = [c for c in cases if Path(c.repo_path).exists()]
    skipped = len(cases) - len(valid_cases)
    if skipped > 0:
        print(f"  Skipping {skipped} cases with missing repos")
    cases = valid_cases

    results_by_config = {}

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Running {config} on {dataset} ({len(cases)} cases, {workers} workers)")
        print(f"{'=' * 60}")

        # Use shared helper for parallel/sequential execution
        results = run_config_evaluation(
            cases, config, workers=workers, desc=config, 
            trajectory_dir=trajectory_dir, context_expand_lines=context_expand_lines
        )

        metrics = compute_metrics(results)
        results_by_config[config] = (
            metrics,
            results,
        )  # Return both metrics and per-case results

        print(f"\n{config} Results:")
        print(f"  Total: {metrics.total_cases}")
        print(f"  Correct: {metrics.correct}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  F1: {metrics.f1:.3f}")
        print(f"  Avg Time: {metrics.avg_time_ms / 1000:.1f}s")

    return results_by_config


def main():
    parser = argparse.ArgumentParser(description="RQ1 Evaluation for AgenticSZZ")
    parser.add_argument(
        "--dataset",
        type=str,
        default="linux",
        choices=["linux", "apache", "github", "all"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["blame_only", "blame_fallback", "tkg_only", "agent_only", "full", "all"],
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of cases (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rq1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (e.g., deepseek-v4-flash, deepseek-v4-pro, qwen3-coder-next). Overrides LLM_MODEL env var.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LLM API base URL for OpenAI-compatible endpoint (e.g., http://localhost:8000/v1). Overrides LLM_BASE_URL env var.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (0 = auto, 1 = sequential)",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help="Use stratified sample from create_pilot_sample.py (overrides --dataset and --max-cases)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Override candidate_limit (K) for sensitivity analysis (e.g., 5, 10, 20, 30, 50)",
    )
    parser.add_argument(
        "--context-expand",
        type=int,
        default=None,
        help="Expand ±N lines around deleted lines for ancestor detection (e.g., 2 for LLM4SZZ-style)",
    )

    args = parser.parse_args()

    # Override model if specified via CLI
    if args.model:
        os.environ["LLM_MODEL"] = args.model
        print(f"Using LLM model: {args.model}")

    # Override base URL if specified via CLI (for local/custom inference endpoints)
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
        print(f"Using LLM base URL: {args.base_url}")

    if args.model or args.base_url:
        # Refresh config singleton to pick up new env vars
        from agentic_szz.config.settings import get_config
        get_config(refresh=True)
    
    # Override candidate_limit (K) if specified via CLI
    if args.candidate_limit is not None:
        from agentic_szz.config.settings import get_config
        config = get_config()
        config.tkg.candidate_limit = args.candidate_limit
        print(f"Using candidate_limit (K) = {args.candidate_limit}")
    
    # Override context_expand_lines if specified via CLI (experimental, for ancestor detection)
    if args.context_expand is not None:
        from agentic_szz.config.settings import get_config
        config = get_config()
        config.tkg.context_expand_lines = args.context_expand
        print(f"Using context_expand_lines = {args.context_expand}")

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine configs to run
    if args.config == "all":
        configs = ["blame_only", "blame_fallback", "tkg_only", "agent_only", "full"]
    else:
        configs = [args.config]

    # Determine number of workers
    workers = args.workers
    if workers == 0:
        workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming APIs
        print(f"Auto-detected {cpu_count()} CPUs, using {workers} workers")
    elif workers > 1:
        print(f"Using {workers} parallel workers")

    # All workers (including single worker) use Docker containers
    # This ensures consistent behavior and no dependency on a local Neo4j instance
    print(f"Using {workers} worker(s) - each will start its own clean Neo4j container")

    # Load cases - either from sample file or from dataset(s)
    if args.sample_file:
        print(f"\nLoading stratified sample from {args.sample_file}...")
        sample_cases = load_sample_cases(args.sample_file)
        # Respect --dataset argument to filter which datasets from sample to run
        if args.dataset == "all":
            datasets = list(sample_cases.keys())
        else:
            datasets = [args.dataset] if args.dataset in sample_cases else []
            if not datasets:
                print(f"WARNING: Dataset '{args.dataset}' not found in sample file")
        print(f"Loaded {sum(len(sample_cases.get(d, [])) for d in datasets)} cases from {len(datasets)} datasets")
    else:
        sample_cases = None
        # Determine datasets to run
        if args.dataset == "all":
            datasets = ["linux", "apache", "github"]
        else:
            datasets = [args.dataset]

    # Generate timestamp once at start for consistent file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []

    # Run evaluation with incremental saving
    for dataset in datasets:
        if sample_cases:
            # Use pre-loaded sample cases for this dataset
            cases = sample_cases.get(dataset, [])
            if not cases:
                continue
            print(f"\n{'='*60}")
            print(f"Running {len(cases)} {dataset} cases from sample ({workers} workers)")
            print(f"{'='*60}")
            # Run each config and save immediately after completion
            for config in configs:
                # Use shared helper for parallel/sequential execution
                # Get context_expand_lines from global config (only for full config)
                from agentic_szz.config.settings import get_config
                ctx_expand = get_config().tkg.context_expand_lines if config == "full" else 0
                config_results = run_config_evaluation(
                    cases, config, workers=workers, desc=f"{dataset}/{config}", 
                    trajectory_dir=str(output_dir / "trajectories"),
                    context_expand_lines=ctx_expand,
                )
                metrics = compute_metrics(config_results)
                print(f"\n{config}: {metrics.correct}/{metrics.total_cases} correct ({metrics.precision*100:.1f}%)")
                # Save immediately after this config completes
                save_config_results(
                    dataset=dataset,
                    config=config,
                    metrics=metrics,
                    per_case_results=config_results,
                    output_dir=output_dir,
                    timestamp=timestamp,
                    summary_rows=summary_rows,
                )
        else:
            from agentic_szz.config.settings import get_config
            cases = load_dataset_cases(dataset, args.max_cases)
            if not cases:
                continue
            valid_cases = [c for c in cases if Path(c.repo_path).exists()]
            skipped = len(cases) - len(valid_cases)
            if skipped > 0:
                print(f"  Skipping {skipped} cases with missing repos")
            cases = valid_cases
            print(f"\n{'='*60}")
            print(f"Running {len(cases)} {dataset} cases ({workers} workers)")
            print(f"{'='*60}")
            # Run each config separately so context_expand only applies to full
            for config in configs:
                ctx_expand = get_config().tkg.context_expand_lines if config == "full" else 0
                config_results = run_config_evaluation(
                    cases, config, workers=workers, desc=f"{dataset}/{config}",
                    trajectory_dir=str(output_dir / "trajectories"),
                    context_expand_lines=ctx_expand,
                )
                metrics = compute_metrics(config_results)
                print(f"\n{config}: {metrics.correct}/{metrics.total_cases} correct ({metrics.precision*100:.1f}%)")
                save_config_results(
                    dataset=dataset,
                    config=config,
                    metrics=metrics,
                    per_case_results=config_results,
                    output_dir=output_dir,
                    timestamp=timestamp,
                    summary_rows=summary_rows,
                )

    # Print final summary
    if summary_rows:
        print(f"\nSummary saved to {output_dir / f'results_{timestamp}.csv'}")
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
