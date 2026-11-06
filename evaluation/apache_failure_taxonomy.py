#!/usr/bin/env python3
"""DS_APACHE failure taxonomy (R1.6).

Classifies the DS_APACHE cases that no RQ1 run solves, using the categories the
reviewer proposes. The first question is whether the ground-truth BIC was ever
available to the agent, because a case where it was not is a candidate-selection
failure and cannot be attributed to the agent's reasoning:

  1. Cross-file BIC   - the BIC shares no file with the fix, so no amount of
                        traversal reaches it. This is the ceiling of the design
                        that cross_file_ceiling.py reports.
  2. Outside top-K    - the BIC is in the graph, but ranked below the K the
                        agent is shown.
  3. Not in graph     - the BIC shares a file with the fix and the candidate set
                        was not truncated, so traversal did not reach it.
  4. Agent selection  - the BIC was in the list the agent saw and it chose
                        another commit.

For group 4 we report the axes the reviewer names: blame-candidate count,
deleted lines in the fix, and whether the case has several ground-truth BICs.

Usage:
    python evaluation/apache_failure_taxonomy.py
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_evaluation import load_dataset_cases  # noqa: E402

RUNS = ["running_1", "running_2", "running_3"]
RUN_ROOT = PROJECT_ROOT / "results" / "rq1"
CANDIDATE_RE = re.compile(r"\[(\w+)\] ([0-9a-f]{7,40})")
TOTAL_RE = re.compile(r"of (\d+) total")


def load_runs():
    """Per-run {case_id: result record} for DS_APACHE."""
    runs = {}
    for run in RUNS:
        matches = sorted((RUN_ROOT / run).glob("apache_full_*.json"))
        with open(matches[-1]) as f:
            runs[run] = {c["case_id"]: c for c in json.load(f)}
    return runs


def candidate_listing(run: str, case_id: str):
    """The list_candidates return the agent saw, or None if no trajectory."""
    path = RUN_ROOT / run / "trajectories" / f"{case_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        traj = json.load(f)["trajectory"]
    for step in traj:
        for part in step.get("parts", []):
            if part.get("type") == "ToolReturnPart" and part.get("tool_name") == "list_candidates":
                return part["content"]
    return ""


def same_commit(a: str, b: str) -> bool:
    """SHAs match when either is a prefix of the other (records are abbreviated)."""
    n = min(len(a), len(b), 8)
    return a[:n] == b[:n]


def changed_files(repo_path: str, sha: str):
    """Files touched by a commit, or None if the commit does not resolve."""
    out = subprocess.run(
        ["git", "-C", repo_path, "show", "--name-only", "--pretty=format:", sha],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def deleted_lines(repo_path: str, sha: str) -> int:
    out = subprocess.run(
        ["git", "-C", repo_path, "show", "--unified=0", "--pretty=format:", sha],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return -1
    return sum(1 for ln in out.stdout.splitlines()
               if ln.startswith("-") and not ln.startswith("---"))


def classify(case, runs, seen_any, truncated_any):
    """Assign one category to a consistently failing case."""
    bfc_files = changed_files(case.repo_path, case.bfc)
    bic_files = [changed_files(case.repo_path, b) for b in case.ground_truth_bics]
    resolved = [f for f in bic_files if f]

    if seen_any:
        return "Agent selection"
    if bfc_files and resolved and not any(bfc_files & f for f in resolved):
        return "Cross-file BIC"
    if truncated_any:
        return "Outside top-K"
    return "Not in graph"


def main():
    runs = load_runs()
    cases = {c.case_id: c for c in load_dataset_cases("apache")}

    case_ids = sorted(runs[RUNS[0]])
    failed = [cid for cid in case_ids
              if not any(runs[r].get(cid, {}).get("success") for r in RUNS)]

    print(f"DS_APACHE cases: {len(case_ids)}")
    for run in RUNS:
        solved = sum(c["success"] for c in runs[run].values())
        print(f"  {run}: {solved} solved")
    print(f"Failed in every run: {len(failed)}\n")

    counts = Counter()
    rows = []
    for cid in failed:
        case = cases.get(cid)
        if case is None:
            counts["case not in dataset"] += 1
            continue

        seen_any = False
        truncated_any = False
        blame_counts = []
        no_prediction = 0
        for run in RUNS:
            listing = candidate_listing(run, cid)
            if listing is None:
                no_prediction += 1
                continue
            labels = CANDIDATE_RE.findall(listing)
            if any(any(same_commit(sha, gt) for _, sha in labels)
                   for gt in case.ground_truth_bics):
                seen_any = True
            total = TOTAL_RE.search(listing)
            if total and int(total.group(1)) > len(labels):
                truncated_any = True
            blame_counts.append(sum(1 for kind, _ in labels if kind == "blame"))

        category = classify(case, runs, seen_any, truncated_any)
        counts[category] += 1
        rows.append({
            "case_id": cid,
            "project": case.project,
            "category": category,
            "blame_candidates": max(blame_counts) if blame_counts else 0,
            "deleted_lines": deleted_lines(case.repo_path, case.bfc),
            "gt_bics": len(case.ground_truth_bics),
            "runs_without_prediction": no_prediction,
        })

    print("Failure categories (cases failing in all three runs):")
    for name, n in counts.most_common():
        print(f"  {name:18} {n:4}  ({100 * n / len(failed):.1f}%)")

    selection = [r for r in rows if r["category"] == "Agent selection"]
    if selection:
        print("\nAgent-selection failures, on the axes the reviewer names:")
        blame = sorted(r["blame_candidates"] for r in selection)
        dels = sorted(r["deleted_lines"] for r in selection)
        multi = sum(1 for r in selection if r["gt_bics"] > 1)
        print(f"  blame candidates: median {blame[len(blame) // 2]}, max {blame[-1]}")
        print(f"  deleted lines:    median {dels[len(dels) // 2]}, max {dels[-1]}")
        print(f"  several GT BICs:  {multi} of {len(selection)}")

    solved_rows = [r for r in rows if r["category"] != "Agent selection"]
    if solved_rows:
        blame = sorted(r["blame_candidates"] for r in solved_rows)
        print("\nCandidate-set failures, same axes:")
        print(f"  blame candidates: median {blame[len(blame) // 2]}, max {blame[-1]}")

    out = PROJECT_ROOT / "results" / "revision" / "apache_failure_taxonomy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"counts": dict(counts), "cases": rows}, f, indent=2)
    print(f"\nPer-case classification written to {out}")


if __name__ == "__main__":
    main()
