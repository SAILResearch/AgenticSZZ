#!/usr/bin/env python3
"""Cross-file ceiling (R2.13): how many BICs share no file with their fix.

AgenticSZZ admits as a candidate only a commit that modified a file the BFC
also modified, so a ground-truth BIC sharing no file with its BFC cannot be
reached by any amount of traversal. That fraction is the recall ceiling of the
approach as designed, and it is the number Section 3.3 reports.

A case counts as cross-file only when no ground-truth BIC shares a file with
the BFC, since one reachable BIC is enough for the case to be solvable.

Usage:
    python evaluation/cross_file_ceiling.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_evaluation import load_dataset_cases  # noqa: E402

DATASETS = ["linux", "github", "apache"]


def changed_files(repo_path, sha):
    """Files touched by a commit, or None if the commit does not resolve."""
    out = subprocess.run(
        ["git", "-C", repo_path, "show", "--name-only", "--pretty=format:", sha],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def main():
    rows = []
    total_cases = total_cross = 0

    for dataset in DATASETS:
        counted = cross = unresolved = 0

        for case in load_dataset_cases(dataset):
            bfc_files = changed_files(case.repo_path, case.bfc)
            if not bfc_files:
                unresolved += 1
                continue

            bic_files = [changed_files(case.repo_path, b) for b in case.ground_truth_bics]
            resolved = [f for f in bic_files if f]
            if not resolved:
                unresolved += 1
                continue

            counted += 1
            if not any(bfc_files & f for f in resolved):
                cross += 1

        rows.append((dataset.upper(), counted, cross, unresolved))
        total_cases += counted
        total_cross += cross

    print(f"{'Dataset':<12}{'cases':>8}{'cross-file':>15}{'unresolved':>12}")
    for name, counted, cross, unresolved in rows:
        pct = cross / counted * 100 if counted else 0
        print(f"DS_{name:<9}{counted:>8}{cross:>9} {pct:>4.1f}%{unresolved:>12}")
    pct = total_cross / total_cases * 100 if total_cases else 0
    print(f"{'All':<12}{total_cases:>8}{total_cross:>9} {pct:>4.1f}%")


if __name__ == "__main__":
    main()
