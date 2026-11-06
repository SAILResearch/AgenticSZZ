#!/usr/bin/env python3
"""TC-SZZ baseline on DS_LINUX (R2.11), scored from the authors' own output.

Tang et al. ship per-case predictions for all 76,046 Linux cases in their
replication package, which covers every DS_LINUX case, so we score their output
rather than re-running their tool. TC-SZZ returns a candidate set per case, so
precision and recall are computed over commits rather than over cases:

    precision = TP / (TP + FP)    TP + FP = all predicted commits
    recall    = TP / (TP + FN)    TP + FN = all ground-truth BICs

--verify applies the identical code to their B-SZZ and AG-SZZ output over their
full dataset, which reproduces Table 2 of their paper and shows the scoring is
theirs rather than ours.

Usage:
    python evaluation/tc_szz_baseline.py [--verify]
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW = PROJECT_ROOT / "vendor/tc-szz/tse_new/rq12/json-output-raw/rq1"
DS_LINUX = PROJECT_ROOT / "dataset/DS_LINUX.json"

# Their published Table 2 numbers, for the --verify comparison.
PUBLISHED = {"bic_b": (0.42, 0.58, 0.49), "bic_ag": (0.42, 0.56, 0.48)}


def score(cases, ground_truth=None):
    """Precision, recall, F1 over predicted commits.

    cases: records with fix_commit_hash, inducing_commit_hash, bug_commit_hash.
    ground_truth: optional fix -> set of BICs, overriding the record's own.
    """
    tp = predicted = actual = 0
    for c in cases:
        truth = set(ground_truth[c["fix_commit_hash"]]) if ground_truth else set(c["bug_commit_hash"])
        pred = set(c["inducing_commit_hash"])
        tp += len(truth & pred)
        predicted += len(pred)
        actual += len(truth)
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, predicted, actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="reproduce their Table 2 first")
    args = ap.parse_args()

    if args.verify:
        print("Their full Linux dataset, our scoring against their published Table 2:")
        for tool, (p_pub, r_pub, f_pub) in PUBLISHED.items():
            cases = json.loads((RAW / f"{tool}_bugfix_commits_all.json").read_text())
            p, r, f, tp, pred, act = score(cases)
            print(
                f"  {tool:<8} P {p:.4f} (pub {p_pub})  R {r:.4f} (pub {r_pub})  "
                f"F1 {f:.4f} (pub {f_pub})  TP+FP {pred}  TP+FN {act}"
            )
        print()

    # Match DS_LINUX by its own BFC list. Filtering their output by repo name is
    # wrong: 'linux' as a substring also matches util-linux and twelve other
    # repositories that belong to DS_GITHUB.
    truth = {c["fix_commit_hash"]: c["bug_commit_hash"] for c in json.loads(DS_LINUX.read_text())}
    cases = json.loads((RAW / "bic_tc_bugfix_commits_all.json").read_text())
    matched = [c for c in cases if c["fix_commit_hash"] in truth]

    p, r, f, tp, pred, act = score(matched, truth)
    print(f"TC-SZZ on DS_LINUX ({len(matched)} of {len(truth)} cases found in their output)")
    print(f"  Precision {p:.4f}  Recall {r:.4f}  F1 {f:.4f}")
    print(f"  TP {tp}  predicted commits {pred}  ground-truth BICs {act}")


if __name__ == "__main__":
    main()
