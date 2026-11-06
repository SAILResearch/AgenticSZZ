#!/usr/bin/env python3
"""Is the agent's confidence usable as a triage signal? (R1.9)

A developer deciding whether to act on a predicted BIC needs to know when the
prediction is likely wrong. The agent reports a confidence with every answer, so
we can measure that from the RQ1 runs without a new experiment:

  - Calibration: accuracy within each confidence band.
  - Selective prediction: accuracy over the cases above a threshold, against the
    fraction of cases that threshold keeps.

The two answer different questions. Calibration asks whether the number means
what it says; selective prediction asks whether ordering by it is useful, which
is the property a triage workflow needs.

Usage:
    python evaluation/confidence_calibration.py
"""

import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RUN_ROOT = PROJECT_ROOT / "results" / "rq1"
RUNS = ["running_1", "running_2", "running_3"]
DATASETS = ["linux", "apache", "github"]

BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
THRESHOLDS = [0.0, 0.7, 0.8, 0.85, 0.9, 0.95]


def load_records():
    """Every RQ1 case record from every run, tagged with its dataset."""
    records = []
    for run in RUNS:
        for dataset in DATASETS:
            matches = sorted(glob.glob(str(RUN_ROOT / run / f"{dataset}_full_*.json")))
            if not matches:
                raise FileNotFoundError(f"no results for {run}/{dataset}")
            with open(matches[-1]) as f:
                for record in json.load(f):
                    record["dataset"] = dataset
                    record["run"] = run
                    records.append(record)
    return records


def accuracy(records):
    return sum(r["success"] for r in records) / len(records) if records else 0.0


def expected_calibration_error(records, n_bins=10):
    """Mean gap between stated confidence and observed accuracy, weighted by band size."""
    error = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        band = [r for r in records if lo <= r["confidence"] < hi or (i == n_bins - 1 and r["confidence"] == 1.0)]
        if not band:
            continue
        stated = sum(r["confidence"] for r in band) / len(band)
        error += abs(accuracy(band) - stated) * len(band) / len(records)
    return error


def main():
    records = load_records()
    n = len(records)
    print(f"{n} case records over {len(RUNS)} runs\n")

    print("Accuracy by confidence band:")
    print(f"  {'band':>11} {'cases':>7} {'accuracy':>9}")
    for lo, hi in BANDS:
        band = [r for r in records if lo <= r["confidence"] < hi]
        if band:
            print(f"  {lo:.2f}-{min(hi, 1.0):.2f} {len(band):>7} {accuracy(band):>9.1%}")

    print("\nSelective prediction:")
    print(f"  {'keep at':>8} {'coverage':>9} {'accuracy':>9}")
    for t in THRESHOLDS:
        kept = [r for r in records if r["confidence"] >= t]
        print(f"  {t:>8.2f} {len(kept) / n:>9.1%} {accuracy(kept):>9.1%}")

    print("\nPer dataset:")
    print(f"  {'dataset':>8} {'all':>8} {'>= 0.95':>9} {'< 0.70':>9}")
    for dataset in DATASETS:
        subset = [r for r in records if r["dataset"] == dataset]
        high = [r for r in subset if r["confidence"] >= 0.95]
        low = [r for r in subset if r["confidence"] < 0.70]
        print(f"  {dataset:>8} {accuracy(subset):>8.1%} {accuracy(high):>9.1%} {accuracy(low):>9.1%}")

    mean_confidence = sum(r["confidence"] for r in records) / n
    print(f"\nMean confidence {mean_confidence:.3f} against accuracy {accuracy(records):.3f}")
    print(f"Expected calibration error {expected_calibration_error(records):.3f}")


if __name__ == "__main__":
    main()
