#!/usr/bin/env python3
"""How graph size and file count affect runtime, memory and accuracy (R1.8).

Every RQ1 result record carries node_count, edge_count, file_count,
tkg_time_ms, time_ms and peak_rss_mb. This bins the cases of the best RQ1 run
by graph size and by the number of files the fix touches, and reports those
costs per bin, plus the median number of tool calls. Tool calls are counted
from the stored trajectories, excluding the final answer. The agent_steps field
counts messages (each call and its return), not tool calls, so it is not used.

Usage:
    python evaluation/scalability_analysis.py
"""

import glob
import json
import re
import statistics
from pathlib import Path

from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
RUN = PROJECT_ROOT / "results/rq1/running_1"

HEADER = (
    f"{'':>12} {'cases':>6} {'nodes':>7} {'edges':>7} "
    f"{'build s':>8} {'total s':>8} {'peak MB':>8} {'calls':>6} {'acc':>8}"
)
TOOL_CALL = re.compile(r"'tool_name': '([a-z_]+)'")


def tool_calls(run):
    """Tool calls per case, excluding final_result, from the run's trajectories."""
    calls = {}
    for f in glob.glob(str(run / "trajectories" / "*.json")):
        d = json.load(open(f))
        calls[d["case_id"]] = sum(
            name != "final_result"
            for step in d["trajectory"] if step["type"] == "ModelResponse"
            for name in TOOL_CALL.findall(str(step["parts"]))
        )
    return calls


def row(label, records):
    med = lambda key: statistics.median(r[key] for r in records if r[key] is not None)
    return (
        f"{label:>12} {len(records):>6} {med('node_count'):>7.0f} {med('edge_count'):>7.0f} "
        f"{med('tkg_time_ms') / 1000:>8.1f} {med('time_ms') / 1000:>8.1f} "
        f"{med('peak_rss_mb'):>8.0f} {med('tool_calls'):>6.0f} "
        f"{100 * sum(r['success'] for r in records) / len(records):>7.1f}%"
    )


def main():
    records = []
    for f in glob.glob(str(RUN / "*_full_*.json")):
        records += [r for r in json.load(open(f)) if r["node_count"]]
    calls = tool_calls(RUN)
    for r in records:
        r["tool_calls"] = calls.get(r["case_id"])  # None when no trajectory was stored
    records.sort(key=lambda r: r["node_count"])
    quartile = len(records) // 4

    print("By graph size (quartiles of node count)")
    print(HEADER)
    for i in range(4):
        chunk = records[i * quartile:(i + 1) * quartile] if i < 3 else records[3 * quartile:]
        print(row(f"{chunk[0]['node_count']} to {chunk[-1]['node_count']}", chunk))

    print("\nBy files the fix touches")
    print(HEADER)
    for lo, hi, label in [(1, 1, "1"), (2, 2, "2"), (3, 5, "3 to 5"), (6, 10**6, "6+")]:
        print(row(label, [r for r in records if lo <= r["file_count"] <= hi]))

    nodes = [r["node_count"] for r in records]
    print("\nSpearman rho against graph size")
    for key in ["tkg_time_ms", "time_ms", "peak_rss_mb"]:
        print(f"  {key:<14} {spearmanr(nodes, [r[key] for r in records]).statistic:>6.3f}")
    with_calls = [r for r in records if r["tool_calls"] is not None]
    rho = spearmanr([r["node_count"] for r in with_calls], [r["tool_calls"] for r in with_calls]).statistic
    print(f"  {'tool_calls':<14} {rho:>6.3f}")

    # Peak RSS is a process high-water mark, so it is worth checking whether it
    # follows the graph at all or only the repository the worker has loaded.
    print("\nPeak memory by project")
    for dataset in ["apache", "github", "linux"]:
        chunk = [r for r in records if r["dataset"] == dataset]
        rho = spearmanr(
            [r["node_count"] for r in chunk], [r["peak_rss_mb"] for r in chunk]
        ).statistic
        print(
            f"  {dataset:<8} cases {len(chunk):>5}  median nodes "
            f"{statistics.median(r['node_count'] for r in chunk):>6.0f}  median peak "
            f"{statistics.median(r['peak_rss_mb'] for r in chunk):>6.0f} MB  "
            f"rho(nodes, peak) {rho:>6.3f}"
        )

    build = sum(r["tkg_time_ms"] for r in records)
    print(f"\nConstruction is {100 * build / sum(r['time_ms'] for r in records):.1f}% of end-to-end time.")


if __name__ == "__main__":
    main()
