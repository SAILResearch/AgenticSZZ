#!/usr/bin/env python3
"""
RQ3 Tool-Use Analysis: Analyze agent tool invocation patterns across LLMs.

Generates Table (tab:rq3-tool-use) showing median steps and per-case tool
call rates for each LLM in the RQ3 sensitivity analysis.

Usage:
    python evaluation/rq3_tool_analysis.py
"""

import json
import glob
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configs: (display_name, trajectory_dir, result_dir)
MODEL_CONFIGS = [
    ("DeepSeek-V3.2", "results/rq1/running_2/trajectories", "results/rq1/running_2"),
    ("DeepSeek-V4-Pro", "results/rq3/deepseek-v4-pro/trajectories", "results/rq3/deepseek-v4-pro"),
    ("Gemma-4-31B", "results/rq3/gemma-4-31b/trajectories", "results/rq3/gemma-4-31b"),
    ("Devstral-Small-2", "results/rq3/devstral-small-2/trajectories", "results/rq3/devstral-small-2"),
    ("Qwen3-Coder-Next", "results/rq3/qwen3-coder-next/trajectories", "results/rq3/qwen3-coder-next"),
]

TOOLS = ["list_candidates", "query_node", "read_node_content", "traverse_graph"]


def analyze_model(traj_dir, result_dir):
    """Analyze tool usage from trajectory files."""
    traj_dir = PROJECT_ROOT / traj_dir
    result_dir = PROJECT_ROOT / result_dir

    files = [f for f in glob.glob(str(traj_dir / "*.json")) if not os.path.isdir(f)]

    # Load success info
    success_map = {}
    for f in glob.glob(str(result_dir / "*_full_*.json")):
        with open(f) as fh:
            for r in json.load(fh):
                key = r.get("case_id", f"{r.get('dataset', '')}_{r['bfc']}")
                success_map[key] = r.get("success", False)

    tool_counts = Counter()
    steps_list = []
    total = len(files)
    correct = 0
    no_diff_read = 0  # cases with 0 read_node_content calls

    for f in files:
        with open(f) as fh:
            traj = json.load(fh)

        case_id = traj.get("case_id", "")
        if success_map.get(case_id, False):
            correct += 1

        case_tools = []
        for step in traj.get("trajectory", []):
            if step.get("type") == "ModelResponse":
                for part in step.get("parts", []):
                    if (
                        isinstance(part, dict)
                        and part.get("tool_name")
                        and part["tool_name"] != "final_result"
                    ):
                        tool = part["tool_name"]
                        tool_counts[tool] += 1
                        case_tools.append(tool)

        steps_list.append(len(case_tools))
        if "read_node_content" not in case_tools:
            no_diff_read += 1

    median_steps = sorted(steps_list)[total // 2] if total else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100 if total else 0,
        "median_steps": median_steps,
        "tool_rates": {t: tool_counts.get(t, 0) / total for t in TOOLS},
        "no_diff_pct": no_diff_read / total * 100 if total else 0,
    }


def generate_latex_table(results):
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Agent tool-use patterns across LLMs (average calls per case). "
        r"\textit{Steps}: median tool calls per case. "
        r"\textit{No-diff}: percentage of cases where the agent decided without reading any code diff.}"
    )
    lines.append(r"    \label{tab:rq3-tool-use}")
    lines.append(r"    \footnotesize")
    lines.append(r"    \begin{tabular}{l r rr r}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{LLM} & \textbf{Steps} & \texttt{query\_node} & \texttt{read\_node\_content} & \textbf{No-diff (\%)} \\"
    )
    lines.append(r"        \midrule")

    for name, stats in results:
        tr = stats["tool_rates"]
        lines.append(
            f"        {name:<24s} & {stats['median_steps']} "
            f"& {tr['query_node']:.2f} "
            f"& {tr['read_node_content']:.2f} "
            f"& {stats['no_diff_pct']:.1f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    results = []
    for name, traj_dir, result_dir in MODEL_CONFIGS:
        traj_path = PROJECT_ROOT / traj_dir
        if not traj_path.exists():
            print(f"Skipping {name}: {traj_dir} not found")
            continue
        stats = analyze_model(traj_dir, result_dir)
        results.append((name, stats))

    # Print summary
    print(f"{'Model':<24s} {'Cases':>6} {'Acc%':>6} {'Steps':>6} {'query_node':>11} {'read_diff':>10} {'No-diff%':>9}")
    print("-" * 80)
    for name, s in results:
        tr = s["tool_rates"]
        print(
            f"{name:<24s} {s['total']:>6} {s['accuracy']:>5.1f}% {s['median_steps']:>6} "
            f"{tr['query_node']:>11.2f} {tr['read_node_content']:>10.2f} {s['no_diff_pct']:>8.1f}%"
        )

    print()
    print("=" * 80)
    print("LaTeX table:")
    print("=" * 80)
    print(generate_latex_table(results))

    # Save LaTeX
    output_path = RESULTS_DIR / "rq3" / "tool_use_table.tex"
    with open(output_path, "w") as f:
        f.write(generate_latex_table(results))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
