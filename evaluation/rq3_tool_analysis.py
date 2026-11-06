#!/usr/bin/env python3
"""
RQ3 Tool-Use Analysis: Analyze agent tool invocation patterns across LLMs.

Generates Table (tab:rq3-tool-use) showing median steps and per-case tool
call rates for each LLM in the RQ3 sensitivity analysis.

The same pass over the trajectories also reports two measurements the revision
needs for the primary model:

  R2.6 (autonomy): how often the two fast-path markers in list_candidates
  output fire, and how often the agent returns the SHA they mark.

  R2.4 (graph): in how many cases the predicted SHA is reachable only through
  a traverse_graph return, which separates what the graph contributes at
  inference time from what it contributes when the candidate set is built.

Usage:
    python evaluation/rq3_tool_analysis.py
"""

import json
import glob
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configs: (display_name, trajectory_dir, result_dir)
# DeepSeek-V4-Pro reuses RQ1 run 1, the highest-F1 of the three RQ1 runs.
MODEL_CONFIGS = [
    ("DeepSeek-V4-Pro", "results/rq1/running_1/trajectories", "results/rq1/running_1"),
    ("DeepSeek-V4.1-Flash", "results/rq3/deepseek-v4.1-flash/trajectories", "results/rq3/deepseek-v4.1-flash"),
    ("Gemma-4-31B", "results/rq3/gemma-4-31b/trajectories", "results/rq3/gemma-4-31b"),
    ("Devstral-Small-2", "results/rq3/devstral-small-2/trajectories", "results/rq3/devstral-small-2"),
    ("Qwen3-Coder-Next", "results/rq3/qwen3-coder-next/trajectories", "results/rq3/qwen3-coder-next"),
]

TOOLS = ["list_candidates", "query_node", "read_node_content", "traverse_graph"]

# The two fast-path markers graphiti_tools.py writes into list_candidates output
# (`:313` and `:319`). They are mutually exclusive branches of the same if/elif.
# The bare ">>> TOP CANDIDATE <<<" at `:209` is a label on the first list entry,
# not a fast path, so it is deliberately not matched here.
SINGLE_MARKER = re.compile(r">>> SINGLE BLAME COMMIT: ([0-9a-f]{7,40}) <<<")
TOP_MARKER = re.compile(r">>> TOP CANDIDATE: ([0-9a-f]{7,40}) \(>70% of blamed lines\) <<<")
SHA_TOKEN = re.compile(r"\b[0-9a-f]{7,40}\b")

# The candidate-list header written at `graphiti_tools.py:332`. Shown < total
# means K truncated the list, which is the only situation in which traverse_graph
# can surface a commit that list_candidates withheld. Counting those cases is
# what shows the traversal-only measure is reachable rather than vacuous.
LIST_HEADER = re.compile(r"=== TOP (\d+) CANDIDATES \(of (\d+) total")


def mentions(text, sha):
    """True if text carries a SHA matching sha, at either abbreviation length."""
    return any(
        sha.startswith(token) or token.startswith(sha)
        for token in SHA_TOKEN.findall(text)
    )


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
    autonomy = defaultdict(Counter)  # dataset -> R2.6 and R2.4 counts
    shortcut = {}  # case_id -> "single", "top" or None, from the list_candidates markers

    for f in files:
        with open(f) as fh:
            traj = json.load(fh)

        case_id = traj.get("case_id", "")
        if success_map.get(case_id, False):
            correct += 1

        predicted = (traj.get("predicted_bic") or "").strip().lower()
        single_sha = None
        top_sha = None
        seen_in_traversal = False
        seen_elsewhere = False
        truncated = False

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
            else:
                for part in step.get("parts", []):
                    if not isinstance(part, dict) or part.get("type") != "ToolReturnPart":
                        continue
                    tool = part.get("tool_name")
                    content = str(part.get("content", ""))
                    if tool == "list_candidates":
                        single = SINGLE_MARKER.search(content)
                        if single and single_sha is None:
                            single_sha = single.group(1)
                        top = TOP_MARKER.search(content)
                        if top and top_sha is None:
                            top_sha = top.group(1)
                        header = LIST_HEADER.search(content)
                        if header and int(header.group(1)) < int(header.group(2)):
                            truncated = True
                    if predicted and mentions(content, predicted):
                        if tool == "traverse_graph":
                            seen_in_traversal = True
                        else:
                            seen_elsewhere = True

        steps_list.append(len(case_tools))
        if "read_node_content" not in case_tools:
            no_diff_read += 1

        shortcut[case_id] = "single" if single_sha else "top" if top_sha else None

        # Cases with no prediction cannot obey a marker, so they are excluded.
        if predicted:
            # The dataset field is null in trajectory files; case_id carries it
            # as a prefix, e.g. "APACHE_00b29b4298".
            counts = autonomy[case_id.split("_")[0] or "UNKNOWN"]
            counts["cases"] += 1
            solved = success_map.get(case_id, False)
            # Accuracy split by whether a marker fired. Whether short-circuiting
            # is a sound design or a crutch depends on how the cases it fires on
            # compare with the ones it leaves to the agent.
            counts["fired_correct" if (single_sha or top_sha) else "unmarked_correct"] += solved
            counts["unmarked"] += not (single_sha or top_sha)
            if single_sha:
                counts["single_fired"] += 1
                counts["single_obeyed"] += predicted.startswith(single_sha)
                counts["single_correct"] += solved
            if top_sha:
                counts["top_fired"] += 1
                counts["top_obeyed"] += predicted.startswith(top_sha)
                counts["top_correct"] += solved
            if single_sha or top_sha:
                counts["any_fired"] += 1
                counts["any_obeyed"] += (
                    single_sha is not None and predicted.startswith(single_sha)
                ) or (top_sha is not None and predicted.startswith(top_sha))
            counts["traverse_called"] += "traverse_graph" in case_tools
            counts["traversal_only"] += seen_in_traversal and not seen_elsewhere
            counts["truncated"] += truncated
            counts["truncated_traversed"] += truncated and "traverse_graph" in case_tools

    median_steps = sorted(steps_list)[total // 2] if total else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100 if total else 0,
        "median_steps": median_steps,
        "mean_steps": sum(steps_list) / total if total else 0,
        "tool_rates": {t: tool_counts.get(t, 0) / total for t in TOOLS},
        "no_diff_pct": no_diff_read / total * 100 if total else 0,
        "autonomy": autonomy,
        "shortcut": shortcut,
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


def print_autonomy(name, stats):
    """Report the R2.6 fast-path and R2.4 graph-use measurements for one model."""
    autonomy = stats["autonomy"]
    totals = Counter()
    for counts in autonomy.values():
        totals.update(counts)

    def pct(numerator, denominator):
        return f"{numerator / denominator * 100:.1f}%" if denominator else "n/a"

    print()
    print("=" * 80)
    print(f"Fast-path autonomy (R2.6) and graph use (R2.4): {name}")
    print("=" * 80)
    print(f"{'':<12}{'cases':>7}{'SINGLE':>16}{'obeyed':>9}{'TOP':>16}{'obeyed':>9}")
    for ds in sorted(autonomy) + ["All"]:
        c = totals if ds == "All" else autonomy[ds]
        print(
            f"{ds:<12}{c['cases']:>7}"
            f"{c['single_fired']:>9} {pct(c['single_fired'], c['cases']):>6}"
            f"{pct(c['single_obeyed'], c['single_fired']):>9}"
            f"{c['top_fired']:>9} {pct(c['top_fired'], c['cases']):>6}"
            f"{pct(c['top_obeyed'], c['top_fired']):>9}"
        )
    print(
        f"\nA fast path fires in {pct(totals['any_fired'], totals['cases'])} of cases, "
        f"and the marked SHA is returned in {pct(totals['any_obeyed'], totals['any_fired'])} of those."
    )
    print(
        f"Accuracy where a marker fired: {pct(totals['fired_correct'], totals['any_fired'])} "
        f"({totals['fired_correct']}/{totals['any_fired']}), "
        f"SINGLE {pct(totals['single_correct'], totals['single_fired'])}, "
        f"TOP {pct(totals['top_correct'], totals['top_fired'])}. "
        f"Where none fired: {pct(totals['unmarked_correct'], totals['unmarked'])} "
        f"({totals['unmarked_correct']}/{totals['unmarked']})."
    )
    print(f"Trajectory length: mean {stats['mean_steps']:.1f} tool calls, "
          f"median {stats['median_steps']}.")

    print(f"\n{'':<12}{'cases':>7}{'traverse_graph called':>24}{'answer only via traversal':>28}")
    for ds in sorted(autonomy) + ["All"]:
        c = totals if ds == "All" else autonomy[ds]
        print(
            f"{ds:<12}{c['cases']:>7}"
            f"{c['traverse_called']:>17} {pct(c['traverse_called'], c['cases']):>6}"
            f"{c['traversal_only']:>21} {pct(c['traversal_only'], c['cases']):>6}"
        )
    print(
        f"\nK truncated the candidate list in {totals['truncated']} of {totals['cases']} cases "
        f"({pct(totals['truncated'], totals['cases'])}), and traverse_graph was called in "
        f"{totals['truncated_traversed']} of those."
    )


def blame_shortcut(record):
    """The shortcut list_candidates would mark for a case, recomputed from blame.

    Runs that end without an answer store no trajectory, so their markers are
    not on disk. The markers depend only on the blame distribution
    (tkg.py:_add_blame_stats_to_bfc), which git blame reproduces.
    """
    from agentic_szz.config.settings import get_config
    from agentic_szz.szz.blame import BlameAnalyzer

    repos = PROJECT_ROOT / "dataset" / "subject_repos"
    if record["dataset"] == "linux":
        repo = repos / "linux"
    elif record["dataset"] == "github":
        with open(PROJECT_ROOT / "dataset" / "DS_GITHUB.json") as fh:
            owner_repo = {d["fix_commit_hash"]: d["repo_name"] for d in json.load(fh)}
        repo = repos / owner_repo.get(record["bfc"], record["project"])
    else:
        repo = repos / record["project"]
    lines = BlameAnalyzer(
        str(repo),
        use_context_fallback=True,
        context_expand_lines=get_config().tkg.context_expand_lines,
    ).analyze(record["bfc"]).blame_distribution
    if len(lines) == 1:
        return "single"
    if lines and max(lines.values()) / sum(lines.values()) > 0.7:
        return "top"
    return None


def print_full_size(stats, result_dir):
    """Shortcut rates and accuracy over every case of the run (R2.6, Section 7.1).

    Runs without an answer count as misses, as in Table 2, and are assigned
    their shortcut from blame. Whether the agent returns the marked commit is
    a property of an answer, so print_autonomy keeps reporting that over
    answered runs only.
    """
    records = [
        r for f in glob.glob(str(PROJECT_ROOT / result_dir / "*_full_*.json"))
        for r in json.load(open(f))
    ]
    shortcut = dict(stats["shortcut"])
    for r in records:
        if r["case_id"] not in shortcut:
            shortcut[r["case_id"]] = blame_shortcut(r)

    groups = defaultdict(list)
    for r in records:
        groups[shortcut[r["case_id"]]].append(r["success"])
        if r["dataset"] == "linux":
            groups["linux"].append(shortcut[r["case_id"]] == "single")

    def line(label, outcomes):
        return f"  {label:<22}{len(outcomes):>6} ({len(outcomes) / len(records) * 100:.1f}%)  accuracy {sum(outcomes) / len(outcomes) * 100:.1f}%"

    print(f"\nShortcuts over all {len(records)} cases (runs without an answer count as misses)")
    print(line("single blame", groups["single"]))
    print(line("dominant commit", groups["top"]))
    print(line("either shortcut", groups["single"] + groups["top"]))
    print(line("no shortcut", groups[None]))
    print(line("not single blame", groups["top"] + groups[None]))
    print(f"  DS_LINUX single-blame rate: {sum(groups['linux']) / len(groups['linux']) * 100:.1f}%")


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
    print(f"{'Model':<24s} {'Cases':>6} {'Acc%':>6} {'Steps':>6} {'list_cand':>10} {'query_node':>11} {'read_diff':>10} {'traverse':>9} {'No-diff%':>9}")
    print("-" * 95)
    for name, s in results:
        tr = s["tool_rates"]
        print(
            f"{name:<24s} {s['total']:>6} {s['accuracy']:>5.1f}% {s['median_steps']:>6} "
            f"{tr['list_candidates']:>10.2f} {tr['query_node']:>11.2f} "
            f"{tr['read_node_content']:>10.2f} {tr['traverse_graph']:>9.2f} {s['no_diff_pct']:>8.1f}%"
        )

    if results:
        print_autonomy(*results[0])
        print_full_size(results[0][1], MODEL_CONFIGS[0][2])

    print()
    print("=" * 80)
    print("LaTeX table:")
    print("=" * 80)
    print(generate_latex_table(results))

    # Save LaTeX
    output_path = RESULTS_DIR / "rq3" / "tool_use_table.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(generate_latex_table(results))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
