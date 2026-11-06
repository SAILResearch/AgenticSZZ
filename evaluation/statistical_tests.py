#!/usr/bin/env python3
"""
Statistical tests for AgenticSZZ paper (TOSEM).

McNemar's test for paired binary outcomes with Cohen's g effect size.
- Cohen's g ranges from 0 to 0.5
- Interpretation: small ≈ 0.05, medium ≈ 0.15, large ≈ 0.25

Usage:
    python evaluation/statistical_tests.py               # all tests
    python evaluation/statistical_tests.py --test main       # Table 2, our best vs LLM4SZZ best
    python evaluation/statistical_tests.py --test robust     # our worst vs LLM4SZZ best
    python evaluation/statistical_tests.py --test ablation   # Table 4, full vs each ablation
    python evaluation/statistical_tests.py --test tc_szz     # Section 6.1, our best vs TC-SZZ on DS_LINUX
"""

import json
import glob
import re
import argparse
from pathlib import Path
from scipy.stats import binom


BASE_DIR = Path(__file__).parent.parent

# Result directories behind the reported numbers. The three AgenticSZZ RQ1 runs
# and the three LLM4SZZ runs both use DeepSeek-V4-Pro-0813.
RQ1_RUNS_DIR = BASE_DIR / 'results/rq1'
RQ1_RUN_PREFIX = 'running_'
LLM4SZZ_DEEPSEEK_DIRS = [
    BASE_DIR / 'results/llm4szz/run_1/save_logs',
    BASE_DIR / 'results/llm4szz/run_2/save_logs',
    BASE_DIR / 'results/llm4szz/run_3/save_logs',
]


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def mcnemar_test(a: int, b: int, c: int, d: int) -> dict:
    """McNemar's exact binomial test with Cohen's g effect size.

    Contingency table (Method A = AgenticSZZ, Method B = baseline):
                      B correct   B wrong
    A correct:            a          b
    A wrong:              c          d

    b = A right, B wrong  (A wins)
    c = A wrong, B right  (B wins)
    """
    n = b + c
    if n == 0:
        return {
            'n_discordant': 0, 'p_value': 1.0, 'odds_ratio': None,
            'cohens_g': 0.0, 'effect_size_interpretation': 'none',
            'significant_001': False, 'significant_01': False, 'significant_05': False,
        }
    p_value = min(1.0, binom.cdf(min(b, c), n, 0.5) * 2)
    odds_ratio = c / b if b > 0 else float('inf')
    proportion = b / n
    cohens_g = abs(proportion - 0.5)
    if cohens_g < 0.05:
        interp = 'negligible'
    elif cohens_g < 0.15:
        interp = 'small'
    elif cohens_g < 0.25:
        interp = 'medium'
    else:
        interp = 'large'
    return {
        'n_discordant': n, 'a': a, 'b': b, 'c': c, 'd': d,
        'p_value': p_value, 'odds_ratio': odds_ratio,
        'cohens_g': cohens_g, 'effect_size_interpretation': interp,
        'significant_001': p_value < 0.001,
        'significant_01': p_value < 0.01,
        'significant_05': p_value < 0.05,
    }


def print_result(name: str, result: dict, method_a: str = "AgenticSZZ", method_b: str = "Baseline"):
    sig = "***" if result['significant_001'] else "**" if result['significant_01'] else "*" if result['significant_05'] else ""
    print(f"\n{name}:")
    print(f"  Contingency: a={result.get('a','?')}, b={result.get('b','?')}, c={result.get('c','?')}, d={result.get('d','?')}")
    print(f"  Discordant pairs: {result['n_discordant']}")
    print(f"  p-value: {result['p_value']:.6f} {sig}")
    if result['odds_ratio'] is not None:
        print(f"  Odds ratio (c/b): {result['odds_ratio']:.3f}")
    print(f"  Cohen's g: {result['cohens_g']:.3f} ({result['effect_size_interpretation']})")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_gt_lookup() -> dict:
    """Load BFC -> list[BIC] ground truth."""
    with open(BASE_DIR / 'dataset/final_all_dataset.json') as f:
        all_info = json.load(f)
    gt = {}
    for item in all_info:
        bfc = item['bug_fixing_commit']
        bic = item['bug_inducing_commit']
        if isinstance(bic, str):
            bic = [bic]
        gt[bfc] = bic
    return gt


def find_agentic_file(run_dir: Path, dataset: str, config: str = 'full') -> Path | None:
    """Find AgenticSZZ result file for a given dataset/config (most recent if multiple)."""
    matches = sorted(run_dir.glob(f"{dataset}_{config}_*.json"))
    return matches[-1] if matches else None


def load_agentic_run(run_dir: Path, datasets=('linux', 'github', 'apache')) -> dict:
    """Load AgenticSZZ full-config results from a run directory.

    Returns:
        dict: bfc -> {'correct': bool, 'dataset': 'DS_LINUX' | ...}
    """
    results = {}
    for ds in datasets:
        f = find_agentic_file(run_dir, ds, 'full')
        if f is None:
            continue
        with open(f) as fh:
            data = json.load(fh)
        for case in data:
            results[case['bfc']] = {'correct': case['success'], 'dataset': f'DS_{ds.upper()}'}
    return results


def run_accuracy(run_dir: Path) -> float:
    """Total correct / total cases across all datasets (full config)."""
    total_correct = total_cases = 0
    for ds in ('linux', 'github', 'apache'):
        f = find_agentic_file(run_dir, ds, 'full')
        if f is None:
            continue
        with open(f) as fh:
            data = json.load(fh)
        total_correct += sum(1 for c in data if c['success'])
        total_cases += len(data)
    return total_correct / total_cases if total_cases else 0.0


def identify_runs() -> dict:
    """Identify best, middle, worst AgenticSZZ runs from RQ1_RUNS_DIR.

    Returns:
        dict with keys 'best', 'middle', 'worst', each a Path
    """
    # Exactly running_<n>; aborted attempts keep a suffix (running_1_FAILED_*) and are skipped.
    run_dirs = [d for d in sorted(RQ1_RUNS_DIR.iterdir())
                if d.is_dir() and re.fullmatch(rf'{RQ1_RUN_PREFIX}\d+', d.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No {RQ1_RUN_PREFIX}<n> directories found in {RQ1_RUNS_DIR}")

    accs = {d: run_accuracy(d) for d in run_dirs}
    sorted_runs = sorted(accs.items(), key=lambda x: x[1])

    print("\nAgenticSZZ run accuracies (full config, all datasets):")
    for run_dir, acc in sorted_runs:
        print(f"  {run_dir.name}: {acc:.4f}")

    worst = sorted_runs[0][0]
    best = sorted_runs[-1][0]
    middle = sorted_runs[len(sorted_runs) // 2][0]
    print(f"  -> Best: {best.name}, Middle: {middle.name}, Worst: {worst.name}")
    return {'best': best, 'middle': middle, 'worst': worst}


def _sha_match(sha1: str, sha2: str) -> bool:
    return bool(sha1 and sha2 and (sha1.startswith(sha2) or sha2.startswith(sha1)))


def _extract_llm4szz_prediction(log_file: Path) -> list:
    with open(log_file) as f:
        logs = json.load(f)
    s2_cids, s1_cids, can_determine = [], [], True
    for log in logs:
        if isinstance(log, dict):
            if 'can_determine' in log:
                can_determine = can_determine and log['can_determine']
            if 's2_final_cid' in log:
                s2_cids.extend(log['s2_final_cid'])
            if 's1_ranked_stmts_infos' in log:
                for info in log['s1_ranked_stmts_infos'][:5]:
                    if 'induce_cid' in info:
                        s1_cids.append(info['induce_cid'])
    return list(set(s2_cids)) if (can_determine and s2_cids) else list(set(s1_cids))


def _llm4szz_correct(preds: list, gt: list) -> bool:
    return any(_sha_match(p, g) for p in preds for g in gt)


def load_llm4szz_runs(save_log_dir: Path, gt_lookup: dict, run_ids: list) -> dict:
    """Load LLM4SZZ results from a save_logs directory.

    Each BFC dir contains llm4szz{run_id}.json files.

    Returns:
        dict: bfc -> {run_id: bool}
    """
    results = {}
    if not save_log_dir.exists():
        return results
    for repo_dir in save_log_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        for bfc_dir in repo_dir.iterdir():
            if not bfc_dir.is_dir():
                continue
            bfc = bfc_dir.name
            gt = gt_lookup.get(bfc, [])
            if not gt:
                for full_bfc in gt_lookup:
                    if _sha_match(bfc, full_bfc[:9]):
                        bfc = full_bfc
                        gt = gt_lookup[full_bfc]
                        break
            results.setdefault(bfc, {})
            for run_id in run_ids:
                log_path = bfc_dir / f'llm4szz{run_id}.json'
                if not log_path.exists():
                    continue
                try:
                    preds = _extract_llm4szz_prediction(log_path)
                    results[bfc][run_id] = _llm4szz_correct(preds, gt)
                except Exception:
                    pass
    return results


def load_llm4szz_best_run(gt_lookup: dict) -> dict:
    """Load the three LLM4SZZ runs and return the best one as bfc -> bool.

    Each run is its own directory holding one llm4szz0.json per case.
    """
    all_runs = {}
    run_correct, run_total = {}, {}
    for run_id, save_log_dir in enumerate(LLM4SZZ_DEEPSEEK_DIRS):
        run_results = load_llm4szz_runs(save_log_dir, gt_lookup, [0])
        correct = total = 0
        for bfc, runs_data in run_results.items():
            if 0 in runs_data:
                total += 1
                correct += bool(runs_data[0])
            all_runs.setdefault(bfc, {})[run_id] = runs_data.get(0, False)
        run_correct[run_id], run_total[run_id] = correct, total

    print("\nLLM4SZZ per-run accuracy:")
    for run_id, total in run_total.items():
        if total > 0:
            print(f"  Run {run_id}: {run_correct[run_id]}/{total} = {run_correct[run_id]/total:.4f}")

    valid = [r for r, total in run_total.items() if total > 0]
    best_run = max(valid, key=lambda r: run_correct[r] / run_total[r])
    print(f"  -> Best run: Run {best_run} ({run_correct[best_run]/run_total[best_run]:.4f})")

    return {bfc: runs[best_run] for bfc, runs in all_runs.items() if best_run in runs}


def match_and_test(agentic: dict, baseline: dict, label: str, method_b: str) -> dict:
    """Pair cases, build contingency table, run McNemar's test per dataset + overall."""
    matched = []
    for bfc, info in agentic.items():
        b_correct = baseline.get(bfc)
        if b_correct is None:
            for b_bfc in baseline:
                if _sha_match(bfc, b_bfc):
                    b_correct = baseline[b_bfc]
                    break
        if b_correct is not None:
            matched.append({'bfc': bfc, 'dataset': info['dataset'],
                             'agentic': info['correct'], 'baseline': b_correct})

    print(f"\nMatched cases: {len(matched)}")
    results = {}

    for ds in ['DS_LINUX', 'DS_GITHUB', 'DS_APACHE', 'ALL']:
        subset = [m for m in matched if ds == 'ALL' or m['dataset'] == ds]
        if not subset:
            continue
        a = sum(1 for m in subset if m['agentic'] and m['baseline'])
        b = sum(1 for m in subset if m['agentic'] and not m['baseline'])
        c = sum(1 for m in subset if not m['agentic'] and m['baseline'])
        d = sum(1 for m in subset if not m['agentic'] and not m['baseline'])
        key = ds.lower().replace('ds_', '')
        results[key] = mcnemar_test(a, b, c, d)
        print_result(f"{label} — {ds} (n={len(subset)})", results[key], "AgenticSZZ", method_b)

    return results


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_main_comparison() -> dict:
    """Table 2: AgenticSZZ best run vs LLM4SZZ best run, both on DeepSeek-V4-Pro."""
    print("=" * 70)
    print("Main comparison: AgenticSZZ BEST vs LLM4SZZ BEST")
    print("=" * 70)
    gt = load_gt_lookup()
    agentic = load_agentic_run(identify_runs()['best'])
    baseline = load_llm4szz_best_run(gt)
    return match_and_test(agentic, baseline, "Best vs Best", "LLM4SZZ")


def test_robustness() -> dict:
    """Robustness: AgenticSZZ WORST run vs LLM4SZZ BEST run."""
    print("=" * 70)
    print("Robustness: AgenticSZZ WORST vs LLM4SZZ BEST")
    print("=" * 70)
    gt = load_gt_lookup()
    agentic = load_agentic_run(identify_runs()['worst'])
    baseline = load_llm4szz_best_run(gt)
    return match_and_test(agentic, baseline, "Worst vs Best", "LLM4SZZ")


def test_tc_szz() -> dict:
    """Section 6.1: AgenticSZZ best run vs TC-SZZ on DS_LINUX.

    TC-SZZ returns every commit in the change history of the fixed lines, so a
    case counts as correct for TC-SZZ when any returned commit is a ground-truth
    BIC. This favours TC-SZZ. Its released predictions cover DS_LINUX only.
    """
    from tc_szz_baseline import RAW, DS_LINUX

    print("=" * 70)
    print("TC-SZZ comparison: AgenticSZZ BEST vs TC-SZZ (DS_LINUX)")
    print("=" * 70)
    with open(DS_LINUX) as f:
        truth = {c['fix_commit_hash']: set(c['bug_commit_hash']) for c in json.load(f)}
    with open(RAW / 'bic_tc_bugfix_commits_all.json') as f:
        cases = json.load(f)
    baseline = {c['fix_commit_hash']: bool(set(c['inducing_commit_hash']) & truth[c['fix_commit_hash']])
                for c in cases if c['fix_commit_hash'] in truth}
    agentic = load_agentic_run(identify_runs()['best'], datasets=('linux',))
    return match_and_test(agentic, baseline, "Best vs TC-SZZ", "TC-SZZ")


def test_ablation_study() -> dict:
    """RQ2 ablation: full pipeline vs each ablation config."""
    print("=" * 70)
    print("Ablation study: Full pipeline vs each ablation")
    print("=" * 70)

    # Non-LLM configurations share one output directory. Agent configurations
    # have separate directories. The no-expansion arm runs the full config with
    # --context-expand 0, so its result filenames contain 'full'.
    ablation_dir = BASE_DIR / 'results/rq2'
    ablation_dirs = {
        'blame_only': (ablation_dir, 'blame_only'),
        'blame_fallback': (ablation_dir, 'blame_fallback'),
        'tkg_only': (ablation_dir, 'tkg_only'),
        'agent_only': (BASE_DIR / 'results/rq2/agent_only', 'agent_only'),
        'full_no_graph': (BASE_DIR / 'results/rq2/full_no_graph', 'full_no_graph'),
        'no_expansion': (BASE_DIR / 'results/rq2/no_expansion', 'full'),
    }
    full_dir = RQ1_RUNS_DIR / 'running_1'  # the run the ablation table compares against
    datasets = ['linux', 'github', 'apache']

    print(f"Full pipeline from: {full_dir.name}")

    # Load full pipeline results
    full_results = {}
    for ds in datasets:
        f = find_agentic_file(full_dir, ds, 'full')
        if f:
            with open(f) as fh:
                data = json.load(fh)
            for case in data:
                full_results[case['bfc']] = {'correct': case['success'], 'dataset': f'DS_{ds.upper()}'}

    results = {}
    for config, (config_dir, file_config) in ablation_dirs.items():
        print(f"\n{'='*70}")
        print(f"Full vs {config} ({config_dir})")
        ablation_results = {}
        for ds in datasets:
            f = find_agentic_file(config_dir, ds, file_config)
            if f:
                with open(f) as fh:
                    data = json.load(fh)
                for case in data:
                    ablation_results[case['bfc']] = case['success']

        matched = []
        for bfc, info in full_results.items():
            if bfc in ablation_results:
                matched.append({'bfc': bfc, 'dataset': info['dataset'],
                                 'agentic': info['correct'], 'baseline': ablation_results[bfc]})

        for ds in ['DS_LINUX', 'DS_GITHUB', 'DS_APACHE', 'ALL']:
            subset = [m for m in matched if ds == 'ALL' or m['dataset'] == ds]
            if not subset:
                continue
            a = sum(1 for m in subset if m['agentic'] and m['baseline'])
            b = sum(1 for m in subset if m['agentic'] and not m['baseline'])
            c = sum(1 for m in subset if not m['agentic'] and m['baseline'])
            d = sum(1 for m in subset if not m['agentic'] and not m['baseline'])
            r = mcnemar_test(a, b, c, d)
            if ds == 'ALL':
                results[config] = r
            print_result(f"Full vs {config} — {ds} (n={len(subset)})", r, "Full", config)

    return results


def generate_summary():
    """Run all tests and print paper-ready summary."""
    print("\n" + "=" * 70)
    print("RUNNING ALL STATISTICAL TESTS")
    print("=" * 70)

    main_r = test_main_comparison()
    robust_r = test_robustness()
    tc_r = test_tc_szz()
    ablation_r = test_ablation_study()

    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)

    r = main_r.get('all', {})
    if r:
        print(f"\nMain comparison (our BEST vs their BEST):")
        print(f"  p={r['p_value']:.4f}, g={r['cohens_g']:.2f} ({r['effect_size_interpretation']})")

    r = robust_r.get('all', {})
    if r:
        print(f"\nRobustness (our WORST vs their BEST):")
        print(f"  p={r['p_value']:.4f}, g={r['cohens_g']:.2f} ({r['effect_size_interpretation']})")

    r = tc_r.get('linux', {})
    if r:
        print(f"\nTC-SZZ (our BEST vs TC-SZZ, DS_LINUX):")
        print(f"  p={r['p_value']:.4f}, g={r['cohens_g']:.2f} ({r['effect_size_interpretation']})")

    if ablation_r:
        gs = [r['cohens_g'] for r in ablation_r.values()]
        print(f"\nAblation study (full vs each ablation, overall):")
        print(f"  g range: {min(gs):.2f} to {max(gs):.2f}")
        for cfg, r in ablation_r.items():
            print(f"  {cfg}: p={r['p_value']:.4f}, g={r['cohens_g']:.2f} ({r['effect_size_interpretation']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical tests for AgenticSZZ paper')
    parser.add_argument('--test', default='all',
                        choices=['all', 'main', 'robust', 'tc_szz', 'ablation'])
    args = parser.parse_args()

    if args.test == 'main':
        test_main_comparison()
    elif args.test == 'robust':
        test_robustness()
    elif args.test == 'tc_szz':
        test_tc_szz()
    elif args.test == 'ablation':
        test_ablation_study()
    else:
        generate_summary()
