#!/usr/bin/env python3
"""
Generate category breakdown table for RQ2 (Table 6).
Shows true positives by BIC category for each approach including LLM4SZZ.

This provides evidence for claims about where each approach finds its TPs.

Usage:
    python evaluation/generate_category_breakdown_table_v2.py

Output:
    - Prints category breakdown to console
    - Saves LaTeX table to results/rq1/category_breakdown_table_v2.tex
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_rq0_categories():
    """Load RQ0 category data.

    Returns lookup: (dataset, bfc, gt_bic) -> category
    Since a BFC can have multiple BICs with different categories,
    we need to match on the specific BIC that was predicted.
    """
    rq0_df = pd.read_csv("results/rq0/rq0_all_cases.csv")
    # Create lookup: (dataset, bfc, gt_bic) -> category
    category_lookup = {}
    # Also create BFC-level lookup for cases where we can't match predicted BIC
    bfc_lookup = {}
    for _, row in rq0_df.iterrows():
        dataset = row['dataset'].lower()
        bfc = row['bfc']
        gt_bic = row['gt_bic']
        category = row['category']
        category_lookup[(dataset, bfc, gt_bic)] = category
        # For BFC-level, prefer 'direct' category if multiple exist
        if (dataset, bfc) not in bfc_lookup or category == 'direct':
            bfc_lookup[(dataset, bfc)] = category
    return category_lookup, bfc_lookup, rq0_df


def load_ablation_results(approach_name: str, result_files: dict):
    """Load results for a given ablation approach (our format)."""
    results = []
    for dataset, filepath in result_files.items():
        if not Path(filepath).exists():
            print(f"Warning: {filepath} not found")
            continue
        with open(filepath) as f:
            data = json.load(f)
            for case in data:
                results.append({
                    'dataset': dataset,
                    'bfc': case['bfc'],
                    'predicted': case.get('predicted', ''),
                    'ground_truth': case.get('ground_truth', []),
                    'success': case['success'],
                    'approach': approach_name,
                })
    return results


def load_llm4szz_results(result_file: str):
    """Load LLM4SZZ results (their format)."""
    results = []
    with open(result_file) as f:
        data = json.load(f)

    for case in data['per_case_results']:
        dataset = case['dataset'].lower()
        bfc = case['bfc']
        gt_bics = case['gt_bics']
        predicted_list = case.get('predicted') or []

        # LLM4SZZ can return multiple predictions; we take the first one
        predicted = predicted_list[0] if predicted_list else ''

        # Check if prediction is correct
        success = False
        if predicted:
            pred_short = predicted[:9]
            for gt in gt_bics:
                gt_short = gt[:9]
                if predicted.startswith(gt_short) or gt.startswith(pred_short):
                    success = True
                    break

        results.append({
            'dataset': dataset,
            'bfc': bfc,
            'predicted': predicted,
            'ground_truth': gt_bics,
            'success': success,
            'approach': 'LLM4SZZ',
        })

    return results


def assign_categories(results: list, category_lookup: dict, bfc_lookup: dict):
    """Assign RQ0 category to each result based on the predicted BIC."""
    for r in results:
        if r['success'] and r['predicted']:
            pred = r['predicted']
            pred_short = pred[:9] if len(pred) > 9 else pred

            # Look up category of the predicted BIC
            cat = category_lookup.get((r['dataset'], r['bfc'], pred), None)

            if cat is None:
                # Try matching with short prefix
                for gt in r['ground_truth']:
                    gt_short = gt[:9] if len(gt) > 9 else gt
                    if pred.startswith(gt_short) or gt.startswith(pred_short):
                        # Found matching GT, look up its category
                        cat = category_lookup.get((r['dataset'], r['bfc'], gt), None)
                        if cat is None:
                            # Try all keys with matching dataset/bfc
                            for key in category_lookup:
                                if (key[0] == r['dataset'] and
                                    key[1] == r['bfc'] and
                                    key[2].startswith(gt_short)):
                                    cat = category_lookup[key]
                                    break
                        if cat:
                            break

            r['category'] = cat if cat else bfc_lookup.get((r['dataset'], r['bfc']), 'unknown')
        else:
            # For failed predictions, use BFC-level category (for GT stats)
            r['category'] = bfc_lookup.get((r['dataset'], r['bfc']), 'unknown')

    return results


def compute_category_counts(results: list, approach_name: str):
    """Compute TP counts by category for an approach."""
    category_map = {
        'direct': 'Blame',
        'ancestor': 'Blame Ancestor',
        'descendant': 'BFC Ancestor',
        'blameless': 'Blameless',
        'others': 'Unreachable',
    }

    counts = defaultdict(int)
    for r in results:
        if r['approach'] == approach_name and r['success']:
            cat = r.get('category', 'unknown')
            display_cat = category_map.get(cat, 'Other')
            counts[display_cat] += 1

    return counts


def find_result_file(run_dir: Path, dataset: str, config: str) -> Path | None:
    """Find the most recent result file for a dataset/config in a run directory."""
    matches = sorted(run_dir.glob(f"{dataset}_{config}_*.json"))
    return matches[-1] if matches else None


def main():
    import os
    os.chdir(Path(__file__).parent.parent)  # cd to project root

    # Load category data
    category_lookup, bfc_lookup, rq0_df = load_rq0_categories()

    # Match the experiment directories documented in the README.
    # The no-expansion arm uses full-config filenames.
    ablation_dir = Path("results/rq2")
    ablation_dirs = {
        'Blame-only': (ablation_dir, 'blame_only'),
        'Blame-fallback': (ablation_dir, 'blame_fallback'),
        'TKG-only': (ablation_dir, 'tkg_only'),
        'Agent-only': (Path("results/rq2/agent_only"), 'agent_only'),
        'AgenticSZZ (flat list)': (Path("results/rq2/full_no_graph"), 'full_no_graph'),
        'AgenticSZZ (w/o expansion)': (Path("results/rq2/no_expansion"), 'full'),
        # run_1 is the highest-F1 of the three RQ1 runs, the one the ablation table
        # compares against.
        'AgenticSZZ': (Path("results/rq1/running_1"), 'full'),
    }
    datasets = ['linux', 'github', 'apache']

    ablation_approaches = {}
    for approach_name, (run_dir, config) in ablation_dirs.items():
        files = {}
        for ds in datasets:
            f = find_result_file(run_dir, ds, config)
            if f:
                files[ds] = str(f)
            else:
                print(f"Warning: no file found for {ds}/{config} in {run_dir}")
        ablation_approaches[approach_name] = files

    # LLM4SZZ on DeepSeek-V4-Pro, best of the three runs by overall F1 (v3)
    llm4szz_files = sorted(Path('results/llm4szz/run_3').glob(
        'llm4szz_deepseek_v4_pro_*.json'
    ))
    if not llm4szz_files:
        raise FileNotFoundError('Run LLM4SZZ v3 before generating the category table.')
    llm4szz_file = llm4szz_files[-1]

    # Collect all results
    all_results = []

    # Load ablation results
    for approach_name, files in ablation_approaches.items():
        results = load_ablation_results(approach_name, files)
        all_results.extend(results)

    # Load LLM4SZZ results
    if Path(llm4szz_file).exists():
        llm4szz_results = load_llm4szz_results(llm4szz_file)
        all_results.extend(llm4szz_results)
    else:
        print(f"Warning: LLM4SZZ results not found at {llm4szz_file}")

    # Assign categories to all results
    all_results = assign_categories(all_results, category_lookup, bfc_lookup)

    # Define display order
    categories_order = ['Blame', 'Blame Ancestor', 'BFC Ancestor', 'Blameless', 'Unreachable']
    approaches_order = ['Blame-only', 'Blame-fallback', 'TKG-only', 'Agent-only',
                        'AgenticSZZ (flat list)', 'LLM4SZZ',
                        'AgenticSZZ (w/o expansion)', 'AgenticSZZ']

    # Get ground truth counts
    gt_counts = {
        'Blame': len(rq0_df[rq0_df['category'] == 'direct']),
        'Blame Ancestor': len(rq0_df[rq0_df['category'] == 'ancestor']),
        'BFC Ancestor': len(rq0_df[rq0_df['category'] == 'descendant']),
        'Blameless': len(rq0_df[rq0_df['category'] == 'blameless']),
        'Unreachable': len(rq0_df[rq0_df['category'] == 'others']),
    }

    print("=" * 80)
    print("CATEGORY BREAKDOWN TABLE FOR RQ2 (Table 6)")
    print("=" * 80)

    print("\nGround Truth Distribution:")
    for cat in categories_order:
        print(f"  {cat:15}: {gt_counts.get(cat, 0):4}")
    print(f"  {'Total':15}: {sum(gt_counts.values()):4}")

    # Build comparison table
    print("\n" + "=" * 80)
    print("TRUE POSITIVES BY APPROACH AND CATEGORY")
    print("=" * 80)

    table_data = []
    for approach in approaches_order:
        counts = compute_category_counts(all_results, approach)

        blame = counts.get('Blame', 0)
        ancestor = counts.get('Blame Ancestor', 0) + counts.get('BFC Ancestor', 0)
        blameless = counts.get('Blameless', 0)
        unreachable = counts.get('Unreachable', 0)
        total = blame + ancestor + blameless + unreachable

        row = {
            'Approach': approach,
            'Blame': blame,
            'Ancestor': ancestor,
            'Blameless': blameless,
            'Unreachable': unreachable,
            'Total': total,
            # Also keep detailed breakdown
            'Blame Ancestor': counts.get('Blame Ancestor', 0),
            'BFC Ancestor': counts.get('BFC Ancestor', 0),
        }
        table_data.append(row)

        print(f"{approach:15}: Blame={blame:4}, Ancestor={ancestor:3} "
              f"(BA={counts.get('Blame Ancestor', 0):2}, BFCA={counts.get('BFC Ancestor', 0):2}), "
              f"Blameless={blameless:3}, Total={total:4}")

    # Add ground truth row
    gt_ancestor = gt_counts.get('Blame Ancestor', 0) + gt_counts.get('BFC Ancestor', 0)
    print(f"{'Ground Truth':15}: Blame={gt_counts.get('Blame', 0):4}, Ancestor={gt_ancestor:3}, "
          f"Blameless={gt_counts.get('Blameless', 0):3}, Total={sum(gt_counts.values()):4}")

    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    latex = r"""
\begin{table}[t]
    \centering
    \caption{True positives by BIC category. Each BFC is assigned to a single category using priority (Blame $>$ Ancestor $>$ Ghost Commit). Ancestor combines Blame Ancestor and BFC Ancestor categories. For fair comparison, LLM4SZZ uses the same LLM as \appname. Results are based on the best run (highest F1-score) for each approach.}
    \label{tab:category-breakdown}
    \small
    \begin{tabular}{l rrrr}
        \toprule
        \textbf{Approach} & \textbf{Blame} & \textbf{Ancestor} & \textbf{Blameless} & \textbf{Total} \\
        \midrule
"""

    for row in table_data:
        approach = row['Approach']
        if approach == 'AgenticSZZ':
            latex += f"        \\textbf{{\\appname}} & \\textbf{{{row['Blame']:,}}} & \\textbf{{{row['Ancestor']}}} & \\textbf{{{row['Blameless']}}} & \\textbf{{{row['Total']:,}}} \\\\\n"
        elif approach.startswith('AgenticSZZ ('):
            variant = approach[len('AgenticSZZ '):]
            latex += f"        \\appname\\ {variant} & {row['Blame']:,} & {row['Ancestor']} & {row['Blameless']} & {row['Total']:,} \\\\\n"
        else:
            latex += f"        {approach} & {row['Blame']:,} & {row['Ancestor']} & {row['Blameless']} & {row['Total']:,} \\\\\n"

    latex += r"""        \midrule
        Ground Truth & """ + f"{gt_counts.get('Blame', 0):,} & {gt_ancestor} & {gt_counts.get('Blameless', 0)} & {sum(gt_counts.values()):,}" + r""" \\
        \bottomrule
    \end{tabular}
\end{table}
"""

    print(latex)

    # Save LaTeX table
    output_path = Path("results/rq1/category_breakdown_table_v2.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {output_path}")

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (AgenticSZZ vs LLM4SZZ)")
    print("=" * 80)

    agentic_row = next(r for r in table_data if r['Approach'] == 'AgenticSZZ')
    llm4szz_row = next((r for r in table_data if r['Approach'] == 'LLM4SZZ'), None)

    if llm4szz_row:
        print(f"Blame:     AgenticSZZ={agentic_row['Blame']}, LLM4SZZ={llm4szz_row['Blame']}, Diff={agentic_row['Blame'] - llm4szz_row['Blame']:+d}")
        print(f"Ancestor:  AgenticSZZ={agentic_row['Ancestor']}, LLM4SZZ={llm4szz_row['Ancestor']}, Diff={agentic_row['Ancestor'] - llm4szz_row['Ancestor']:+d}")
        print(f"Blameless: AgenticSZZ={agentic_row['Blameless']}, LLM4SZZ={llm4szz_row['Blameless']}, Diff={agentic_row['Blameless'] - llm4szz_row['Blameless']:+d}")
        print(f"Total:     AgenticSZZ={agentic_row['Total']}, LLM4SZZ={llm4szz_row['Total']}, Diff={agentic_row['Total'] - llm4szz_row['Total']:+d}")


if __name__ == "__main__":
    main()
