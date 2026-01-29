#!/usr/bin/env python3
"""
Stratified Sampling Script for AgenticSZZ Pilot Experiment

Creates a stratified sample of cases for preliminary testing before
full-scale evaluation. Samples proportionally by:
- Dataset (Linux, Apache, GitHub)
- Category (Direct, Ancestor, Descendant, Others, Blameless)

Usage:
    python evaluation/create_pilot_sample.py --fraction 0.1 --output results/pilot/pilot_cases.json
    python evaluation/create_pilot_sample.py --fraction 0.2 --output results/pilot/pilot_cases_20pct.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def load_rq0_cases() -> pd.DataFrame:
    """Load all cases from RQ0 analysis, deduplicated by BFC.
    
    The rq0_all_cases.csv has one row per GT BIC, so BFCs with multiple
    GT BICs appear multiple times. We deduplicate by BFC, keeping the
    most informative category for stratified sampling.
    """
    csv_path = PROJECT_ROOT / "results" / "rq0" / "rq0_all_cases.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"RQ0 cases not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Deduplicate by BFC, keeping one representative row per BFC
    # Priority: direct > ancestor > descendant > others > blameless
    # This ensures we sample based on the "most actionable" category
    category_priority = {
        'direct': 0,
        'ancestor': 1, 
        'descendant': 2,
        'others': 3,
        'blameless': 4,
        'error': 5,
    }
    
    # Add priority column for sorting
    df['_priority'] = df['category'].map(lambda x: category_priority.get(x, 99))
    
    # Sort by BFC and priority, then drop duplicates keeping first (lowest priority = best)
    df = df.sort_values(['bfc', '_priority'])
    original_count = len(df)
    df = df.drop_duplicates(subset=['bfc'], keep='first')
    df = df.drop(columns=['_priority'])
    
    print(f"  Deduplicated: {original_count} rows -> {len(df)} unique BFCs")
    
    return df


def stratified_sample(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """
    Perform stratified sampling by dataset and category.
    
    Args:
        df: DataFrame with 'dataset' and 'category' columns
        fraction: Fraction of cases to sample (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    random.seed(seed)
    
    # Group by dataset and category
    samples = []
    for (dataset, category), group in df.groupby(['dataset', 'category']):
        n_total = len(group)
        n_sample = max(1, int(n_total * fraction))  # At least 1 per stratum
        
        # Sample
        sampled = group.sample(n=min(n_sample, n_total), random_state=seed)
        samples.append(sampled)
        print(f"  {dataset}/{category}: {n_sample}/{n_total} cases")
    
    return pd.concat(samples, ignore_index=True)


def check_repos_available(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to cases with available repos."""
    repos_dir = PROJECT_ROOT / "dataset" / "subject_repos"
    
    available = []
    for _, row in df.iterrows():
        dataset = row['dataset']
        repo_name = row['repo_name']
        
        if dataset == 'LINUX':
            repo_path = repos_dir / "linux"
        elif dataset == 'APACHE':
            # Apache repos are in apache/project format or just project
            repo_path = repos_dir / repo_name
            if not repo_path.exists():
                repo_path = repos_dir / "apache" / repo_name
        else:  # GITHUB
            # GitHub repos can have various structures
            repo_path = repos_dir / repo_name
            if not repo_path.exists():
                # Try nested structure (e.g., SimpleServer/SimpleServer)
                repo_path = repos_dir / repo_name / repo_name
        
        if repo_path.exists():
            available.append(row)
    
    return pd.DataFrame(available)


def main():
    parser = argparse.ArgumentParser(description="Create stratified sample for pilot experiment")
    parser.add_argument("--fraction", type=float, default=0.1, 
                       help="Fraction of cases to sample (default: 0.1 = 10%%)")
    parser.add_argument("--output", type=str, default="results/pilot/pilot_cases.json",
                       help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--check-repos", action="store_true",
                       help="Only include cases with available repos")
    
    args = parser.parse_args()
    
    print(f"Loading RQ0 cases...")
    df = load_rq0_cases()
    print(f"  Total cases: {len(df)}")
    
    # Exclude error cases (analysis failures in RQ0)
    error_count = len(df[df['category'] == 'error'])
    if error_count > 0:
        df = df[df['category'] != 'error']
        print(f"  Excluded {error_count} error cases")
    
    if args.check_repos:
        print(f"\nFiltering to available repos...")
        df = check_repos_available(df)
        print(f"  Available cases: {len(df)}")
    
    print(f"\nStratified sampling ({args.fraction*100:.0f}% of cases):")
    sample = stratified_sample(df, args.fraction, args.seed)
    
    # Summary by dataset
    print(f"\nSample summary:")
    for dataset, group in sample.groupby('dataset'):
        print(f"  {dataset}: {len(group)} cases")
        for category, cat_group in group.groupby('category'):
            print(f"    - {category}: {len(cat_group)}")
    
    # Convert to list of cases for evaluation script
    cases = []
    for _, row in sample.iterrows():
        # Handle NaN values (convert to None or appropriate type)
        is_blameable = row["is_blameable"]
        if pd.isna(is_blameable):
            is_blameable = None
        else:
            is_blameable = bool(is_blameable)
        
        cases.append({
            "dataset": row["dataset"],
            "repo_name": row["repo_name"],
            "bfc": row["bfc"],
            "gt_bic": row["gt_bic"],
            "category": row["category"],
            "is_blameable": is_blameable,
        })
    
    # Save
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "fraction": args.fraction,
                "seed": args.seed,
                "total_sampled": len(cases),
                "check_repos": args.check_repos,
            },
            "cases": cases,
        }, f, indent=2)
    
    print(f"\nSaved {len(cases)} cases to {output_path}")
    print(f"\nTo run pilot experiment:")
    print(f"  python evaluation/run_pilot.py --input {args.output}")


if __name__ == "__main__":
    main()
