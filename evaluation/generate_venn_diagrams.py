#!/usr/bin/env python3
"""
Generate Venn diagrams comparing true positives of LLM4SZZ vs AgenticSZZ
across BIC categories (Blame, Ancestor, Blameless).

Uses the same data sources and category assignment logic as 
generate_category_breakdown_table_v2.py to ensure consistency with Table 5.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RQ0_DIR = RESULTS_DIR / "rq0"
RQ1_BEST_DIR = RESULTS_DIR / "rq1" / "running_3"   # best run (highest F1) for AgenticSZZ
LLM4SZZ_FILE = RESULTS_DIR / "llm4szz_deepseek_full_v2" / "llm4szz_deepseek_chat_20260129_100447.json"
OUTPUT_DIR = PROJECT_ROOT / "overleaf_tosem"


def load_rq0_categories():
    """Load RQ0 category data.
    
    Returns:
    - category_lookup: (dataset, bfc, gt_bic) -> category
    - bfc_category: (dataset, bfc) -> single category (using priority: Blame > Ancestor > Blameless)
    
    Each BFC is assigned to exactly ONE category following RQ0's approach.
    """
    rq0_df = pd.read_csv(RQ0_DIR / "rq0_all_cases.csv")
    
    # Create lookup: (dataset, bfc, gt_bic) -> category
    category_lookup = {}
    # Track all categories per BFC
    bfc_all_cats = defaultdict(set)
    
    for _, row in rq0_df.iterrows():
        dataset = row['dataset'].lower()
        bfc = row['bfc']
        gt_bic = row['gt_bic']
        rq0_category = row['category']
        
        # Map RQ0 categories to display categories (matching Table 5)
        if rq0_category == 'direct':
            category = 'Blame'
        elif rq0_category in ['ancestor', 'descendant']:
            category = 'Ancestor'
        elif rq0_category == 'blameless':
            category = 'Blameless'
        else:
            category = 'Other'  # unreachable, others, error
        
        category_lookup[(dataset, bfc, gt_bic)] = category
        bfc_all_cats[(dataset, bfc)].add(category)
    
    # Assign single category per BFC using priority: Blame > Ancestor > Blameless
    bfc_category = {}
    for key, cats in bfc_all_cats.items():
        if 'Blame' in cats:
            bfc_category[key] = 'Blame'
        elif 'Ancestor' in cats:
            bfc_category[key] = 'Ancestor'
        elif 'Blameless' in cats:
            bfc_category[key] = 'Blameless'
        else:
            bfc_category[key] = 'Other'
    
    return category_lookup, bfc_category


def load_agenticszz_results():
    """Load AgenticSZZ results from all three datasets (best run = running_3)."""
    results = []

    for dataset in ['linux', 'github', 'apache']:
        matches = sorted(RQ1_BEST_DIR.glob(f"{dataset}_full_*.json"))
        if not matches:
            raise FileNotFoundError(f"No full result for {dataset} in {RQ1_BEST_DIR}")
        json_path = matches[-1]
        with open(json_path) as f:
            data = json.load(f)

        for case in data:
            results.append({
                'dataset': dataset,
                'bfc': case['bfc'],
                'predicted': case.get('predicted', ''),
                'ground_truth': case.get('ground_truth', []),
                'success': case['success'],
            })

    return results


def load_llm4szz_results():
    """Load LLM4SZZ DeepSeek results."""
    results = []
    
    with open(LLM4SZZ_FILE) as f:
        data = json.load(f)
    
    for case in data.get('per_case_results', []):
        dataset = case['dataset'].lower()
        bfc = case['bfc']
        gt_bics = case['gt_bics']
        predicted_list = case.get('predicted') or []
        
        # LLM4SZZ can return multiple predictions; take the first one
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
        })
    
    return results


def compute_venn_data(agenticszz_results, llm4szz_results, bfc_category, target_category):
    """Compute Venn diagram data for a specific category.
    
    Uses BFC-level category assignment (each BFC belongs to exactly one category).
    """
    
    # Find true positives for each approach
    agenticszz_tp = set()
    llm4szz_tp = set()
    
    for r in agenticszz_results:
        if r['success']:
            key = (r['dataset'], r['bfc'])
            if bfc_category.get(key) == target_category:
                agenticszz_tp.add(key)
    
    for r in llm4szz_results:
        if r['success']:
            key = (r['dataset'], r['bfc'])
            if bfc_category.get(key) == target_category:
                llm4szz_tp.add(key)
    
    # Venn diagram areas
    only_agenticszz = len(agenticszz_tp - llm4szz_tp)
    only_llm4szz = len(llm4szz_tp - agenticszz_tp)
    both = len(agenticszz_tp & llm4szz_tp)
    
    # Count total ground truth BFCs in this category
    gt_cases = set(key for key, cat in bfc_category.items() if cat == target_category)
    
    return {
        'only_agenticszz': only_agenticszz,
        'only_llm4szz': only_llm4szz,
        'both': both,
        'total': len(gt_cases),
        'agenticszz_total': len(agenticszz_tp),
        'llm4szz_total': len(llm4szz_tp)
    }


def create_venn_figure(venn_data, category_name, output_path):
    """Create a single Venn diagram for a category with proportional circle sizes."""
    from matplotlib.patches import Circle
    import math
    
    fig, ax = plt.subplots(figsize=(3, 2.0))
    
    only_a = venn_data['only_agenticszz']
    only_b = venn_data['only_llm4szz']
    both = venn_data['both']
    
    # Total for each set
    total_a = only_a + both
    total_b = only_b + both
    
    # Calculate radii proportional to sqrt of area (so area is proportional to count)
    # Normalize so the larger circle has radius 0.35
    max_total = max(total_a, total_b)
    if max_total > 0:
        r_a = 0.35 * math.sqrt(total_a / max_total)
        r_b = 0.35 * math.sqrt(total_b / max_total)
    else:
        r_a, r_b = 0.35, 0.35
    
    # Ensure minimum visible radius
    r_a = max(r_a, 0.15)
    r_b = max(r_b, 0.15)
    
    if both == 0:
        # No overlap - circles touch but don't overlap
        gap = 0.02
        x1 = -r_a - gap/2
        x2 = r_b + gap/2
    else:
        # Overlap - position circles to show overlap region
        overlap_dist = 0.2  # How much circles overlap
        x1 = -overlap_dist
        x2 = overlap_dist
    
    # Draw circles
    # Use muted publication-quality colors consistent with other figures
    circle1 = Circle((x1, 0), r_a, alpha=0.7, facecolor='#7FB069', edgecolor='black', linewidth=1)  # Muted green (AgenticSZZ)
    circle2 = Circle((x2, 0), r_b, alpha=0.7, facecolor='#E8A86B', edgecolor='black', linewidth=1)  # Muted orange (LLM4SZZ)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Add number labels
    if both == 0:
        # No overlap - numbers in circle centers
        ax.text(x1, 0, str(only_a), ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x2, 0, str(only_b), ha='center', va='center', fontsize=12, fontweight='bold')
    else:
        # With overlap - position numbers at geometric centers of each region
        d = x2 - x1
        # x-coordinate of circle intersection line (radical axis)
        x_intersect = x1 + (d**2 + r_a**2 - r_b**2) / (2 * d)
        # visual center of lens = midpoint of lens's x-extent
        x_both = ((x2 - r_b) + (x1 + r_a)) / 2
        ax.text((x1 - r_a + x_intersect) / 2, 0, str(only_a), ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text((x_intersect + x2 + r_b) / 2, 0, str(only_b), ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x_both, 0, str(both), ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Category label removed - now provided by LaTeX subcaption
    
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.38, 0.38)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading RQ0 category data...")
    category_lookup, bfc_category = load_rq0_categories()
    print(f"  Loaded {len(category_lookup)} (dataset, bfc, bic) entries")
    print(f"  Assigned {len(bfc_category)} unique BFCs to single categories")
    
    # Show category distribution
    cat_counts = defaultdict(int)
    for cat in bfc_category.values():
        cat_counts[cat] += 1
    print(f"  Category distribution: {dict(cat_counts)}")
    
    print("\nLoading AgenticSZZ results...")
    agenticszz_results = load_agenticszz_results()
    print(f"  Loaded {len(agenticszz_results)} cases")
    
    print("\nLoading LLM4SZZ results...")
    llm4szz_results = load_llm4szz_results()
    print(f"  Loaded {len(llm4szz_results)} cases")
    
    print("\nComputing Venn diagram data...")
    all_venn_data = {}
    for category in ['Blame', 'Ancestor', 'Blameless']:
        venn_data = compute_venn_data(
            agenticszz_results, llm4szz_results, 
            bfc_category, category
        )
        all_venn_data[category] = venn_data
        print(f"\n{category}:")
        print(f"  Ground truth cases: {venn_data['total']}")
        print(f"  AgenticSZZ TP: {venn_data['agenticszz_total']}")
        print(f"  LLM4SZZ TP: {venn_data['llm4szz_total']}")
        print(f"  Only AgenticSZZ: {venn_data['only_agenticszz']}")
        print(f"  Only LLM4SZZ: {venn_data['only_llm4szz']}")
        print(f"  Both correct: {venn_data['both']}")
    
    print("\nGenerating Venn diagrams...")
    
    # Generate individual figures only (no combined figure)
    for category in ['Blame', 'Ancestor', 'Blameless']:
        output_path = OUTPUT_DIR / f"venn_{category.lower()}.pdf"
        create_venn_figure(all_venn_data[category], category, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
