#!/usr/bin/env python3
"""
RQ0 Analysis: Analyze blameable and blameless distribution from the experiment results.
This script generates comprehensive insights about the distribution of blameable vs blameless cases.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(results_dir):
    """Load the RQ0 CSV files."""
    all_cases_path = Path(results_dir) / "rq0_all_cases.csv"
    summary_path = Path(results_dir) / "rq0_summary.csv"

    all_cases = pd.read_csv(all_cases_path)
    summary = pd.read_csv(summary_path)

    return all_cases, summary


def analyze_blameable_distribution(all_cases, summary):
    """Analyze the distribution of blameable vs blameless cases."""
    print("=" * 80)
    print("RQ0: BLAMEABLE VS BLAMELESS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Overall summary
    print("1. OVERALL SUMMARY")
    print("-" * 80)
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        original_size = row["original_dataset_size"]
        analyzed = row["analyzed_cases"]
        blameable_cases = row["blameable_cases"]
        blameless_cases = row["blameless_cases"]

        blameable_pct = blameable_cases / analyzed * 100
        blameless_pct = blameless_cases / analyzed * 100

        print(f"\n{dataset}:")
        print(f"  Total Cases:      {original_size:>6} (analyzed: {analyzed})")
        print(f"  Blameable Cases:  {blameable_cases:>6} ({blameable_pct:>5.2f}%)")
        print(f"  Blameless Cases:  {blameless_cases:>6} ({blameless_pct:>5.2f}%)")
        print(
            f"  Blameable Ratio:  {blameable_cases}/{analyzed} = {blameable_cases / analyzed:.3f}"
        )

    print()

    # BIC statistics
    print("2. BUG-INDUCING COMMIT (BIC) STATISTICS")
    print("-" * 80)
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        total_bics = row["total_bics"]
        blameable_bics = row["blameable_bics"]
        blameless_bics = row["blameless_bics"]

        print(f"\n{dataset}:")
        print(f"  Total BICs:       {total_bics:>6}")
        print(
            f"  Blameable BICs:   {blameable_bics:>6} ({blameable_bics / total_bics * 100:>5.2f}%)"
        )
        print(
            f"  Blameless BICs:   {blameless_bics:>6} ({blameless_bics / total_bics * 100:>5.2f}%)"
        )
        print(f"  Avg BICs per case: {total_bics / row['analyzed_cases']:.2f}")

    print()

    # Category distribution
    print("3. BIC CATEGORY DISTRIBUTION")
    print("-" * 80)
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        direct_bics = row["direct_bics"]
        ancestor_bics = row["ancestor_bics"]
        descendant_bics = row["descendant_bics"]
        others_bics = row["others_bics"]
        blameless_bics = row["blameless_bics"]
        total_bics = row["total_bics"]

        print(f"\n{dataset}:")
        print(f"  Direct BICs:      {direct_bics:>6} ({direct_bics / total_bics * 100:>5.2f}%)")
        print(f"  Ancestor BICs:    {ancestor_bics:>6} ({ancestor_bics / total_bics * 100:>5.2f}%)")
        print(
            f"  Descendant BICs:  {descendant_bics:>6} ({descendant_bics / total_bics * 100:>5.2f}%)"
        )
        print(f"  Others BICs:      {others_bics:>6} ({others_bics / total_bics * 100:>5.2f}%)")
        print(
            f"  Blameless BICs:   {blameless_bics:>6} ({blameless_bics / total_bics * 100:>5.2f}%)"
        )

    print()

    # Change characteristics
    print("4. CHANGE CHARACTERISTICS")
    print("-" * 80)
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        avg_del = row["avg_deleted_lines"]
        avg_add = row["avg_added_lines"]
        avg_commits = row["avg_blame_commits"]

        print(f"\n{dataset}:")
        print(f"  Avg Deleted Lines:  {avg_del:>8.2f}")
        print(f"  Avg Added Lines:    {avg_add:>8.2f}")
        print(f"  Avg Blame Commits:  {avg_commits:>8.2f}")

    print()

    # Ancestor depth analysis
    print("5. ANCESTOR DEPTH ANALYSIS (for ancestor category)")
    print("-" * 80)
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        avg_depth = row["avg_ancestor_depth"]
        median_depth = row["median_ancestor_depth"]
        max_depth = row["max_ancestor_depth"]

        print(f"\n{dataset}:")
        print(f"  Average Depth:    {avg_depth:>10.2f}")
        print(f"  Median Depth:     {median_depth:>10.0f}")
        print(f"  Max Depth:        {max_depth:>10}")

    print()


def analyze_detailed_patterns(all_cases):
    """Analyze detailed patterns in the data."""
    print("6. DETAILED PATTERN ANALYSIS")
    print("-" * 80)

    for dataset in all_cases["dataset"].unique():
        dataset_cases = all_cases[all_cases["dataset"] == dataset]

        print(f"\n{dataset}:")

        # Category breakdown for blameable vs blameless
        blameable = dataset_cases[dataset_cases["is_blameable"] == True]
        blameless = dataset_cases[dataset_cases["is_blameable"] == False]

        print("\n  Category distribution for BLAMEABLE cases:")
        category_counts = blameable["category"].value_counts()
        for cat, count in category_counts.items():
            print(f"    {cat:12s}: {count:>5} ({count / len(blameable) * 100:>5.2f}%)")

        # Number of BICs distribution
        print("\n  Number of BICs per case (BLAMEABLE):")
        bic_counts = blameable["num_blame_commits"].value_counts().sort_index()
        for num_bics, count in list(bic_counts.items())[:10]:  # Top 10
            print(f"    {num_bics} BIC(s): {count:>5} cases")
        if len(bic_counts) > 10:
            print(f"    ... ({len(bic_counts) - 10} more values)")

        # Change size analysis
        print("\n  Change size statistics (BLAMEABLE):")
        print(f"    Median deleted lines: {blameable['total_deleted_lines'].median():.0f}")
        print(f"    Median added lines:   {blameable['total_added_lines'].median():.0f}")
        print(f"    75th percentile deleted: {blameable['total_deleted_lines'].quantile(0.75):.0f}")
        print(f"    75th percentile added:   {blameable['total_added_lines'].quantile(0.75):.0f}")

    print()


def generate_insights(all_cases, summary):
    """Generate key insights and recommendations."""
    print("7. KEY INSIGHTS & IMPLICATIONS FOR AGENTIC-SZZ")
    print("=" * 80)

    insights = []

    # Calculate overall statistics
    total_analyzed = summary["analyzed_cases"].sum()
    total_blameable = summary["blameable_cases"].sum()
    blameable_ratio = total_blameable / total_analyzed

    insights.append(
        f"1. Blameable Rate: {blameable_ratio * 100:.1f}% of all cases have at least one"
    )
    insights.append("   blameable BIC, indicating that most bugs CAN be traced back to specific")
    insights.append("   code changes through git blame.")

    # Direct vs ancestor
    total_direct = summary["direct_bics"].sum()
    total_ancestor = summary["ancestor_bics"].sum()
    total_bics = summary["total_bics"].sum()

    insights.append(
        f"\n2. Direct BICs dominate: {total_direct / total_bics * 100:.1f}% of all BICs are 'direct'"
    )
    insights.append(
        f"   (blame points directly to the BIC), while {total_ancestor / total_bics * 100:.1f}% are"
    )
    insights.append("   'ancestor' BICs (requiring deeper history traversal).")

    # Descendant BICs (new category)
    total_descendant = (
        summary["descendant_bics"].sum() if "descendant_bics" in summary.columns else 0
    )

    if total_descendant > 0:
        insights.append(
            f"\n2b. Descendant BICs discovered: {total_descendant / total_bics * 100:.1f}% of all BICs came"
        )
        insights.append(
            "    AFTER the blame commits (forward in time). These are searchable via forward"
        )
        insights.append("    history search, unlike truly unrelated BICs!")

    # Ancestor depth implications
    linux_row = summary[summary["dataset"] == "LINUX"].iloc[0]
    median_depth = linux_row["median_ancestor_depth"]
    max_depth = linux_row["max_ancestor_depth"]

    insights.append("\n3. Ancestor Depth Challenge: For Linux, the median ancestor depth is")
    insights.append(
        f"   {median_depth:.0f} commits, with a maximum of {max_depth:.0f}. This suggests"
    )
    insights.append("   that agentic-SZZ needs robust strategies for deep history traversal.")

    # Multiple BICs
    multi_bic_cases = all_cases[all_cases["num_blame_commits"] > 1]
    multi_bic_pct = len(multi_bic_cases) / len(all_cases) * 100

    insights.append(
        f"\n4. Multiple BICs: {multi_bic_pct:.1f}% of cases have multiple blame commits,"
    )
    insights.append("   indicating that bugs often have multiple contributing factors that need")
    insights.append("   to be considered by the agentic approach.")

    # Change size
    median_changes = all_cases[all_cases["is_blameable"] == True][
        ["total_deleted_lines", "total_added_lines"]
    ].median()

    insights.append(
        f"\n5. Change Size: The median blameable case involves {median_changes['total_deleted_lines']:.0f}"
    )
    insights.append(
        f"   deleted lines and {median_changes['total_added_lines']:.0f} added lines. This relatively"
    )
    insights.append("   small change size suggests targeted fixes are common.")

    # Design implications
    insights.append("\n" + "=" * 80)
    insights.append("DESIGN IMPLICATIONS FOR AGENTIC-SZZ:")
    insights.append("=" * 80)

    insights.append("\na) The high blameable rate validates the git-blame-based approach, but")
    insights.append(
        "   the {:.1f}% blameless rate shows the need for fallback strategies.".format(
            (1 - blameable_ratio) * 100
        )
    )

    insights.append("\nb) The dominance of direct BICs suggests optimizing for the common case,")
    insights.append("   while still handling ancestor traversal when needed.")

    if total_descendant > 0:
        insights.append(
            "\nc) The discovery of descendant BICs (16-34%) validates the need for BIDIRECTIONAL"
        )
        insights.append(
            "   search: backward for ancestors, forward for descendants. This dramatically"
        )
        insights.append("   increases our searchable BIC coverage from ~70% to ~97-100%!")

    insights.append("\nd) The deep ancestor depths in Linux highlight the importance of efficient")
    insights.append("   history traversal algorithms in the agentic agent.")

    insights.append("\ne) Multiple BICs per case motivate an agentic approach that can reason")
    insights.append("   about multiple potential bug-inducing commits and their interactions.")

    for line in insights:
        print(line)

    print()


def generate_publication_figure(summary, output_path):
    """Generate publication-quality figure of BIC category distribution"""
    import matplotlib.pyplot as plt

    # Professional color scheme - soft, pastel, academic-friendly
    # Colors swapped for Blame/Blame_Ancestor to match rq0_commits_category.png
    colors = {
        "Blame": "#90B77D",  # Soft sage green (was Direct with blue)
        "Blame_Ancestor": "#5B8FA3",  # Soft steel blue (was Ancestor with green)
        "BFC_Ancestor": "#E8B55D",  # Soft golden yellow (unchanged)
        "Unreachable": "#D08A85",  # Soft dusty rose (was Others)
        "Blameless": "#B8A9C9",  # Soft lavender
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    # Dynamically get available datasets
    available_datasets = []
    dataset_labels = []
    for ds_name, ds_label in [
        ("LINUX", "Linux"),
        ("APACHE", "Apache"),
        ("GITHUB", "GitHub"),
    ]:
        if ds_name in summary["dataset"].values:
            available_datasets.append(ds_name)
            dataset_labels.append(ds_label)

    categories = ["Blame", "Blame_Ancestor", "BFC_Ancestor", "Unreachable", "Blameless"]

    # Prepare data - 5 categories: Blame, Blame_Ancestor, BFC_Ancestor, Unreachable, Blameless
    data = []
    for dataset_name in available_datasets:
        row = summary[summary["dataset"] == dataset_name].iloc[0]
        total = row["total_bics"]

        data.append(
            [
                row["direct_bics"] / total * 100,
                row["ancestor_bics"] / total * 100,
                row["descendant_bics"] / total * 100,
                row["others_bics"] / total * 100,
                row["blameless_bics"] / total * 100,
            ]
        )

    data = np.array(data)

    # Create stacked bars with professional styling
    x = np.arange(len(dataset_labels))
    width = 0.5
    bottom = np.zeros(len(dataset_labels))

    for i, cat in enumerate(categories):
        bars = ax.bar(
            x,
            data[:, i],
            width,
            label=cat,
            bottom=bottom,
            color=colors[cat],
            edgecolor="white",
            linewidth=1.5,
        )

        # Add percentage labels for all visible segments (> 3%)
        for j, (bar, val) in enumerate(zip(bars, data[:, i])):
            if val > 3:  # Show label for segments > 3%
                height = bar.get_height()
                label_y = bottom[j] + height / 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="600",
                )

        bottom += data[:, i]

    # Styling
    ax.set_ylabel("Percentage of BICs (%)", fontsize=11, fontweight="500")
    ax.set_xlabel("Dataset", fontsize=11, fontweight="500")
    # Title removed for publication - add via LaTeX \\caption{} instead
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=11)
    ax.set_ylim(0, 100)

    # Legend at bottom - horizontal, outside plot area
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=10,
    )

    # Grid for readability
    ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
    ax.set_axisbelow(True)

    # Clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\n✓ Publication figure saved: {output_path}")


def generate_csv_summary(summary, output_path):
    """Generate CSV summary table"""
    data_rows = []

    for _, row in summary.iterrows():
        dataset = row["dataset"]
        total = row["total_bics"]
        row["blameless_bics"] if "blameless_bics" in row.index else 0

        data_rows.append(
            {
                "Dataset": dataset,
                "Total_Cases": int(row["original_dataset_size"]),
                "Total_BICs": int(total),
                "Direct": int(row["direct_bics"]),
                "Direct_Pct": f"{row['direct_bics'] / total * 100:.1f}%",
                "Ancestor": int(row["ancestor_bics"]),
                "Ancestor_Pct": f"{row['ancestor_bics'] / total * 100:.1f}%",
                "Descendant": int(row["descendant_bics"]),
                "Descendant_Pct": f"{row['descendant_bics'] / total * 100:.1f}%",
                "Others": int(row["others_bics"]),
                "Others_Pct": f"{row['others_bics'] / total * 100:.1f}%",
                "Blameless": int(row["blameless_bics"]),
                "Blameless_Pct": f"{row['blameless_bics'] / total * 100:.1f}%",
                "Searchable": int(
                    row["direct_bics"] + row["ancestor_bics"] + row["descendant_bics"]
                ),
                "Searchable_Pct": f"{(row['direct_bics'] + row['ancestor_bics'] + row['descendant_bics']) / total * 100:.1f}%",
            }
        )

    import pandas as pd

    df = pd.DataFrame(data_rows)
    df.to_csv(output_path, index=False)

    print(f"✓ CSV table saved: {output_path}")


def generate_blame_complexity_figures(all_cases, output_dir):
    """Generate publication-quality figures for blame complexity analysis.

    Creates two separate PDF figures with per-dataset distributions:
    - rq0_deleted_lines.pdf: Distribution of deleted lines per case by dataset
    - rq0_blame_commits.pdf: Distribution of unique blame commits per case by dataset

    Focuses on 'Direct' category cases to motivate the ranking approach.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    # Filter to Direct cases only (the main target for ranking)
    direct_cases = all_cases[all_cases["category"] == "direct"]

    # Dataset colors - consistent with category distribution figure
    dataset_colors = {
        "LINUX": "#5B8FA3",  # Soft steel blue
        "APACHE": "#E8B55D",  # Soft golden yellow
        "GITHUB": "#7FB685",  # Soft sage green
    }
    datasets = ["LINUX", "APACHE", "GITHUB"]

    # Common styling function
    def apply_styling(ax):
        ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(0.8)

    # ====== Figure 1: Deleted Lines Distribution by Dataset ======
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    # Use log-scale bins
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 500, 20000]
    bin_labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]

    x = np.arange(len(bin_labels))
    width = 0.25

    for i, dataset in enumerate(datasets):
        dataset_data = direct_cases[direct_cases["dataset"] == dataset][
            "total_deleted_lines"
        ].values
        hist_counts, _ = np.histogram(dataset_data, bins=bins)
        offset = (i - 1) * width
        ax1.bar(
            x + offset,
            hist_counts,
            width,
            label=dataset,
            color=dataset_colors[dataset],
            edgecolor="white",
            linewidth=0.8,
        )

    ax1.set_xlabel("Deleted Lines", fontsize=10, fontweight="500")
    ax1.set_ylabel("Cases", fontsize=10, fontweight="500")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=8, rotation=45, ha="right")
    ax1.legend(title="Dataset", fontsize=8, title_fontsize=9)

    apply_styling(ax1)
    plt.tight_layout()

    output_path1 = output_dir / "rq0_deleted_lines.pdf"
    plt.savefig(output_path1, bbox_inches="tight", dpi=300)
    plt.close()

    # ====== Figure 2: Blame Commits Distribution by Dataset ======
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    labels = ["1", "2", "3", "4", "5", "6+"]
    x = np.arange(len(labels))
    width = 0.25

    for i, dataset in enumerate(datasets):
        dataset_data = direct_cases[direct_cases["dataset"] == dataset]["num_blame_commits"].values
        commit_counts = pd.Series(dataset_data).value_counts().sort_index()

        grouped_counts = {}
        for num, count in commit_counts.items():
            if num <= 5:
                grouped_counts[str(num)] = count
            else:
                grouped_counts["6+"] = grouped_counts.get("6+", 0) + count

        counts = [grouped_counts.get(label, 0) for label in labels]
        offset = (i - 1) * width
        ax2.bar(
            x + offset,
            counts,
            width,
            label=dataset,
            color=dataset_colors[dataset],
            edgecolor="white",
            linewidth=0.8,
        )

    ax2.set_xlabel("Blame Commits", fontsize=10, fontweight="500")
    ax2.set_ylabel("Cases", fontsize=10, fontweight="500")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(title="Dataset", fontsize=8, title_fontsize=9)

    apply_styling(ax2)
    plt.tight_layout()

    output_path2 = output_dir / "rq0_blame_commits.pdf"
    plt.savefig(output_path2, bbox_inches="tight", dpi=300)
    plt.close()

    # ====== Figure 3: Relationship Heatmaps (Per-Dataset) ======
    x_bins = [1, 2, 3, 5, 10, 20, 50, 100, 500, 20000]
    y_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 100]
    x_labels_hm = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]
    y_labels_hm = ["1", "2", "3", "4", "5", "6+"]

    heatmap_paths = []

    for dataset in datasets:
        dataset_data = direct_cases[direct_cases["dataset"] == dataset]
        deleted_lines = dataset_data["total_deleted_lines"].values
        blame_commits = dataset_data["num_blame_commits"].values

        fig3, ax3 = plt.subplots(figsize=(5, 4))

        hist2d, xedges, yedges = np.histogram2d(deleted_lines, blame_commits, bins=[x_bins, y_bins])

        im = ax3.imshow(
            hist2d.T,
            origin="lower",
            aspect="auto",
            cmap="Greens",  # Match Blame category color (green)
            interpolation="nearest",
        )

        for i in range(len(x_labels_hm)):
            for j in range(len(y_labels_hm)):
                value = int(hist2d[i, j])
                if value > 0:
                    text_color = "white" if value > hist2d.max() * 0.5 else "black"
                    ax3.text(
                        i,
                        j,
                        f"{value}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=text_color,
                    )

        ax3.set_xticks(range(len(x_labels_hm)))
        ax3.set_xticklabels(x_labels_hm, fontsize=11, rotation=45, ha="right")
        ax3.set_yticks(range(len(y_labels_hm)))
        ax3.set_yticklabels(y_labels_hm, fontsize=11)
        ax3.set_xlabel("Deleted Lines", fontsize=12, fontweight="500")
        ax3.set_ylabel("Blame Commits", fontsize=12, fontweight="500")

        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label("Cases", fontsize=11)

        apply_styling(ax3)
        plt.tight_layout()

        output_path3 = output_dir / f"rq0_complexity_heatmap_{dataset.lower()}.pdf"
        plt.savefig(output_path3, bbox_inches="tight", dpi=300)
        plt.close()
        heatmap_paths.append(output_path3)

    # Print summary stats per dataset
    print("\n✓ Blame complexity figures saved:")
    print(f"  - {output_path1}")
    print(f"  - {output_path2}")
    for hp in heatmap_paths:
        print(f"  - {hp}")
    for dataset in datasets:
        ds_data = direct_cases[direct_cases["dataset"] == dataset]
        single = len(ds_data[ds_data["num_blame_commits"] == 1])
        total = len(ds_data)
        print(
            f"  {dataset}: {total} Direct cases, {single} ({single / total * 100:.1f}%) with single blame commit"
        )


def generate_file_depth_figures(all_cases, output_dir):
    """Generate file-depth distribution figures for ancestor and descendant cases.

    Creates figures showing:
    - Distribution of ancestor_file_depth (blame → BIC) for ancestor cases
    - Distribution of descendant_file_depth (BIC → BFC) for descendant cases
    - Coverage analysis at different depth limits
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    # Check if the new columns exist
    if (
        "ancestor_file_depth" not in all_cases.columns
        or "descendant_file_depth" not in all_cases.columns
    ):
        print("⚠ File depth columns not found. Run rq0_run_experiments.py first to generate them.")
        return

    # Filter to valid ancestor and descendant cases
    ancestor_cases = all_cases[
        (all_cases["category"] == "ancestor") & (all_cases["ancestor_file_depth"] >= 0)
    ]
    descendant_cases = all_cases[
        (all_cases["category"] == "descendant") & (all_cases["descendant_file_depth"] >= 0)
    ]

    # Dataset colors
    dataset_colors = {
        "LINUX": "#5B8FA3",  # Soft steel blue
        "APACHE": "#E8B55D",  # Soft golden yellow
        "GITHUB": "#7FB685",  # Soft sage green
    }

    # Common styling function
    def apply_styling(ax):
        ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(0.8)

    # ====== Figure 1: Ancestor File-Depth Distribution (blame → BIC) ======
    if len(ancestor_cases) > 0:
        fig1, ax1 = plt.subplots(figsize=(8, 5))

        # Use bins appropriate for the data
        bins = [0, 2, 5, 10, 20, 50, 100, 200, 500, 10000]
        bin_labels = [
            "0-1",
            "2-4",
            "5-9",
            "10-19",
            "20-49",
            "50-99",
            "100-199",
            "200-499",
            "500+",
        ]

        x = np.arange(len(bin_labels))
        width = 0.25
        datasets = ["LINUX", "APACHE", "GITHUB"]

        for i, dataset in enumerate(datasets):
            dataset_data = ancestor_cases[ancestor_cases["dataset"] == dataset][
                "ancestor_file_depth"
            ].values
            if len(dataset_data) > 0:
                hist_counts, _ = np.histogram(dataset_data, bins=bins)
                offset = (i - 1) * width
                ax1.bar(
                    x + offset,
                    hist_counts,
                    width,
                    label=dataset,
                    color=dataset_colors.get(dataset, "#888888"),
                    edgecolor="white",
                    linewidth=0.8,
                )

        ax1.set_xlabel("Ancestor File-Depth (blame → BIC)", fontsize=10, fontweight="500")
        ax1.set_ylabel("Cases", fontsize=10, fontweight="500")
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, fontsize=8, rotation=45, ha="right")
        ax1.legend(title="Dataset", fontsize=8, title_fontsize=9)

        apply_styling(ax1)
        plt.tight_layout()

        output_path1 = output_dir / "rq0_ancestor_file_depth.pdf"
        plt.savefig(output_path1, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  - {output_path1}")

    # ====== Figure 2: Descendant File-Depth Distribution (BIC → BFC) ======
    if len(descendant_cases) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        bins = [0, 2, 5, 10, 20, 50, 100, 200, 500, 10000]
        bin_labels = [
            "0-1",
            "2-4",
            "5-9",
            "10-19",
            "20-49",
            "50-99",
            "100-199",
            "200-499",
            "500+",
        ]

        x = np.arange(len(bin_labels))
        width = 0.25
        datasets = ["LINUX", "APACHE", "GITHUB"]

        for i, dataset in enumerate(datasets):
            dataset_data = descendant_cases[descendant_cases["dataset"] == dataset][
                "descendant_file_depth"
            ].values
            if len(dataset_data) > 0:
                hist_counts, _ = np.histogram(dataset_data, bins=bins)
                offset = (i - 1) * width
                ax2.bar(
                    x + offset,
                    hist_counts,
                    width,
                    label=dataset,
                    color=dataset_colors.get(dataset, "#888888"),
                    edgecolor="white",
                    linewidth=0.8,
                )

        ax2.set_xlabel("Descendant File-Depth (BIC → BFC)", fontsize=10, fontweight="500")
        ax2.set_ylabel("Cases", fontsize=10, fontweight="500")
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_labels, fontsize=8, rotation=45, ha="right")
        ax2.legend(title="Dataset", fontsize=8, title_fontsize=9)

        apply_styling(ax2)
        plt.tight_layout()

        output_path2 = output_dir / "rq0_descendant_file_depth.pdf"
        plt.savefig(output_path2, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  - {output_path2}")

    # ====== Print Coverage Analysis ======
    print("\n  Coverage Analysis:")
    for limit in [10, 20, 50, 100]:
        if len(ancestor_cases) > 0:
            anc_cov = len(ancestor_cases[ancestor_cases["ancestor_file_depth"] <= limit])
            anc_total = len(ancestor_cases)
            anc_pct = anc_cov / anc_total * 100
        else:
            anc_pct = 0

        if len(descendant_cases) > 0:
            desc_cov = len(descendant_cases[descendant_cases["descendant_file_depth"] <= limit])
            desc_total = len(descendant_cases)
            desc_pct = desc_cov / desc_total * 100
        else:
            desc_pct = 0

        print(f"    Limit {limit:3d}: Ancestor {anc_pct:5.1f}%, Descendant {desc_pct:5.1f}%")


def generate_file_depth_cdf(all_cases, output_dir):
    """Generate CDF (Cumulative Distribution Function) figure for file-depth coverage.

    Creates a single publication-quality figure showing:
    - Ancestor file-depth coverage (blame → BIC)
    - Descendant file-depth coverage (BIC → BFC)

    This directly answers: "What depth limit achieves X% coverage?"
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    # Check if columns exist
    if (
        "ancestor_file_depth" not in all_cases.columns
        or "descendant_file_depth" not in all_cases.columns
    ):
        print("⚠ File depth columns not found.")
        return

    # Filter to valid cases
    ancestor_depths = all_cases[
        (all_cases["category"] == "ancestor") & (all_cases["ancestor_file_depth"] >= 0)
    ]["ancestor_file_depth"].values
    descendant_depths = all_cases[
        (all_cases["category"] == "descendant") & (all_cases["descendant_file_depth"] >= 0)
    ]["descendant_file_depth"].values

    if len(ancestor_depths) == 0 and len(descendant_depths) == 0:
        print("⚠ No valid depth data found.")
        return

    # Create figure - compact size with larger fonts
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Colors
    ancestor_color = "#5B8FA3"  # Soft steel blue
    descendant_color = "#E8B55D"  # Soft golden yellow

    # Plot CDFs
    x_range = np.arange(0, 201, 1)  # 0 to 200

    if len(ancestor_depths) > 0:
        ancestor_cdf = [(ancestor_depths <= x).sum() / len(ancestor_depths) * 100 for x in x_range]
        ax.plot(
            x_range,
            ancestor_cdf,
            label="Blame Ancestor",
            color=ancestor_color,
            linewidth=2.5,
        )

    if len(descendant_depths) > 0:
        descendant_cdf = [
            (descendant_depths <= x).sum() / len(descendant_depths) * 100 for x in x_range
        ]
        ax.plot(
            x_range,
            descendant_cdf,
            label="BFC Ancestor",
            color=descendant_color,
            linewidth=2.5,
        )

    # Add horizontal reference lines at 80% and 90% with depth labels
    for y_val in [80, 90]:
        ax.axhline(y=y_val, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            190,
            y_val + 1.5,
            f"{y_val}%",
            fontsize=11,
            color="#666666",
            ha="left",
            fontweight="500",
        )

        # Find and label depth where each curve crosses this threshold
        if len(ancestor_depths) > 0:
            # Find smallest depth that achieves this coverage
            for d in range(1, 500):
                if (ancestor_depths <= d).sum() / len(ancestor_depths) * 100 >= y_val:
                    # Position below the line
                    ax.annotate(
                        f"{d}",
                        xy=(d, y_val),
                        xytext=(d, y_val - 8),
                        fontsize=11,
                        color=ancestor_color,
                        fontweight="bold",
                        ha="center",
                    )
                    break
        if len(descendant_depths) > 0:
            for d in range(1, 500):
                if (descendant_depths <= d).sum() / len(descendant_depths) * 100 >= y_val:
                    # Position above the line
                    ax.annotate(
                        f"{d}",
                        xy=(d, y_val),
                        xytext=(d, y_val + 5),
                        fontsize=11,
                        color=descendant_color,
                        fontweight="bold",
                        ha="center",
                    )
                    break

    # Styling - larger fonts for compact figure
    ax.set_xlabel("File-History Depth Limit", fontsize=12, fontweight="500")
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight="500")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 102)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.10),
        fontsize=11,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    # Grid
    ax.grid(axis="both", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
    ax.set_axisbelow(True)

    # Clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout()

    output_path = output_dir / "rq0_file_depth_coverage.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"  - {output_path}")

    # Print summary stats
    if len(ancestor_depths) > 0:
        anc_25 = (ancestor_depths <= 25).sum() / len(ancestor_depths) * 100
        anc_50 = (ancestor_depths <= 50).sum() / len(ancestor_depths) * 100
        print(f"    Ancestor: 25→{anc_25:.1f}%, 50→{anc_50:.1f}%")
    if len(descendant_depths) > 0:
        desc_25 = (descendant_depths <= 25).sum() / len(descendant_depths) * 100
        desc_50 = (descendant_depths <= 50).sum() / len(descendant_depths) * 100
        print(f"    Descendant: 25→{desc_25:.1f}%, 50→{desc_50:.1f}%")


def generate_blameless_coverage_figures(all_cases, output_dir):
    """Generate coverage analysis figures for blameless cases.

    Reads pre-computed context coverage data from the CSV (context_category,
    context_blame_depth, context_bfc_depth columns) generated by rq0_run_experiments.py.

    Creates:
    - rq0_blameless_coverage.pdf: CDF showing coverage at different depths
    - Prints coverage statistics by category
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    # Filter to blameless cases with context coverage data
    blameless_cases = all_cases[all_cases["category"] == "blameless"].copy()

    if len(blameless_cases) == 0:
        print("⚠ No blameless cases found.")
        return

    # Check if context columns exist
    if "context_category" not in blameless_cases.columns:
        print("⚠ No context coverage data found. Run rq0_run_experiments.py first.")
        return

    print(f"\n{'=' * 70}")
    print(f"BLAMELESS COVERAGE ANALYSIS ({len(blameless_cases)} cases)")
    print(f"{'=' * 70}")

    total = len(blameless_cases)

    # Category breakdown
    print("\n  Category Breakdown (Context Fallback ±2 lines):")
    categories = ["direct", "ancestor", "descendant", "not_found"]
    for cat in categories:
        count = len(blameless_cases[blameless_cases["context_category"] == cat])
        pct = count / total * 100 if total > 0 else 0
        print(f"    {cat:20s}: {count:4d} ({pct:5.1f}%)")

    # Total found (direct + ancestor + descendant)
    found_mask = blameless_cases["context_category"].isin(["direct", "ancestor", "descendant"])
    found_count = found_mask.sum()
    print(f"    {'---':20s}")
    print(f"    {'Total Found':20s}: {found_count:4d} ({found_count / total * 100:.1f}%)")

    # Get depths separately for ancestor and descendant
    direct_cases = blameless_cases[blameless_cases["context_category"] == "direct"]
    ancestor_cases = blameless_cases[blameless_cases["context_category"] == "ancestor"]
    descendant_cases = blameless_cases[blameless_cases["context_category"] == "descendant"]

    direct_count = len(direct_cases)
    direct_pct = direct_count / total * 100

    # Ancestor depths (from context_blame_depth)
    ancestor_depths = ancestor_cases["context_blame_depth"].values
    ancestor_depths = ancestor_depths[ancestor_depths >= 0]

    # Descendant depths (from context_bfc_depth)
    descendant_depths = descendant_cases["context_bfc_depth"].values
    descendant_depths = descendant_depths[descendant_depths >= 0]

    # Print depth stats
    if len(ancestor_depths) > 0 or len(descendant_depths) > 0:
        print(f"\n  Depth Coverage (cumulative with direct={direct_count}):")
        for limit in [5, 10, 20, 50, 100]:
            anc_cov = (ancestor_depths <= limit).sum() if len(ancestor_depths) > 0 else 0
            desc_cov = (descendant_depths <= limit).sum() if len(descendant_depths) > 0 else 0
            total_cov = direct_count + anc_cov + desc_cov
            pct = total_cov / total * 100
            print(f"    Depth ≤{limit:3d}: {total_cov:4d}/{total} ({pct:.1f}%)")

    # Generate CDF figure with TWO curves (consistent with file_depth_coverage)
    # Compact size with larger fonts for paper
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x_range = np.arange(0, 101, 1)

    # Colors matching file_depth_coverage
    ancestor_color = "#5B8FA3"  # Soft steel blue
    descendant_color = "#E8B55D"  # Soft golden yellow

    # Ancestor CDF: % of ancestor cases covered at depth <= X (0-100% of category)
    if len(ancestor_depths) > 0:
        ancestor_cdf = [(ancestor_depths <= x).sum() / len(ancestor_depths) * 100 for x in x_range]
        ax.plot(
            x_range,
            ancestor_cdf,
            label="Blame Ancestor",
            color=ancestor_color,
            linewidth=2.5,
        )

    # Descendant CDF: % of descendant cases covered at depth <= X (0-100% of category)
    if len(descendant_depths) > 0:
        descendant_cdf = [
            (descendant_depths <= x).sum() / len(descendant_depths) * 100 for x in x_range
        ]
        ax.plot(
            x_range,
            descendant_cdf,
            label="BFC Ancestor",
            color=descendant_color,
            linewidth=2.5,
        )

    # 50% threshold line with depth annotation (practical threshold for blameless)
    y_threshold = 50
    ax.axhline(y=y_threshold, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        95,
        y_threshold + 1.5,
        f"{y_threshold}%",
        fontsize=11,
        color="#666666",
        ha="left",
        fontweight="500",
    )

    # Find and label depth where each curve crosses 80% threshold
    if len(ancestor_depths) > 0:
        for d in range(1, 101):
            if (ancestor_depths <= d).sum() / len(ancestor_depths) * 100 >= y_threshold:
                ax.annotate(
                    f"{d}",
                    xy=(d, y_threshold),
                    xytext=(d, y_threshold - 8),
                    fontsize=11,
                    color=ancestor_color,
                    fontweight="bold",
                    ha="center",
                )
                break
    if len(descendant_depths) > 0:
        for d in range(1, 101):
            if (descendant_depths <= d).sum() / len(descendant_depths) * 100 >= y_threshold:
                ax.annotate(
                    f"{d}",
                    xy=(d, y_threshold),
                    xytext=(d, y_threshold + 5),
                    fontsize=11,
                    color=descendant_color,
                    fontweight="bold",
                    ha="center",
                )
                break

    # Add annotation for Fallback Direct cases
    ax.annotate(
        f"+ Fallback Direct: {direct_count} cases ({direct_pct:.0f}% of blameless)",
        xy=(5, 5),
        fontsize=10,
        color="#666666",
        fontweight="500",
    )

    # Styling - larger fonts for compact figure
    ax.set_xlabel("File-History Depth Limit", fontsize=12, fontweight="500")
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight="500")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.12),
        fontsize=11,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    ax.grid(axis="both", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout()

    output_path = output_dir / "rq0_blameless_coverage.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\n  ✓ Figure saved: {output_path}")


def save_analysis_report(all_cases, summary, output_file):
    """Save a detailed analysis report to a file."""
    with open(output_file, "w") as f:
        # Redirect print to file
        import sys

        original_stdout = sys.stdout
        sys.stdout = f

        analyze_blameable_distribution(all_cases, summary)
        analyze_detailed_patterns(all_cases)
        generate_insights(all_cases, summary)

        sys.stdout = original_stdout

    print(f"\n✓ Analysis report saved to: {output_file}")


def generate_file_depth_cdf_per_dataset(all_cases, output_dir):
    """Generate individual CDF figures for file-depth coverage, one per dataset.

    Creates separate publication-quality figures for each dataset showing:
    - Ancestor file-depth coverage (blame → BIC)
    - Descendant file-depth coverage (BIC → BFC)

    Output: rq0_file_depth_coverage_{dataset}.pdf (3 files)
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    datasets = ["LINUX", "APACHE", "GITHUB"]

    # Colors
    ancestor_color = "#5B8FA3"  # Soft steel blue
    descendant_color = "#E8B55D"  # Soft golden yellow

    print("\n  Generating per-dataset file-depth CDF figures...")

    for dataset in datasets:
        ds_cases = all_cases[all_cases["dataset"] == dataset]

        # Filter to valid cases with measurable file-history depth
        ancestor_depths = ds_cases[
            (ds_cases["category"] == "ancestor") & (ds_cases["ancestor_file_depth"] >= 0)
        ]["ancestor_file_depth"].values
        descendant_depths = ds_cases[
            (ds_cases["category"] == "descendant") & (ds_cases["descendant_file_depth"] >= 0)
        ]["descendant_file_depth"].values

        if len(ancestor_depths) == 0 and len(descendant_depths) == 0:
            print(f"    {dataset}: No valid depth data, skipping")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))

        x_range = np.arange(0, 201, 1)

        if len(ancestor_depths) > 0:
            ancestor_cdf = [
                (ancestor_depths <= x).sum() / len(ancestor_depths) * 100 for x in x_range
            ]
            ax.plot(
                x_range,
                ancestor_cdf,
                label=f"Blame Ancestor (n={len(ancestor_depths)})",
                color=ancestor_color,
                linewidth=2.5,
            )

        if len(descendant_depths) > 0:
            descendant_cdf = [
                (descendant_depths <= x).sum() / len(descendant_depths) * 100 for x in x_range
            ]
            ax.plot(
                x_range,
                descendant_cdf,
                label=f"BFC Ancestor (n={len(descendant_depths)})",
                color=descendant_color,
                linewidth=2.5,
            )

        # Add horizontal reference lines at 80% and 90%
        for y_val in [80, 90]:
            ax.axhline(y=y_val, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(
                190,
                y_val + 1.5,
                f"{y_val}%",
                fontsize=9,
                color="#666666",
                ha="left",
                fontweight="500",
            )

            if len(ancestor_depths) > 0:
                for d in range(1, 500):
                    if (ancestor_depths <= d).sum() / len(ancestor_depths) * 100 >= y_val:
                        ax.annotate(
                            f"{d}",
                            xy=(d, y_val),
                            xytext=(d, y_val - 8),
                            fontsize=9,
                            color=ancestor_color,
                            fontweight="bold",
                            ha="center",
                        )
                        break
            if len(descendant_depths) > 0:
                for d in range(1, 500):
                    if (descendant_depths <= d).sum() / len(descendant_depths) * 100 >= y_val:
                        ax.annotate(
                            f"{d}",
                            xy=(d, y_val),
                            xytext=(d, y_val + 5),
                            fontsize=9,
                            color=descendant_color,
                            fontweight="bold",
                            ha="center",
                        )
                        break

        # Styling
        ax.set_xlabel("File-History Depth Limit", fontsize=11, fontweight="500")
        ax.set_ylabel("Coverage (%)", fontsize=11, fontweight="500")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 102)
        ax.legend(
            loc="lower right",
            fontsize=10,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#CCCCCC",
        )

        ax.grid(axis="both", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(0.8)

        plt.tight_layout()

        output_path = output_dir / f"rq0_blameable_depth_{dataset.lower()}.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"    {dataset}: {output_path}")


def generate_blameless_coverage_per_dataset(all_cases, output_dir):
    """Generate individual blameless coverage figures, one per dataset.

    Creates separate publication-quality CDF figures for each dataset showing
    blameless case coverage at different depths.

    Output: rq0_blameless_depth_{dataset}.pdf (3 files) - consistent with blameable naming
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    datasets = ["LINUX", "APACHE", "GITHUB"]

    ancestor_color = "#5B8FA3"
    descendant_color = "#E8B55D"

    print("\n  Generating per-dataset blameless coverage figures...")

    for dataset in datasets:
        ds_cases = all_cases[
            (all_cases["dataset"] == dataset) & (all_cases["category"] == "blameless")
        ]

        if len(ds_cases) == 0 or "context_category" not in ds_cases.columns:
            print(f"    {dataset}: No blameless data, skipping")
            continue

        # Get cases by context category
        ancestor_cases = ds_cases[ds_cases["context_category"] == "ancestor"]
        descendant_cases = ds_cases[ds_cases["context_category"] == "descendant"]

        # Get valid depths
        ancestor_depths = ancestor_cases["context_blame_depth"].values
        ancestor_depths = ancestor_depths[ancestor_depths >= 0]

        descendant_depths = descendant_cases["context_bfc_depth"].values
        descendant_depths = descendant_depths[descendant_depths >= 0]

        if len(ancestor_depths) == 0 and len(descendant_depths) == 0:
            print(f"    {dataset}: No valid blameless context depths, skipping")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))

        x_range = np.arange(0, 201, 1)

        if len(ancestor_depths) > 0:
            ancestor_cdf = [
                (ancestor_depths <= x).sum() / len(ancestor_depths) * 100 for x in x_range
            ]
            ax.plot(
                x_range,
                ancestor_cdf,
                label=f"Blame Ancestor (n={len(ancestor_depths)})",
                color=ancestor_color,
                linewidth=2.5,
            )

        if len(descendant_depths) > 0:
            descendant_cdf = [
                (descendant_depths <= x).sum() / len(descendant_depths) * 100 for x in x_range
            ]
            ax.plot(
                x_range,
                descendant_cdf,
                label=f"BFC Ancestor (n={len(descendant_depths)})",
                color=descendant_color,
                linewidth=2.5,
            )

        # Add horizontal reference lines at 80% and 90%
        for y_val in [80, 90]:
            ax.axhline(y=y_val, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(
                190,
                y_val + 1.5,
                f"{y_val}%",
                fontsize=9,
                color="#666666",
                ha="left",
                fontweight="500",
            )

            if len(ancestor_depths) > 0:
                for d in range(1, 500):
                    if (ancestor_depths <= d).sum() / len(ancestor_depths) * 100 >= y_val:
                        ax.annotate(
                            f"{d}",
                            xy=(d, y_val),
                            xytext=(d, y_val - 8),
                            fontsize=9,
                            color=ancestor_color,
                            fontweight="bold",
                            ha="center",
                        )
                        break
            if len(descendant_depths) > 0:
                for d in range(1, 500):
                    if (descendant_depths <= d).sum() / len(descendant_depths) * 100 >= y_val:
                        ax.annotate(
                            f"{d}",
                            xy=(d, y_val),
                            xytext=(d, y_val + 5),
                            fontsize=9,
                            color=descendant_color,
                            fontweight="bold",
                            ha="center",
                        )
                        break

        # Styling - consistent with blameable figures
        ax.set_xlabel("File-History Depth Limit", fontsize=11, fontweight="500")
        ax.set_ylabel("Coverage (%)", fontsize=11, fontweight="500")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 102)
        ax.legend(
            loc="lower right",
            fontsize=10,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#CCCCCC",
        )

        ax.grid(axis="both", alpha=0.25, linestyle="-", linewidth=0.5, color="#BBBBBB")
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(0.8)

        plt.tight_layout()

        output_path = output_dir / f"rq0_blameless_depth_{dataset.lower()}.pdf"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"    {dataset}: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RQ0 results for blameable/blameless distribution"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/rq0",
        help="Directory containing RQ0 results CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rq0/rq0_analysis_report.txt",
        help="Output file for the analysis report",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.results_dir}...")
    all_cases, summary = load_data(args.results_dir)

    # Perform analysis
    analyze_blameable_distribution(all_cases, summary)
    analyze_detailed_patterns(all_cases)
    generate_insights(all_cases, summary)

    # Save report
    save_analysis_report(all_cases, summary, args.output)

    # Generate publication outputs (in results/rq0 directory)
    from pathlib import Path

    print("\nGenerating publication outputs...")
    generate_publication_figure(summary, Path(args.results_dir) / "rq0_category_distribution.pdf")
    generate_blame_complexity_figures(all_cases, args.results_dir)
    print("\nGenerating file-depth figures...")
    generate_file_depth_figures(all_cases, args.results_dir)
    print("\nGenerating file-depth CDF...")
    generate_file_depth_cdf(all_cases, args.results_dir)
    print("\\nGenerating blameless coverage analysis...")
    generate_blameless_coverage_figures(all_cases, args.results_dir)

    # Per-dataset figures for paper subfigures
    print("\nGenerating per-dataset figures...")
    generate_file_depth_cdf_per_dataset(all_cases, args.results_dir)
    generate_blameless_coverage_per_dataset(all_cases, args.results_dir)

    generate_csv_summary(summary, Path(args.results_dir) / "rq0_summary_table.csv")
    print(f"\n✓ Publication outputs saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
