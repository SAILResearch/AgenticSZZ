#!/bin/bash
#
# Sensitivity Analysis for Candidate Limit (K)
# Runs experiments sequentially with different K values
#
# Usage:
#   ./evaluation/run_sensitivity_analysis.sh
#   # Or from anywhere:
#   bash /path/to/evaluation/run_sensitivity_analysis.sh
#
# Results saved to: results/sensitivity_analysis/sensitivity_k{5,10,15,20,30,50}
#
# Every flag apart from --candidate-limit must match the main full run, so that
# the curve isolates K. --context-expand 2 is passed explicitly because
# TKGConfig.context_expand_lines defaults to 0.
#

set -e  # Exit on error

# Auto-detect project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Configuration
SAMPLE_FILE="results/pilot/pilot_cases_20pct.json"
OUTPUT_BASE="results/sensitivity_analysis"
WORKERS=16
CONFIG="full"
DATASET="all"

# K values to test
K_VALUES=(5 10 15 20 30 50)

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "AgenticSZZ Sensitivity Analysis"
echo "=============================================="
echo "Sample file: $SAMPLE_FILE"
echo "Workers: $WORKERS"
echo "K values: ${K_VALUES[*]}"
echo "Output: $OUTPUT_BASE/sensitivity_k*"
echo "=============================================="
echo ""

# Check if sample file exists
if [ ! -f "$SAMPLE_FILE" ]; then
    echo "ERROR: Sample file not found: $SAMPLE_FILE"
    exit 1
fi

# Run experiments sequentially
for K in "${K_VALUES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/sensitivity_k$K"

    echo ""
    echo "=============================================="
    echo "Starting K=$K experiment"
    echo "Output: $OUTPUT_DIR"
    echo "Time: $(date)"
    echo "=============================================="

    # Skip if already completed (check for CSV files)
    if ls "$OUTPUT_DIR"/*results_*.csv 1> /dev/null 2>&1; then
        echo "SKIPPING: Results already exist in $OUTPUT_DIR"
        continue
    fi

    # Run evaluation
    python evaluation/run_evaluation.py \
        --candidate-limit "$K" \
        --output-dir "$OUTPUT_DIR" \
        --sample-file "$SAMPLE_FILE" \
        --config "$CONFIG" \
        --context-expand 2 \
        --workers "$WORKERS" \
        --dataset "$DATASET"

    echo ""
    echo "Completed K=$K at $(date)"
    echo ""
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""

# Generate figure
echo "Generating sensitivity analysis figure..."
python evaluation/sensitivity_analysis.py \
    --results-dir "$OUTPUT_BASE" \
    --output "figures/sensitivity_k.pdf" \
    --show-datasets

echo ""
echo "Done! Figure saved to figures/sensitivity_k.pdf"
