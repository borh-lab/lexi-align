#!/bin/bash
# Batch evaluation script for all golden-data files

set -e

ADAPTER="${1:-litellm:azure_ai/gpt-5-nano}"
OUTPUT_DIR="${2:-results}"
shift 2 2>/dev/null || true  # Remove first two args if they exist, ignore errors

# Capture all remaining arguments to pass through
EXTRA_ARGS="$@"

mkdir -p "$OUTPUT_DIR"

echo "Evaluating all datasets with adapter: $ADAPTER"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra arguments: $EXTRA_ARGS"
fi
echo ""

for file in golden-data/*-lexi_review_with_sure.jsonl; do
    basename=$(basename "$file" .jsonl)
    output_file="$OUTPUT_DIR/results_${basename}.json"

    echo "=========================================="
    echo "Processing: $basename"
    echo "=========================================="

    uv run --dev python golden-data/evaluate_golden.py "$file" \
        --adapter "$ADAPTER" \
        --output "$output_file" \
        --use-dynamic-schema \
        --batch-size 8 \
        -v \
        $EXTRA_ARGS

    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""
echo "Generating comparison report..."

python golden-data/compare_results.py \
    "$OUTPUT_DIR"/results_*.json \
    --output "$OUTPUT_DIR/comparison_report.md" \
    --format markdown

echo "Comparison report saved to: $OUTPUT_DIR/comparison_report.md"
