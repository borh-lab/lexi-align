#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "tabulate",
# ]
# ///
"""Compare results from multiple evaluation runs."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd


def load_results(files: List[Path]) -> pd.DataFrame:
    """Load all result files into a DataFrame."""
    data = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            result = json.load(fp)

            # Handle both old and new result formats
            metrics = result.get("metrics", {})
            micro_metrics = metrics.get("micro", result.get("micro_metrics", {}))

            data.append(
                {
                    "file": f.stem,
                    "dataset": Path(result["data_file"]).stem,
                    "adapter": result["adapter_spec"],
                    "examples": result["num_examples"],
                    "successful": result["num_successful"],
                    "failed": result["num_failed"],
                    "precision": micro_metrics.get("precision", 0.0),
                    "recall": micro_metrics.get("recall", 0.0),
                    "f_measure": micro_metrics.get("f_measure", 0.0),
                    "aer": micro_metrics.get("aer", 1.0),
                }
            )
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(
        description="Compare results from multiple evaluation runs"
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="JSON result files to compare",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for comparison table (default: print to stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "latex"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    args = parser.parse_args()

    # Load results
    df = load_results(args.result_files)

    if df.empty:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    # Generate output
    output = []
    output.append("\n" + "=" * 100)
    output.append("COMPARISON OF ALIGNMENT RESULTS")
    output.append("=" * 100)
    output.append(df.to_string(index=False))
    output.append("=" * 100)

    # Summary by adapter
    output.append("\nSummary by Adapter:")
    summary = (
        df.groupby("adapter")
        .agg(
            {
                "precision": "mean",
                "recall": "mean",
                "f_measure": "mean",
                "aer": "mean",
                "successful": "sum",
                "failed": "sum",
            }
        )
        .round(4)
    )
    output.append(summary.to_string())
    output.append("=" * 100)

    # Summary by dataset
    if len(df["dataset"].unique()) > 1:
        output.append("\nSummary by Dataset:")
        dataset_summary = (
            df.groupby("dataset")
            .agg(
                {
                    "precision": "mean",
                    "recall": "mean",
                    "f_measure": "mean",
                    "aer": "mean",
                    "successful": "sum",
                    "failed": "sum",
                }
            )
            .round(4)
        )
        output.append(dataset_summary.to_string())
        output.append("=" * 100)

    result_text = "\n".join(output)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            if args.format == "markdown":
                # Convert to markdown table
                f.write("# Alignment Results Comparison\n\n")
                f.write("## Detailed Results\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n\n## Summary by Adapter\n\n")
                f.write(summary.to_markdown())
                if len(df["dataset"].unique()) > 1:
                    f.write("\n\n## Summary by Dataset\n\n")
                    f.write(dataset_summary.to_markdown())
            elif args.format == "csv":
                df.to_csv(f, index=False)
            elif args.format == "latex":
                f.write(df.to_latex(index=False))
        print(f"Results written to {args.output}")
    else:
        print(result_text)


if __name__ == "__main__":
    main()
