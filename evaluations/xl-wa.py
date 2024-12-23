#!/usr/bin/env python3
import argparse
import json
import logging
import math
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, TypedDict, Union, cast
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import requests  # type: ignore
import seaborn as sns  # type: ignore
from loguru import logger


def setup_logging(verbosity: int = 0):
    """Setup logging to use loguru for everything."""
    # Remove default loguru handler
    logger.remove()

    # Define log levels
    log_levels = {
        0: "WARNING",  # default
        1: "INFO",     # -v
        2: "DEBUG",    # -vv
    }
    level = log_levels[min(verbosity, 2)]

    # Define format with colors
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:"
        "<cyan>{function}</cyan>:"
        "<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Add loguru handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True
    )

    # Create handler that routes standard logging to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Remove any existing handlers from root logger
    logging.root.handlers = []

    # Add intercept handler to root logger
    logging.root.addHandler(InterceptHandler())

    # Set level for root logger
    logging.root.setLevel(level)
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binomtest  # type: ignore
from tqdm import tqdm  # type: ignore

from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
from lexi_align.adapters.llama_cpp_adapter import LlamaCppAdapter
from lexi_align.adapters.outlines_adapter import OutlinesAdapter
from lexi_align.core import (
    MetricsDict,
    ValidationErrorStats,
    align_tokens,
    align_tokens_batched,
)
from lexi_align.metrics import calculate_metrics
from lexi_align.models import (
    AlignmentResult,
    TextAlignment,
    TokenMapping,
    ValidationErrorType,
)
from lexi_align.text_processing import (
    MarkerGenerator,
    create_subscript_generator,
    create_underscore_generator,
)
from lexi_align.utils import (
    create_token_mapping,
    parse_pharaoh_format,
    read_pharaoh_file,
    remove_unique_one,
    validate_token_lists,
)
from lexi_align.visualize import visualize_alignments


class TokenStats(TypedDict):
    correct: int
    incorrect: int
    missed: int
    extra: int
    alignment_ratio: float
    accuracy: float
    coverage: float


def analyze_token_alignments(
    predicted: TextAlignment,
    gold: TextAlignment,
) -> dict[str, Dict[str, TokenStats]]:
    """Analyze token alignment accuracy and patterns.

    Args:
        predicted: Predicted alignment
        gold: Gold standard alignment

    Returns:
        Dictionary with source and target token statistics

    Example:
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> pred = TextAlignment(alignment=[
        ...     TokenAlignment(source="the₁", target="le₁"),
        ...     TokenAlignment(source="cat", target="chat")
        ... ])
        >>> gold = TextAlignment(alignment=[
        ...     TokenAlignment(source="the₁", target="le₁"),
        ...     TokenAlignment(source="cat", target="chat"),
        ...     TokenAlignment(source="the₂", target="le₂")
        ... ])
        >>> stats = analyze_token_alignments(pred, gold)
        >>> stats["source"]["the₁"]["correct"]  # Changed to use full token
        1
        >>> stats["source"]["the₂"]["missed"]  # Changed to use full token
        1
        >>> stats["target"]["chat"]["correct"]
        1
    """
    # Track per-token statistics using full tokens (with markers)
    source_stats: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "missed": 0, "extra": 0}
    )
    target_stats: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "missed": 0, "extra": 0}
    )

    # Get alignment pairs directly
    pred_pairs = {(a.source, a.target) for a in predicted.alignment}
    gold_pairs = {(a.source, a.target) for a in gold.alignment}

    # Find correct alignments (true positives)
    for src, tgt in pred_pairs & gold_pairs:
        source_stats[src]["correct"] += 1
        target_stats[tgt]["correct"] += 1

    # Find incorrect alignments (false positives)
    for src, tgt in pred_pairs - gold_pairs:
        source_stats[src]["incorrect"] += 1
        target_stats[tgt]["incorrect"] += 1

    # Find missed alignments (false negatives)
    for src, tgt in gold_pairs - pred_pairs:
        source_stats[src]["missed"] += 1
        target_stats[tgt]["missed"] += 1

    # Calculate alignment ratios
    for stats in [source_stats, target_stats]:
        for token_stats in stats.values():
            token_stats["alignment_ratio"] = float(
                (token_stats["correct"] + token_stats["incorrect"])
                / max(token_stats["correct"] + token_stats["missed"], 1)
            )

    return {
        "source": cast(Dict[str, TokenStats], dict(source_stats)),
        "target": cast(Dict[str, TokenStats], dict(target_stats)),
    }


def aggregate_token_statistics(
    all_stats: list[dict],
) -> dict[str, Dict[str, TokenStats]]:
    """Aggregate token statistics across multiple examples."""
    combined_stats: Dict[str, DefaultDict[str, Dict[str, Any]]] = {
        "source": defaultdict(
            lambda: {"correct": 0, "incorrect": 0, "missed": 0, "extra": 0}
        ),
        "target": defaultdict(
            lambda: {"correct": 0, "incorrect": 0, "missed": 0, "extra": 0}
        ),
    }

    # Combine stats
    for stats in all_stats:
        for lang in ["source", "target"]:
            for token, token_stats in stats[lang].items():
                for key in ["correct", "incorrect", "missed", "extra"]:
                    combined_stats[lang][token][key] += token_stats[key]

    # Calculate final metrics
    for lang in ["source", "target"]:
        for token, stats in combined_stats[lang].items():
            total_gold = stats["correct"] + stats["missed"]
            total_pred = stats["correct"] + stats["incorrect"]
            stats["accuracy"] = float(stats["correct"] / max(total_pred, 1))
            stats["coverage"] = float(stats["correct"] / max(total_gold, 1))
            stats["alignment_ratio"] = float(total_pred / max(total_gold, 1))

    return {
        "source": cast(Dict[str, TokenStats], dict(combined_stats["source"])),
        "target": cast(Dict[str, TokenStats], dict(combined_stats["target"])),
    }


ADAPTER_TYPES = {
    "litellm": LiteLLMAdapter,
    "outlines": OutlinesAdapter,
    "llama-cpp": LlamaCppAdapter,
}

LANGUAGE_MAP = {
    "BG": "Bulgarian",
    "DA": "Danish",
    "ES": "Spanish",
    "ET": "Estonian",
    "HU": "Hungarian",
    "IT": "Italian",
    "NL": "Dutch",
    "PT": "Portuguese",
    "RU": "Russian",
    "SL": "Slovenian",
}

# All available language pairs in XL-WA (publicly available)
ALL_LANG_PAIRS = [
    # "EN-AR",
    "EN-BG",
    "EN-DA",
    "EN-ES",
    "EN-ET",
    "EN-HU",
    "EN-IT",
    # "EN-KO",
    "EN-NL",
    "EN-PT",
    "EN-RU",
    "EN-SL",
    # "EN-SV",
    # "EN-ZH",
]


def download_xl_wa(target_dir: Path) -> None:
    """Download and extract the XL-WA dataset zip file, using cached version if available."""
    url = "https://github.com/SapienzaNLP/XL-WA/archive/f5c9ea26daa4e53e5f3fa133a45e1bede1db816d.zip"
    cache_dir = Path(__file__).parent
    cached_zip = cache_dir / "xl-wa.zip"

    # Check if we need to download
    if not cached_zip.exists():
        logger.info("Downloading XL-WA dataset...")
        response = requests.get(url)
        response.raise_for_status()

        # Save to cache
        cached_zip.write_bytes(response.content)
        logger.info(f"Saved dataset to {cached_zip}")
    else:
        logger.info(f"Using cached dataset from {cached_zip}")

    logger.info("Extracting dataset...")
    with ZipFile(cached_zip) as zip_file:
        zip_file.extractall(target_dir)

        # The zip creates a subdirectory with the commit hash - we need to account for this
        extracted_dir = target_dir / "XL-WA-f5c9ea26daa4e53e5f3fa133a45e1bede1db816d"

        # Move contents up one level if needed
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                item.rename(target_dir / item.name)
            extracted_dir.rmdir()


def export_results(results_file: str, output_dir: Path) -> None:
    """Export alignments from results JSON to Pharaoh format files."""
    with open(results_file) as f:
        results = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for lang_pair, data in results["language_pairs"].items():
        output_file = output_dir / f"{lang_pair.lower()}.align"
        logger.info(f"Writing alignments to {output_file}")

        with open(output_file, "w") as f:
            for item in data["alignments"]:
                # Get the original source and target sentences
                source_sent = " ".join(item["source_tokens"])
                target_sent = " ".join(item["target_tokens"])

                # Get the predicted alignment
                predicted = TextAlignment.model_validate(item["predicted"])

                # Create token mappings
                source_mapping = create_token_mapping(item["source_tokens"])
                target_mapping = create_token_mapping(item["target_tokens"])

                # Get alignment positions
                alignment_pairs = predicted.get_alignment_positions(
                    source_mapping, target_mapping
                )

                # Format alignment string
                alignment_str = " ".join(f"{s}-{t}" for s, t in sorted(alignment_pairs))

                # Write in Pharaoh format
                f.write(f"{source_sent}\t{target_sent}\t{alignment_str}\n")


def calculate_overall_metrics(results: dict) -> dict:
    """Calculate micro-averaged metrics across all language pairs."""
    total_true_positives = 0
    total_predicted = 0
    total_gold = 0

    for data in results["language_pairs"].values():
        metrics = data["metrics"]
        total_true_positives += metrics["total_true_positives"]
        total_predicted += metrics["total_predicted"]
        total_gold += metrics["total_gold"]

    precision = total_true_positives / total_predicted if total_predicted > 0 else 0
    recall = total_true_positives / total_gold if total_gold > 0 else 0
    f_measure = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f_measure": f_measure}


def create_alignment_visualizations(results_files: list[str], pdf_path: str) -> None:
    """Create PDF with visualizations of alignments from multiple results files."""
    pdf_path = ensure_extension(pdf_path, "pdf")
    all_alignments = {}  # Dict to store alignments by source/target pair

    # Load all results files
    for file in results_files:
        with open(file) as f:
            results = json.load(f)

            # Create model identifier including key parameters
            params = results["parameters"]
            model_id = f"{params['model']}"
            if params.get("num_train_examples"):
                model_id += f" ({params['num_train_examples']}shot)"
            if params.get("model_seed"):
                model_id += f" (seed={params['model_seed']})"

            # Process each language pair
            for lang_pair, data in results["language_pairs"].items():
                for alignment_data in data["alignments"]:
                    # Create key from source and target tokens
                    source_tokens = alignment_data["source_tokens"]
                    target_tokens = alignment_data["target_tokens"]
                    key = (tuple(source_tokens), tuple(target_tokens), lang_pair)

                    # Initialize dict for this source/target pair if needed
                    if key not in all_alignments:
                        all_alignments[key] = {
                            "Gold": TextAlignment.model_validate(alignment_data["gold"])
                        }

                    # Add this model's alignment
                    all_alignments[key][model_id] = TextAlignment.model_validate(
                        alignment_data["predicted"]
                    )

    # Create PDF with all visualizations
    with PdfPages(pdf_path) as pdf:
        # Sort keys for consistent ordering
        for key in sorted(all_alignments.keys()):
            source_tokens, target_tokens, lang_pair = key
            alignments = all_alignments[key]

            # Only visualize if we have multiple models (including gold)
            if len(alignments) > 2:  # More than just gold + one model
                # Calculate metrics comparing each model to Gold
                metrics_str = ""
                for model_id in sorted(alignments.keys()):
                    if model_id != "Gold":
                        metrics = calculate_metrics(
                            alignments[model_id], alignments["Gold"]
                        )
                        metrics_str += f"\n{model_id}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} f_measure={metrics['f_measure']:.2f}"

                # Create more compact title with token counts
                title = (
                    f"{lang_pair} ({len(source_tokens)}-{len(target_tokens)} tokens)\n"
                    f"S: {' '.join(source_tokens)}\n"
                    f"T: {' '.join(target_tokens)}"
                )

                # Create visualization with gold as reference
                visualize_alignments(
                    source_tokens=list(source_tokens),
                    target_tokens=list(target_tokens),
                    alignments=alignments,
                    title=title,
                    reference_model="Gold",  # Use gold alignments as reference
                )

                # Save current figure to PDF
                pdf.savefig(bbox_inches="tight")
                plt.close()


def calculate_token_accuracy(
    correct: int,
    occurrences: int,
) -> tuple[float, float]:
    """Calculate accuracy and Clopper-Pearson exact method lower bound for token alignments.

    Args:
        correct: Number of correct alignments
        occurrences: Total number of occurrences

    Returns:
        Tuple of (accuracy, lower_bound)

    Example:
        >>> acc, lower = calculate_token_accuracy(8, 10)
        >>> f"{acc:.3f}"
        '0.800'
        >>> f"{lower:.3f}"
        '0.444'
        >>> acc, lower = calculate_token_accuracy(95, 100)
        >>> f"{acc:.3f}"
        '0.950'
        >>> f"{lower:.3f}"
        '0.887'
    """

    if occurrences == 0:
        return 0.0, 0.0

    accuracy = correct / occurrences

    # Use scipy's binomtest which implements Clopper-Pearson exact method
    result = binomtest(correct, occurrences)
    lower_bound = result.proportion_ci()[0]  # Get lower bound of confidence interval

    return accuracy, lower_bound


def analyze_token_statistics(alignments_data: list[dict]) -> dict:
    """Analyze token alignment patterns across all examples."""
    # Track token statistics with type annotation
    token_stats: dict[str, dict[str, dict[str, int]]] = {
        "source": defaultdict(lambda: {"correct": 0, "occurrences": 0}),
        "target": defaultdict(lambda: {"correct": 0, "occurrences": 0}),
    }

    # Collect statistics
    for item in alignments_data:
        source_tokens = item["source_tokens"]
        target_tokens = item["target_tokens"]
        predicted = TextAlignment.model_validate(item["predicted"])
        gold = TextAlignment.model_validate(item["gold"])

        # Create token mappings
        source_mapping = create_token_mapping(source_tokens)
        target_mapping = create_token_mapping(target_tokens)

        # Get alignment positions
        pred_positions = set(
            predicted.get_alignment_positions(source_mapping, target_mapping)
        )
        gold_positions = set(
            gold.get_alignment_positions(source_mapping, target_mapping)
        )
        correct_positions = pred_positions & gold_positions

        # Track occurrences and correct alignments for source tokens
        for src_token in source_tokens:
            base_token = remove_unique_one(src_token, source_mapping.marker_pattern)
            token_stats["source"][base_token]["occurrences"] += 1

            # Find position of this token
            src_idx = source_mapping.get_position(src_token)
            # Count correct alignments for this specific token occurrence
            if any(
                (s_idx, t_idx) in correct_positions
                for s_idx, t_idx in gold_positions
                if s_idx == src_idx
            ):
                token_stats["source"][base_token]["correct"] += 1

        # Track occurrences and correct alignments for target tokens
        for tgt_token in target_tokens:
            base_token = remove_unique_one(tgt_token, target_mapping.marker_pattern)
            token_stats["target"][base_token]["occurrences"] += 1

            # Find position of this token
            tgt_idx = target_mapping.get_position(tgt_token)
            # Count correct alignments for this specific token occurrence
            if any(
                (s_idx, t_idx) in correct_positions
                for s_idx, t_idx in gold_positions
                if t_idx == tgt_idx
            ):
                token_stats["target"][base_token]["correct"] += 1

    # Calculate statistics for each token
    results: dict[str, dict[str, dict[str, Union[float, int]]]] = {
        "source": {},
        "target": {},
    }

    for lang in ["source", "target"]:
        for token, stats in token_stats[lang].items():
            # Skip tokens with too few occurrences
            if stats["occurrences"] < 5:
                continue

            # Calculate accuracy
            accuracy = stats["correct"] / stats["occurrences"]

            # Calculate Wilson score interval
            # Using z=1.96 for 95% confidence interval
            n = stats["occurrences"]
            p = accuracy

            # Wilson score interval calculation
            z = 1.96  # 95% confidence
            denominator = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denominator
            spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
            lower_bound = max(0, center - spread)

            results[lang][token] = {
                "accuracy": accuracy,
                "lower_bound": lower_bound,
                "occurrences": stats["occurrences"],
                "correct": stats["correct"],
                "weighted_score": accuracy * math.log2(1 + stats["occurrences"]),
            }

    return results


def evaluate_results(
    results_files: list[str], output_path: Optional[str] = None
) -> str:
    """Generate markdown table and plots comparing results from multiple JSON files."""
    all_results = {}
    metrics_data = []  # For plotting

    for file in results_files:
        with open(file) as f:
            results = json.load(f)
            # Create model identifier including key parameters
            params = results["parameters"]
            model_id = f"{params['model']}"
            if params.get("num_train_examples"):
                model_id += f" ({params['num_train_examples']}shot)"
            if params.get("model_seed"):
                model_id += f" (seed={params['model_seed']})"

            all_results[model_id] = results

            # Collect individual alignment metrics for plotting
            for lang_pair, data in results["language_pairs"].items():
                for alignment_data in data["alignments"]:
                    metrics = alignment_data[
                        "metrics"
                    ]  # Get metrics for each individual alignment
                    metrics_data.append(
                        {
                            "Model": model_id,
                            "Language Pair": lang_pair,
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "f_measure": metrics["f_measure"],
                        }
                    )

    # Create DataFrame for plotting
    df = pd.DataFrame(metrics_data)
    print(df)

    # Create distribution plots
    plt.figure(figsize=(15, 8))  # Increased figure size

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12

    # Create violin plots for each metric
    metrics = ["Precision", "Recall", "f_measure"]
    num_models = len(df["Model"].unique())
    width = 0.8 / num_models  # Adjust width based on number of models

    for i, model in enumerate(df["Model"].unique()):
        model_data = df[df["Model"] == model]
        offset = (i - (num_models - 1) / 2) * width  # Center the groups

        # Create violin plot
        for j, metric in enumerate(metrics):
            # Position violins with offset
            pos = j + offset
            violin_parts = plt.violinplot(
                model_data[metric],
                positions=[pos],
                widths=width,
                showmeans=True,
                showextrema=True,
            )

            # Customize violin colors and style
            color = plt.colormaps["Set3"](i / num_models)
            for pc in violin_parts["bodies"]:  # type: ignore[attr-defined]
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            # Customize other parts
            violin_parts["cmeans"].set_color("black")
            violin_parts["cmaxes"].set_color("black")
            violin_parts["cmins"].set_color("black")
            violin_parts["cbars"].set_color("black")

        # Add to legend
        plt.plot([], [], color=color, label=model, linewidth=10, alpha=0.7)

    # Customize plot
    plt.xticks(range(len(metrics)), metrics)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    # Get unique language pairs for title
    lang_pairs_str = ", ".join(sorted(df["Language Pair"].unique()))

    # Update title with language pairs
    plt.title(f"Distribution of Alignment Metrics\n({lang_pairs_str})")

    # Move legend below plot and adjust y-axis limit
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)

    # Add space at top of plot
    plt.ylim(0.0, 1.05)  # Increased upper limit to 1.05

    # Add horizontal gridlines at 0.1 intervals
    plt.grid(True, axis="y", alpha=0.3)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add space at bottom for legend

    # Build tables per model
    tables = []
    for model, results in all_results.items():
        # Build table header
        header = ["Language Pair", "Precision", "Recall", "f_measure"]

        # Build table rows
        rows = []
        for lang_pair in sorted(results["language_pairs"].keys()):
            metrics = results["language_pairs"][lang_pair]["metrics"]
            rows.append(
                [
                    lang_pair,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f_measure']:.3f}",
                ]
            )

        # Add overall averages
        metrics = calculate_overall_metrics(results)
        rows.append(
            [
                "**Average**",
                f"**{metrics['precision']:.3f}**",
                f"**{metrics['recall']:.3f}**",
                f"**{metrics['f_measure']:.3f}**",
            ]
        )

        # Format as markdown table with caption
        table = [
            f"### {model}",
            "",  # Empty line after caption
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        table.extend("| " + " | ".join(row) + " |" for row in rows)
        table.append("")  # Empty line after table

        tables.append("\n".join(table))

    # Save plot if output path provided
    if output_path:
        plot_path = output_path.replace(".md", ".png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

    # Add token analysis section
    tables.append("\n### Token Analysis\n")

    for model, results in all_results.items():
        tables.append(f"\n#### {model}\n")

        for lang_pair, data in results["language_pairs"].items():
            # Analyze tokens for this language pair
            token_stats = analyze_token_statistics(data["alignments"])

            tables.append(f"\n##### {lang_pair}")

            for lang in ["source", "target"]:
                tables.append(f"\n###### {lang.title()} Language")

                # Sort tokens by weighted score for most reliable
                most_reliable = sorted(
                    token_stats[lang].items(),
                    key=lambda x: x[1]["weighted_score"],
                    reverse=True,  # Highest scores first
                )

                # Sort tokens by weighted score for least reliable
                least_reliable = sorted(
                    token_stats[lang].items(),
                    key=lambda x: x[1]["weighted_score"],
                    reverse=False,  # Lowest scores first
                )

                # Show most reliable tokens (high accuracy, many occurrences)
                tables.append("\nMost reliable alignments:")
                for token, stats in most_reliable[:25]:  # First 25 (highest scores)
                    tables.append(
                        f"- {token}: {stats['accuracy']:.2%} accuracy "
                        f"({stats['correct']}/{stats['occurrences']} occurrences, "
                        f"95% CI: [{stats['lower_bound']:.2%}, {stats['accuracy']:.2%}])"
                    )

                # Show least reliable tokens (lowest scores)
                tables.append("\nLeast reliable alignments:")
                for token, stats in least_reliable[:25]:  # First 25 (lowest scores)
                    tables.append(
                        f"- {token}: {stats['accuracy']:.2%} accuracy "
                        f"({stats['correct']}/{stats['occurrences']} occurrences, "
                        f"95% CI: [{stats['lower_bound']:.2%}, {stats['accuracy']:.2%}])"
                    )

    # Add validation error analysis section
    tables.append("\n### Validation Error Analysis\n")

    for model, results in all_results.items():
        tables.append(f"\n#### {model}\n")

        # Aggregate validation error frequencies across all language pairs
        aggregated_stats: dict[ValidationErrorType, Dict[str, int]] = {
            error_type: {} for error_type in ValidationErrorType
        }

        total_errors: dict[ValidationErrorType, int] = {
            error_type: 0 for error_type in ValidationErrorType
        }

        for lang_pair, data in results["language_pairs"].items():
            diagnostics = data["metrics"]["diagnostics"]
            for error_type_str, stats in diagnostics["validation_error_stats"].items():
                error_type = ValidationErrorType(error_type_str)
                total_errors[error_type] += stats["count"]

                # Aggregate frequencies
                for token, freq in stats["frequencies"].items():
                    if token not in aggregated_stats[error_type]:
                        aggregated_stats[error_type][token] = 0
                    aggregated_stats[error_type][token] += freq

        # Output aggregated statistics
        for error_type in ValidationErrorType:
            if total_errors[error_type] > 0:
                tables.append(
                    f"\n##### {error_type.value} (Total: {total_errors[error_type]})"
                )

                # Sort tokens by frequency
                sorted_tokens = sorted(
                    aggregated_stats[error_type].items(),
                    key=lambda x: (
                        -x[1],
                        x[0],
                    ),  # Sort by frequency desc, then token asc
                )

                # Create frequency table
                tables.append("\nToken | Frequency | % of Error Type")
                tables.append("------|-----------|----------------")
                for token, freq in sorted_tokens:
                    percentage = (freq / total_errors[error_type]) * 100
                    tables.append(f"`{token}` | {freq} | {percentage:.1f}%")
                tables.append("")  # Empty line after table

    return "\n".join(tables)


def get_run_parameters(args: argparse.Namespace) -> dict:
    """Collect all run parameters into a dictionary."""
    return {
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "model_seed": args.model_seed,
        "num_train_examples": args.num_train_examples,
        "sample_size": args.sample_size,
        "marker_type": args.marker_type,
    }


def get_marker_generator(marker_type: str) -> MarkerGenerator:
    """Get marker generator based on type string."""
    if marker_type == "subscript":
        return create_subscript_generator()
    elif marker_type == "underscore":
        return create_underscore_generator()
    else:
        raise ValueError(f"Unknown marker type: {marker_type}")


def ensure_extension(filepath: str, extension: str) -> str:
    """Ensure filepath has the specified extension.

    Args:
        filepath: Path to file
        extension: Extension to ensure (without dot)

    Returns:
        Path with extension added if needed

    Example:
        >>> ensure_extension("results", "json")
        'results.json'
        >>> ensure_extension("results.json", "json")
        'results.json'
        >>> ensure_extension("path/to/results", "pdf")
        'path/to/results.pdf'
    """
    if not filepath.lower().endswith(f".{extension.lower()}"):
        return f"{filepath}.{extension}"
    return filepath


def get_language_pairs(lang_pairs: Optional[List[str]]) -> List[str]:
    """Validate and return language pairs to evaluate."""
    if lang_pairs is None:
        return ["EN-SL"]  # Default
    elif lang_pairs == ["all"]:
        return ALL_LANG_PAIRS
    else:
        # Validate language pairs
        invalid_pairs = set(lang_pairs) - set(ALL_LANG_PAIRS)
        if invalid_pairs:
            raise ValueError(f"Invalid language pairs: {invalid_pairs}")
        return lang_pairs


def load_training_examples(
    repo_path: Path, lang_pair: str, num_examples: Optional[int] = None
) -> List[Tuple[List[str], List[str], TextAlignment]]:
    """Load training examples for a language pair."""
    target_lang = lang_pair.split("-")[1].lower()
    train_file = repo_path / "data" / target_lang / "train.tsv"

    if not train_file.exists():
        logger.warning(f"Training file not found: {train_file}")
        return []

    try:
        examples = read_pharaoh_file(str(train_file))

        if num_examples is not None:
            examples = random.sample(examples, min(num_examples, len(examples)))

        # Convert to expected format with tokenized lists
        return [(src.split(), tgt.split(), align) for src, tgt, align in examples]

    except Exception as e:
        logger.error(f"Error loading training examples: {e}")
        return []


def evaluate_language_pair(
    repo_path: Path,
    lang_pair: str,
    llm_adapter: Union[LiteLLMAdapter, OutlinesAdapter, LlamaCppAdapter],
    args: argparse.Namespace,
) -> tuple[MetricsDict, list[dict]]:
    """Evaluate alignment performance for a single language pair using micro-averaging."""
    target_lang_code = lang_pair.split("-")[1]
    target_lang_lower = target_lang_code.lower()

    test_file = repo_path / "data" / target_lang_lower / "test.tsv"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    logger.info(f"Evaluating {lang_pair}...")

    # Get marker generator based on args
    marker_generator = get_marker_generator(args.marker_type)

    training_examples = None
    if args.num_train_examples is not None:
        training_examples = load_training_examples(
            repo_path, lang_pair, args.num_train_examples
        )
        logger.info(f"Loaded {len(training_examples)} training examples")

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = f.readlines()

    if args.sample_size and len(test_cases) > args.sample_size:
        test_cases = test_cases[: args.sample_size]

    # Process in batches if batch size is specified
    if args.batch_size and hasattr(llm_adapter, "batch"):
        logger.info(f"Processing in batches of size {args.batch_size}")
        return _evaluate_language_pair_batch(
            test_cases,
            lang_pair,
            llm_adapter,
            args,
            training_examples,
            marker_generator,
        )
    else:
        logger.info("Processing examples sequentially")
        return _evaluate_language_pair_sequential(
            test_cases,
            lang_pair,
            llm_adapter,
            args,
            training_examples,
            marker_generator,
        )


def _process_alignment_result(
    alignment_result: AlignmentResult,
    gold_alignment: TextAlignment,
    source_tokens: list[str],
    target_tokens: list[str],
    source_mapping: TokenMapping,
    target_mapping: TokenMapping,
) -> tuple[Optional[dict], Optional[dict]]:
    """Process a single alignment result and return metrics and alignment data.

    Returns:
        Tuple of (metrics_dict, alignment_data_dict) or (None, None) if processing fails
    """
    if not alignment_result.alignment:
        return None, None

    # Validate alignment tokens
    is_valid, errors = validate_token_lists(
        [a.source_token for a in alignment_result.alignment.alignment],
        [a.target_token for a in alignment_result.alignment.alignment],
        source_mapping,
        target_mapping,
    )

    if not is_valid:
        logger.error(f"Invalid alignment tokens: {errors}")
        return None, None

    # Ensure alignment is sorted via model validation
    alignment_result.alignment = TextAlignment.model_validate(
        alignment_result.alignment.model_dump()
    )

    metrics = calculate_metrics(alignment_result.alignment, gold_alignment)

    # Create alignment data dictionary
    alignment_data = {
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "predicted": alignment_result.alignment.model_dump(),
        "gold": gold_alignment.model_dump(),
        "metrics": metrics,
        "diagnostics": {
            "total_attempts": len(alignment_result.attempts),
            "attempts": [attempt.model_dump() for attempt in alignment_result.attempts],
        },
    }

    return metrics, alignment_data


def _evaluate_language_pair_sequential(
    test_cases: list[str],
    lang_pair: str,
    llm_adapter: Union[LiteLLMAdapter, OutlinesAdapter, LlamaCppAdapter],
    args: argparse.Namespace,
    training_examples: Optional[
        List[Tuple[List[str], List[str], TextAlignment]]
    ] = None,
    marker_generator: Optional[MarkerGenerator] = None,
) -> tuple[MetricsDict, list[dict]]:
    """Process test cases sequentially."""
    target_lang_code = lang_pair.split("-")[1]
    target_lang = LANGUAGE_MAP.get(target_lang_code, target_lang_code)

    # Micro-averaged metrics and diagnostics
    total_true_positives = 0
    total_predicted = 0
    total_gold = 0
    failed_calls = 0
    total_attempts = 0
    total_validation_errors = 0
    exception_counts: dict[str, int] = {}
    validation_error_stats: dict[ValidationErrorType, ValidationErrorStats] = {
        error_type: {"count": 0, "frequencies": {}}
        for error_type in ValidationErrorType
    }
    alignments_data = []

    for i, line in enumerate(tqdm(test_cases, desc=f"Processing {lang_pair}"), 1):
        try:
            source_sent, target_sent, gold_alignment = parse_pharaoh_format(line)
            source_tokens = source_sent.split()
            target_tokens = target_sent.split()

            # Create token mappings once and reuse
            source_mapping = create_token_mapping(source_tokens, marker_generator)
            target_mapping = create_token_mapping(target_tokens, marker_generator)

            logger.debug(
                f"Processing example {i}:\n  Source: {source_sent}\n  Target: {target_sent}\n"
                f"  Unique source: {' '.join(source_mapping.uniquified)}\n"
                f"  Unique target: {' '.join(target_mapping.uniquified)}"
            )

            alignment_result = align_tokens(
                llm_adapter,
                source_mapping.uniquified,  # Use pre-uniquified tokens
                target_mapping.uniquified,  # Use pre-uniquified tokens
                source_language="English",
                target_language=target_lang,
                examples=training_examples,
                marker_generator=marker_generator,
                max_retries=args.max_retries,
            )

            metrics, alignment_data = _process_alignment_result(
                alignment_result,
                gold_alignment,
                source_tokens,
                target_tokens,
                source_mapping,
                target_mapping,
            )

            if metrics and alignment_data:
                # Update counters
                total_true_positives += metrics["true_positives"]
                total_predicted += metrics["predicted"]
                total_gold += metrics["gold"]
                alignments_data.append(alignment_data)
            else:
                failed_calls += 1

            # Update diagnostic counters
            total_attempts += len(alignment_result.attempts)
            for attempt in alignment_result.attempts:
                if not attempt.validation_passed:
                    total_validation_errors += 1
                    for error_type, _description, tokens in attempt.validation_errors:
                        validation_error_stats[error_type]["count"] += 1
                        # Update frequencies for each token
                        for token in tokens:
                            validation_error_stats[error_type]["frequencies"][token] = (
                                validation_error_stats[error_type]["frequencies"].get(
                                    token, 0
                                )
                                + 1
                            )
                if attempt.exception:
                    exc_type = attempt.exception.split(":")[0].strip()
                    exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1

        except Exception as e:
            logger.error(
                f"Unexpected error processing example {i}:\n"
                f"  Raw line: {line!r}\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error: {str(e)}"
            )
            failed_calls += 1
            continue

    # Calculate micro-averaged metrics with enhanced diagnostics
    micro_precision = (
        total_true_positives / total_predicted if total_predicted > 0 else 0.0
    )
    micro_recall = total_true_positives / total_gold if total_gold > 0 else 0.0

    # Calculate AER
    micro_aer = (
        1.0 - ((total_true_positives * 2) / (total_predicted + total_gold))
        if (total_predicted + total_gold) > 0
        else 1.0
    )

    # Calculate weighted F-measure with α=0.5 (equivalent to f_measure)
    if micro_precision > 0 and micro_recall > 0:
        f_divident = (0.5 / micro_precision) + (0.5 / micro_recall)
        micro_f_measure = 1.0 / f_divident
    else:
        micro_f_measure = 0.0

    final_metrics: MetricsDict = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f_measure": micro_f_measure,
        "aer": micro_aer,
        "total_predicted": total_predicted,
        "total_gold": total_gold,
        "total_true_positives": total_true_positives,
        "diagnostics": {
            "total_attempts": total_attempts,
            "total_validation_errors": total_validation_errors,
            "avg_attempts_per_pair": total_attempts / (len(test_cases) - failed_calls)
            if len(test_cases) > failed_calls
            else 0,
            "validation_error_stats": {
                ValidationErrorType(error_type.value): {
                    "count": stats["count"],
                    "frequencies": stats["frequencies"],
                }
                for error_type, stats in validation_error_stats.items()
                if isinstance(stats, dict)
                and isinstance(stats.get("count"), int)
                and stats["count"] > 0
            },
            "exception_types": exception_counts,
            "failed_calls": failed_calls,
            "failure_rate": failed_calls / len(test_cases) if test_cases else 0,
        },
    }

    return final_metrics, alignments_data


def _evaluate_language_pair_batch(
    test_cases: list[str],
    lang_pair: str,
    llm_adapter: Union[LiteLLMAdapter, OutlinesAdapter, LlamaCppAdapter],
    args: argparse.Namespace,
    training_examples: Optional[
        List[Tuple[List[str], List[str], TextAlignment]]
    ] = None,
    marker_generator: Optional[MarkerGenerator] = None,
) -> tuple[MetricsDict, list[dict]]:
    """Process test cases in batches."""
    target_lang_code = lang_pair.split("-")[1]
    target_lang = LANGUAGE_MAP.get(target_lang_code, target_lang_code)

    # Initialize counters and storage
    total_true_positives = 0
    total_predicted = 0
    total_gold = 0
    failed_calls = 0
    total_attempts = 0
    total_validation_errors = 0
    exception_counts: dict[str, int] = {}
    validation_error_stats: dict[ValidationErrorType, ValidationErrorStats] = {
        error_type: {"count": 0, "frequencies": {}}
        for error_type in ValidationErrorType
    }
    alignments_data = []
    # TODO most and least correctly aligned tokens
    # TODO most over-aligned and under-aligned tokens

    # Parse all test cases first
    source_batch = []
    target_batch = []
    source_tokens_list = []
    target_tokens_list = []
    gold_alignments = []
    source_mappings = []
    target_mappings = []

    # Create marker generator if not provided
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    # Add progress bar for parsing phase
    for line in tqdm(test_cases, desc=f"Parsing {lang_pair} test cases"):
        try:
            source_sent, target_sent, gold_alignment = parse_pharaoh_format(line)
            source_tokens = source_sent.split()
            target_tokens = target_sent.split()

            # Store tokens
            source_tokens_list.append(source_tokens)
            target_tokens_list.append(target_tokens)

            # Create and store mappings
            source_mapping = create_token_mapping(source_tokens, marker_generator)
            target_mapping = create_token_mapping(target_tokens, marker_generator)

            source_batch.append(source_mapping.uniquified)
            target_batch.append(target_mapping.uniquified)
            gold_alignments.append(gold_alignment)
            source_mappings.append(source_mapping)
            target_mappings.append(target_mapping)
            source_mappings.append(source_mapping)
            target_mappings.append(target_mapping)
        except Exception as e:
            logger.error(f"Error parsing line: {line!r}\nError: {str(e)}")
            continue

    try:
        # First create progress bar for batch processing
        pbar = tqdm(
            total=len(source_batch), desc=f"Processing batched {lang_pair} results"
        )

        try:
            # Process in smaller sub-batches to show progress
            sub_batch_size = args.batch_size or len(source_batch)
            for i in range(0, len(source_batch), sub_batch_size):
                sub_source = source_batch[i : i + sub_batch_size]
                sub_target = target_batch[i : i + sub_batch_size]

                # Process sub-batch
                sub_results = align_tokens_batched(
                    llm_adapter,
                    sub_source,
                    sub_target,
                    source_language="English",
                    target_language=target_lang,
                    examples=training_examples,
                    max_retries=args.max_retries,
                    marker_generator=marker_generator,
                    batch_size=args.batch_size,
                )

                # Process results for this sub-batch
                for idx, result in enumerate(sub_results):
                    batch_idx = i + idx
                    try:
                        metrics, alignment_data = _process_alignment_result(
                            result,
                            gold_alignments[batch_idx],
                            source_tokens_list[batch_idx],
                            target_tokens_list[batch_idx],
                            source_mappings[batch_idx],
                            target_mappings[batch_idx],
                        )

                        if metrics and alignment_data:
                            # Update counters
                            total_true_positives += metrics["true_positives"]
                            total_predicted += metrics["predicted"]
                            total_gold += metrics["gold"]
                            alignments_data.append(alignment_data)
                        else:
                            failed_calls += 1

                        # Update diagnostic counters
                        total_attempts += len(result.attempts)
                        for attempt in result.attempts:
                            if not attempt.validation_passed:
                                total_validation_errors += 1
                                for (
                                    error_type,
                                    description,
                                    tokens,
                                ) in attempt.validation_errors:
                                    validation_error_stats[error_type]["count"] += 1
                                    # Update frequencies for each token
                                    for token in tokens:
                                        validation_error_stats[error_type][
                                            "frequencies"
                                        ][token] = (
                                            validation_error_stats[error_type][
                                                "frequencies"
                                            ].get(token, 0)
                                            + 1
                                        )
                            if attempt.exception:
                                exc_type = attempt.exception.split(":")[0].strip()
                                exception_counts[exc_type] = (
                                    exception_counts.get(exc_type, 0) + 1
                                )

                    except Exception as e:
                        failed_calls += 1
                        logger.error(f"Error processing result: {str(e)}")
                        continue
                    finally:
                        pbar.update(1)

        finally:
            pbar.close()

    except Exception as e:
        failed_calls += len(source_batch)
        logger.error(f"Batch processing error: {str(e)}")

    # Calculate metrics
    final_metrics: MetricsDict = {
        "precision": total_true_positives / total_predicted
        if total_predicted > 0
        else 0.0,
        "recall": total_true_positives / total_gold if total_gold > 0 else 0.0,
        "f_measure": (2 * total_true_positives / (total_predicted + total_gold))
        if (total_predicted + total_gold) > 0
        else 0.0,
        "aer": (1.0 - ((2 * total_true_positives) / (total_predicted + total_gold)))
        if (total_predicted + total_gold) > 0
        else 1.0,  # If no alignments, AER is 1.0 (worst case)
        "total_predicted": total_predicted,
        "total_gold": total_gold,
        "total_true_positives": total_true_positives,
        "diagnostics": {
            "total_attempts": total_attempts,
            "total_validation_errors": total_validation_errors,
            "avg_attempts_per_pair": total_attempts / (len(test_cases) - failed_calls)
            if len(test_cases) > failed_calls
            else 0,
            "validation_error_stats": {
                ValidationErrorType(error_type): {
                    "count": stats["count"],
                    "frequencies": dict(
                        sorted(
                            stats["frequencies"].items(),
                            key=lambda x: (
                                -x[1],
                                x[0],
                            ),  # Sort by frequency desc, then token asc
                        )
                    ),
                }
                for error_type, stats in validation_error_stats.items()
                if isinstance(stats, dict)
                and isinstance(stats.get("count"), int)
                and stats["count"] > 0
            },
            "exception_types": exception_counts,
            "failed_calls": failed_calls,
            "failure_rate": failed_calls / len(test_cases) if test_cases else 0,
        },
    }

    return final_metrics, alignments_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate and analyze word alignments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run alignment analysis")
    analyze_parser.add_argument(
        "--lang-pairs",
        nargs="+",
        help='Language pairs to evaluate (e.g., EN-SL EN-DE) or "all"',
    )
    analyze_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for example selection"
    )
    analyze_parser.add_argument(
        "--model-seed",
        type=int,
        help="Seed for LLM sampling (optional)",
    )
    analyze_parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for LLM sampling"
    )
    analyze_parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    analyze_parser.add_argument(
        "--adapter",
        choices=list(ADAPTER_TYPES.keys()),
        default="litellm",
        help="Adapter type to use (litellm, outlines, or llama-cpp)",
    )
    analyze_parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for llama.cpp (sets n_gpu_layers=-1)",
    )
    analyze_parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU for llama.cpp (-1 for all)",
    )
    analyze_parser.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        help="Context window size for llama.cpp (0 to use model's default)",
    )
    analyze_parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="Maximum number of prompt tokens to batch for llama.cpp",
    )
    analyze_parser.add_argument(
        "--n-threads",
        type=int,
        help="Number of threads to use for llama.cpp (default: use all)",
    )
    analyze_parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples for multinomial sampling (outlines only)",
    )
    analyze_parser.add_argument(
        "--model-device",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: auto-detect)",
    )
    analyze_parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        help="JSON string of kwargs for model initialization",
    )
    analyze_parser.add_argument(
        "--transformers-kwargs",
        type=json.loads,
        help="JSON string of kwargs for transformers.AutoModelForCausalLM.from_pretrained()",
    )
    analyze_parser.add_argument(
        "--tokenizer-model",
        help="HuggingFace model ID for tokenizer (llama-cpp only, defaults to model ID if not specified)",
    )
    analyze_parser.add_argument(
        "--model-dtype",
        choices=["float32", "float16", "bfloat16", "int8", "int4"],
        default="bfloat16",
        help="Data type for model weights (outlines only)",
    )
    analyze_parser.add_argument(
        "--beam-size",
        type=int,
        help="Number of beams for beam search (outlines only, overrides other sampling parameters)",
    )
    analyze_parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k filtering parameter (outlines only)",
    )
    analyze_parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p filtering parameter (outlines only)",
    )
    analyze_parser.add_argument(
        "--num-train-examples",
        type=int,
        help="Number of training examples to use for few-shot learning",
    )
    analyze_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity level (-v for INFO, -vv for DEBUG)",
    )
    analyze_parser.add_argument("--output", "-o", help="Path to save results JSON file")
    analyze_parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of test examples to evaluate per language pair",
    )
    analyze_parser.add_argument(
        "--marker-type",
        choices=["subscript", "underscore"],
        default="subscript",
        help="Type of marker to use for unique tokens",
    )
    analyze_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed alignments",
    )
    analyze_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing multiple examples at once (if supported by adapter)",
    )
    analyze_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate (overrides model-specific defaults)",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export alignments to Pharaoh format"
    )
    export_parser.add_argument("results_file", help="Input JSON results file")
    export_parser.add_argument(
        "output_dir", help="Output directory for alignment files"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare results across models"
    )
    evaluate_parser.add_argument(
        "results_files", nargs="+", help="Input JSON results files"
    )
    evaluate_parser.add_argument(
        "--output", "-o", help="Output markdown file (optional)"
    )
    evaluate_parser.add_argument(
        "--pdf", help="Output PDF file for alignment visualizations"
    )

    args = parser.parse_args()
    
    # Setup logging before anything else
    setup_logging(getattr(args, "verbose", 0))

    if args.command == "analyze":
        # Setup random seed for example selection only
        random.seed(args.seed)

        # Create LLM adapter based on type
        if args.adapter == "litellm":
            model_params = {
                "model": args.model,
                "temperature": args.temperature,
            }
            if args.model_seed is not None:
                model_params["seed"] = args.model_seed
            llm_adapter: Union[LiteLLMAdapter, OutlinesAdapter, LlamaCppAdapter] = (
                LiteLLMAdapter(model_params=model_params)
            )
        elif args.adapter == "llama-cpp":
            # If use_gpu is specified, override n_gpu_layers to use all layers
            n_gpu_layers = 99 if args.use_gpu else args.n_gpu_layers

            llm_adapter = LlamaCppAdapter(
                model_path=args.model,
                temperature=args.temperature,
                n_gpu_layers=n_gpu_layers,
                n_ctx=args.n_ctx,
                n_batch=args.n_batch,
                n_threads=args.n_threads,
                tokenizer_repo_id=args.tokenizer_model,
                **(args.model_kwargs or {}),
            )
        else:  # outlines
            max_tokens = args.max_tokens if args.max_tokens else 4096
            logger.info(f"Using max_tokens={max_tokens} for model {args.model}")

            llm_adapter = OutlinesAdapter(
                model_name=args.model,
                # Sampling parameters
                temperature=args.temperature,
                samples=args.samples,
                top_k=args.top_k,
                top_p=args.top_p,
                beam_size=args.beam_size,
                max_tokens=max_tokens,
                # Model configuration
                device=args.model_device,
                dtype=args.model_dtype,
                model_kwargs=args.model_kwargs,
                **(args.transformers_kwargs or {}),
            )

        lang_pairs = get_language_pairs(args.lang_pairs)

        run_params = get_run_parameters(args)

        results = {
            "parameters": run_params,
            "language_pairs": {},
            "training_examples": {},
        }

        # Download and cache dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            # Download and extract dataset
            download_xl_wa(repo_path)

            # Evaluate each language pair
            for lang_pair in lang_pairs:
                try:
                    # Load training examples first so they're in scope
                    training_examples = None
                    if args.num_train_examples is not None:
                        training_examples = load_training_examples(
                            repo_path, lang_pair, args.num_train_examples
                        )
                        logger.info(
                            f"Loaded {len(training_examples)} training examples"
                        )

                    metrics, alignments = evaluate_language_pair(
                        repo_path, lang_pair, llm_adapter, args
                    )
                    results["language_pairs"][lang_pair] = {
                        "metrics": metrics,
                        "alignments": alignments,
                    }
                    # Add training examples if they exist
                    if training_examples:
                        results["language_pairs"][lang_pair]["training_examples"] = [
                            {
                                "source_tokens": src,
                                "target_tokens": tgt,
                                "alignment": align.model_dump(),
                            }
                            for src, tgt, align in training_examples
                        ]
                    logger.info(f"Results for {lang_pair}:")
                    logger.info(f"Precision: {metrics['precision']:.4f}")
                    logger.info(f"Recall: {metrics['recall']:.4f}")
                    logger.info(f"F-measure: {metrics['f_measure']:.4f}")
                except Exception as e:
                    logger.error(f"Failed to evaluate {lang_pair}: {e}")

            # Print final results with enhanced diagnostics
            print("\nFinal Results:")
            print("-" * 50)

            # Aggregate diagnostics across all language pairs
            total_validation_errors = 0
            total_attempts = 0
            all_validation_error_types: dict[str, int] = {}
            all_exception_types: dict[str, int] = {}

            for lang_pair, data in results["language_pairs"].items():
                metrics = data["metrics"]
                diagnostics = metrics["diagnostics"]

                # Update aggregates
                total_validation_errors += diagnostics["total_validation_errors"]
                total_attempts += diagnostics["total_attempts"]

                # Merge error type counts
                for error_type_str, stats in diagnostics[
                    "validation_error_stats"
                ].items():
                    if isinstance(stats, dict):
                        count = stats.get("count", 0)
                        if isinstance(count, int):
                            all_validation_error_types[error_type_str] = (
                                all_validation_error_types.get(error_type_str, 0)
                                + count
                            )

                # Merge error type counts
                for error_type_str, stats in diagnostics[
                    "validation_error_stats"
                ].items():
                    if isinstance(stats, dict):
                        count = stats.get("count", 0)
                        if isinstance(count, int):
                            all_validation_error_types[error_type_str] = (
                                all_validation_error_types.get(error_type_str, 0)
                                + count
                            )

                # Print per-language pair results
                print(f"\n{lang_pair}:")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F-measure: {metrics['f_measure']:.4f}")
                print(f"AER: {metrics['aer']:.4f}")
                print(
                    f"Average attempts per alignment pair: {diagnostics['avg_attempts_per_pair']:.2f}"
                )
                print(f"Validation errors: {diagnostics['total_validation_errors']}")
                print(
                    f"Failed calls: {diagnostics['failed_calls']} ({diagnostics['failure_rate']:.1%})"
                )

            # Print aggregate diagnostics
            print("\nAggregate Diagnostics:")
            print("-" * 50)
            print(
                f"Total alignment pairs processed: {sum(len(data['alignments']) for data in results['language_pairs'].values())}"
            )
            print(f"Total attempts across all pairs: {total_attempts}")
            print(f"Total validation errors: {total_validation_errors}")
            print("\nValidation error stats:")
            for error_type, count in sorted(
                all_validation_error_types.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {error_type}: {count}")
            print("\nException types:")
            for exc_type, count in sorted(
                all_exception_types.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {exc_type}: {count}")

            # Save complete results if output path provided
            if args.output:
                output_path = ensure_extension(args.output, "json")
                with open(output_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Results saved to {output_path}")

    elif args.command == "export":
        export_results(args.results_file, Path(args.output_dir))

    elif args.command == "evaluate":
        table = evaluate_results(args.results_files, args.output)
        if args.output:
            md_path = ensure_extension(args.output, "md")
            with open(md_path, "w") as f:
                f.write(table)

            # Save plot
            plot_path = ensure_extension(md_path.replace(".md", ""), "png")
            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            print(table)

        # Create visualizations if PDF output is specified
        if args.pdf:
            create_alignment_visualizations(
                args.results_files, ensure_extension(args.pdf, "pdf")
            )


if __name__ == "__main__":
    main()
