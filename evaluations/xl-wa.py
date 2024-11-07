#!/usr/bin/env python3
import json
import argparse
import random
from pathlib import Path
import tempfile
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
from lexi_align.models import TextAlignment
from lexi_align.utils import (
    export_pharaoh_format,
    make_unique,
    read_pharaoh_file,
)
from tqdm import tqdm  # type: ignore
import requests  # type: ignore
from zipfile import ZipFile
from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
from lexi_align.adapters.outlines_adapter import OutlinesAdapter
from lexi_align.adapters.llama_cpp_adapter import LlamaCppAdapter
from lexi_align.core import align_tokens
from lexi_align.utils import parse_pharaoh_format
from lexi_align.metrics import calculate_metrics
import logging

logger = logging.getLogger(__name__)

ADAPTER_TYPES = {
    "litellm": LiteLLMAdapter,
    "outlines": OutlinesAdapter,
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
        alignments = []

        for item in data["alignments"]:
            source_tokens = item["source_tokens"]  # Now using the stored tokens
            target_tokens = item["target_tokens"]
            alignment = TextAlignment.model_validate(item["predicted"])
            alignments.append((source_tokens, target_tokens, alignment))

        logger.info(f"Writing {len(alignments)} alignments to {output_file}")

        with open(output_file, "w") as f:
            for source_tokens, target_tokens, alignment in alignments:
                f.write(
                    export_pharaoh_format(source_tokens, target_tokens, alignment)
                    + "\n"
                )


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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


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
                            "F1": metrics["f1"],
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
    metrics = ["Precision", "Recall", "F1"]
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
            color = plt.cm.get_cmap("Set3")(i / num_models)  # type: ignore
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
        header = ["Language Pair", "Precision", "Recall", "F1"]

        # Build table rows
        rows = []
        for lang_pair in sorted(results["language_pairs"].keys()):
            metrics = results["language_pairs"][lang_pair]["metrics"]
            rows.append(
                [
                    lang_pair,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1']:.3f}",
                ]
            )

        # Add overall averages
        metrics = calculate_overall_metrics(results)
        rows.append(
            [
                "**Average**",
                f"**{metrics['precision']:.3f}**",
                f"**{metrics['recall']:.3f}**",
                f"**{metrics['f1']:.3f}**",
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
    }


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
) -> tuple[dict, list[dict], list[str]]:
    """Evaluate alignment performance for a single language pair using micro-averaging."""
    target_lang_code = lang_pair.split("-")[1]
    target_lang = LANGUAGE_MAP.get(target_lang_code, target_lang_code)
    target_lang_lower = target_lang_code.lower()

    test_file = repo_path / "data" / target_lang_lower / "test.tsv"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    logger.info(f"Evaluating {lang_pair}...")

    training_examples = None
    training_pharaoh = []
    if args.num_train_examples is not None:
        training_examples = load_training_examples(
            repo_path, lang_pair, args.num_train_examples
        )
        logger.info(f"Loaded {len(training_examples)} training examples")
        # Convert training examples to Pharaoh format for logging
        for source_tokens, target_tokens, alignment in training_examples:
            pharaoh_str = export_pharaoh_format(source_tokens, target_tokens, alignment)
            training_pharaoh.append(pharaoh_str)
        logger.debug(f"Training examples: {training_pharaoh}")

    # Micro-averaged metrics
    total_true_positives = 0
    total_predicted = 0
    total_gold = 0
    alignments_data = []

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = f.readlines()

    if args.sample_size and len(test_cases) > args.sample_size:
        test_cases = test_cases[: args.sample_size]

    for i, line in enumerate(tqdm(test_cases, desc=f"Processing {lang_pair}"), 1):
        try:
            source_sent, target_sent, gold_alignment = parse_pharaoh_format(line)
            source_tokens = source_sent.split()
            target_tokens = target_sent.split()

            # Create unique versions for prediction
            unique_source = make_unique(source_tokens)
            unique_target = make_unique(target_tokens)

            logger.debug(
                f"Processing example {i}:\n  Source: {source_sent}\n  Target: {target_sent}\n"
                f"  Unique source: {unique_source}\n  Unique target: {unique_target}"
            )

            try:
                predicted_alignment = align_tokens(
                    llm_adapter,
                    unique_source,
                    unique_target,
                    source_language="English",
                    target_language=target_lang,
                    examples=training_examples,
                )
            except Exception as e:
                logger.error(
                    f"API call failed for example {i}:\n"
                    f"  Source: {source_sent}\n"
                    f"  Target: {target_sent}\n"
                    f"  Error: {str(e)}"
                )
                continue

            metrics = calculate_metrics(predicted_alignment, gold_alignment)

            logging.debug(f"Metrics: {metrics}")

        except Exception as e:
            logger.error(
                f"Unexpected error processing example {i}:\n"
                f"  Raw line: {line!r}\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error: {str(e)}"
            )
            continue

        # Update micro-average counters
        total_true_positives += metrics["precision"] * len(
            predicted_alignment.alignment
        )
        total_predicted += len(predicted_alignment.alignment)
        total_gold += len(gold_alignment.alignment)

        # Store alignment data with unique tokens everywhere
        alignment_data = {
            "source_tokens": unique_source,
            "target_tokens": unique_target,
            "predicted": predicted_alignment.model_dump(),
            "gold": gold_alignment.model_dump(),
            "metrics": metrics,
        }
        alignments_data.append(alignment_data)

    # Calculate micro-averaged metrics
    micro_precision = (
        total_true_positives / total_predicted if total_predicted > 0 else 0.0
    )
    micro_recall = total_true_positives / total_gold if total_gold > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    metrics = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        # Add total counts for reference
        "total_predicted": total_predicted,
        "total_gold": total_gold,
        "total_true_positives": total_true_positives,
    }

    return metrics, alignments_data, training_pharaoh


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
        help="Adapter type to use (litellm or outlines)",
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

    args = parser.parse_args()

    log_levels = {
        0: logging.WARNING,  # default
        1: logging.INFO,  # -v
        2: logging.DEBUG,  # -vv
    }
    verbosity = min(getattr(args, "verbose", 0), 2)
    logging.basicConfig(
        level=log_levels[verbosity],
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

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
            llm_adapter = LlamaCppAdapter(
                model_path=args.model,
                temperature=args.temperature,
                n_gpu_layers=args.n_gpu_layers,
                n_ctx=args.n_ctx,
                n_batch=args.n_batch,
                n_threads=args.n_threads,
                **(args.model_kwargs or {}),
            )
        else:  # outlines
            llm_adapter = OutlinesAdapter(
                model_name=args.model,
                # Sampling parameters
                temperature=args.temperature,
                samples=args.samples,
                top_k=args.top_k,
                top_p=args.top_p,
                beam_size=args.beam_size,
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
                    metrics, alignments, training_pharaoh = evaluate_language_pair(
                        repo_path, lang_pair, llm_adapter, args
                    )
                    results["language_pairs"][lang_pair] = {
                        "metrics": metrics,
                        "alignments": alignments,
                    }
                    # Add training examples if they exist
                    if training_pharaoh:
                        results["training_examples"][lang_pair] = training_pharaoh
                    logger.info(f"Results for {lang_pair}:")
                    logger.info(f"Precision: {metrics['precision']:.4f}")
                    logger.info(f"Recall: {metrics['recall']:.4f}")
                    logger.info(f"F1: {metrics['f1']:.4f}")
                except Exception as e:
                    logger.error(f"Failed to evaluate {lang_pair}: {e}")

            # Print final results (metrics only)
            print("\nFinal Results:")
            print("-" * 50)
            for lang_pair, data in results["language_pairs"].items():
                metrics = data["metrics"]
                print(f"\n{lang_pair}:")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1: {metrics['f1']:.4f}")

            # Save complete results if output path provided
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Results saved to {args.output}")

    elif args.command == "export":
        export_results(args.results_file, Path(args.output_dir))

    elif args.command == "evaluate":
        table = evaluate_results(args.results_files, args.output)
        if args.output:
            with open(args.output, "w") as f:
                f.write(table)
        else:
            print(table)


if __name__ == "__main__":
    main()
