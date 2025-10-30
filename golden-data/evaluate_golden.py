#!/usr/bin/env python3
"""Evaluate alignment models on golden-data Japanese↔Japanese dataset."""

import argparse
import json
import logging
import sys
from pathlib import Path
from types import FrameType
from typing import Optional

from lexi_align_prompt import EXAMPLES_JP_JP, GUIDELINES
from loguru import logger

from lexi_align import align_and_evaluate_dataset, create_adapter
from lexi_align.models import TextAlignment, TokenAlignment
from lexi_align.text_processing import create_subscript_generator
from lexi_align.utils import create_token_mapping


def load_golden_data(filepath: Path) -> list[dict]:
    """Load JSONL golden data file."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def convert_index_pairs_to_alignment(
    src_tokens: list[str],
    tgt_tokens: list[str],
    index_pairs: list[list[int]],
    marker_generator=None,
) -> TextAlignment:
    """Convert index-based pairs to TextAlignment with unique tokens."""
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    # Create mappings to get uniquified tokens
    src_mapping = create_token_mapping(src_tokens, marker_generator)
    tgt_mapping = create_token_mapping(tgt_tokens, marker_generator)

    # Build alignments using uniquified tokens
    alignments = []
    for src_idx, tgt_idx in index_pairs:
        src_token = src_mapping.uniquified[src_idx]
        tgt_token = tgt_mapping.uniquified[tgt_idx]
        alignments.append(TokenAlignment(source=src_token, target=tgt_token))

    return TextAlignment(
        alignment=alignments,
        source_mapping=src_mapping,
        target_mapping=tgt_mapping,
    )


def _print_summary(
    dataset_name: str,
    adapter_spec: str,
    total_examples: int,
    successful: int,
    failed: int,
    metrics: dict,
) -> None:
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Adapter: {adapter_spec}")
    print("=" * 60)
    print(f"Examples: {total_examples} (successful: {successful}, failed: {failed})")
    print("\nMicro-averaged Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F-measure: {metrics['f_measure']:.4f}")
    print(f"  AER:       {metrics['aer']:.4f}")
    print("\nAlignment counts:")
    print(f"  Predicted: {metrics['total_predicted']}")
    print(f"  Gold:      {metrics['total_gold']}")
    print(f"  Correct:   {metrics['total_true_positives']}")
    print("=" * 60)


def create_visualizations_from_results(
    results_file: Path,
    output_pdf: Path,
    max_examples: Optional[int] = None,
) -> None:
    """Create PDF with visualizations from a result JSON file.

    Args:
        results_file: Path to result JSON file
        output_pdf: Path to output PDF file
        max_examples: Optional maximum number of examples to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from lexi_align.visualize import visualize_alignments

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    adapter_spec = results["adapter_spec"]
    alignments_data = results["alignments"]

    # Limit examples if requested
    if max_examples and len(alignments_data) > max_examples:
        alignments_data = alignments_data[:max_examples]

    logger.info(f"Creating visualization PDF with {len(alignments_data)} examples...")

    with PdfPages(output_pdf) as pdf:
        for i, alignment_data in enumerate(alignments_data, 1):
            try:
                example_id = alignment_data["id"]
                src_tokens = alignment_data["src_tokens"]
                tgt_tokens = alignment_data["tgt_tokens"]

                # Load alignments
                predicted = TextAlignment.model_validate(alignment_data["predicted"])
                gold = TextAlignment.model_validate(alignment_data["gold"])

                # Create title
                title = (
                    f"Example {example_id} ({len(src_tokens)}-{len(tgt_tokens)} tokens)"
                )

                # Create visualization dictionary
                alignments = {
                    "Gold": gold,
                    adapter_spec: predicted,
                }

                # Visualize
                visualize_alignments(
                    source_tokens=src_tokens,
                    target_tokens=tgt_tokens,
                    alignments=alignments,
                    title=title,
                    reference_model="Gold",
                )

                # Save to PDF
                pdf.savefig(bbox_inches="tight")
                plt.close()

            except Exception as e:
                logger.error(f"Failed to visualize example {i}: {e}")
                plt.close()

    logger.info(f"Visualization PDF saved to {output_pdf}")


def evaluate_golden_data(
    data_file: Path,
    adapter_spec: str,
    output_file: Optional[Path] = None,
    sample_size: Optional[int] = None,
    max_retries: int = 3,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    visualize: bool = False,
    visualize_output: Optional[Path] = None,
    visualize_max_examples: Optional[int] = None,
    use_guidelines: bool = True,
    **adapter_kwargs,
) -> dict:
    """Evaluate model on golden data using simplified API."""

    # Load data
    examples = load_golden_data(data_file)
    if sample_size:
        examples = examples[:sample_size]

    logger.info(f"Loaded {len(examples)} examples from {data_file.name}")

    # Create adapter
    adapter = create_adapter(adapter_spec, **adapter_kwargs)
    marker_generator = create_subscript_generator()

    # Prepare sequences and gold alignments
    source_sequences = [ex["src_tokens"] for ex in examples]
    target_sequences = [ex["tgt_tokens"] for ex in examples]
    gold_alignments = [
        convert_index_pairs_to_alignment(
            ex["src_tokens"],
            ex["tgt_tokens"],
            ex.get("sure_idx") if ex.get("sure_idx") else ex["index_pairs"],
            marker_generator,
        )
        for ex in examples
    ]

    # Verify all gold alignments were created successfully
    assert len(gold_alignments) == len(examples), (
        f"Expected {len(examples)} gold alignments but got {len(gold_alignments)} in {gold_alignments}"
    )

    for i, gold_align in enumerate(gold_alignments):
        example_id = examples[i].get("id", f"index_{i}")

        assert gold_align is not None, (
            f"Gold alignment for example {i} (id={example_id}) is None"
        )

        assert isinstance(gold_align, TextAlignment), (
            f"Gold alignment for example {i} (id={example_id}) is not a TextAlignment, "
            f"got {type(gold_align).__name__}"
        )

        assert len(gold_align.alignment) > 0, (
            f"Gold alignment for example {i} (id={example_id}) has no alignment pairs. "
            f"Source tokens: {len(examples[i]['src_tokens'])}, "
            f"Target tokens: {len(examples[i]['tgt_tokens'])}, "
            f"Index pairs: {len(examples[i]['index_pairs'])}, "
            f"Gold: {gold_align}"
        )

    logger.info(f"Validated {len(gold_alignments)} gold alignments successfully")

    # Run alignment and evaluation
    logger.info("Starting alignment evaluation...")

    results, metrics = align_and_evaluate_dataset(
        adapter,
        source_sequences,
        target_sequences,
        gold_alignments,
        source_language="Japanese",
        target_language="Japanese",
        guidelines=GUIDELINES if use_guidelines else None,
        examples=EXAMPLES_JP_JP if use_guidelines else None,
        max_retries=max_retries,
        marker_generator=marker_generator,
        show_progress=True,
        batch_size=batch_size,
        concurrency=concurrency,
    )

    # Count successful/failed
    successful_results = [r for r in results if r.alignment]
    failed_count = len(results) - len(successful_results)

    if not successful_results:
        logger.error("No successful alignments!")
        return {
            "data_file": str(data_file),
            "adapter_spec": adapter_spec,
            "num_examples": len(examples),
            "num_successful": 0,
            "num_failed": failed_count,
            "error": "No successful alignments",
        }

    # Build detailed results
    alignments_data = []
    for i, result in enumerate(results):
        if result.alignment:
            example = examples[i]
            alignment_dict = {
                "id": example["id"],
                "src_tokens": example["src_tokens"],
                "tgt_tokens": example["tgt_tokens"],
                "predicted": result.alignment.model_dump(),
                "gold": gold_alignments[i].model_dump(),
                "metrics": metrics["per_example"][len(alignments_data)],
                "attempts": len(result.attempts),
            }

            # Add reasoning if present for easier access
            if result.alignment.reasoning:
                alignment_dict["reasoning"] = result.alignment.reasoning

            alignments_data.append(alignment_dict)

    # Compile results
    results_dict = {
        "data_file": str(data_file),
        "adapter_spec": adapter_spec,
        "parameters": {
            "adapter_spec": adapter_spec,
            "use_guidelines": use_guidelines,
            "max_retries": max_retries,
            "batch_size": batch_size,
            "concurrency": concurrency,
            **adapter_kwargs,  # Include adapter kwargs for reference
        },
        "adapter_kwargs": adapter_kwargs,
        "num_examples": len(examples),
        "num_successful": len(successful_results),
        "num_failed": failed_count,
        "metrics": {
            "micro": metrics["micro"],
            "macro": metrics["macro"],
        },
        "alignments": alignments_data,
    }

    # Save results
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")

    # Create visualizations if requested
    if visualize and output_file:
        if visualize_output is None:
            # Generate default PDF filename from JSON filename
            visualize_output = output_file.with_suffix(".pdf")

        logger.info("Creating visualizations...")
        try:
            create_visualizations_from_results(
                output_file, visualize_output, max_examples=visualize_max_examples
            )
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")

    # Print summary
    _print_summary(
        data_file.name,
        adapter_spec,
        len(examples),
        len(successful_results),
        failed_count,
        metrics["micro"],
    )

    return results_dict


def get_run_parameters(args: argparse.Namespace) -> dict:
    """Collect all run parameters into a dictionary."""
    return {
        "adapter": args.adapter,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "sample_size": getattr(args, "sample_size", None),
        "use_guidelines": getattr(args, "use_guidelines", True),
        "use_dynamic_schema": getattr(args, "use_dynamic_schema", True),
        "use_reasoning": getattr(args, "use_reasoning", False),
        "batch_size": getattr(args, "batch_size", None),
        "concurrency": getattr(args, "concurrency", None),
        "num_train_examples": getattr(args, "num_train_examples", None),
        "visualize": getattr(args, "visualize", False),
        "visualize_max_examples": getattr(args, "visualize_max_examples", None),
        # Include sampling parameters if present
        "top_k": getattr(args, "top_k", None),
        "top_p": getattr(args, "top_p", None),
        "beam_size": getattr(args, "beam_size", None),
        "presence_penalty": getattr(args, "presence_penalty", None),
        "min_p": getattr(args, "min_p", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "sglang_url": getattr(args, "sglang_url", None),
        "sglang_api_key": "<redacted>"
        if getattr(args, "sglang_api_key", None)
        else None,
    }


def setup_logging(verbosity: int = 0):
    """Setup logging to use loguru for everything."""
    # Remove default loguru handler
    logger.remove()

    # Define log levels
    log_levels = {
        0: "WARNING",  # default
        1: "INFO",  # -v
        2: "DEBUG",  # -vv
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
    logger.add(sys.stderr, format=log_format, level=level, colorize=True)

    # Create handler that routes standard logging to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level_name = logger.level(record.levelname).name
            except ValueError:
                level_name = record.levelno

            # Find caller from where originated the logged message
            frame: Optional[FrameType] = sys._getframe(6)
            depth = 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level_name, record.getMessage()
            )

    # Remove any existing handlers from root logger
    logging.root.handlers = []

    # Add intercept handler to root logger
    logging.root.addHandler(InterceptHandler())

    # Set level for root logger
    logging.root.setLevel(level)

    # Reduce verbosity of noisy third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate alignment models on golden-data dataset"
    )
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        help="Path to JSONL data file (e.g., golden-data/1928_1913-lexi_review_with_sure.jsonl)",
    )
    parser.add_argument(
        "--data-files",
        type=Path,
        nargs="+",
        help="Process multiple JSONL data files at once",
    )
    parser.add_argument(
        "--adapter",
        default="litellm:gpt-4o-mini",
        help="Adapter specification (e.g., 'litellm:gpt-4o', 'transformers:Qwen/Qwen3-0.6B')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file for results (single file mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (multi-file mode, auto-generates filenames)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per alignment (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size if adapter supports batching",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent async requests for API-based adapters (default: 8)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate PDF visualization of alignments after evaluation",
    )
    parser.add_argument(
        "--visualize-output",
        type=Path,
        help="Path to visualization PDF (default: <output>.pdf)",
    )
    parser.add_argument(
        "--visualize-max-examples",
        type=int,
        help="Maximum number of examples to visualize (default: all)",
    )
    parser.add_argument(
        "--sglang-url",
        help="Base URL for SGLang server (e.g., http://localhost:30000/v1)",
    )
    parser.add_argument(
        "--sglang-api-key",
        help="API key for SGLang server (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="Beam size for beam search",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Presence penalty for sampling",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        help="Minimum probability threshold",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--use-dynamic-schema",
        action="store_true",
        default=True,
        help="Use dynamic schema with token constraints (default: True)",
    )
    parser.add_argument(
        "--no-dynamic-schema",
        dest="use_dynamic_schema",
        action="store_false",
        help="Disable dynamic schema",
    )
    parser.add_argument(
        "--use-reasoning",
        action="store_true",
        help="Request step-by-step reasoning from model before alignment (increases output size)",
    )
    parser.add_argument(
        "--no-guidelines",
        dest="use_guidelines",
        action="store_false",
        default=True,
        help="Disable use of Japanese-specific guidelines and examples (enabled by default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.data_file and args.data_files:
        parser.error("Cannot specify both data_file and --data-files")

    if not args.data_file and not args.data_files:
        parser.error("Must specify either data_file or --data-files")

    # Multi-file mode
    if args.data_files:
        if not args.output_dir:
            parser.error("--output-dir is required when using --data-files")

        # Setup logging
        setup_logging(args.verbose)

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(args.data_files)} data files")
        logger.info(f"Output directory: {args.output_dir}")

        # Build adapter kwargs once (shared across all files)
        adapter_kwargs = {
            "temperature": args.temperature,
            "use_dynamic_schema": args.use_dynamic_schema,
            "use_reasoning": args.use_reasoning,
        }

        # Add optional parameters
        if args.batch_size:
            adapter_kwargs["batch_size"] = args.batch_size
        if hasattr(args, "sglang_url") and args.sglang_url:
            adapter_kwargs["base_url"] = args.sglang_url
        if hasattr(args, "sglang_api_key") and args.sglang_api_key:
            adapter_kwargs["api_key"] = args.sglang_api_key
        if hasattr(args, "top_k") and args.top_k:
            adapter_kwargs["top_k"] = args.top_k
        if hasattr(args, "top_p") and args.top_p is not None:
            adapter_kwargs["top_p"] = args.top_p
        if hasattr(args, "beam_size") and args.beam_size:
            adapter_kwargs["beam_size"] = args.beam_size
        if hasattr(args, "presence_penalty") and args.presence_penalty is not None:
            adapter_kwargs["presence_penalty"] = args.presence_penalty
        if hasattr(args, "min_p") and args.min_p is not None:
            adapter_kwargs["min_p"] = args.min_p
        if hasattr(args, "max_tokens") and args.max_tokens:
            adapter_kwargs["max_tokens"] = args.max_tokens

        # Override with environment variables if set
        import os

        if "SGLANG_URL" in os.environ and not adapter_kwargs.get("base_url"):
            adapter_kwargs["base_url"] = os.environ["SGLANG_URL"]

        # Process each file
        all_results = []
        all_predictions = []
        all_gold_alignments = []

        for data_file in args.data_files:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing: {data_file.name}")
            logger.info(f"{'=' * 60}")

            # Generate output filename
            output_file = args.output_dir / f"results_{data_file.stem}.json"

            # Determine visualize output path if requested
            visualize_output = None
            if args.visualize:
                visualize_output = args.output_dir / f"results_{data_file.stem}.pdf"

            try:
                result = evaluate_golden_data(
                    data_file,
                    args.adapter,
                    output_file,
                    args.sample_size,
                    args.max_retries,
                    concurrency=args.concurrency,
                    visualize=args.visualize,
                    visualize_output=visualize_output,
                    visualize_max_examples=args.visualize_max_examples,
                    use_guidelines=args.use_guidelines,
                    **adapter_kwargs,
                )
                all_results.append(
                    {
                        "file": data_file.name,
                        "output": str(output_file),
                        "success": True,
                        "num_examples": result.get("num_examples", 0),
                        "num_successful": result.get("num_successful", 0),
                        "metrics": result.get("metrics", {}),
                    }
                )

                # Collect alignments for overall metrics calculation
                for alignment_data in result.get("alignments", []):
                    predicted = TextAlignment.model_validate(
                        alignment_data["predicted"]
                    )
                    gold = TextAlignment.model_validate(alignment_data["gold"])
                    all_predictions.append(predicted)
                    all_gold_alignments.append(gold)

            except Exception as e:
                logger.error(f"Failed to process {data_file}: {e}")
                all_results.append(
                    {
                        "file": data_file.name,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Calculate aggregate metrics across all files
        aggregate_metrics = None
        if all_predictions and all_gold_alignments:
            from lexi_align.core import build_micro_metrics

            aggregate_metrics = build_micro_metrics(
                list(zip(all_predictions, all_gold_alignments))
            )

        # Print summary
        print("\n" + "=" * 60)
        print("MULTI-FILE EVALUATION SUMMARY")
        print("=" * 60)
        successful = sum(1 for r in all_results if r["success"])
        failed = len(all_results) - successful
        print(f"Total files: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if aggregate_metrics:
            print("\nAggregate Metrics (across all files):")
            print(f"  Precision: {aggregate_metrics['precision']:.4f}")
            print(f"  Recall:    {aggregate_metrics['recall']:.4f}")
            print(f"  F-measure: {aggregate_metrics['f_measure']:.4f}")
            print(f"  AER:       {aggregate_metrics['aer']:.4f}")
            print("  Total alignments:")
            print(f"    Predicted: {aggregate_metrics['total_predicted']}")
            print(f"    Gold:      {aggregate_metrics['total_gold']}")
            print(f"    Correct:   {aggregate_metrics['total_true_positives']}")

        print("=" * 60)

        for result in all_results:
            if result["success"]:
                metrics = result.get("metrics", {}).get("micro", {})
                f_measure = metrics.get("f_measure", 0)
                print(
                    f"✓ {result['file']}: {result['num_successful']}/{result['num_examples']} alignments (F1: {f_measure:.4f})"
                )
            else:
                print(f"✗ {result['file']}: {result.get('error', 'Unknown error')}")

        print("\nResults saved to:", args.output_dir)

        # Save aggregate summary to JSON
        if aggregate_metrics:
            summary_file = args.output_dir / "aggregate_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "parameters": get_run_parameters(args),
                        "aggregate_metrics": aggregate_metrics,
                        "per_file_results": all_results,
                        "adapter": args.adapter,
                        "adapter_kwargs": adapter_kwargs,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Aggregate summary saved to: {summary_file}")

        return

    # Single file mode (original behavior)
    # Setup logging
    setup_logging(args.verbose)

    # Build adapter kwargs
    adapter_kwargs = {
        "temperature": args.temperature,
        "use_dynamic_schema": args.use_dynamic_schema,
        "use_reasoning": args.use_reasoning,
    }

    # Add optional parameters
    if args.batch_size:
        adapter_kwargs["batch_size"] = args.batch_size
    if hasattr(args, "sglang_url") and args.sglang_url:
        adapter_kwargs["base_url"] = args.sglang_url
    if hasattr(args, "sglang_api_key") and args.sglang_api_key:
        adapter_kwargs["api_key"] = args.sglang_api_key
    if hasattr(args, "top_k") and args.top_k:
        adapter_kwargs["top_k"] = args.top_k
    if hasattr(args, "top_p") and args.top_p is not None:
        adapter_kwargs["top_p"] = args.top_p
    if hasattr(args, "beam_size") and args.beam_size:
        adapter_kwargs["beam_size"] = args.beam_size
    if hasattr(args, "presence_penalty") and args.presence_penalty is not None:
        adapter_kwargs["presence_penalty"] = args.presence_penalty
    if hasattr(args, "min_p") and args.min_p is not None:
        adapter_kwargs["min_p"] = args.min_p
    if hasattr(args, "max_tokens") and args.max_tokens:
        adapter_kwargs["max_tokens"] = args.max_tokens

    # Override with environment variables if set
    import os

    if "SGLANG_URL" in os.environ and not adapter_kwargs.get("base_url"):
        adapter_kwargs["base_url"] = os.environ["SGLANG_URL"]

    # Run evaluation
    evaluate_golden_data(
        args.data_file,
        args.adapter,
        args.output,
        args.sample_size,
        args.max_retries,
        concurrency=args.concurrency,
        visualize=args.visualize,
        visualize_output=args.visualize_output,
        visualize_max_examples=args.visualize_max_examples,
        use_guidelines=args.use_guidelines,
        **adapter_kwargs,
    )


if __name__ == "__main__":
    main()
