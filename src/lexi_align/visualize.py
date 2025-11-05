from logging import getLogger

__all__ = ["visualize_alignments", "visualize_alignments_altair", "visualize_alignment"]
from typing import Dict, Optional

import altair as alt
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns  # type: ignore
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pyfonts import load_google_font, set_default_font  # type: ignore[import-untyped]

from lexi_align.models import TextAlignment
from lexi_align.utils import create_token_mapping, make_unique

logger = getLogger(__name__)

font = load_google_font("IBM Plex Sans JP")
set_default_font(font)  # Sets font for all text


def visualize_alignments(
    source_tokens: list[str],
    target_tokens: list[str],
    alignments: Dict[str, TextAlignment],
    title: str,
    output_path: Optional[str] = None,
    reference_model: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    """
    Visualize one or more token alignments using matplotlib.

    Args:
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens
        alignments: Dictionary mapping model names to their TextAlignment results
        title: Title for the visualization
        output_path: Optional path to save the visualization (PNG/PDF/etc)
        reference_model: Optional model name to use as reference for highlighting differences
        figsize: Optional figure size as (width, height) tuple. If None, size is calculated dynamically.

    Example:
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> source = "The cat sat".split()
        >>> target = "Le chat assis".split()
        >>> alignments = {
        ...     "model1": TextAlignment(alignment=[
        ...         TokenAlignment(source="The", target="Le"),
        ...         TokenAlignment(source="cat", target="chat")
        ...     ]),
        ...     "model2": TextAlignment(alignment=[
        ...         TokenAlignment(source="The", target="Le"),
        ...         TokenAlignment(source="cat", target="chat"),
        ...         TokenAlignment(source="sat", target="assis")
        ...     ])
        ... }
        >>> visualize_alignments(source, target, alignments, "Test Alignment")  # doctest: +SKIP
    """

    # Create token mappings once for reuse
    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)

    # Use uniquified tokens from mappings
    source_tokens = source_mapping.uniquified
    target_tokens = target_mapping.uniquified

    # Keep only models with at least one declared TokenAlignment
    alignments = {m: a for m, a in alignments.items() if a.alignment}

    # Further filter out any model that yields ZERO actual plot‐cells
    alignments = {
        m: a
        for m, a in alignments.items()
        if a.get_alignment_positions(source_mapping, target_mapping)
    }
    if not alignments:
        logger.info("Skipping visualization – no valid alignments to display")
        return

    # Calculate dynamic figure size if not provided
    if figsize is None:
        # Base width and height (minimum sizes)
        base_width = 12
        base_height = 8

        # Scale factors
        width_per_token = 0.5
        height_per_token = 0.3

        # Calculate dimensions based on token counts
        width = max(
            base_width, base_width + (len(target_tokens) - 20) * width_per_token
        )
        height = max(
            base_height, base_height + (len(source_tokens) - 20) * height_per_token
        )

        # Add extra width for legend
        legend_width = 5

        figsize = (width + legend_width, height)

    # Create the plot with calculated or provided figsize
    fig, ax = plt.subplots(figsize=figsize)

    # Get color palette for models
    # Filter out reference model from regular visualization
    model_names = sorted(name for name in alignments.keys() if name != reference_model)
    colors = sns.color_palette("Pastel1", n_colors=len(model_names))

    # Create mapping of positions for each alignment using TextAlignment methods
    cell_models: Dict[tuple[int, int], set[str]] = {
        (i, j): set()
        for i in range(len(source_tokens))
        for j in range(len(target_tokens))
    }

    # Collect which models align each cell (excluding reference model)
    for model, alignment in alignments.items():
        if model != reference_model:  # Skip reference model
            # Get position-based alignments
            for s_idx, t_idx in alignment.get_alignment_positions(
                source_mapping, target_mapping
            ):
                cell_models[(s_idx, t_idx)].add(model)

    # Precompute reference (gold) positions if provided
    ref_positions: set[tuple[int, int]] = set()
    if reference_model and reference_model in alignments:
        ref_positions = set(
            alignments[reference_model].get_alignment_positions(
                source_mapping, target_mapping
            )
        )

    # Draw the alignments
    for i, _source_token in enumerate(source_tokens):
        for j, _target_token in enumerate(target_tokens):
            models_for_cell = cell_models[(i, j)]
            if models_for_cell:
                # Draw reference model highlighting if specified
                if reference_model and ref_positions:
                    is_in_reference = (i, j) in ref_positions
                    color = "black" if is_in_reference else "red"
                    ax.add_patch(
                        Rectangle(
                            (j + 0.1, i + 0.1),  # Increased margin
                            0.8,  # Reduced width
                            0.8,  # Reduced height
                            fill=False,
                            color=color,
                            alpha=1.0,
                            linewidth=1.5,  # Make lines more visible
                        )
                    )

                # Draw pie chart for model agreement
                total_models = len(models_for_cell)
                if total_models > 1:
                    # Create donut chart for multiple models
                    _wedges = ax.pie(
                        [1] * total_models,
                        colors=[colors[model_names.index(m)] for m in models_for_cell],
                        radius=0.35,
                        center=(j + 0.5, i + 0.5),
                        wedgeprops=dict(width=0.15),
                        startangle=90,
                    )[0]  # Just take the patches (wedges)
                    # Add count in center
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        str(total_models),
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                        weight="bold",
                    )
                else:
                    # Solid circle for single model
                    ax.pie(
                        [1],
                        colors=[colors[model_names.index(next(iter(models_for_cell)))]],
                        radius=0.35,
                        center=(j + 0.5, i + 0.5),
                        wedgeprops=dict(width=0.35),
                        startangle=90,
                    )

    # Draw missing gold-only cells (in gold but not predicted by any model)
    if reference_model and ref_positions:
        for i, j in ref_positions:
            if not cell_models[(i, j)]:
                ax.add_patch(
                    Rectangle(
                        (j + 0.1, i + 0.1),
                        0.8,
                        0.8,
                        fill=False,
                        color="blue",
                        linestyle="--",
                        alpha=1.0,
                        linewidth=1.5,
                    )
                )

    # Configure axes
    ax.set_xlim(-0.5, len(target_tokens) + 0.5)
    ax.set_ylim(len(source_tokens) + 0.5, -0.5)
    ax.set_xticks([i + 0.5 for i in range(len(target_tokens))])
    ax.set_yticks([i + 0.5 for i in range(len(source_tokens))])

    # Set labels with unique markers if needed
    ax.set_xticklabels(
        make_unique(target_tokens), rotation=45, weight="bold", ha="right"
    )
    ax.set_yticklabels(make_unique(source_tokens), weight="bold")

    # Configure grid and spines
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray")

    # Add legend with model colors and reference indicators
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            label=model,
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markersize=10,
        )
        for i, model in enumerate(model_names)
    ]

    # Add reference model indicators to legend if applicable
    if reference_model:
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    label="Correct alignment",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=10,
                    markeredgewidth=1.5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    label="Misalignment",
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markersize=10,
                    markeredgewidth=1.5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    label="Missing (gold only)",
                    markerfacecolor="none",
                    markeredgecolor="blue",
                    markersize=10,
                    markeredgewidth=1.5,
                ),
            ]
        )

    ax.set_title(title, fontsize=14, weight="bold", wrap=True, loc="left")

    # Create metrics text if reference model exists
    if reference_model:
        metrics_text = "Metrics vs Reference:\n"
        for model in model_names:
            metrics = alignments[model].compare_alignments(
                alignments[reference_model], source_mapping, target_mapping
            )
            metrics_text += f"{model}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}\n"

        # Add metrics text above legend, right-aligned and close to the left side
        plt.figtext(
            0.82,
            0.7,
            metrics_text,
            fontsize="small",
            ha="right",
            va="top",
        )

    # Add legend below metrics
    ax.legend(
        handles=legend_handles,
        title="Models",
        loc="center left",
        bbox_to_anchor=(1.02, 0.4),
        ncol=1,
        fontsize="small",
        title_fontsize="small",
    )

    plt.tight_layout(rect=(0, 0, 0.85, 1))

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        if plt.isinteractive():
            plt.show()
        else:
            logger.debug("Matplotlib non-interactive backend; skipping plt.show()")


def visualize_alignment(
    source_tokens: list[str],
    target_tokens: list[str],
    alignment: TextAlignment,
    title: str,
    output_path: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    """
    Convenience wrapper to visualize a single alignment.
    """
    visualize_alignments(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        alignments={"model": alignment},
        title=title,
        output_path=output_path,
        reference_model=None,
        figsize=figsize,
    )


def visualize_alignments_altair(
    source_tokens: list[str],
    target_tokens: list[str],
    alignments: Dict[str, TextAlignment],
    title: str,
    output_path: Optional[str] = None,
) -> Optional[alt.Chart]:
    """
    Altair‐based scatterplot of token alignments.
    Returns an Altair Chart (or saves it as HTML if output_path ends in .html).

    Args:
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens
        alignments: Dictionary mapping model names to TextAlignment objects
        title: Title for the chart
        output_path: Optional path to save as HTML

    Returns:
        Altair Chart object if alignments exist, None otherwise

    Example:
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> source = ["The", "cat"]
        >>> target = ["Le", "chat"]
        >>> alignments = {
        ...     "model1": TextAlignment(alignment=[
        ...         TokenAlignment(source="The", target="Le"),
        ...         TokenAlignment(source="cat", target="chat")
        ...     ])
        ... }
        >>> chart = visualize_alignments_altair(source, target, alignments, "Test")
        >>> chart is not None
        True
        >>> # Empty alignments return None
        >>> empty_chart = visualize_alignments_altair(source, target, {}, "Empty")
        >>> empty_chart is None
        True
    """
    # Recreate the mappings and uniquified tokens:
    source_map = create_token_mapping(source_tokens)
    target_map = create_token_mapping(target_tokens)
    src_uni = source_map.uniquified
    tgt_uni = target_map.uniquified

    # Prepare records
    records = []
    for model_name, ta in alignments.items():
        # skip empty or invalid
        for s, t in ta.get_alignment_positions(source_map, target_map):
            records.append(
                {
                    "model": model_name,
                    "source_pos": s,
                    "target_pos": t,
                    "source_token": src_uni[s],
                    "target_token": tgt_uni[t],
                }
            )
    if not records:
        logger.info("Skipping Altair visualization – no valid alignments")
        return None

    # Build a Polars DataFrame and convert to pandas for Altair
    df = pl.DataFrame(records).to_pandas()

    chart = (
        alt.Chart(df)
        .mark_circle(size=100)
        .encode(
            x=alt.X(
                "target_pos:O",
                axis=alt.Axis(
                    title="Target tokens",
                    labelExpr="datum.target_token",
                    labels=True,
                ),
            ),
            y=alt.Y(
                "source_pos:O",
                axis=alt.Axis(
                    title="Source tokens",
                    labelExpr="datum.source_token",
                    labels=True,
                ),
            ),
            color=alt.Color("model:N", legend=alt.Legend(title="Model")),
            tooltip=["model", "source_token", "target_token"],
        )
        .properties(title=title)
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )

    # Save or return
    if output_path and output_path.endswith(".html"):
        chart.save(output_path)
    return chart
