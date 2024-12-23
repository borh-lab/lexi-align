from logging import getLogger
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from lexi_align.models import TextAlignment
from lexi_align.utils import create_token_mapping, make_unique

logger = getLogger(__name__)


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
    Visualize multiple token alignments using matplotlib.

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

    # Filter out empty alignments
    alignments = {k: v for k, v in alignments.items() if v.alignment}

    if len(alignments) <= 1:
        logger.info(
            f"Skipping visualization - need multiple alignments, got: {len(alignments)}"
        )
        return

    # Calculate dynamic figure size if not provided
    if figsize is None:
        # Base width and height (minimum sizes)
        base_width = 12
        base_height = 8

        # Scale factors
        width_per_token = 0.5  # How much width to add per target token
        height_per_token = 0.3  # How much height to add per source token

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
    _fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

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

    # Draw the alignments
    for i, _source_token in enumerate(source_tokens):
        for j, _target_token in enumerate(target_tokens):
            models_for_cell = cell_models[(i, j)]
            if models_for_cell:
                # Draw reference model highlighting if specified
                if reference_model:
                    # Check if this cell is in the reference alignment using positions
                    ref_alignment = alignments[reference_model]
                    ref_positions = set(
                        ref_alignment.get_alignment_positions(
                            source_mapping, target_mapping
                        )
                    )
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
                        radius=0.35,  # Reduced from 0.45
                        center=(j + 0.5, i + 0.5),
                        wedgeprops=dict(width=0.15),  # Reduced from 0.2
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
                        radius=0.35,  # Reduced from 0.45
                        center=(j + 0.5, i + 0.5),
                        wedgeprops=dict(width=0.35),  # Reduced from 0.45
                        startangle=90,
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
            color="w",
            label=model,
            markerfacecolor=colors[i],
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
                    color="black",
                    label="Correct alignment",
                    markerfacecolor="none",
                    markersize=10,
                    markeredgewidth=1.5,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="red",
                    label="Misalignment",
                    markerfacecolor="none",
                    markersize=10,
                    markeredgewidth=1.5,
                ),
            ]
        )

    ax.legend(
        handles=legend_handles,
        title="Models",
        loc="upper left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize="small",
        title_fontsize="small",
    )

    # Set title with left-aligned text
    ax.set_title(title, fontsize=14, weight="bold", wrap=True, loc="left")

    # Create metrics text if reference model exists
    if reference_model:
        metrics_text = "Metrics vs Reference:\n"
        for model in model_names:
            metrics = alignments[model].compare_alignments(
                alignments[reference_model], source_mapping, target_mapping
            )
            metrics_text += f"{model}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}\n"

        # Add metrics text above legend, right-aligned and closer to left side
        plt.figtext(
            0.82,  # x position moved left from 0.87
            0.7,  # y position stays the same
            metrics_text,
            fontsize="small",
            ha="right",  # changed from 'center' to 'right'
            va="top",
        )

    # Add legend below metrics
    ax.legend(
        handles=legend_handles,
        title="Models",
        loc="center left",
        bbox_to_anchor=(1.02, 0.4),  # Adjusted y position to be below metrics
        ncol=1,
        fontsize="small",
        title_fontsize="small",
    )

    plt.tight_layout(rect=(0, 0, 0.85, 1))

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
