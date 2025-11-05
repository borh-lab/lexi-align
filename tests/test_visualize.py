import os
import tempfile

import altair as alt
import matplotlib.pyplot as plt
import pytest

from lexi_align.models import TextAlignment, TokenAlignment
from lexi_align.visualize import visualize_alignments, visualize_alignments_altair


@pytest.mark.parametrize(
    "reference_model,figsize",
    [
        (None, None),  # Default behavior
        ("model1", None),  # With reference model
        (None, (15, 6)),  # Custom figsize
        ("model1", (15, 6)),  # Both reference and custom figsize
    ],
)
def test_visualize_alignments_matplotlib(
    sample_tokens, sample_alignments, reference_model, figsize
):
    """Test matplotlib visualization with various configurations."""
    visualize_alignments(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments=sample_alignments,
        title="Test Alignment",
        reference_model=reference_model,
        figsize=figsize,
    )

    # Check that a figure was created
    assert plt.get_fignums(), "No figure was created"

    # If custom figsize provided, verify it
    if figsize:
        fig = plt.gcf()
        assert fig.get_size_inches().tolist() == list(figsize)

    plt.close()


def test_visualize_alignments_output_file(sample_tokens, sample_alignments):
    """Test saving visualization to file"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        try:
            visualize_alignments(
                source_tokens=sample_tokens["source"],
                target_tokens=sample_tokens["target"],
                alignments=sample_alignments,
                title="Test Alignment",
                output_path=tmp.name,
            )

            # Check that file exists and has content
            assert os.path.exists(tmp.name), "Output file was not created"
            assert os.path.getsize(tmp.name) > 0, "Output file is empty"

        finally:
            plt.close()
            os.unlink(tmp.name)


def test_visualize_alignments_empty():
    """Test handling of empty alignments"""
    visualize_alignments(
        source_tokens=["test"],
        target_tokens=["test"],
        alignments={},
        title="Empty Test",
    )
    assert not plt.get_fignums(), "Figure was created for empty alignments"
    plt.close()


def test_visualize_alignments_invalid_tokens():
    """Test handling of invalid token alignments"""
    invalid_alignments = {
        "model1": TextAlignment(
            alignment=[TokenAlignment(source="invalid", target="nonexistent")]
        )
    }

    visualize_alignments(
        source_tokens=["test"],
        target_tokens=["test"],
        alignments=invalid_alignments,
        title="Invalid Test",
    )
    assert not plt.get_fignums(), "Figure was created for invalid alignments"
    plt.close()


def test_visualize_alignments_altair_basic(sample_tokens, sample_alignments):
    """Test Altair visualization returns Chart for valid alignments."""
    chart = visualize_alignments_altair(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments=sample_alignments,
        title="Altair Test",
    )
    assert chart is not None, "Chart should not be None for valid alignments"
    assert isinstance(chart, alt.Chart), "Result should be an Altair Chart"


def test_visualize_alignments_altair_empty():
    """Test Altair visualization returns None for empty alignments."""
    chart = visualize_alignments_altair(
        source_tokens=["test"],
        target_tokens=["test"],
        alignments={},
        title="Empty Altair Test",
    )
    assert chart is None, "Chart should be None for empty alignments"
