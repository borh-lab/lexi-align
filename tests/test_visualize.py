from lexi_align.visualize import visualize_alignments
from lexi_align.models import TextAlignment, TokenAlignment
import matplotlib.pyplot as plt
import tempfile
import os


def test_visualize_alignments_basic(sample_tokens, sample_alignments):
    """Test basic visualization functionality"""
    # Create visualization
    visualize_alignments(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments=sample_alignments,
        title="Test Alignment",
    )

    # Check that a figure was created
    assert plt.get_fignums(), "No figure was created"
    plt.close()


def test_visualize_alignments_output_file(sample_tokens, sample_alignments):
    """Test saving visualization to file"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        try:
            # Create and save visualization
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
            # Cleanup
            plt.close()
            os.unlink(tmp.name)


def test_visualize_alignments_reference_model(sample_tokens, sample_alignments):
    """Test visualization with reference model highlighting"""
    visualize_alignments(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments=sample_alignments,
        title="Test Alignment",
        reference_model="model1",
    )

    # Check that a figure was created
    assert plt.get_fignums(), "No figure was created"
    plt.close()


def test_visualize_alignments_single_model(sample_tokens, sample_alignment):
    """Test that visualization is skipped for single model"""
    visualize_alignments(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments={"model1": sample_alignment},
        title="Test Alignment",
    )

    # Check that no figure was created
    assert not plt.get_fignums(), "Figure was created for single model"
    plt.close()


def test_visualize_alignments_custom_figsize(sample_tokens, sample_alignments):
    """Test custom figure size"""
    custom_figsize = (15, 6)  # Explicitly provided size should override dynamic sizing
    visualize_alignments(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments=sample_alignments,
        title="Test Alignment",
        figsize=custom_figsize,
    )

    # Check figure size
    fig = plt.gcf()
    assert fig.get_size_inches().tolist() == list(custom_figsize)
    plt.close()


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
    from lexi_align.models import TextAlignment, TokenAlignment

    invalid_alignments = {
        "model1": TextAlignment(
            alignment=[
                TokenAlignment(source_token="invalid", target_token="nonexistent")
            ]
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


def test_visualize_alignments_legend(sample_tokens, sample_alignments):
    """Test legend generation"""
    visualize_alignments(
        source_tokens=sample_tokens["source"],
        target_tokens=sample_tokens["target"],
        alignments=sample_alignments,
        title="Legend Test",
    )

    fig = plt.gcf()
    legend = fig.axes[0].get_legend()
    assert legend is not None, "Legend not created"
    assert len(legend.get_texts()) == len(sample_alignments), "Incorrect legend entries"
    plt.close()


def test_visualize_alignments_unique_labels():
    """Test uniquification of duplicate tokens in labels"""
    source = ["the", "the", "cat"]
    target = ["le", "le", "chat"]
    alignments = {
        "model1": TextAlignment(
            alignment=[
                TokenAlignment(source_token="the", target_token="le"),
                TokenAlignment(source_token="cat", target_token="chat"),
            ]
        ),
        "model2": TextAlignment(
            alignment=[
                TokenAlignment(source_token="the", target_token="le"),
                TokenAlignment(source_token="cat", target_token="chat"),
            ]
        ),
    }

    visualize_alignments(
        source_tokens=source,
        target_tokens=target,
        alignments=alignments,
        title="Unique Labels Test",
    )

    fig = plt.gcf()
    xticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    yticklabels = [t.get_text() for t in fig.axes[0].get_yticklabels()]

    assert len(set(xticklabels)) == len(xticklabels), "X-axis labels not unique"
    assert len(set(yticklabels)) == len(yticklabels), "Y-axis labels not unique"
    plt.close()
