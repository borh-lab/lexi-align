"""Test alignment metrics calculation."""

import pytest

from lexi_align.metrics import calculate_metrics
from lexi_align.models import TextAlignment, TokenAlignment


def test_calculate_metrics_perfect_match():
    """Test metrics with perfect alignment match."""
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    metrics = calculate_metrics(alignment, alignment)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f_measure"] == 1.0
    assert metrics["aer"] == 0.0
    assert metrics["true_positives"] == 2
    assert metrics["predicted"] == 2
    assert metrics["gold"] == 2


def test_calculate_metrics_no_match():
    """Test metrics with no matching alignments."""
    predicted = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="chat"),
            TokenAlignment(source="cat", target="le"),
        ]
    )
    gold = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f_measure"] == 0.0
    assert metrics["aer"] == 1.0
    assert metrics["true_positives"] == 0
    assert metrics["predicted"] == 2
    assert metrics["gold"] == 2


def test_calculate_metrics_partial_match():
    """Test metrics with partial alignment match."""
    predicted = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    gold = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
            TokenAlignment(source="sat", target="assis"),
        ]
    )
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == 1.0  # 2/2 predicted are correct
    assert metrics["recall"] == pytest.approx(0.667, abs=0.01)  # 2/3 gold found
    assert metrics["f_measure"] == pytest.approx(0.8, abs=0.01)
    assert metrics["aer"] == pytest.approx(0.2, abs=0.01)
    assert metrics["true_positives"] == 2
    assert metrics["predicted"] == 2
    assert metrics["gold"] == 3


def test_calculate_metrics_empty_predicted():
    """Test metrics with empty predicted alignment."""
    predicted = TextAlignment(alignment=[])
    gold = TextAlignment(alignment=[TokenAlignment(source="the", target="le")])
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f_measure"] == 0.0
    assert metrics["aer"] == 1.0
    assert metrics["true_positives"] == 0
    assert metrics["predicted"] == 0
    assert metrics["gold"] == 1


def test_calculate_metrics_empty_gold():
    """Test metrics with empty gold alignment."""
    predicted = TextAlignment(alignment=[TokenAlignment(source="the", target="le")])
    gold = TextAlignment(alignment=[])
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f_measure"] == 0.0
    assert metrics["aer"] == 1.0
    assert metrics["true_positives"] == 0
    assert metrics["predicted"] == 1
    assert metrics["gold"] == 0


def test_calculate_metrics_both_empty():
    """Test metrics with both alignments empty."""
    predicted = TextAlignment(alignment=[])
    gold = TextAlignment(alignment=[])
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f_measure"] == 0.0
    assert metrics["aer"] == 1.0
    assert metrics["true_positives"] == 0
    assert metrics["predicted"] == 0
    assert metrics["gold"] == 0


def test_calculate_metrics_over_prediction():
    """Test metrics when predicted has more alignments than gold."""
    predicted = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
            TokenAlignment(source="sat", target="assis"),
            TokenAlignment(source="on", target="sur"),
        ]
    )
    gold = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == 0.5  # 2/4 predicted are correct
    assert metrics["recall"] == 1.0  # 2/2 gold found
    assert metrics["true_positives"] == 2
    assert metrics["predicted"] == 4
    assert metrics["gold"] == 2


def test_calculate_metrics_custom_f_alpha():
    """Test metrics with custom f_alpha parameter."""
    predicted = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    gold = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
            TokenAlignment(source="sat", target="assis"),
        ]
    )

    # Test with different f_alpha values
    # In the formula f = 1 / ((f_alpha/precision) + ((1-f_alpha)/recall)):
    # - Higher f_alpha weights precision more (gives result closer to precision)
    # - Lower f_alpha weights recall more (gives result closer to recall)
    metrics_default = calculate_metrics(predicted, gold, f_alpha=0.5)
    metrics_precision_weighted = calculate_metrics(predicted, gold, f_alpha=0.75)
    metrics_recall_weighted = calculate_metrics(predicted, gold, f_alpha=0.25)

    # Since precision=1.0 and recall=0.667:
    # - With higher precision weight (f_alpha=0.75), f_measure should be higher (closer to 1.0)
    # - With higher recall weight (f_alpha=0.25), f_measure should be lower (closer to 0.667)
    assert metrics_precision_weighted["f_measure"] > metrics_default["f_measure"]
    assert metrics_recall_weighted["f_measure"] < metrics_default["f_measure"]


def test_calculate_metrics_duplicate_alignments():
    """Test that duplicate alignments are counted correctly."""
    # Predicted has duplicate
    predicted = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="the", target="le"),  # duplicate
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    gold = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )
    metrics = calculate_metrics(predicted, gold)

    # Duplicates should be counted in set operations (only once)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
