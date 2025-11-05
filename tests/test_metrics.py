"""Test alignment metrics calculation."""

import pytest

from lexi_align.metrics import calculate_metrics
from lexi_align.models import TextAlignment, TokenAlignment


@pytest.mark.parametrize(
    "predicted_pairs,gold_pairs,expected_precision,expected_recall,expected_f1",
    [
        # Perfect match
        (
            [("the", "le"), ("cat", "chat")],
            [("the", "le"), ("cat", "chat")],
            1.0,
            1.0,
            1.0,
        ),
        # No match
        (
            [("the", "chat"), ("cat", "le")],
            [("the", "le"), ("cat", "chat")],
            0.0,
            0.0,
            0.0,
        ),
        # Partial match: 2/2 predicted correct, 2/3 gold found
        (
            [("the", "le"), ("cat", "chat")],
            [("the", "le"), ("cat", "chat"), ("sat", "assis")],
            1.0,
            pytest.approx(0.667, abs=0.01),
            pytest.approx(0.8, abs=0.01),
        ),
        # Over-prediction: 2/4 predicted correct, 2/2 gold found
        (
            [("the", "le"), ("cat", "chat"), ("sat", "assis"), ("on", "sur")],
            [("the", "le"), ("cat", "chat")],
            0.5,
            1.0,
            pytest.approx(0.667, abs=0.01),
        ),
        # Empty predicted
        ([], [("the", "le")], 0.0, 0.0, 0.0),
        # Empty gold
        ([("the", "le")], [], 0.0, 0.0, 0.0),
        # Both empty
        ([], [], 0.0, 0.0, 0.0),
    ],
)
def test_calculate_metrics_scenarios(
    predicted_pairs, gold_pairs, expected_precision, expected_recall, expected_f1
):
    """Test metrics across various alignment scenarios."""
    predicted = TextAlignment(
        alignment=[TokenAlignment(source=s, target=t) for s, t in predicted_pairs]
    )
    gold = TextAlignment(
        alignment=[TokenAlignment(source=s, target=t) for s, t in gold_pairs]
    )
    metrics = calculate_metrics(predicted, gold)

    assert metrics["precision"] == expected_precision
    assert metrics["recall"] == expected_recall
    assert metrics["f_measure"] == expected_f1


def test_calculate_metrics_f_alpha_weighting():
    """Test that f_alpha parameter correctly weights precision vs recall.

    With precision=1.0 and recall≈0.667:
    - Higher f_alpha (0.75) weights precision more → f_measure closer to 1.0
    - Lower f_alpha (0.25) weights recall more → f_measure closer to 0.667
    """
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

    metrics_default = calculate_metrics(predicted, gold, f_alpha=0.5)
    metrics_precision_weighted = calculate_metrics(predicted, gold, f_alpha=0.75)
    metrics_recall_weighted = calculate_metrics(predicted, gold, f_alpha=0.25)

    # Verify monotonic ordering: precision-weighted > default > recall-weighted
    assert metrics_precision_weighted["f_measure"] > metrics_default["f_measure"]
    assert metrics_default["f_measure"] > metrics_recall_weighted["f_measure"]


# Duplicate alignment test covered by parametrized test (perfect match scenario)
