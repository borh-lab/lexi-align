import pytest
from lexi_align.adapters.outlines_adapter import OutlinesAdapter
from lexi_align.core import align_tokens
from lexi_align.models import TextAlignment, TokenAlignment
from lexi_align.metrics import calculate_metrics
from logging import getLogger

logger = getLogger(__name__)


@pytest.mark.llm
@pytest.mark.slow
def test_outlines_adapter():
    # Test with greedy sampling (temperature=0.0)
    adapter_greedy = OutlinesAdapter(temperature=0.0)

    # Test with multinomial sampling (temperature=0.3)
    adapter_multinomial = OutlinesAdapter(temperature=0.3, samples=1)

    # Simple example in English-French
    examples = [
        (
            "The cat".split(),
            "Le chat".split(),
            TextAlignment(
                alignment=[
                    TokenAlignment(source_token="The", target_token="Le"),
                    TokenAlignment(source_token="cat", target_token="chat"),
                ]
            ),
        )
    ]

    source = "I see a dog"
    target = "Je vois un chien"

    source_tokens = source.split()
    target_tokens = target.split()

    # Expected alignment
    expected = TextAlignment(
        alignment=[
            TokenAlignment(source_token="I", target_token="Je"),
            TokenAlignment(source_token="see", target_token="vois"),
            TokenAlignment(source_token="a", target_token="un"),
            TokenAlignment(source_token="dog", target_token="chien"),
        ]
    )

    # Test greedy sampling
    result_greedy = align_tokens(
        adapter_greedy,
        source_tokens,
        target_tokens,
        source_language="English",
        target_language="French",
        examples=examples,
    )

    # Test multinomial sampling
    result_multinomial = align_tokens(
        adapter_multinomial,
        source_tokens,
        target_tokens,
        source_language="English",
        target_language="French",
        examples=examples,
    )

    # Validate both results
    for result, sampler_type in [
        (result_greedy, "greedy"),
        (result_multinomial, "multinomial"),
    ]:
        # Validate the structure of the result
        assert isinstance(result, TextAlignment)
        assert len(result.alignment) > 0

        # Calculate alignment quality metrics
        metrics = calculate_metrics(result, expected)

        # Check if metrics meet minimum thresholds
        min_threshold = 0.25
        assert (
            metrics["precision"] >= min_threshold
        ), f"{sampler_type} precision {metrics['precision']} below threshold {min_threshold}"
        assert (
            metrics["recall"] >= min_threshold
        ), f"{sampler_type} recall {metrics['recall']} below threshold {min_threshold}"
        assert (
            metrics["f1"] >= min_threshold
        ), f"{sampler_type} F1 score {metrics['f1']} below threshold {min_threshold}"

        # Log the metrics for visibility
        logger.info(f"{sampler_type} alignment metrics: {metrics}")
