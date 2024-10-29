import os
import pytest
from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
from lexi_align.core import align_tokens
from lexi_align.models import TextAlignment, TokenAlignment
from lexi_align.metrics import calculate_metrics
from logging import getLogger

logger = getLogger(__name__)


@pytest.mark.skipif(
    "TEST_LLM_MODEL" not in os.environ,
    reason="TEST_LLM_MODEL environment variable not set",
)
def test_litellm_adapter():
    model = os.environ["TEST_LLM_MODEL"]
    adapter = LiteLLMAdapter(model_params={"model": model})

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

    # Update expected to match the actual source/target tokens
    expected = TextAlignment(
        alignment=[
            TokenAlignment(source_token="I", target_token="Je"),
            TokenAlignment(source_token="see", target_token="vois"),
            TokenAlignment(source_token="a", target_token="un"),
            TokenAlignment(source_token="dog", target_token="chien"),
        ]
    )

    result = align_tokens(
        adapter,
        source_tokens,
        target_tokens,
        source_language="English",
        target_language="French",
        examples=examples,
    )

    # Validate the structure of the result
    assert isinstance(result, TextAlignment)
    assert len(result.alignment) > 0

    # Calculate alignment quality metrics
    metrics = calculate_metrics(result, expected)

    # Check if metrics meet minimum thresholds
    min_threshold = 0.25
    assert (
        metrics["precision"] >= min_threshold
    ), f"Precision {metrics['precision']} below threshold {min_threshold}"
    assert (
        metrics["recall"] >= min_threshold
    ), f"Recall {metrics['recall']} below threshold {min_threshold}"
    assert (
        metrics["f1"] >= min_threshold
    ), f"F1 score {metrics['f1']} below threshold {min_threshold}"

    # Log the metrics for visibility
    logger.info(f"Alignment metrics: {metrics}")
