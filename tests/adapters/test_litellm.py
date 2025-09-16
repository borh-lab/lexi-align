import os
from logging import getLogger

import pytest

from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
from lexi_align.core import (
    align_tokens,
    align_tokens_async,
)
from lexi_align.metrics import calculate_metrics
from lexi_align.models import AlignmentResult, TextAlignment, TokenAlignment

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
                    TokenAlignment(source="The", target="Le"),
                    TokenAlignment(source="cat", target="chat"),
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
            TokenAlignment(source="I", target="Je"),
            TokenAlignment(source="see", target="vois"),
            TokenAlignment(source="a", target="un"),
            TokenAlignment(source="dog", target="chien"),
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
    assert isinstance(result, AlignmentResult)
    assert result.alignment is not None
    assert len(result.alignment.alignment) > 0

    # Calculate alignment quality metrics
    metrics = calculate_metrics(result.alignment, expected)

    # Check if metrics meet minimum thresholds
    min_threshold = 0.25
    assert metrics["precision"] >= min_threshold, (
        f"Precision {metrics['precision']} below threshold {min_threshold}"
    )
    assert metrics["recall"] >= min_threshold, (
        f"Recall {metrics['recall']} below threshold {min_threshold}"
    )
    assert metrics["f_measure"] >= min_threshold, (
        f"F-measure {metrics['f_measure']} below threshold {min_threshold}"
    )

    # Log the metrics for visibility
    logger.info(f"Alignment metrics: {metrics}")


@pytest.mark.skipif(
    "TEST_LLM_MODEL" not in os.environ,
    reason="TEST_LLM_MODEL environment variable not set",
)
@pytest.mark.asyncio
async def test_litellm_adapter_async():
    model = os.environ["TEST_LLM_MODEL"]
    adapter = LiteLLMAdapter(model_params={"model": model})

    # Simple example in English-French
    examples = [
        (
            "The cat".split(),
            "Le chat".split(),
            TextAlignment(
                alignment=[
                    TokenAlignment(source="The", target="Le"),
                    TokenAlignment(source="cat", target="chat"),
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
            TokenAlignment(source="I", target="Je"),
            TokenAlignment(source="see", target="vois"),
            TokenAlignment(source="a", target="un"),
            TokenAlignment(source="dog", target="chien"),
        ]
    )

    result = await align_tokens_async(
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
    assert metrics["precision"] >= min_threshold, (
        f"Precision {metrics['precision']} below threshold {min_threshold}"
    )
    assert metrics["recall"] >= min_threshold, (
        f"Recall {metrics['recall']} below threshold {min_threshold}"
    )
    assert metrics["f_measure"] >= min_threshold, (
        f"F-measure {metrics['f_measure']} below threshold {min_threshold}"
    )

    # Log the metrics for visibility
    logger.info(f"Alignment metrics: {metrics}")
