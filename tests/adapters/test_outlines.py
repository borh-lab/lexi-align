from logging import getLogger

import pytest

from lexi_align.adapters.outlines_adapter import OutlinesAdapter
from lexi_align.core import align_tokens, align_tokens_batched
from lexi_align.metrics import calculate_metrics
from lexi_align.models import AlignmentResult, TextAlignment, TokenAlignment

logger = getLogger(__name__)


def validate_alignment_tokens(
    alignment: TextAlignment, src_tokens: list[str], tgt_tokens: list[str]
):
    """Helper function to validate alignment tokens."""
    src_aligned = {align.source for align in alignment.alignment}
    tgt_aligned = {align.target for align in alignment.alignment}
    assert src_aligned.issubset(set(src_tokens)), (
        f"Aligned source tokens {src_aligned} not subset of input tokens {set(src_tokens)}"
    )
    assert tgt_aligned.issubset(set(tgt_tokens)), (
        f"Aligned target tokens {tgt_aligned} not subset of input tokens {set(tgt_tokens)}"
    )


@pytest.mark.llm
@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_outlines_adapter_different_batch_sizes(batch_size):
    """Test that batching works correctly with different batch sizes."""
    adapter = OutlinesAdapter(temperature=0.3, samples=1, batch_size=batch_size)

    # Create fixed size input
    source_tokens_batch = [
        ["I", "see", "a", "dog"],
        ["The", "cat", "sleeps"],
        ["She", "runs", "fast"],
        ["They", "eat", "lunch"],
    ]
    target_tokens_batch = [
        ["Je", "vois", "un", "chien"],
        ["Le", "chat", "dort"],
        ["Elle", "court", "vite"],
        ["Ils", "mangent", "déjeuner"],
    ]

    results = align_tokens_batched(
        adapter,
        source_tokens_batch,
        target_tokens_batch,
        source_language="English",
        target_language="French",
    )

    # Results should be the same regardless of batch size
    assert len(results) == len(source_tokens_batch)
    for result, src, tgt in zip(results, source_tokens_batch, target_tokens_batch):
        assert isinstance(result, AlignmentResult)
        assert isinstance(result.alignment, TextAlignment)
        assert result.alignment.alignment and len(result.alignment.alignment) > 0
        validate_alignment_tokens(result.alignment, src, tgt)


@pytest.mark.llm
@pytest.mark.slow
def test_outlines_adapter_empty_batch():
    """Test handling of empty batch."""
    adapter = OutlinesAdapter(temperature=0.3, samples=1, batch_size=2)

    # Test with empty batch
    results = align_tokens_batched(
        adapter, [], [], source_language="English", target_language="French"
    )
    assert len(results) == 0
    assert all(isinstance(r, AlignmentResult) for r in results)


@pytest.mark.llm
@pytest.mark.slow
def test_outlines_adapter_single_item_batch():
    """Test handling of single-item batch."""
    adapter = OutlinesAdapter(temperature=0.3, samples=1, batch_size=2)

    source_tokens_batch = [["Hello", "world"]]
    target_tokens_batch = [["Bonjour", "monde"]]

    results = align_tokens_batched(
        adapter,
        source_tokens_batch,
        target_tokens_batch,
        source_language="English",
        target_language="French",
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, AlignmentResult)
    assert result.alignment is not None, (
        f"Alignment failed after {len(result.attempts)} attempts.\n"
        f"Last attempt details:\n"
        f"- Raw response type: {type(result.attempts[-1].raw_response)}\n"
        f"- Raw response: {result.attempts[-1].raw_response}\n"
        f"- Validation errors: {[f'{err[0]}: {err[1]}' for err in result.attempts[-1].validation_errors]}\n"
        f"- Exception: {result.attempts[-1].exception}\n\n"
        f"First message sent:\n{result.attempts[0].messages_sent[0]}"
    )
    validate_alignment_tokens(
        result.alignment, source_tokens_batch[0], target_tokens_batch[0]
    )
    assert isinstance(result, AlignmentResult)


@pytest.mark.llm
@pytest.mark.slow
def test_outlines_adapter_batch_error_handling():
    """Test error handling in batch processing."""
    adapter = OutlinesAdapter(temperature=0.3, samples=1, batch_size=2)

    # Test mismatched batch sizes
    with pytest.raises(
        ValueError, match="Number of source and target sequences must match"
    ):
        align_tokens_batched(
            adapter,
            [["Hello", "world"]],
            [],
            source_language="English",
            target_language="French",
        )


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

    # Expected alignment
    expected = TextAlignment(
        alignment=[
            TokenAlignment(source="I", target="Je"),
            TokenAlignment(source="see", target="vois"),
            TokenAlignment(source="a", target="un"),
            TokenAlignment(source="dog", target="chien"),
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
        assert isinstance(result, AlignmentResult)
        assert isinstance(result.alignment, TextAlignment)
        assert result.alignment.alignment and len(result.alignment.alignment) > 0

        # Calculate alignment quality metrics
        metrics = calculate_metrics(result.alignment, expected)

        # Check if metrics meet minimum thresholds
        min_threshold = 0.25
        assert metrics["precision"] >= min_threshold, (
            f"{sampler_type} precision {metrics['precision']} below threshold {min_threshold}"
        )
        assert metrics["recall"] >= min_threshold, (
            f"{sampler_type} recall {metrics['recall']} below threshold {min_threshold}"
        )
        assert metrics["f_measure"] >= min_threshold, (
            f"{sampler_type} F-measure {metrics['f_measure']} below threshold {min_threshold}"
        )

        # Log the metrics for visibility
        logger.info(f"{sampler_type} alignment metrics: {metrics}")
