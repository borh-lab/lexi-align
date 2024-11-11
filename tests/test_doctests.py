import pytest
from lexi_align.core import align_tokens, align_tokens_raw
from lexi_align.models import TextAlignment, TokenAlignment, AlignmentResult


# TODO: Currently we mirror the doctests that need LLM API access here.
# In the future, we should try to remove this code duplication.
@pytest.mark.llm
def test_align_tokens_doctest(doctest_namespace):
    """Run the doctest for align_tokens with LLM access."""
    # Set up the namespace with required imports and objects
    doctest_namespace["align_tokens"] = align_tokens
    from lexi_align.adapters.outlines_adapter import OutlinesAdapter

    doctest_namespace["OutlinesAdapter"] = OutlinesAdapter
    doctest_namespace["TextAlignment"] = TextAlignment
    doctest_namespace["TokenAlignment"] = TokenAlignment

    # Create test data
    source_tokens = ["The", "cat", "is", "on", "the", "mat"]
    target_tokens = ["Le", "chat", "est", "sur", "le", "tapis"]

    # Run the example from the docstring
    adapter = OutlinesAdapter(temperature=0.0)
    alignment = align_tokens(
        adapter,
        source_tokens,
        target_tokens,
        source_language="English",
        target_language="French",
    )

    # Verify the result matches expected format
    assert isinstance(alignment, AlignmentResult)
    assert isinstance(alignment.alignment, TextAlignment)
    assert len(alignment.alignment.alignment) > 0
    for align in alignment.alignment.alignment:
        assert isinstance(align, TokenAlignment)
        assert align.source_token in source_tokens
        assert align.target_token in target_tokens


@pytest.mark.llm
def test_align_tokens_raw_doctest(doctest_namespace):
    """Run the doctest for align_tokens_raw with LLM access."""
    # Set up the namespace with required imports and objects
    doctest_namespace["align_tokens_raw"] = align_tokens_raw
    from lexi_align.adapters.outlines_adapter import OutlinesAdapter

    doctest_namespace["OutlinesAdapter"] = OutlinesAdapter
    doctest_namespace["TextAlignment"] = TextAlignment
    doctest_namespace["TokenAlignment"] = TokenAlignment

    # Create test data
    source_tokens = ["The", "cat", "is", "on", "the", "mat"]
    target_tokens = ["Le", "chat", "est", "sur", "le", "tapis"]
    custom_messages = [
        {
            "role": "system",
            "content": "You are a translator aligning English to French.",
        },
        {
            "role": "user",
            "content": "Align these tokens:\n"
            f"English: {' '.join(source_tokens)}\n"
            f"French: {' '.join(target_tokens)}",
        },
    ]

    # Run the example from the docstring
    adapter = OutlinesAdapter(temperature=0.0)
    alignment = align_tokens_raw(adapter, source_tokens, target_tokens, custom_messages)

    # Verify the result matches expected format
    assert isinstance(alignment, AlignmentResult)
    assert isinstance(alignment.alignment, TextAlignment)
    assert len(alignment.alignment.alignment) > 0
    for align in alignment.alignment.alignment:
        assert isinstance(align, TokenAlignment)
        assert align.source_token in source_tokens
        assert align.target_token in target_tokens
