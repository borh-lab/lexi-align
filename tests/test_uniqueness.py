import pytest
from lexi_align.utils import (
    make_unique,
    remove_unique,
    remove_unique_one,
    export_pharaoh_format,
    parse_pharaoh_format,
)
from lexi_align.text_processing import create_subscript_generator
from lexi_align.models import TextAlignment, TokenAlignment


def test_make_unique_basic():
    """Test basic uniqueness functionality."""
    tokens = ["the", "cat", "the", "mat"]
    unique_tokens = make_unique(tokens)
    assert unique_tokens == ["the₁", "cat", "the₂", "mat"]

    # Test removal
    assert remove_unique(unique_tokens) == tokens
    pattern = create_subscript_generator().pattern
    assert [remove_unique_one(t, pattern) for t in unique_tokens] == tokens


def test_make_unique_multiple_repeats():
    """Test handling of tokens that appear more than twice."""
    tokens = ["the", "the", "the", "cat", "cat", "mat"]
    unique_tokens = make_unique(tokens)
    assert unique_tokens == ["the₁", "the₂", "the₃", "cat₁", "cat₂", "mat"]

    # Test removal
    assert remove_unique(unique_tokens) == tokens
    pattern = create_subscript_generator().pattern
    assert [remove_unique_one(t, pattern) for t in unique_tokens] == tokens


def test_make_unique_already_unique():
    """Test that already unique tokens are not modified."""
    tokens = ["the", "cat", "mat"]
    unique_tokens = make_unique(tokens)
    assert unique_tokens == tokens

    # Test removal has no effect
    assert remove_unique(unique_tokens) == tokens
    pattern = create_subscript_generator().pattern
    assert [remove_unique_one(t, pattern) for t in unique_tokens] == tokens


def test_make_unique_empty_and_edge_cases():
    """Test edge cases for uniqueness handling."""
    # Empty list
    assert make_unique([]) == []

    # Single token
    assert make_unique(["token"]) == ["token"]

    # Tokens with spaces or special characters
    tokens = ["hello world", "hello world", "test!"]
    unique_tokens = make_unique(tokens)
    assert unique_tokens == ["hello world₁", "hello world₂", "test!"]


def test_subscript_handling():
    """Test handling of existing subscripts and ensuring they're properly managed."""
    # Test that existing subscripts are handled correctly
    tokens = ["word₁", "word₂", "word₁"]
    unique_tokens = make_unique(tokens)
    assert unique_tokens == ["word₁", "word₂", "word₃"]

    pattern = create_subscript_generator().pattern
    # Test removal of multiple subscripts
    assert remove_unique_one("word₁₂", pattern) == "word"
    assert remove_unique_one("word₂", pattern) == "word"
    assert remove_unique_one("word₁₂₃", pattern) == "word"

    # Test full removal
    assert remove_unique(unique_tokens) == ["word", "word", "word"]

    # Test complex subscript combinations
    tokens = ["word₁₂", "word₁₂", "word₂₁"]
    unique_tokens = make_unique(tokens)
    assert unique_tokens == ["word₁", "word₂", "word₃"]


def test_invalid_inputs():
    """Test handling of invalid inputs for uniqueness functions."""
    # Test None values
    with pytest.raises(TypeError):
        make_unique(None)  # type: ignore[arg-type]


def test_make_unique_idempotent():
    """Test that make_unique is idempotent."""
    tokens = ["the", "cat", "the", "mat"]
    unique_tokens = make_unique(tokens)
    # Running make_unique again should return the same result
    assert make_unique(unique_tokens) == unique_tokens


def test_whitespace_preservation():
    """Test that whitespace and special characters are preserved."""
    # Original test was wrong - these tokens are all different so they shouldn't get subscripts
    tokens = ["hello  world", "hello\tworld", "hello\nworld"]
    unique_tokens = make_unique(tokens)
    # They should remain unchanged since they're already unique
    assert unique_tokens == ["hello  world", "hello\tworld", "hello\nworld"]

    # Test removal preserves whitespace
    assert remove_unique(unique_tokens) == tokens

    # Test case with repeating tokens
    repeated_tokens = ["hello  world", "hello  world", "hello\tworld"]
    unique_repeated = make_unique(repeated_tokens)
    assert unique_repeated == ["hello  world₁", "hello  world₂", "hello\tworld"]


def test_pharaoh_format_uniqueness():
    """Test that Pharaoh format correctly handles unique tokens in complex scenarios."""
    # Create uniquified tokens matching the alignment
    source_tokens = ["the₁", "big", "cat₁", "saw", "the₂", "cat₂"]
    target_tokens = ["le₁", "gros", "chat₁", "a", "vu", "le₂", "chat₂"]

    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source_token="the₁", target_token="le₁"),
            TokenAlignment(source_token="big", target_token="gros"),
            TokenAlignment(source_token="cat₁", target_token="chat₁"),
            # Split multi-word alignment into separate token alignments
            TokenAlignment(source_token="saw", target_token="a"),
            TokenAlignment(source_token="saw", target_token="vu"),
            TokenAlignment(source_token="the₂", target_token="le₂"),
            TokenAlignment(source_token="cat₂", target_token="chat₂"),
        ]
    )

    # Export to Pharaoh format
    pharaoh = export_pharaoh_format(source_tokens, target_tokens, alignment)

    # Expected format should match word-level alignments with uniquified tokens
    expected = "the₁ big cat₁ saw the₂ cat₂\tle₁ gros chat₁ a vu le₂ chat₂\t0-0 1-1 2-2 3-3 3-4 4-5 5-6"
    assert pharaoh == expected

    # Parse back and verify
    parsed_source, parsed_target, parsed_alignment = parse_pharaoh_format(pharaoh)
    assert parsed_source == " ".join(source_tokens)
    assert parsed_target == " ".join(target_tokens)

    # Verify alignments are identical after round-trip
    original_pairs = {(a.source_token, a.target_token) for a in alignment.alignment}
    parsed_pairs = {
        (a.source_token, a.target_token) for a in parsed_alignment.alignment
    }
    assert original_pairs == parsed_pairs
