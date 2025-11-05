"""Test alignment validation functions."""

from lexi_align.constants import UNALIGNED_MARKER
from lexi_align.core import (
    _create_retry_message,
    _validate_alignment,
    categorize_validation_errors,
    normalize_validation_errors,
)
from lexi_align.models import (
    TextAlignment,
    TokenAlignment,
    ValidationErrorType,
)


def test_categorize_validation_errors_single_type():
    """Test categorizing errors of single type."""
    errors = [
        (ValidationErrorType.INVALID_SOURCE_TOKEN, "Invalid 'foo'", ["foo"]),
        (ValidationErrorType.INVALID_SOURCE_TOKEN, "Invalid 'bar'", ["bar"]),
    ]
    result = categorize_validation_errors(errors)

    assert result[ValidationErrorType.INVALID_SOURCE_TOKEN]["count"] == 2
    assert result[ValidationErrorType.INVALID_SOURCE_TOKEN]["frequencies"]["foo"] == 1
    assert result[ValidationErrorType.INVALID_SOURCE_TOKEN]["frequencies"]["bar"] == 1


def test_categorize_validation_errors_multiple_types():
    """Test categorizing errors of multiple types."""
    errors = [
        (ValidationErrorType.INVALID_SOURCE_TOKEN, "Invalid 'foo'", ["foo"]),
        (ValidationErrorType.MISSING_TARGET_ALIGNMENTS, "Missing 'bar'", ["bar"]),
    ]
    result = categorize_validation_errors(errors)

    assert result[ValidationErrorType.INVALID_SOURCE_TOKEN]["count"] == 1
    assert result[ValidationErrorType.MISSING_TARGET_ALIGNMENTS]["count"] == 1


def test_categorize_validation_errors_empty():
    """Test categorizing empty error list."""
    result = categorize_validation_errors([])

    # Should have entries for all error types with count 0
    for error_type in ValidationErrorType:
        assert result[error_type]["count"] == 0
        assert result[error_type]["frequencies"] == {}


def test_normalize_validation_errors():
    """Test normalizing validation errors to dicts."""
    errors = [
        (ValidationErrorType.INVALID_SOURCE_TOKEN, "bad token", ["foo"]),
        (ValidationErrorType.MISSING_TARGET_ALIGNMENTS, "missing", ["bar"]),
    ]
    result = normalize_validation_errors(errors)

    assert len(result) == 2
    assert result[0]["type"] == ValidationErrorType.INVALID_SOURCE_TOKEN
    assert result[0]["message"] == "bad token"
    assert result[0]["tokens"] == ["foo"]


def test_validate_alignment_valid():
    """Test validation of valid alignment."""
    source_tokens = ["the", "cat"]
    target_tokens = ["le", "chat"]
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )

    is_valid, errors, valid_aligns, rem_src, rem_tgt = _validate_alignment(
        alignment, source_tokens, target_tokens
    )

    assert is_valid
    assert len(errors) == 0
    assert len(valid_aligns) == 2
    assert len(rem_src) == 0
    assert len(rem_tgt) == 0


def test_validate_alignment_invalid_source_token():
    """Test validation with invalid source token."""
    source_tokens = ["the", "cat"]
    target_tokens = ["le", "chat"]
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="dog", target="chat"),  # 'dog' not in source
        ]
    )

    is_valid, errors, valid_aligns, rem_src, rem_tgt = _validate_alignment(
        alignment, source_tokens, target_tokens
    )

    assert not is_valid
    assert len(errors) > 0
    assert any(e[0] == ValidationErrorType.INVALID_SOURCE_TOKEN for e in errors)
    assert len(valid_aligns) == 1  # Only 'the'->'le' is valid


def test_validate_alignment_missing_tokens():
    """Test validation with unaligned tokens."""
    source_tokens = ["the", "cat", "sat"]
    target_tokens = ["le", "chat"]
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
        ]
    )

    is_valid, errors, valid_aligns, rem_src, rem_tgt = _validate_alignment(
        alignment, source_tokens, target_tokens
    )

    assert not is_valid
    assert "cat" in rem_src
    assert "sat" in rem_src
    assert "chat" in rem_tgt


def test_validate_alignment_with_unaligned_marker():
    """Test validation with explicit unaligned markers."""
    source_tokens = ["the", "cat"]
    target_tokens = ["chat"]
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target=UNALIGNED_MARKER),
            TokenAlignment(source="cat", target="chat"),
        ]
    )

    is_valid, errors, valid_aligns, rem_src, rem_tgt = _validate_alignment(
        alignment, source_tokens, target_tokens
    )

    assert is_valid
    # 'the' explicitly marked as unaligned, so not in remaining
    assert "the" not in rem_src
    assert len(rem_tgt) == 0


def test_validate_alignment_with_existing_alignments():
    """Test validation accumulating with existing alignments."""
    source_tokens = ["the", "cat", "sat"]
    target_tokens = ["le", "chat", "assis"]

    existing = [TokenAlignment(source="the", target="le")]

    new_alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="cat", target="chat"),
        ]
    )

    is_valid, errors, valid_aligns, rem_src, rem_tgt = _validate_alignment(
        new_alignment, source_tokens, target_tokens, existing_alignments=existing
    )

    assert not is_valid  # Still missing 'sat'
    assert len(valid_aligns) == 2  # existing + new
    assert "sat" in rem_src
    assert "assis" in rem_tgt


def test_create_retry_message_basic():
    """Test creating retry message."""
    source_tokens = ["the", "cat", "sat"]
    target_tokens = ["le", "chat", "assis"]
    valid_alignments = [TokenAlignment(source="the", target="le")]
    remaining_source = {"cat", "sat"}
    remaining_target = {"chat", "assis"}

    message = _create_retry_message(
        valid_alignments,
        remaining_source,
        remaining_target,
        source_tokens,
        target_tokens,
    )

    content = message.content
    assert "partial alignments" in content
    assert "remaining_source_tokens:" in content
    assert "remaining_target_tokens:" in content


def test_create_retry_message_preserves_order():
    """Test that retry message preserves token order."""
    source_tokens = ["a", "b", "c", "d"]
    target_tokens = ["w", "x", "y", "z"]
    valid_alignments = [TokenAlignment(source="a", target="w")]
    remaining_source = {"d", "b", "c"}  # Unordered set
    remaining_target = {"z", "x", "y"}  # Unordered set

    message = _create_retry_message(
        valid_alignments,
        remaining_source,
        remaining_target,
        source_tokens,
        target_tokens,
    )

    content = message.content
    # Type guard: ensure content is a string
    assert isinstance(content, str), "Message content should be a string"

    # Extract just the remaining tokens lines to avoid false matches
    lines = content.split("\n")
    remaining_src_line = next(
        (line for line in lines if line.startswith("remaining_source_tokens:")), ""
    )
    remaining_tgt_line = next(
        (line for line in lines if line.startswith("remaining_target_tokens:")), ""
    )

    # Extract just the token portion (after the colon and space)
    src_tokens_str = (
        remaining_src_line.split(":", 1)[1].strip() if ":" in remaining_src_line else ""
    )
    tgt_tokens_str = (
        remaining_tgt_line.split(":", 1)[1].strip() if ":" in remaining_tgt_line else ""
    )

    # Should appear in original order in the token string, not set order
    assert (
        src_tokens_str.index("b")
        < src_tokens_str.index("c")
        < src_tokens_str.index("d")
    )
    assert (
        tgt_tokens_str.index("x")
        < tgt_tokens_str.index("y")
        < tgt_tokens_str.index("z")
    )
