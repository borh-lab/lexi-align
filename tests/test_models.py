"""Test core data models."""

import pytest

from lexi_align.models import (
    UNALIGNED_MARKER,
    SpecialTokens,
    TextAlignment,
    TokenAlignment,
    calculate_max_alignments,
    create_dynamic_alignment_schema,
    create_token_mapping,
)


def test_calculate_max_alignments():
    """Test max alignments calculation."""
    assert calculate_max_alignments(["a", "b"], ["x", "y"]) == 4
    assert calculate_max_alignments(["a"], ["x"]) == 2
    assert calculate_max_alignments(["a", "b", "c"], ["x"]) == 5
    assert calculate_max_alignments([], []) == 1  # minimum of 1


def test_text_alignment_equality():
    """Test TextAlignment equality comparison."""
    a1 = TextAlignment(
        alignment=[
            TokenAlignment(source="a", target="b"),
            TokenAlignment(source="c", target="d"),
        ]
    )
    a2 = TextAlignment(
        alignment=[
            TokenAlignment(source="a", target="b"),
            TokenAlignment(source="c", target="d"),
        ]
    )
    a3 = TextAlignment(
        alignment=[
            TokenAlignment(source="a", target="b"),
        ]
    )

    assert a1 == a2
    assert a1 != a3
    assert a1 != "not an alignment"  # type: ignore


def test_text_alignment_pairs():
    """Test TextAlignment.pairs() method."""
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="a", target="b"),
            TokenAlignment(source="c", target="d"),
        ]
    )
    assert alignment.pairs() == [("a", "b"), ("c", "d")]


def test_text_alignment_deduplication():
    """Test that TextAlignment removes duplicate pairs."""
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="a", target="b"),
            TokenAlignment(source="a", target="b"),  # duplicate
            TokenAlignment(source="c", target="d"),
        ]
    )

    # Should have deduplicated
    assert len(alignment.alignment) == 2
    pairs = set(alignment.pairs())
    assert ("a", "b") in pairs
    assert ("c", "d") in pairs


def test_text_alignment_filters_both_unaligned():
    """Test that alignments with both tokens unaligned are filtered."""
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source=UNALIGNED_MARKER, target=UNALIGNED_MARKER),
            TokenAlignment(source="a", target="b"),
        ]
    )

    # Should filter out the both-unaligned pair
    assert len(alignment.alignment) == 1
    assert alignment.alignment[0].source == "a"


def test_text_alignment_sort_by_position():
    """Test TextAlignment sorting by position."""
    source_tokens = ["a", "b", "c"]
    target_tokens = ["x", "y", "z"]
    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)

    # Create alignment in wrong order
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="c", target="z"),
            TokenAlignment(source="a", target="x"),
            TokenAlignment(source="b", target="y"),
        ],
        source_mapping=source_mapping,
        target_mapping=target_mapping,
    )

    # Should be sorted by source position
    sorted_alignment = alignment.sort_by_position(source_mapping, target_mapping)
    sources = [a.source for a in sorted_alignment.alignment]
    assert sources == ["a", "b", "c"]


def test_text_alignment_from_token_alignments():
    """Test creating TextAlignment from token lists."""
    source = ["the", "cat"]
    target = ["le", "chat"]
    alignments = [
        TokenAlignment(source="the", target="le"),
        TokenAlignment(source="cat", target="chat"),
    ]

    result = TextAlignment.from_token_alignments(alignments, source, target)

    assert result.source_mapping is not None
    assert result.target_mapping is not None
    assert len(result.alignment) == 2


def test_text_alignment_get_alignment_positions():
    """Test getting alignment positions."""
    source_tokens = ["the", "cat", "the"]
    target_tokens = ["le", "chat", "le"]
    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)

    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the₁", target="le₁"),
            TokenAlignment(source="cat", target="chat"),
        ],
        source_mapping=source_mapping,
        target_mapping=target_mapping,
    )

    positions = alignment.get_alignment_positions()
    assert positions == [(0, 0), (1, 1)]


def test_text_alignment_get_aligned_tokens():
    """Test getting aligned token sets."""
    alignment = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ]
    )

    source_tokens, target_tokens = alignment.get_aligned_tokens()
    assert source_tokens == {"the", "cat"}
    assert target_tokens == {"le", "chat"}


def test_text_alignment_compare_alignments():
    """Test comparing alignments."""
    source_tokens = ["the", "cat"]
    target_tokens = ["le", "chat"]
    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)

    predicted = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
            TokenAlignment(source="cat", target="chat"),
        ],
        source_mapping=source_mapping,
        target_mapping=target_mapping,
    )

    gold = TextAlignment(
        alignment=[
            TokenAlignment(source="the", target="le"),
        ],
        source_mapping=source_mapping,
        target_mapping=target_mapping,
    )

    metrics = predicted.compare_alignments(gold, source_mapping, target_mapping)

    assert metrics["precision"] == 0.5  # 1/2
    assert metrics["recall"] == 1.0  # 1/1
    assert metrics["true_positives"] == 1
    assert metrics["predicted"] == 2
    assert metrics["gold"] == 1


def test_create_dynamic_alignment_schema_constraints():
    """Test dynamic schema with min/max constraints."""
    source = ["the", "cat"]
    target = ["le", "chat"]

    Schema = create_dynamic_alignment_schema(source, target, min_length=1, max_length=3)

    # Valid: within range
    valid_data = {
        "alignment": [
            {"source": "the", "target": "le"},
            {"source": "cat", "target": "chat"},
        ]
    }
    schema = Schema.model_validate(valid_data)
    assert len(schema.alignment) == 2

    # Invalid: too short
    with pytest.raises(Exception):
        Schema.model_validate({"alignment": []})

    # Invalid: too long
    with pytest.raises(Exception):
        Schema.model_validate(
            {
                "alignment": [
                    {"source": "the", "target": "le"},
                    {"source": "cat", "target": "chat"},
                    {"source": "the", "target": "le"},
                    {"source": "cat", "target": "chat"},
                ]
            }
        )


def test_create_dynamic_alignment_schema_with_reasoning():
    """Test dynamic schema with reasoning field."""
    source = ["the", "cat"]
    target = ["le", "chat"]

    Schema = create_dynamic_alignment_schema(source, target, use_reasoning=True)

    # Should require reasoning
    valid_data = {
        "reasoning": "These are simple one-to-one correspondences between English and French words.",
        "alignment": [
            {"source": "the", "target": "le"},
        ],
    }
    schema = Schema.model_validate(valid_data)
    assert schema.reasoning is not None
    assert len(schema.reasoning) >= 50  # min_length constraint


def test_special_tokens_enum():
    """Test SpecialTokens enum values."""
    assert SpecialTokens.UNALIGNED.value == "<unaligned>"
    assert SpecialTokens.SOURCE_SPECIFIC.value == "<source_specific>"
    assert SpecialTokens.TARGET_SPECIFIC.value == "<target_specific>"
    assert UNALIGNED_MARKER == "<unaligned>"
