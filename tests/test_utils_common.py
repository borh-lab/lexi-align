"""Test common utility functions."""

import pytest
import torch

from lexi_align.models import TextAlignment, TokenAlignment, TextAlignmentSchema
from lexi_align.utils.common import (
    ensure_text_alignment,
    temporary_torch_seed,
    filter_none_values,
    batch_iterable,
)


def test_ensure_text_alignment_identity():
    """Test that TextAlignment passes through unchanged."""
    ta = TextAlignment(alignment=[TokenAlignment(source="a", target="b")])
    result = ensure_text_alignment(ta)
    assert result is ta


def test_ensure_text_alignment_from_schema():
    """Test conversion from TextAlignmentSchema."""
    schema = TextAlignmentSchema(alignment=[TokenAlignment(source="a", target="b")])
    result = ensure_text_alignment(schema)
    assert isinstance(result, TextAlignment)
    assert len(result.alignment) == 1
    assert result.alignment[0].source == "a"


def test_ensure_text_alignment_from_dict():
    """Test conversion from dictionary."""
    data = {"alignment": [{"source": "a", "target": "b"}]}
    result = ensure_text_alignment(data)
    assert isinstance(result, TextAlignment)
    assert len(result.alignment) == 1


def test_ensure_text_alignment_from_json_string():
    """Test conversion from JSON string."""
    json_str = '{"alignment": [{"source": "a", "target": "b"}]}'
    result = ensure_text_alignment(json_str)
    assert isinstance(result, TextAlignment)
    assert len(result.alignment) == 1


def test_ensure_text_alignment_invalid_type():
    """Test that invalid types raise TypeError."""
    with pytest.raises(TypeError, match="Cannot convert"):
        ensure_text_alignment(123)  # type: ignore

    with pytest.raises(TypeError, match="Cannot convert"):
        ensure_text_alignment([1, 2, 3])  # type: ignore


def test_temporary_torch_seed_none():
    """Test that None seed does nothing."""
    original = torch.rand(1).item()
    with temporary_torch_seed(None):
        after = torch.rand(1).item()
    # Different random values (not reset)
    assert original != after


def test_temporary_torch_seed_reproducibility():
    """Test that seed makes generation reproducible."""
    with temporary_torch_seed(42):
        value1 = torch.rand(1).item()

    with temporary_torch_seed(42):
        value2 = torch.rand(1).item()

    assert value1 == value2


def test_temporary_torch_seed_restoration():
    """Test that seed is restored after context."""
    # Get initial state
    initial_state = torch.random.get_rng_state()

    # Use temporary seed
    with temporary_torch_seed(42):
        torch.rand(1)  # Generate something

    # State should be restored
    restored_state = torch.random.get_rng_state()
    assert torch.equal(initial_state, restored_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_temporary_torch_seed_cuda():
    """Test that CUDA seed is also managed."""
    with temporary_torch_seed(42):
        value1 = torch.randn(1, device="cuda").item()

    with temporary_torch_seed(42):
        value2 = torch.randn(1, device="cuda").item()

    assert value1 == value2


def test_filter_none_values_basic():
    """Test filtering None values from dict."""
    data = {"a": 1, "b": None, "c": 2, "d": None}
    result = filter_none_values(data)
    assert result == {"a": 1, "c": 2}


def test_filter_none_values_empty():
    """Test filtering empty dict."""
    assert filter_none_values({}) == {}


def test_filter_none_values_all_none():
    """Test filtering dict with all None values."""
    data = {"a": None, "b": None}
    result = filter_none_values(data)
    assert result == {}


def test_filter_none_values_no_none():
    """Test filtering dict with no None values."""
    data = {"a": 1, "b": 2, "c": 3}
    result = filter_none_values(data)
    assert result == data


def test_filter_none_values_preserves_falsy():
    """Test that other falsy values are preserved."""
    data = {"a": 0, "b": "", "c": False, "d": None, "e": []}
    result = filter_none_values(data)
    # Only None should be filtered
    assert result == {"a": 0, "b": "", "c": False, "e": []}


def test_batch_iterable_even_division():
    """Test batching with even division."""
    items = [1, 2, 3, 4, 5, 6]
    result = batch_iterable(items, 2)
    assert result == [[1, 2], [3, 4], [5, 6]]


def test_batch_iterable_uneven_division():
    """Test batching with remainder."""
    items = [1, 2, 3, 4, 5]
    result = batch_iterable(items, 2)
    assert result == [[1, 2], [3, 4], [5]]


def test_batch_iterable_size_one():
    """Test batching with size 1."""
    items = [1, 2, 3]
    result = batch_iterable(items, 1)
    assert result == [[1], [2], [3]]


def test_batch_iterable_size_larger_than_list():
    """Test batching with batch size larger than list."""
    items = [1, 2, 3]
    result = batch_iterable(items, 10)
    assert result == [[1, 2, 3]]


def test_batch_iterable_empty():
    """Test batching empty list."""
    result = batch_iterable([], 5)
    assert result == []
