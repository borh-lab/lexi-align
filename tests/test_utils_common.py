"""Test common utility functions."""

import pytest
import torch

from lexi_align.utils.common import (
    temporary_torch_seed,
)


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
