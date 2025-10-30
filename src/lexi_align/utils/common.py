"""Common utility functions and patterns."""

from contextlib import contextmanager
from functools import cache
from typing import Any, Iterator, Optional

from lexi_align.models import TextAlignment, TextAlignmentSchema


def ensure_text_alignment(obj: Any) -> TextAlignment:
    """Single source of truth for converting objects to TextAlignment.

    Args:
        obj: Object to convert (TextAlignment, TextAlignmentSchema, dict, or str)

    Returns:
        TextAlignment instance

    Raises:
        TypeError: If object cannot be converted

    Example:
        >>> from lexi_align.models import TokenAlignment
        >>> ta = TextAlignment(alignment=[TokenAlignment(source="a", target="b")])
        >>> ensure_text_alignment(ta) is ta
        True
    """
    if isinstance(obj, TextAlignment):
        return obj
    if isinstance(obj, (TextAlignmentSchema, dict, str)):
        from lexi_align.utils import to_text_alignment

        return to_text_alignment(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__} to TextAlignment")


@contextmanager
def temporary_torch_seed(seed: Optional[int]) -> Iterator[None]:
    """Context manager for temporarily setting PyTorch random seed.

    Args:
        seed: Seed to set (None to skip)

    Example:
        >>> import torch
        >>> with temporary_torch_seed(42):
        ...     x = torch.rand(1)
    """
    if seed is None:
        yield
        return

    try:
        import torch
    except ImportError:
        yield
        return

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@cache
def get_default_marker_generator():
    """Cached default marker generator creation."""
    from lexi_align.text_processing import create_subscript_generator

    return create_subscript_generator()


def filter_none_values(d: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from dictionary.

    Example:
        >>> filter_none_values({"a": 1, "b": None, "c": 2})
        {'a': 1, 'c': 2}
    """
    return {k: v for k, v in d.items() if v is not None}


def batch_iterable(iterable, n: int):
    """Batch an iterable into chunks of size n.

    For Python 3.12+, uses itertools.batched; otherwise provides fallback.

    Example:
        >>> list(batch_iterable([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    try:
        from itertools import batched  # Python 3.12+

        return [list(batch) for batch in batched(iterable, n)]
    except ImportError:
        # Fallback for Python < 3.12
        result = []
        for i in range(0, len(iterable), n):
            result.append(iterable[i : i + n])
        return result
