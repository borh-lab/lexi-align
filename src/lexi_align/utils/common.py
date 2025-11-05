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
        >>> from lexi_align.models import TokenAlignment, TextAlignmentSchema
        >>> # Identity: TextAlignment passes through unchanged
        >>> ta = TextAlignment(alignment=[TokenAlignment(source="a", target="b")])
        >>> ensure_text_alignment(ta) is ta
        True
        >>> # From TextAlignmentSchema
        >>> schema = TextAlignmentSchema(alignment=[TokenAlignment(source="a", target="b")])
        >>> result = ensure_text_alignment(schema)
        >>> isinstance(result, TextAlignment)
        True
        >>> result.alignment[0].source
        'a'
        >>> # From dict
        >>> data = {"alignment": [{"source": "a", "target": "b"}]}
        >>> result = ensure_text_alignment(data)
        >>> isinstance(result, TextAlignment)
        True
        >>> len(result.alignment)
        1
        >>> # From JSON string
        >>> json_str = '{"alignment": [{"source": "a", "target": "b"}]}'
        >>> result = ensure_text_alignment(json_str)
        >>> isinstance(result, TextAlignment)
        True
        >>> # Invalid type raises TypeError
        >>> ensure_text_alignment(123)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: Cannot convert int to TextAlignment
        >>> ensure_text_alignment([1, 2, 3])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: Cannot convert list to TextAlignment
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
    """Remove None values from dictionary, preserving other falsy values.

    Example:
        >>> filter_none_values({"a": 1, "b": None, "c": 2})
        {'a': 1, 'c': 2}
        >>> # Falsy values other than None are preserved
        >>> filter_none_values({"a": 0, "b": "", "c": False, "d": None, "e": []})
        {'a': 0, 'b': '', 'c': False, 'e': []}
        >>> # Empty dict
        >>> filter_none_values({})
        {}
        >>> # All None values
        >>> filter_none_values({"a": None, "b": None})
        {}
    """
    return {k: v for k, v in d.items() if v is not None}


def batch_iterable(iterable, n: int):
    """Batch an iterable into chunks of size n.

    For Python 3.12+, uses itertools.batched; otherwise provides fallback.

    Example:
        >>> batch_iterable([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
        >>> # Uneven division
        >>> batch_iterable([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
        >>> # Size 1
        >>> batch_iterable([1, 2, 3], 1)
        [[1], [2], [3]]
        >>> # Batch size larger than list
        >>> batch_iterable([1, 2, 3], 10)
        [[1, 2, 3]]
        >>> # Empty list
        >>> batch_iterable([], 5)
        []
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


def redact_for_logging(obj: Any, max_string_length: int = 1000) -> Any:
    """Redact sensitive information from objects for safe logging.

    Recursively processes dicts, lists, and tuples to redact sensitive keys
    and truncate long strings. Sensitive keys include api_key, authorization,
    access_token, token, messages, and headers (case-insensitive).

    Args:
        obj: Object to redact
        max_string_length: Maximum string length before truncation (default: 1000)

    Returns:
        Redacted copy of the object

    Example:
        >>> redact_for_logging({"api_key": "secret123", "model": "gpt-4"})
        {'api_key': '<redacted>', 'model': 'gpt-4'}
        >>> redact_for_logging({"messages": [{"role": "user", "content": "hi"}]})
        {'messages': '<redacted 1 messages>'}
        >>> redact_for_logging({"text": "a" * 2000})  # doctest: +ELLIPSIS
        {'text': 'aaa...'}
        >>> # Nested dict with mixed sensitive/non-sensitive keys
        >>> redact_for_logging({"config": {"api_key": "secret", "model": "gpt-4"}})
        {'config': {'api_key': '<redacted>', 'model': 'gpt-4'}}
        >>> # Headers redaction
        >>> redact_for_logging({"headers": {"Authorization": "Bearer token123"}})
        {'headers': '<redacted>'}
        >>> # List of dicts with sensitive keys
        >>> redact_for_logging([{"token": "abc123"}, {"data": "ok"}])
        [{'token': '<redacted>'}, {'data': 'ok'}]
    """
    sensitive_keys = {
        "api_key",
        "authorization",
        "access_token",
        "token",
        "messages",
        "headers",
    }

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            key_lower = key.lower()
            if key_lower in sensitive_keys:
                if key_lower == "messages" and isinstance(value, list):
                    result[key] = f"<redacted {len(value)} messages>"
                else:
                    result[key] = "<redacted>"
            else:
                result[key] = redact_for_logging(value, max_string_length)
        return result
    elif isinstance(obj, list):
        return [redact_for_logging(item, max_string_length) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(redact_for_logging(item, max_string_length) for item in obj)
    elif isinstance(obj, str):
        if len(obj) > max_string_length:
            truncated_chars = len(obj) - max_string_length
            return obj[:max_string_length] + f"... [truncated {truncated_chars} chars]"
        return obj
    else:
        return obj
