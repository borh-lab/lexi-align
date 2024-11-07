import re
from typing import Callable, Pattern
from dataclasses import dataclass


@dataclass
class MarkerGenerator:
    """Pairs a marker generation function with its removal pattern."""

    generate: Callable[[int], str]
    pattern: Pattern[str]


def create_subscript_generator() -> MarkerGenerator:
    """Create a generator for subscript number markers."""
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"

    def generator(num: int) -> str:
        return "".join(subscript_digits[int(digit)] for digit in str(num))

    return MarkerGenerator(generate=generator, pattern=re.compile(r"[₀₁₂₃₄₅₆₇₈₉]+$"))


def create_underscore_generator() -> MarkerGenerator:
    """Create a generator for underscore number markers."""
    return MarkerGenerator(generate=lambda n: f"_{n}", pattern=re.compile(r"_[0-9]+$"))


def remove_unique_one(token: str, marker_pattern: Pattern[str]) -> str:
    """Remove unique marker from a token using the specified pattern.

    Args:
        token: Input token possibly containing a unique marker
        marker_pattern: Regular expression pattern for removing the marker

    Returns:
        Token with unique marker removed

    Example:
        >>> pattern = create_subscript_generator().pattern
        >>> remove_unique_one("cat₁", pattern)
        'cat'
        >>> pattern = create_underscore_generator().pattern
        >>> remove_unique_one("dog_2", pattern)
        'dog'
    """
    return marker_pattern.sub("", token)
