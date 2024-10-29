import re


def remove_unique_one(token: str) -> str:
    """Remove subscript numbers from a token.

    Args:
        token: Token possibly containing subscript numbers

    Returns:
        Token with subscript numbers removed

    Example:
        >>> remove_unique_one("cat₁")
        'cat'
        >>> remove_unique_one("the₂")
        'the'
        >>> remove_unique_one("normal")
        'normal'
    """
    return re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+$", "", token)
