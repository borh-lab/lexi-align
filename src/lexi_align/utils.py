from pydantic import BaseModel
from typing import Union
import re
from lexi_align.models import TextAlignment, TokenAlignment
from lexi_align.text_processing import remove_unique_one
from logging import getLogger

logger = getLogger(__name__)


class SystemMessage:
    def __init__(self, content: str):
        self.role = "system"
        self.content = content


class UserMessage:
    def __init__(self, content: Union[str, BaseModel]):
        self.role = "user"
        if isinstance(content, BaseModel):  # Compact output to save on tokens
            self.content = content.model_dump_json(indent=None)
        else:
            self.content = content


class AssistantMessage:
    def __init__(self, content: Union[str, BaseModel]):
        self.role = "assistant"
        if isinstance(content, BaseModel):  # Compact output to save on tokens
            self.content = content.model_dump_json(indent=None)
        else:
            self.content = content


def format_messages(*messages):
    # Handle both individual messages and lists of messages
    if len(messages) == 1 and isinstance(messages[0], list):
        messages = messages[0]
    return [{"role": msg.role, "content": msg.content} for msg in messages]


STRIP_RE = re.compile(r"[^\w\s']")


def strip_punctuation(s: str) -> str:
    """Remove punctuation from a string, keeping spaces and apostrophes.

    Args:
        s: Input string

    Returns:
        String with punctuation removed

    Example:
        >>> strip_punctuation("Hello, world!")
        'Hello world'
        >>> strip_punctuation("don't")
        "don't"
        >>> strip_punctuation("«quoted»")
        'quoted'
    """
    return STRIP_RE.sub("", s)
    # return re.sub(r"[^A-Za-zぁ-ゟァ-ヿ一-鿿 ]+", "", s)


def remove_unique(tokens: list[str]) -> list[str]:
    """Remove subscript numbers from all tokens.

    Args:
        tokens: List of tokens

    Returns:
        List of tokens with subscript numbers removed

    Example:
        >>> remove_unique(["cat₁", "the₂", "normal"])
        ['cat', 'the', 'normal']
    """
    return [remove_unique_one(token) for token in tokens]


def make_unique(names: list[str]) -> list[str]:
    """Add subscript numbers to disambiguate repeated tokens.

    Args:
        names: List of tokens

    Returns:
        List of tokens with subscript numbers added to duplicates

    Example:
        >>> make_unique(["the", "cat", "the", "mat"])
        ['the₁', 'cat', 'the₂', 'mat']
    """
    if not isinstance(names, list):
        raise TypeError("Input must be a list")

    for name in names:
        if not isinstance(name, str):
            raise TypeError("All tokens must be strings")

    # Strip existing subscripts and count base tokens
    base_tokens = [remove_unique_one(name) for name in names]
    base_counts: dict[str, int] = {}
    base_seen: dict[str, int] = {}
    unique_names = []
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"

    def to_subscript(num):
        return "".join(subscript_digits[int(digit)] for digit in str(num))

    # First pass: count base token occurrences
    for base_token in base_tokens:
        base_counts[base_token] = base_counts.get(base_token, 0) + 1

    # Second pass: add subscripts
    for i, base_token in enumerate(base_tokens):
        if base_counts[base_token] > 1:
            count = base_seen.get(base_token, 0) + 1
            base_seen[base_token] = count
            unique_names.append(f"{base_token}{to_subscript(count)}")
        else:
            unique_names.append(base_token)
    return unique_names


def export_pharaoh_format(
    source_tokens: list[str],
    target_tokens: list[str],
    alignment: TextAlignment,
    sep: str = "\t",
) -> str:
    """Export alignment data in Pharaoh format.

    Args:
        source_tokens: Pre-tokenized source text as list of strings
        target_tokens: Pre-tokenized target text as list of strings
        alignment: TextAlignment object containing the token alignments
        sep: Separator character for Pharaoh format fields (default: tab)

    Returns:
        String in Pharaoh format: "source target alignments"
    """
    # Create unique versions of tokens
    unique_source = make_unique(source_tokens)
    unique_target = make_unique(target_tokens)

    # Create mapping of tokens to their positions
    source_positions = {token: i for i, token in enumerate(unique_source)}
    target_positions = {token: i for i, token in enumerate(unique_target)}

    # Process alignments
    alignment_pairs: list[tuple[int, int]] = []
    for align in alignment.alignment:
        # Get base tokens and find their uniquified versions
        s_token = align.source_token
        t_token = align.target_token

        # If tokens aren't already uniquified, they'll be in the positions dict
        # If they are uniquified, they should match entries in the positions dict
        if s_token in source_positions:
            s_pos = source_positions[s_token]
        else:
            # Handle already uniquified tokens
            s_pos = source_positions.get(
                s_token, source_positions[remove_unique_one(s_token)]
            )

        if t_token in target_positions:
            t_pos = target_positions[t_token]
        else:
            t_pos = target_positions.get(
                t_token, target_positions[remove_unique_one(t_token)]
            )

        alignment_pairs.append((s_pos, t_pos))

    # Sort alignment pairs
    alignment_pairs.sort()
    alignment_str = " ".join(f"{s}-{t}" for s, t in alignment_pairs)

    # Join tokens into sentences using uniquified versions
    source_sentence = " ".join(unique_source)
    target_sentence = " ".join(unique_target)

    return f"{source_sentence}{sep}{target_sentence}{sep}{alignment_str}"


def parse_pharaoh_format(line: str) -> tuple[str, str, TextAlignment]:
    """Parse a line in Pharaoh format.

    Args:
        line: Tab-separated line in Pharaoh format

    Returns:
        Tuple of (source_sentence, target_sentence, TextAlignment)
    """
    try:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            raise ValueError("Input must have exactly 3 tab-separated parts")

        source_sentence, target_sentence, alignments = parts

        # Split sentences into tokens
        source_tokens = source_sentence.split()
        target_tokens = target_sentence.split()

        # Create unique versions of tokens
        unique_source = make_unique(source_tokens)
        unique_target = make_unique(target_tokens)

        # Parse alignments and create TokenAlignment objects directly with unique tokens
        alignment_list = []
        for align_pair in alignments.split():
            s_idx, t_idx = map(int, align_pair.split("-"))
            if s_idx >= len(source_tokens) or t_idx >= len(target_tokens):
                raise ValueError(f"Alignment indices {s_idx}-{t_idx} out of bounds")
            alignment_list.append(
                TokenAlignment(
                    source_token=unique_source[s_idx], target_token=unique_target[t_idx]
                )
            )

        # Create TextAlignment object
        text_alignment = TextAlignment(alignment=alignment_list)

        # Verify alignment by round-tripping through export_pharaoh_format
        _ = export_pharaoh_format(source_tokens, target_tokens, text_alignment)

        # Return the original sentences and alignment
        return source_sentence, target_sentence, text_alignment
    except Exception as e:
        raise ValueError(f"Failed to parse Pharaoh format: {str(e)}") from e


def read_pharaoh_file(filepath: str) -> list[tuple[str, str, TextAlignment]]:
    """Read alignments from a file in Pharaoh format.

    Args:
        filepath: Path to input file

    Returns:
        List of (source_sentence, target_sentence, TextAlignment) tuples

    Example:
        >>> # Create a temporary file for testing
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        ...     _ = f.write("the cat\\tle chat\\t0-0 1-1\\n")
        ...     _ = f.write("invalid line\\n")  # This line will be skipped
        ...     filepath = f.name
        >>> alignments = read_pharaoh_file(filepath)
        >>> import os; os.unlink(filepath)  # Clean up
        >>> len(alignments)
        1
        >>> source, target, align = alignments[0]
        >>> source
        'the cat'
    """
    alignments = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                alignments.append(parse_pharaoh_format(line))
            except ValueError as e:
                logger.warning(f"Skipping line {line_num} due to error: {e}")
    return alignments


def write_pharaoh_file(
    filepath: str, alignments: list[tuple[str, str, TextAlignment]]
) -> None:
    """Write alignments to a file in Pharaoh format.

    Args:
        filepath: Path to output file
        alignments: List of alignment tuples

    Example:
        >>> # Create test data
        >>> align_data = [
        ...     ("the cat", "le chat",
        ...      TextAlignment(alignment=[
        ...          TokenAlignment(source_token="the", target_token="le"),
        ...          TokenAlignment(source_token="cat", target_token="chat")
        ...      ]))
        ... ]
        >>> # Write to temporary file
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        ...     filepath = f.name
        >>> write_pharaoh_file(filepath, align_data)
        >>> # Verify contents
        >>> with open(filepath) as f:
        ...     print(f.read().strip())
        the cat	le chat	0-0 1-1
        >>> import os; os.unlink(filepath)  # Clean up
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for source, target, alignment in alignments:
            try:
                # Convert source and target strings to token lists
                source_tokens = source.split()
                target_tokens = target.split()

                line = export_pharaoh_format(source_tokens, target_tokens, alignment)
                f.write(line + "\n")
            except Exception as e:
                logger.warning(f"Failed to write alignment: {e}")
