import re
from logging import getLogger
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel

from lexi_align.models import (
    TextAlignment,
    TokenAlignment,
    TokenMapping,
    create_token_mapping,
    make_unique,
)
from lexi_align.text_processing import (
    create_subscript_generator,
    remove_unique_one,
)

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


Message = Union[SystemMessage, UserMessage, AssistantMessage]


def format_messages(*messages) -> list[dict[str, str]]:
    # Handle both individual messages and lists of messages
    message_list: list[Any]
    if len(messages) == 1 and isinstance(messages[0], list):
        message_list = messages[0]
    else:
        message_list = list(messages)  # Convert tuple to list
    return [{"role": msg.role, "content": msg.content} for msg in message_list]


def format_tokens(source_tokens: list[str], target_tokens: list[str]) -> str:
    """Format source and target tokens for the LLM prompt.

    Args:
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens

    Returns:
        Formatted string with source and target tokens

    Example:
        >>> format_tokens(["the", "cat"], ["le", "chat"])
        'Source tokens: the cat\\nTarget tokens: le chat'
    """
    return (
        f"Source tokens: {' '.join(make_unique(source_tokens))}\n"
        f"Target tokens: {' '.join(make_unique(target_tokens))}"
    )


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
    marker_generator = create_subscript_generator()  # Get default marker generator
    return [remove_unique_one(token, marker_generator.pattern) for token in tokens]


def normalize_tokens(
    tokens: List[str], marker_pattern: Optional[re.Pattern] = None
) -> List[str]:
    """Remove markers from tokens.

    Args:
        tokens: List of tokens to normalize
        marker_pattern: Optional regex pattern for markers (defaults to subscript)

    Returns:
        List of tokens with markers removed

    Example:
        >>> tokens = ['the₁', 'cat', 'the₂', 'mat']
        >>> normalize_tokens(tokens)
        ['the', 'cat', 'the', 'mat']
        >>> # With custom marker pattern
        >>> import re
        >>> pattern = re.compile(r'_\\d+$')
        >>> normalize_tokens(['the_1', 'cat', 'the_2'], pattern)
        ['the', 'cat', 'the']
    """
    if marker_pattern is None:
        marker_pattern = create_subscript_generator().pattern
    return [remove_unique_one(token, marker_pattern) for token in tokens]


def validate_token_lists(
    source_tokens: List[str],
    target_tokens: List[str],
    source_mapping: TokenMapping,
    target_mapping: TokenMapping,
) -> Tuple[bool, List[str]]:
    """Validate that token lists are consistent with their mappings.

    Args:
        source_tokens: Source language tokens
        target_tokens: Target language tokens
        source_mapping: TokenMapping for source tokens
        target_mapping: TokenMapping for target tokens

    Returns:
        Tuple of (is_valid, error_messages)

    Example:
        >>> source = ["the", "cat", "the"]
        >>> target = ["le", "chat", "le"]
        >>> source_map = create_token_mapping(source)
        >>> target_map = create_token_mapping(target)
        >>> # Test with valid tokens
        >>> valid, errors = validate_token_lists(
        ...     ['the₁', 'cat', 'the₂'],
        ...     ['le₁', 'chat', 'le₂'],
        ...     source_map,
        ...     target_map
        ... )
        >>> valid
        True
        >>> len(errors)
        0
        >>> # Test with invalid tokens
        >>> valid, errors = validate_token_lists(
        ...     ['the₁', 'dog', 'the₂'],  # 'dog' is not in mapping
        ...     ['le₁', 'chat', 'le₂'],
        ...     source_map,
        ...     target_map
        ... )
        >>> valid
        False
        >>> errors  # doctest: +NORMALIZE_WHITESPACE
        ["Source token 'dog' not found in mapping"]
    """
    errors = []

    # Check source tokens
    for token in source_tokens:
        if source_mapping.get_position(token) == -1:
            errors.append(f"Source token '{token}' not found in mapping")

    # Check target tokens
    for token in target_tokens:
        if target_mapping.get_position(token) == -1:
            errors.append(f"Target token '{token}' not found in mapping")

    return len(errors) == 0, errors


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
        String in Pharaoh format: "source target alignments" with custom separator
    """
    # Get default marker generator
    marker_generator = create_subscript_generator()

    # Create unique versions of tokens
    unique_source = make_unique(source_tokens)
    unique_target = make_unique(target_tokens)

    # Create mapping of tokens to their positions
    source_positions = {token: i for i, token in enumerate(unique_source)}
    target_positions = {token: i for i, token in enumerate(unique_target)}

    # Also create mapping for base tokens to handle non-uniquified tokens
    base_source_positions = {
        remove_unique_one(token, marker_generator.pattern): i
        for i, token in enumerate(source_tokens)
    }
    base_target_positions = {
        remove_unique_one(token, marker_generator.pattern): i
        for i, token in enumerate(target_tokens)
    }

    # Process alignments
    alignment_pairs: list[tuple[int, int]] = []
    for align in alignment.alignment:
        try:
            s_token = align.source_token
            t_token = align.target_token

            # Try to get position, first from uniquified tokens, then from base tokens
            try:
                s_pos = source_positions[s_token]
            except KeyError:
                base_s_token = remove_unique_one(s_token, marker_generator.pattern)
                if base_s_token not in base_source_positions:
                    logger.warning(
                        f"Source token '{s_token}' not found in source tokens"
                    )
                    continue
                s_pos = base_source_positions[base_s_token]

            try:
                t_pos = target_positions[t_token]
            except KeyError:
                base_t_token = remove_unique_one(t_token, marker_generator.pattern)
                if base_t_token not in base_target_positions:
                    logger.warning(
                        f"Target token '{t_token}' not found in target tokens"
                    )
                    continue
                t_pos = base_target_positions[base_t_token]

            alignment_pairs.append((s_pos, t_pos))
        except Exception as e:
            logger.warning(f"Error processing alignment {align}: {e}")
            continue

    # Sort alignment pairs
    alignment_pairs.sort()
    alignment_str = " ".join(f"{s}-{t}" for s, t in alignment_pairs)

    # Join tokens into sentences using original tokens
    source_sentence = " ".join(source_tokens)
    target_sentence = " ".join(target_tokens)

    return f"{source_sentence}{sep}{target_sentence}{sep}{alignment_str}"


def parse_pharaoh_format(line: str, sep: str = "\t") -> tuple[str, str, TextAlignment]:
    """Parse a line in Pharaoh format.

    Args:
        line: Separator-delimited line in Pharaoh format
        sep: Separator character (default: tab)

    Returns:
        Tuple of (source_sentence, target_sentence, TextAlignment)
    """
    try:
        parts = line.strip().split(sep)
        if len(parts) != 3:
            raise ValueError(f"Input must have exactly 3 {sep}-separated parts")

        source_sentence, target_sentence, alignments = parts

        # Split sentences into tokens
        source_tokens = source_sentence.split()
        target_tokens = target_sentence.split()

        # Create marker generator
        marker_generator = create_subscript_generator()

        # Create mappings for source and target tokens
        source_mapping = create_token_mapping(source_tokens, marker_generator)
        target_mapping = create_token_mapping(target_tokens, marker_generator)

        # Create unique versions of tokens directly
        unique_source = make_unique(source_tokens)
        unique_target = make_unique(target_tokens)

        # Parse alignments and create TokenAlignment objects directly with unique tokens
        alignment_list = []
        for align_pair in alignments.split():
            s_idx, t_idx = map(int, align_pair.split("-"))
            if s_idx >= len(source_tokens) or t_idx >= len(target_tokens):
                raise ValueError(f"Alignment indices {s_idx}-{t_idx} out of bounds")
            alignment_list.append(
                TokenAlignment(source=unique_source[s_idx], target=unique_target[t_idx])
            )

        # Create sorted alignment
        text_alignment = TextAlignment(
            alignment=alignment_list,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
        )

        # Verify alignment by round-tripping through export_pharaoh_format
        _ = export_pharaoh_format(source_tokens, target_tokens, text_alignment)

        # Return the original sentences and alignment
        return source_sentence, target_sentence, text_alignment
    except Exception as e:
        raise ValueError(f"Failed to parse Pharaoh format: {str(e)}") from e


def read_pharaoh_file(
    filepath: str, sep: str = "\t"
) -> list[tuple[str, str, TextAlignment]]:
    """Read alignments from a file in Pharaoh format.

    Args:
        filepath: Path to input file
        sep: Separator character (default: tab)

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
                alignments.append(parse_pharaoh_format(line, sep=sep))
            except ValueError as e:
                logger.warning(f"Skipping line {line_num} due to error: {e}")
    return alignments


def write_pharaoh_file(
    filepath: str, alignments: list[tuple[str, str, TextAlignment]], sep: str = "\t"
) -> None:
    """Write alignments to a file in Pharaoh format.

    Args:
        filepath: Path to output file
        alignments: List of alignment tuples
        sep: Separator character (default: tab)

    Example:
        >>> # Create test data
        >>> source = ["the", "cat"]
        >>> target = ["le", "chat"]
        >>> alignments = [
        ...     TokenAlignment(source="the", target="le"),
        ...     TokenAlignment(source="cat", target="chat")
        ... ]
        >>> align = TextAlignment.from_token_alignments(alignments, source, target)
        >>> align_data = [(" ".join(source), " ".join(target), align)]
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

                line = export_pharaoh_format(
                    source_tokens, target_tokens, alignment, sep=sep
                )
                f.write(line + "\n")
            except Exception as e:
                logger.warning(f"Failed to write alignment: {e}")
