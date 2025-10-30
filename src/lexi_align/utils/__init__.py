import re
import unicodedata as ud
from logging import getLogger
from typing import Any, List, Literal, Optional, Tuple, Type, Union, cast

from llm_schema_lite import loads as llm_loads
from pydantic import BaseModel

from lexi_align.constants import (
    REMAINING_SOURCE_PREFIX,
    REMAINING_TARGET_PREFIX,
    SOURCE_TOKENS_PREFIX,
    TARGET_TOKENS_PREFIX,
    UNALIGNED_MARKER,
)
from lexi_align.models import (
    ChatMessageDict as ModelChatMessageDict,
)
from lexi_align.models import (
    TextAlignment,
    TextAlignmentSchema,
    TokenAlignment,
    TokenMapping,
    calculate_max_alignments,
    create_dynamic_alignment_schema,
    create_token_mapping,
    make_unique,
)
from lexi_align.text_processing import (
    MarkerGenerator,
    create_subscript_generator,
    remove_unique_one,
)

from .common import get_default_marker_generator

logger = getLogger(__name__)


class SystemMessage:
    role: Literal["system"]
    content: str

    def __init__(self, content: str):
        self.role = "system"
        self.content = content


class UserMessage:
    role: Literal["user"]
    content: Union[str, BaseModel]

    def __init__(self, content: Union[str, BaseModel]):
        self.role = "user"
        if isinstance(content, BaseModel):  # Compact output to save on tokens
            self.content = content.model_dump_json(indent=None)
        else:
            self.content = content


class AssistantMessage:
    role: Literal["assistant"]
    content: Union[str, BaseModel]

    def __init__(self, content: Union[str, BaseModel]):
        self.role = "assistant"
        if isinstance(content, BaseModel):  # Compact output to save on tokens
            self.content = content.model_dump_json(indent=None)
        else:
            self.content = content


Message = Union[SystemMessage, UserMessage, AssistantMessage]


ChatMessageDict = ModelChatMessageDict  # re-export canonical type


def to_text_alignment(obj: Any) -> TextAlignment:
    """Normalize various LLM outputs to a TextAlignment instance.

    Args:
        obj: A TextAlignment, TextAlignmentSchema, JSON string, or dict

    Returns:
        TextAlignment

    Example:
        >>> from lexi_align.models import TokenAlignment
        >>> ta = TextAlignment(alignment=[TokenAlignment(source="a", target="b")])
        >>> to_text_alignment(ta) is ta
        True
        >>> schema = TextAlignmentSchema(alignment=[TokenAlignment(source="a", target="b")])
        >>> to_text_alignment(schema).alignment[0].source
        'a'
    """
    if isinstance(obj, TextAlignment):
        return obj
    if isinstance(obj, TextAlignmentSchema):
        return TextAlignment(alignment=obj.alignment)
    if isinstance(obj, dict):
        return TextAlignment.model_validate(obj, strict=True)
    if isinstance(obj, str):
        data = llm_loads(obj, mode="json")
        return TextAlignment.model_validate(data, strict=True)
    raise TypeError(f"Cannot convert object of type {type(obj)} to TextAlignment")


def format_messages(*messages: Any) -> list[ChatMessageDict]:
    """Normalize chat messages (SystemMessage/UserMessage/AssistantMessage or list thereof) to dicts.

    Example:
        >>> msgs = format_messages(SystemMessage("sys"), UserMessage("hi"))
        >>> msgs[0]["role"], msgs[1]["content"]
        ('system', 'hi')
    """
    # Handle both individual messages and lists of messages
    message_list: list[Any]
    if len(messages) == 1 and isinstance(messages[0], list):
        message_list = messages[0]
    else:
        message_list = list(messages)
    out: list[ChatMessageDict] = []
    for msg in message_list:
        if isinstance(msg, (SystemMessage, UserMessage, AssistantMessage)):
            content = msg.content
            if isinstance(content, BaseModel):
                content = content.model_dump_json(indent=None)
            role_lit = cast(Literal["system", "user", "assistant"], msg.role)
            out.append({"role": role_lit, "content": content})
        elif isinstance(msg, dict):
            role_val = msg.get("role", "user")
            if role_val not in ("system", "user", "assistant"):
                role_val = "user"
            role = cast(Literal["system", "user", "assistant"], role_val)
            content = msg.get("content", "")
            if isinstance(content, BaseModel):
                content = content.model_dump_json(indent=None)
            if not isinstance(content, str):
                content = str(content)
            out.append({"role": role, "content": content})
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")
    return out


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


def extract_tokens_and_retry_flag(
    messages: list[dict],
) -> tuple[list[str], list[str], bool]:
    """Extract source/target tokens and retry flag from messages.

    Returns:
        (source_tokens, target_tokens, is_retry)

    Example:
        >>> msgs = [
        ...     {"role": "system", "content": "x"},
        ...     {"role": "user", "content": "source_tokens: the cat\\ntarget_tokens: le chat"},
        ... ]
        >>> extract_tokens_and_retry_flag(msgs)
        (['the', 'cat'], ['le', 'chat'], False)
        >>> msgs.append({"role": "user", "content": "Here are partial alignments:\\n{\\"alignment\\": []}\\nPlease provide alignments for the remaining tokens:"})
        >>> extract_tokens_and_retry_flag(msgs)[2]
        True
    """
    source_tokens: list[str] = []
    target_tokens: list[str] = []
    is_retry = False

    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue

        if "Please provide alignments for the remaining tokens:" in content:
            is_retry = True

        if not source_tokens or not target_tokens:
            lines = [ln.strip() for ln in content.splitlines()]
            st = next((ln for ln in lines if ln.startswith(SOURCE_TOKENS_PREFIX)), None)
            tt = next((ln for ln in lines if ln.startswith(TARGET_TOKENS_PREFIX)), None)
            if st and tt:
                # Add validation for split operation
                st_parts = st.split(SOURCE_TOKENS_PREFIX, 1)
                tt_parts = tt.split(TARGET_TOKENS_PREFIX, 1)

                if len(st_parts) == 2 and st_parts[1].strip():
                    source_tokens = st_parts[1].split()
                else:
                    logger.warning(f"Invalid source token format: {st}")

                if len(tt_parts) == 2 and tt_parts[1].strip():
                    target_tokens = tt_parts[1].split()
                else:
                    logger.warning(f"Invalid target token format: {tt}")

        if source_tokens and target_tokens and is_retry:
            break

    if not source_tokens or not target_tokens:
        raise ValueError("Could not find original source and target tokens in messages")

    return source_tokens, target_tokens, is_retry


def extract_existing_alignments_from_messages(
    messages: list[dict],
) -> Optional[list[TokenAlignment]]:
    """Extract existing valid alignments from a retry message if present.

    Looks for a JSON block starting at the first '{' after the line
    'Here are partial alignments:'.

    Example:
        >>> msgs = [{"role": "user", "content": "Here are partial alignments:\\n{\\"alignment\\": [{\\"source\\": \\"a\\", \\"target\\": \\"b\\"}]}" }]
        >>> aligns = extract_existing_alignments_from_messages(msgs)
        >>> [(a.source, a.target) for a in (aligns or [])]
        [('a', 'b')]
    """
    pattern = re.compile(
        r"Here are partial alignments:\s*(\{.*?\})(?:\s*Please provide alignments for the remaining tokens:|\s*$)",
        re.S,
    )
    for message in reversed(messages):
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        m = pattern.search(content)
        if not m:
            # Fallback: locate cue and try balanced-braces extraction
            cue = "Here are partial alignments:"
            idx = content.find(cue)
            if idx != -1:
                brace_start = content.find("{", idx)
                if brace_start != -1:
                    depth = 0
                    end = -1
                    for i, ch in enumerate(content[brace_start:], start=brace_start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if end != -1:
                        candidate = content[brace_start:end]
                        try:
                            data = llm_loads(candidate, mode="json")
                            ta = TextAlignment.model_validate(data, strict=True)
                            return ta.alignment
                        except Exception:
                            pass
                # Fallback 2: fenced code block ```json ... ```
                fence_start = content.find("```", idx)
                if fence_start != -1:
                    fence_end = content.find("```", fence_start + 3)
                    if fence_end != -1:
                        block = content[fence_start + 3 : fence_end].strip()
                        # Strip optional language tag (e.g., 'json')
                        first_nl = block.find("\n")
                        if first_nl > 0 and block[:first_nl].strip().lower() in {
                            "json",
                            "jsonc",
                        }:
                            block = block[first_nl + 1 :].strip()
                        try:
                            data = llm_loads(block, mode="json")
                            ta = TextAlignment.model_validate(data, strict=True)
                            return ta.alignment
                        except Exception:
                            pass
            continue
        try:
            data = llm_loads(m.group(1), mode="json")
            ta = TextAlignment.model_validate(data, strict=True)
            return ta.alignment
        except Exception:
            return None
    return None


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
    return "".join(ch for ch in s if not ud.category(ch).startswith("P") or ch == "'")


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
    marker_generator = get_default_marker_generator()
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
        marker_pattern = get_default_marker_generator().pattern
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

    # build errors via list‐comprehensions
    errors = [
        f"Source token '{t}' not found in mapping"
        for t in source_tokens
        if t != UNALIGNED_MARKER and source_mapping.get_position(t) == -1
    ] + [
        f"Target token '{t}' not found in mapping"
        for t in target_tokens
        if t != UNALIGNED_MARKER and target_mapping.get_position(t) == -1
    ]
    return (not errors, errors)


def extract_remaining_tokens_from_messages(
    messages: list[dict],
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Extract remaining source/target tokens listed in retry prompts, if present.

    Returns:
        (remaining_source_tokens, remaining_target_tokens) or (None, None) if not found.

    Example:
        >>> msgs = [
        ...     {"role": "user", "content": "remaining_source_tokens: the₁ cat\\nremaining_target_tokens: le₁ chat"}
        ... ]
        >>> extract_remaining_tokens_from_messages(msgs)
        (['the₁', 'cat'], ['le₁', 'chat'])
        >>> msgs = [{"role": "user", "content": "remaining_source_tokens: the₁"}]
        >>> extract_remaining_tokens_from_messages(msgs)
        (['the₁'], None)
    """
    rem_src: Optional[list[str]] = None
    rem_tgt: Optional[list[str]] = None
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        lines = [ln.strip() for ln in content.splitlines()]
        st = next((ln for ln in lines if ln.startswith(REMAINING_SOURCE_PREFIX)), None)
        tt = next((ln for ln in lines if ln.startswith(REMAINING_TARGET_PREFIX)), None)

        # Add validation for split operations
        if st:
            st_parts = st.split(REMAINING_SOURCE_PREFIX, 1)
            if len(st_parts) == 2 and st_parts[1].strip():
                rem_src = st_parts[1].split()
            else:
                logger.warning(f"Invalid remaining source token format: {st}")

        if tt:
            tt_parts = tt.split(REMAINING_TARGET_PREFIX, 1)
            if len(tt_parts) == 2 and tt_parts[1].strip():
                rem_tgt = tt_parts[1].split()
            else:
                logger.warning(f"Invalid remaining target token format: {tt}")

        if rem_src is not None or rem_tgt is not None:
            break
    return rem_src, rem_tgt


def select_alignment_schema(
    source_tokens: list[str],
    target_tokens: list[str],
    min_alignments: int = 0,
    is_retry: bool = False,
    existing_alignments: Optional[List[TokenAlignment]] = None,
    remaining_source_tokens: Optional[list[str]] = None,
    remaining_target_tokens: Optional[list[str]] = None,
    use_reasoning: bool = False,
) -> Type[TextAlignmentSchema]:
    """
    Choose a dynamic alignment schema with retry-aware max_length and an
    effective minimum length based on token counts and already-aligned items.

    On first attempt the default minimum is max(#source_tokens, #target_tokens).
    On retry, if existing_alignments is provided, the default minimum is the
    number of remaining source/target tokens that still need alignment.
    If the caller provides a positive min_alignments, it is honored; otherwise
    the computed default is used. The effective minimum is clamped to [0, max_length].
    """
    # Determine "remaining" tokens. Prefer explicit lists from the retry prompt,
    # else fall back to subtracting already-used tokens from existing_alignments.
    if is_retry:
        if remaining_source_tokens is not None or remaining_target_tokens is not None:
            rem_src = list(remaining_source_tokens or [])
            rem_tgt = list(remaining_target_tokens or [])
        elif existing_alignments:
            aligned_source = {
                a.source for a in existing_alignments if a.source != UNALIGNED_MARKER
            }
            aligned_target = {
                a.target for a in existing_alignments if a.target != UNALIGNED_MARKER
            }
            # Compute remaining tokens deterministically (preserve order)
            rem_src = [t for t in source_tokens if t not in aligned_source]
            rem_tgt = [t for t in target_tokens if t not in aligned_target]
        else:
            rem_src = list(source_tokens)
            rem_tgt = list(target_tokens)
        if len(rem_src) == 0 and len(rem_tgt) == 0:
            # Nothing left to align: force empty alignment
            max_len = 0
            default_min = 0
        else:
            max_len = calculate_max_alignments(rem_src, rem_tgt)
            default_min = max(len(rem_src), len(rem_tgt))
    else:
        rem_src = list(source_tokens)
        rem_tgt = list(target_tokens)
        max_len = calculate_max_alignments(source_tokens, target_tokens)
        default_min = max(len(source_tokens), len(target_tokens))

    # If caller provided a positive min_alignments, honor it; otherwise use default_min
    eff_min = (
        int(min_alignments) if (min_alignments and min_alignments > 0) else default_min
    )
    # Clamp to [0, max_len]
    eff_min = min(max(0, eff_min), max_len)

    # Always use full token lists for schema - the prompt will guide which tokens to align
    enum_src = source_tokens
    enum_tgt = target_tokens

    return create_dynamic_alignment_schema(
        enum_src,
        enum_tgt,
        min_length=eff_min,
        max_length=max_len,
        tokens_are_uniquified=True,  # tokens from messages are already uniquified
        use_reasoning=use_reasoning,
    )


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
    # Get marker generator and mappings
    marker_generator = create_subscript_generator()
    src_map = create_token_mapping(source_tokens, marker_generator)
    tgt_map = create_token_mapping(target_tokens, marker_generator)

    # Process alignments using mappings
    alignment_pairs: list[tuple[int, int]] = []
    for pair in alignment.alignment:
        try:
            s_pos = src_map.get_position(pair.source)
            t_pos = tgt_map.get_position(pair.target)
            if s_pos < 0:
                logger.warning(
                    f"Source token '{pair.source}' not found in source tokens"
                )
                continue
            if t_pos < 0:
                logger.warning(
                    f"Target token '{pair.target}' not found in target tokens"
                )
                continue
            alignment_pairs.append((s_pos, t_pos))
        except Exception as e:
            logger.warning(f"Error processing alignment {pair}: {e}")
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


def parse_case_line(
    line: str,
    marker_generator: Optional[MarkerGenerator] = None,
) -> tuple[list[str], list[str], TextAlignment, TokenMapping, TokenMapping]:
    """
    Parse a Pharaoh-format line and return token lists, gold alignment, and mappings.
    """
    src_sent, tgt_sent, gold = parse_pharaoh_format(line)
    src_tokens = src_sent.split()
    tgt_tokens = tgt_sent.split()
    if marker_generator is None:
        marker_generator = create_subscript_generator()
    src_map = create_token_mapping(src_tokens, marker_generator)
    tgt_map = create_token_mapping(tgt_tokens, marker_generator)
    return src_tokens, tgt_tokens, gold, src_map, tgt_map


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
