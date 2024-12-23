import asyncio
import json
import logging
from logging import getLogger
from typing import (
    Any,
    Dict,
    List,
    LiteralString,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
)

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    UNALIGNED_MARKER,
    AlignmentAttempt,
    AlignmentResult,
    SpecialTokens,
    TextAlignment,
    TextAlignmentSchema,
    TokenAlignment,
    ValidationErrorType,
    create_dynamic_alignment_schema,
)
from lexi_align.text_processing import MarkerGenerator, create_subscript_generator
from lexi_align.utils import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
    create_token_mapping,
    format_messages,
    format_tokens,
    make_unique,
)

logging.basicConfig(level=logging.DEBUG)

logger = getLogger(__name__)


class ValidationErrorStats(TypedDict):
    count: int
    frequencies: Dict[str, int]


class DiagnosticsDict(TypedDict):
    total_attempts: int
    total_validation_errors: int
    avg_attempts_per_pair: float
    validation_error_stats: Dict[ValidationErrorType, ValidationErrorStats]
    exception_types: Dict[str, int]
    failed_calls: int
    failure_rate: float


class MetricsDict(TypedDict):
    precision: float
    recall: float
    f_measure: float
    aer: float
    total_predicted: int
    total_gold: int
    total_true_positives: int
    diagnostics: DiagnosticsDict


def categorize_validation_errors(
    errors: list[tuple[ValidationErrorType, str, list[str]]],
) -> dict[ValidationErrorType, ValidationErrorStats]:
    """Categorize and count validation errors.

    Args:
        errors: List of validation error tuples

    Returns:
        Dictionary mapping error types to statistics

    Example:
        >>> from lexi_align.models import ValidationErrorType
        >>> errors = [
        ...     (ValidationErrorType.INVALID_SOURCE_TOKEN, "Invalid token 'foo'", ["foo"]),
        ...     (ValidationErrorType.INVALID_SOURCE_TOKEN, "Invalid token 'bar'", ["bar"]),
        ...     (ValidationErrorType.MISSING_TARGET_ALIGNMENTS, "Missing target", ["le"])
        ... ]
        >>> stats = categorize_validation_errors(errors)
        >>> stats[ValidationErrorType.INVALID_SOURCE_TOKEN]["count"]
        2
        >>> stats[ValidationErrorType.INVALID_SOURCE_TOKEN]["frequencies"]["foo"]
        1
        >>> stats[ValidationErrorType.MISSING_TARGET_ALIGNMENTS]["count"]
        1
    """
    # Initialize with proper typing
    stats: dict[ValidationErrorType, ValidationErrorStats] = {
        error_type: {"count": 0, "frequencies": {}}
        for error_type in ValidationErrorType
    }

    for error_type, _, tokens in errors:
        stats[error_type]["count"] += 1
        # Update frequencies for each token
        for token in tokens:
            stats[error_type]["frequencies"][token] = (
                stats[error_type]["frequencies"].get(token, 0) + 1
            )

    return stats


def _validate_alignment(
    alignment: TextAlignment,
    source_tokens: list[str],
    target_tokens: list[str],
    marker_generator: Optional[MarkerGenerator] = None,
    existing_alignments: Optional[List[TokenAlignment]] = None,
) -> tuple[
    bool,
    list[tuple[ValidationErrorType, str, list[str]]],
    list[TokenAlignment],
    set[str],
    set[str],
]:
    """
    Validate alignment and extract valid alignments and remaining tokens.
    Now handles explicit unaligned tokens and improved error reporting.
    Returns tuple of:
    - is_valid: bool
    - errors: list of (error_type, description, affected_tokens)
    - valid_alignments: list of valid TokenAlignment objects
    - remaining_source: set of unaligned source tokens
    - remaining_target: set of unaligned target tokens
    """
    source_mapping = create_token_mapping(source_tokens, marker_generator)
    target_mapping = create_token_mapping(target_tokens, marker_generator)

    valid_alignments = list(existing_alignments) if existing_alignments else []
    errors: list[tuple[ValidationErrorType, str, list[str]]] = []

    # Get special tokens for validation
    special_tokens = {
        SpecialTokens.UNALIGNED.value,
        # SpecialTokens.SOURCE_SPECIFIC.value,
        # SpecialTokens.TARGET_SPECIFIC.value
    }

    # Track explicitly unaligned tokens
    explicitly_unaligned_source = set()
    explicitly_unaligned_target = set()

    # Track invalid tokens with improved handling
    invalid_source: list[str] = []
    invalid_target: list[str] = []

    # Validate each alignment pair
    for align in alignment.alignment:
        # Skip empty or whitespace-only tokens
        if not align.source or not align.source.strip():
            invalid_source.append("<empty>")
            continue
        if not align.target or not align.target.strip():
            invalid_target.append("<empty>")
            continue

        # Check for multi-token strings
        if len(align.source.split()) > 1:
            invalid_source.append(repr(align.source))
            continue
        if len(align.target.split()) > 1:
            invalid_target.append(repr(align.target))
            continue

        # Handle special token alignments
        if align.source in special_tokens or align.target in special_tokens:
            valid_alignments.append(align)
            if align.source == UNALIGNED_MARKER:
                explicitly_unaligned_target.add(align.target)
            elif align.target == UNALIGNED_MARKER:
                explicitly_unaligned_source.add(align.source)
            continue

        # Validate regular alignments
        s_valid = source_mapping.get_position(align.source) != -1
        t_valid = target_mapping.get_position(align.target) != -1

        if s_valid and t_valid:
            valid_alignments.append(align)
        else:
            if not s_valid:
                invalid_source.append(repr(align.source))
            if not t_valid:
                invalid_target.append(repr(align.target))

    # Helper function to format token counts
    def format_token_counts(tokens: list[str]) -> str:
        # Count occurrences of each token
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        # Format each token with count if > 1
        formatted = []
        for token, count in sorted(counts.items()):
            if count > 1:
                formatted.append(f"{token} (x{count})")
            else:
                formatted.append(token)

        return ", ".join(formatted)

    # Add error messages for invalid tokens with counts
    if invalid_source:
        errors.append(
            (
                ValidationErrorType.INVALID_SOURCE_TOKEN,
                f"Invalid source tokens: {format_token_counts(invalid_source)}",
                invalid_source,
            )
        )
    if invalid_target:
        errors.append(
            (
                ValidationErrorType.INVALID_TARGET_TOKEN,
                f"Invalid target tokens: {format_token_counts(invalid_target)}",
                invalid_target,
            )
        )

    # Calculate remaining tokens, excluding aligned and explicitly unaligned ones
    aligned_sources = {
        align.source for align in valid_alignments if align.source not in special_tokens
    }
    aligned_targets = {
        align.target for align in valid_alignments if align.target not in special_tokens
    }

    remaining_source = (
        set(source_mapping.uniquified) - aligned_sources - explicitly_unaligned_source
    )
    remaining_target = (
        set(target_mapping.uniquified) - aligned_targets - explicitly_unaligned_target
    )

    # Add validation errors
    if invalid_source:
        errors.append(
            (
                ValidationErrorType.INVALID_SOURCE_TOKEN,
                f"Invalid source tokens: {', '.join(invalid_source)}",
                invalid_source,
            )
        )
    if invalid_target:
        errors.append(
            (
                ValidationErrorType.INVALID_TARGET_TOKEN,
                f"Invalid target tokens: {', '.join(invalid_target)}",
                invalid_target,
            )
        )
    if remaining_source:
        errors.append(
            (
                ValidationErrorType.MISSING_SOURCE_ALIGNMENTS,
                f"Unaligned source tokens: {', '.join(remaining_source)}",
                list(remaining_source),
            )
        )
    if remaining_target:
        errors.append(
            (
                ValidationErrorType.MISSING_TARGET_ALIGNMENTS,
                f"Unaligned target tokens: {', '.join(remaining_target)}",
                list(remaining_target),
            )
        )

    # Consider alignment valid if we have valid alignments and all tokens are accounted for
    is_valid = bool(valid_alignments) and not remaining_source and not remaining_target

    return (
        is_valid,
        errors,
        valid_alignments,
        remaining_source,
        remaining_target,
    )


def _create_retry_message(
    valid_alignments: List[TokenAlignment],
    remaining_source: set[str],
    remaining_target: set[str],
    source_tokens: List[str],
    target_tokens: List[str],
) -> UserMessage:
    """Create message for retry attempts with partial alignments."""
    message_parts = []

    # First show the complete token lists
    message_parts.append(format_tokens(source_tokens, target_tokens))
    message_parts.append("")

    # Add partial alignments
    if valid_alignments:
        alignment_dicts = [
            {"source": align.source, "target": align.target}
            for align in valid_alignments
        ]
        alignment_str = json.dumps({"alignment": alignment_dicts})
        message_parts.append("Here are partial alignments:")
        message_parts.append(alignment_str)
        message_parts.append("")

    # Add remaining unaligned tokens
    message_parts.append("Please provide alignments for the remaining tokens:")
    if remaining_source:
        message_parts.append(f"Source tokens: {' '.join(sorted(remaining_source))}")
    if remaining_target:
        message_parts.append(f"Target tokens: {' '.join(sorted(remaining_target))}")

    return UserMessage("\n".join(message_parts))


def _process_alignment_sync(
    llm_adapter: LLMAdapter,
    messages: List[Message],
    source_tokens: List[str],
    target_tokens: List[str],
    marker_generator: Optional[MarkerGenerator],
    max_retries: int,
    schema_class: Type[TextAlignmentSchema] = TextAlignmentSchema,
) -> AlignmentResult:
    """
    Synchronous core alignment processing logic.
    """
    attempts: List[AlignmentAttempt] = []
    valid_alignments: List[TokenAlignment] = []
    alignment: Optional[TextAlignment] = None

    # Use existing token mappings
    source_mapping = create_token_mapping(source_tokens, marker_generator)
    target_mapping = create_token_mapping(target_tokens, marker_generator)

    # Track explicitly unaligned tokens
    unaligned_source: set[str] = set()
    unaligned_target: set[str] = set()
    remaining_source: set[str] = set(source_mapping.uniquified)
    remaining_target: set[str] = set(target_mapping.uniquified)

    for attempt in range(max_retries):
        logger.debug(f"Attempt {attempt + 1} for alignment")
        current_messages = format_messages(messages)
        current_attempt = AlignmentAttempt(
            attempt_number=attempt + 1,
            messages_sent=current_messages.copy(),
            raw_response=None,
            validation_passed=False,
            validation_errors=[],
        )

        try:
            raw_response = llm_adapter(current_messages)

            # Handle schema conversion
            if schema_class != TextAlignmentSchema:
                # If we're using dynamic schema
                if isinstance(raw_response, TextAlignmentSchema) and not isinstance(
                    raw_response, schema_class
                ):
                    # Convert base schema to dynamic schema
                    raw_response = TextAlignment(alignment=raw_response.alignment)
                elif not isinstance(raw_response, TextAlignment):
                    # Try direct validation with dynamic schema
                    validated = schema_class.model_validate(raw_response)
                    raw_response = TextAlignment(alignment=validated.alignment)

            # Convert to TextAlignment if needed
            if not isinstance(raw_response, TextAlignment):
                raw_response = TextAlignment(alignment=raw_response.alignment)

            current_attempt.raw_response = raw_response
            logger.debug(f"Raw response: {raw_response}")

            (
                _,  # is_valid not needed
                error_messages,
                new_valid_alignments,
                remaining_source,
                remaining_target,
            ) = _validate_alignment(
                raw_response,
                source_tokens,
                target_tokens,
                marker_generator,
                valid_alignments,
            )

            # Update unaligned token sets from new alignments
            for align in raw_response.alignment:
                if align.target_token == UNALIGNED_MARKER:
                    unaligned_source.add(align.source_token)
                if align.source_token == UNALIGNED_MARKER:
                    unaligned_target.add(align.target_token)

            # Filter out alignments containing UNALIGNED_MARKER
            new_valid_alignments = [
                align
                for align in new_valid_alignments
                if align.source_token != UNALIGNED_MARKER
                and align.target_token != UNALIGNED_MARKER
            ]

            # Deduplicate and sort new alignments
            if new_valid_alignments:
                # Convert to set of tuples for deduplication
                existing_pairs = {(a.source, a.target) for a in valid_alignments}
                new_pairs = {(a.source, a.target) for a in new_valid_alignments}

                # Only add alignments we don't already have
                unique_new_pairs = new_pairs - existing_pairs

                # Convert back to TokenAlignment objects
                new_unique_alignments = [
                    TokenAlignment(source=s, target=t) for s, t in unique_new_pairs
                ]

                # Add to valid alignments
                valid_alignments.extend(new_unique_alignments)

                # Create TextAlignment to trigger automatic sorting
                temp_alignment = TextAlignment(
                    alignment=valid_alignments,
                    source_mapping=source_mapping,
                    target_mapping=target_mapping,
                )
                valid_alignments = temp_alignment.alignment

            # Remove unaligned tokens from remaining sets
            remaining_source = remaining_source - unaligned_source
            remaining_target = remaining_target - unaligned_target

            is_complete = not (remaining_source or remaining_target)
            current_attempt.validation_passed = bool(new_valid_alignments)
            current_attempt.validation_errors = error_messages

            if is_complete:
                alignment = TextAlignment(
                    alignment=valid_alignments,
                    source_mapping=source_mapping,
                    target_mapping=target_mapping,
                )
                attempts.append(current_attempt)
                break

            # Add retry message with partial alignments and remaining tokens
            # Remove the last assistant message since we'll incorporate its content
            messages.pop()
            # Replace the last user message with retry message including all context
            messages[-1] = _create_retry_message(
                valid_alignments,
                remaining_source,
                remaining_target,
                source_tokens,
                target_tokens,
            )

        except Exception as e:
            current_attempt.exception = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

        attempts.append(current_attempt)

    # Create final alignment if we have valid alignments but didn't complete
    if not alignment and valid_alignments:
        logger.debug(
            f"""Alignment not complete, returning partial valid alignments: {valid_alignments}
            Missing source: {remaining_source}
            Missing target: {remaining_target}"""
        )
        alignment = TextAlignment(
            alignment=valid_alignments,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
        )

    return AlignmentResult(
        alignment=alignment,
        attempts=attempts,
    )


def _create_alignment_messages(
    source_tokens: list[str],
    target_tokens: list[str],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    marker_generator: Optional[MarkerGenerator] = None,
    include_schema: bool = False,
) -> List[Message]:
    """
    Create the message list for alignment tasks.

    Args:
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens
        source_language: Optional source language name
        target_language: Optional target language name
        guidelines: Optional alignment guidelines
        examples: Optional list of example alignments
        marker_generator: Optional MarkerGenerator for unique markers (defaults to subscript)

    Returns:
        List of messages for the LLM
    """

    # Use default subscript generator if none provided
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    # Create example with duplicates to show marker usage
    example_source = ["a", "a", "b", "a"]
    example_target = ["c", "b", "c"]
    unique_source = make_unique(example_source, marker_generator)
    unique_target = make_unique(example_target, marker_generator)

    system_msg_parts = [
        "You are an expert translator and linguistic annotator"
        + (
            f" from {source_language} to {target_language}."
            if source_language and target_language
            else "."
        ),
        "Given a list of tokens in the source and target, your task is to align them. Do not further split or merge the tokens and use the exact case/form of the tokens provided as-is.",
        f"For duplicate tokens, unique markers will be added like this: source='{' '.join(unique_source)}', target='{' '.join(unique_target)}'",
        f"Special token to use when alignment is not possible: {UNALIGNED_MARKER}",
        # f"Special tokens: {UNALIGNED_MARKER} (cannot align), <source_specific> (source-only), <target_specific> (target-only). Example: articles→<target_specific>, <source_specific>→particles, punct→{UNALIGNED_MARKER}",
    ]

    if include_schema:
        schema_str = str(TextAlignmentSchema.model_json_schema())
        system_msg_parts.append(f"\nExpected JSON format:\n```json\n{schema_str}\n```")

    if guidelines:
        system_msg_parts.append(
            f"\nHere are annotation guidelines you should strictly follow:\n\n{guidelines}"
        )
    if examples:
        system_msg_parts.append(
            "\nReturn alignments in the same format as the following examples:"
        )

    messages: List[Message] = [SystemMessage("\n".join(system_msg_parts))]

    if examples:
        for example_source_tokens, example_target_tokens, example_alignment in examples:
            messages.append(
                UserMessage(format_tokens(example_source_tokens, example_target_tokens))
            )
            messages.append(AssistantMessage(example_alignment))

    messages.append(UserMessage(format_tokens(source_tokens, target_tokens)))

    return messages


def align_tokens(
    llm_adapter: LLMAdapter,
    source_tokens: List[str | LiteralString],
    target_tokens: List[str | LiteralString],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
) -> AlignmentResult:
    """
    Align tokens from source language to target language using a language model.

    Args:
        llm_adapter: An adapter instance for running the language model
        source_tokens: List of source language tokens
        target_tokens: List of target language tokens
        source_language: Optional source language name
        target_language: Optional target language name
        guidelines: Optional alignment guidelines
        examples: Optional list of example alignments
        max_retries: Maximum number of retries for invalid alignments
        marker_generator: Optional generator for unique markers

    Returns:
        AlignmentResult object containing the alignment (if successful) and diagnostic information

    Example:
        >>> from lexi_align.adapters.outlines_adapter import OutlinesAdapter
        >>> adapter = OutlinesAdapter("meta-llama/Llama-3.2-3B-Instruct")
        >>> source = ["The", "cat", "sat"]
        >>> target = ["Le", "chat", "assis"]
        >>> result = align_tokens(adapter, source, target, "English", "French")
        >>> result.alignment.alignment  # doctest: +NORMALIZE_WHITESPACE
        [TokenAlignment(source='cat', target='chat'),
         TokenAlignment(source='sat', target='assis')]
    """
    # Create mappings before processing
    source_mapping = create_token_mapping(source_tokens, marker_generator)
    target_mapping = create_token_mapping(target_tokens, marker_generator)

    # Create dynamic schema if adapter supports length constraints
    schema_class: Type[TextAlignmentSchema] = (
        create_dynamic_alignment_schema(source_tokens, target_tokens, marker_generator)
        if llm_adapter.supports_length_constraints()
        else TextAlignmentSchema
    )

    messages = _create_alignment_messages(
        source_tokens,
        target_tokens,
        source_language,
        target_language,
        guidelines,
        examples,
        marker_generator,
    )

    logger.debug(f"Source mapping: {source_mapping.uniquified}")
    logger.debug(f"Target mapping: {target_mapping.uniquified}")

    result = _process_alignment_sync(
        llm_adapter,
        messages,
        source_tokens,
        target_tokens,
        marker_generator,
        max_retries,
        schema_class=schema_class,
    )

    # Sort alignment by position if we have a valid result
    if result.alignment:
        logger.debug(f"Result before sorting: {result.alignment.alignment}")
        result.alignment = result.alignment.sort_by_position(
            source_mapping, target_mapping
        )
        logger.debug(f"Result after sorting: {result.alignment.alignment}")

    return result


async def align_tokens_async(
    llm_adapter: LLMAdapter,
    source_tokens: List[str],
    target_tokens: List[str],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
) -> Union[TextAlignment, AlignmentResult]:
    """
    Async version of align_tokens.

    Example:
        >>> import asyncio
        >>> from lexi_align.adapters.outlines_adapter import OutlinesAdapter
        >>> adapter = OutlinesAdapter("meta-llama/Llama-3.2-3B-Instruct")
        >>> source = ["The", "cat", "sat"]
        >>> target = ["Le", "chat", "assis"]
        >>> result = asyncio.run(align_tokens_async(adapter, source, target, "English", "French"))
        >>> result.alignment  # doctest: +NORMALIZE_WHITESPACE
        [TokenAlignment(source='cat', target='chat'), TokenAlignment(source='sat', target='assis'), TokenAlignment(source='The', target='<unaligned>')]
    """
    # Create mappings at the start
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    source_mapping = create_token_mapping(source_tokens, marker_generator)
    target_mapping = create_token_mapping(target_tokens, marker_generator)
    messages = _create_alignment_messages(
        source_tokens,
        target_tokens,
        source_language,
        target_language,
        guidelines,
        examples,
        marker_generator,
    )

    logger.debug(f"Async - Source mapping: {source_mapping.uniquified}")
    logger.debug(f"Async - Target mapping: {target_mapping.uniquified}")

    result = None
    error_msg: list[tuple[ValidationErrorType, str, list[str]]] = []
    existing_alignments: List[TokenAlignment] = []

    for attempt in range(max_retries):
        try:
            # Use acall if available, otherwise fall back to sync call
            if hasattr(llm_adapter, "acall"):
                result = await llm_adapter.acall(format_messages(messages))
            else:
                result = llm_adapter(format_messages(messages))

            logger.debug(f"Async - Raw result: {result}")

            # Convert DynamicTokenAlignment to TokenAlignment if needed
            if result and hasattr(result, "alignment"):
                result = TextAlignment(
                    alignment=[
                        TokenAlignment(source=align.source, target=align.target)
                        for align in result.alignment
                    ]
                )

            # Validate the alignment
            (
                is_valid,
                error_msg,
                valid_alignments,
                remaining_source,
                remaining_target,
            ) = _validate_alignment(
                result,
                source_tokens,
                target_tokens,
                marker_generator,
                existing_alignments=existing_alignments,
            )

            # Store valid alignments for next iteration
            if valid_alignments:
                existing_alignments.extend(valid_alignments)

            if is_valid or existing_alignments:
                # Create final alignment with all valid alignments
                alignment = TextAlignment(
                    alignment=existing_alignments,
                    source_mapping=source_mapping,
                    target_mapping=target_mapping,
                )
                return alignment

            # Add retry message if not valid
            messages.append(AssistantMessage(result))
            messages.append(
                UserMessage(
                    f"The previous alignment was partially valid. Please provide alignments for the remaining tokens:\n\n"
                    f"Unaligned source tokens: {str(remaining_source)[1:-1]}\n"
                    f"Unaligned target tokens: {str(remaining_target)[1:-1]}"
                )
            )

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return AlignmentResult(
                    alignment=None,
                    attempts=[
                        AlignmentAttempt(
                            attempt_number=attempt + 1,
                            messages_sent=format_messages(messages),
                            raw_response=None,
                            validation_passed=False,
                            validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                            exception=str(e),
                        )
                    ],
                )

    # If we get here, create an AlignmentResult with any partial alignments we have
    final_alignment: TextAlignment | None = None
    if existing_alignments:
        final_alignment = TextAlignment(
            alignment=existing_alignments,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
        )
        final_alignment = final_alignment.sort_by_position(
            source_mapping, target_mapping
        )

    return AlignmentResult(
        alignment=final_alignment,
        attempts=[
            AlignmentAttempt(
                attempt_number=max_retries,
                messages_sent=format_messages(messages),
                raw_response=result,
                validation_passed=False,
                validation_errors=error_msg,
            )
        ],
    )


def batch_sequences(sequences: list, chunk_size: int) -> list[list]:
    """Split sequences into chunks of specified size."""
    return [sequences[i : i + chunk_size] for i in range(0, len(sequences), chunk_size)]


def align_tokens_batched(
    llm_adapter: LLMAdapter,
    source_sequences: list[list[str]],
    target_sequences: list[list[str]],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
    batch_size: int = 5,
) -> Sequence[AlignmentResult]:
    """Process multiple sequences of tokens for alignment with proper retry handling."""
    if len(source_sequences) != len(target_sequences):
        raise ValueError("Number of source and target sequences must match")

    if not llm_adapter.supports_true_batching():
        logger.warning(
            f"Adapter {llm_adapter.__class__.__name__} does not support true batching (batch_size={batch_size}), falling back to sequential processing"
        )
        return [
            align_tokens(
                llm_adapter,
                src_tokens,
                tgt_tokens,
                source_language,
                target_language,
                guidelines,
                examples,
                max_retries,
                marker_generator,
            )
            for src_tokens, tgt_tokens in zip(source_sequences, target_sequences)
        ]

    # Create marker generator if not provided
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    # Create mappings and enums for each sequence pair at the start
    source_mappings = [
        create_token_mapping(src, marker_generator) for src in source_sequences
    ]
    target_mappings = [
        create_token_mapping(tgt, marker_generator) for tgt in target_sequences
    ]

    # Track attempts and results for each sequence
    sequence_attempts: list[list[AlignmentAttempt]] = [[] for _ in source_sequences]
    final_results: list[Optional[TextAlignment]] = [None] * len(source_sequences)

    # Track which sequences need retries
    retry_indices = list(range(len(source_sequences)))

    for attempt in range(max_retries):
        if not retry_indices:
            break

        # Prepare retry batch
        retry_sources = [source_sequences[i] for i in retry_indices]
        retry_targets = [target_sequences[i] for i in retry_indices]

        # Create messages for retry batch
        retry_messages = [
            _create_alignment_messages(
                src,
                tgt,
                source_language,
                target_language,
                guidelines,
                examples,
                marker_generator,
            )
            for src, tgt in zip(retry_sources, retry_targets)
        ]

        formatted_messages = [format_messages(*msgs) for msgs in retry_messages]

        try:
            # Process batch with validation parameters
            batch_results = llm_adapter.batch(
                formatted_messages,
            )

            # Process results and track which need retries
            new_retry_indices = []

            for batch_idx, (result, msgs) in enumerate(
                zip(batch_results, formatted_messages)
            ):
                seq_idx = retry_indices[batch_idx]

                if result is None:
                    # Failed generation
                    sequence_attempts[seq_idx].append(
                        AlignmentAttempt(
                            attempt_number=attempt + 1,
                            messages_sent=msgs,
                            raw_response=None,
                            validation_passed=False,
                            validation_errors=[
                                (ValidationErrorType.OTHER, "Generation failed", [])
                            ],
                        )
                    )
                    new_retry_indices.append(seq_idx)
                    continue

                # Add type guard for alignment access
                existing_alignments = None
                if seq_idx < len(final_results):
                    current_result = final_results[seq_idx]
                    if isinstance(current_result, TextAlignment):
                        existing_alignments = current_result.alignment

                # Validate alignment and get valid alignments
                (
                    is_valid,
                    error_msg,
                    valid_alignments,
                    remaining_source,
                    remaining_target,
                ) = _validate_alignment(
                    result,
                    source_sequences[seq_idx],
                    target_sequences[seq_idx],
                    marker_generator,
                    # Pass any existing valid alignments from previous attempts
                    existing_alignments=existing_alignments,
                )

                sequence_attempts[seq_idx].append(
                    AlignmentAttempt(
                        attempt_number=attempt + 1,
                        messages_sent=msgs,
                        raw_response=result,
                        validation_passed=is_valid,
                        validation_errors=error_msg if not is_valid else [],
                    )
                )

                if valid_alignments:  # Store partial results even if not fully valid
                    if final_results[seq_idx] is None:
                        final_results[seq_idx] = TextAlignment(
                            alignment=valid_alignments,
                            source_mapping=source_mappings[seq_idx],
                            target_mapping=target_mappings[seq_idx],
                        )
                    elif isinstance(final_results[seq_idx], TextAlignment):
                        # Add type guard to ensure alignment access is safe
                        current_alignment = final_results[seq_idx]
                        if current_alignment is not None:
                            current_alignment.alignment.extend(valid_alignments)
                            # Re-sort after extending
                            final_results[seq_idx] = TextAlignment(
                                alignment=current_alignment.alignment,
                                source_mapping=source_mappings[seq_idx],
                                target_mapping=target_mappings[seq_idx],
                            )

                if not is_valid:
                    # Add retry message with information about remaining tokens
                    retry_messages[batch_idx].append(
                        UserMessage(
                            "The previous alignment was partially valid. Please provide alignments for the remaining tokens:\n\n"
                            + (
                                f"Remaining source tokens: {str(remaining_source)[1:-1]}\n"
                                if remaining_source
                                else ""
                            )
                            + (
                                f"Remaining target tokens: {str(remaining_target)[1:-1]}"
                                if remaining_target
                                else ""
                            )
                        )
                    )
                    new_retry_indices.append(seq_idx)

            retry_indices = new_retry_indices

        except Exception as e:
            logger.warning(f"Batch attempt {attempt + 1} failed: {e}")
            # On complete batch failure, all sequences need retry
            for seq_idx in retry_indices:
                sequence_attempts[seq_idx].append(
                    AlignmentAttempt(
                        attempt_number=attempt + 1,
                        messages_sent=formatted_messages[retry_indices.index(seq_idx)],
                        raw_response=None,
                        validation_passed=False,
                        validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                        exception=str(e),
                    )
                )

    # Create final AlignmentResults
    final_alignment_results = []
    for i, (result, attempts) in enumerate(zip(final_results, sequence_attempts)):
        sorted_result = None
        if result is not None and isinstance(result, TextAlignment):
            # Sort by position using the mappings we created at the start
            sorted_result = result.sort_by_position(
                source_mappings[i],
                target_mappings[i],
            )
        final_alignment_results.append(
            AlignmentResult(
                alignment=sorted_result,
                attempts=attempts,
            )
        )
    return final_alignment_results


def align_tokens_raw(
    llm_adapter: LLMAdapter,
    source_tokens: List[str],
    target_tokens: List[str],
    custom_messages: List[Dict[str, Any]],
) -> AlignmentResult:
    """
    Align tokens using custom messages instead of the default system/guidelines/examples template.

    Example:
        >>> from lexi_align.adapters.outlines_adapter import OutlinesAdapter
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> source = ["The", "cat", "sat"]
        >>> target = ["Le", "chat", "assis"]
        >>> # Create mock adapter for testing
        >>> class MockAdapter(LLMAdapter):
        ...     def __call__(self, messages: list[dict]) -> TextAlignment:
        ...         return TextAlignment(alignment=[
        ...             TokenAlignment(source="The", target="Le"),
        ...             TokenAlignment(source="cat", target="chat"),
        ...             TokenAlignment(source="sat", target="assis")
        ...         ])
        >>> adapter = MockAdapter()
        >>> messages = [
        ...     {"role": "system", "content": "You are a translator aligning English to French."},
        ...     {"role": "user", "content": "Align these tokens:\\n"
        ...         f"English: {' '.join(source)}\\n"
        ...         f"French: {' '.join(target)}"}
        ... ]
        >>> result = align_tokens_raw(adapter, source, target, messages)
        >>> result.alignment.alignment  # doctest: +NORMALIZE_WHITESPACE
        [TokenAlignment(source='The', target='Le'),
         TokenAlignment(source='cat', target='chat'),
         TokenAlignment(source='sat', target='assis')]
    """
    messages = custom_messages.copy()  # Make a copy to not modify the input
    messages.append(
        {
            "role": "user",
            "content": (
                f"source_tokens: {make_unique(source_tokens)}\n"
                f"target_tokens: {make_unique(target_tokens)}"
            ),
        }
    )

    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)
    try:
        if asyncio.iscoroutinefunction(getattr(llm_adapter, "acall", None)):
            result = asyncio.run(llm_adapter.acall(messages))
        else:
            result = llm_adapter(messages)

        # Validate the alignment
        (
            is_valid,
            error_messages,
            valid_alignments,
            _,  # remaining_source
            _,  # remaining_target
        ) = _validate_alignment(
            result,
            source_tokens,
            target_tokens,
            marker_generator=None,
            existing_alignments=None,
        )

        # Create alignment from valid alignments if any
        alignment = (
            TextAlignment(
                alignment=valid_alignments,
                source_mapping=source_mapping,
                target_mapping=target_mapping,
            )
            if valid_alignments
            else None
        )

        # Sort alignment by position if we have valid alignments
        if alignment:
            source_mapping = create_token_mapping(source_tokens)
            target_mapping = create_token_mapping(target_tokens)
            alignment = alignment.sort_by_position(source_mapping, target_mapping)

        return AlignmentResult(
            alignment=alignment,
            attempts=[
                AlignmentAttempt(
                    attempt_number=1,
                    messages_sent=messages,
                    raw_response=result,
                    validation_passed=is_valid,
                    validation_errors=error_messages,
                )
            ],
        )
    except Exception as e:
        return AlignmentResult(
            alignment=None,
            attempts=[
                AlignmentAttempt(
                    attempt_number=1,
                    messages_sent=messages,
                    raw_response=None,
                    validation_passed=False,
                    validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                    exception=str(e),
                )
            ],
        )
