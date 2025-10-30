from asyncio import run as asyncio_run
from logging import getLogger
from typing import (
    Any,
    Dict,
    List,
    LiteralString,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

from llm_schema_lite import simplify_schema
from tqdm import tqdm

from lexi_align.adapters import LLMAdapter
from lexi_align.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    REMAINING_SOURCE_PREFIX,
    REMAINING_TARGET_PREFIX,
    SOURCE_TOKENS_PREFIX,
    TARGET_TOKENS_PREFIX,
    UNALIGNED_MARKER,
)
from lexi_align.models import (
    AlignmentAttempt,
    AlignmentResult,
    SpecialTokens,
    TextAlignment,
    TextAlignmentSchema,
    TokenAlignment,
    TokenMapping,
    ValidationErrorDict,
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
    make_unique,
)
from lexi_align.utils.common import batch_iterable, ensure_text_alignment

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
    # Count error-type occurrences and token frequencies
    from collections import Counter

    error_counter: Counter[ValidationErrorType] = Counter(err[0] for err in errors)
    token_counters: dict[ValidationErrorType, Counter[str]] = {}
    for err_type, _, tokens in errors:
        token_counters.setdefault(err_type, Counter()).update(tokens)

    # Build final stats dict in one shot
    return {
        et: {
            "count": error_counter.get(et, 0),
            "frequencies": dict(token_counters.get(et, {})),
        }
        for et in ValidationErrorType
    }


def normalize_validation_errors(
    errors: list[tuple[ValidationErrorType, str, list[str]]],
) -> list[ValidationErrorDict]:
    """Convert tuple-based validation errors to typed dicts.

    Example:
        >>> from lexi_align.models import ValidationErrorType
        >>> errs = [(ValidationErrorType.INVALID_SOURCE_TOKEN, "bad token", ["foo"])]
        >>> out = normalize_validation_errors(errs)
        >>> out[0]["type"] == ValidationErrorType.INVALID_SOURCE_TOKEN and out[0]["message"] == "bad token"
        True
    """
    return [{"type": et, "message": msg, "tokens": toks} for et, msg, toks in errors]


def _validate_alignment(
    alignment: TextAlignment,
    source_tokens: list[str],
    target_tokens: list[str],
    marker_generator: Optional[MarkerGenerator] = None,
    existing_alignments: Optional[List[TokenAlignment]] = None,
    source_mapping: Optional[TokenMapping] = None,
    target_mapping: Optional[TokenMapping] = None,
) -> tuple[
    bool,
    list[tuple[ValidationErrorType, str, list[str]]],
    list[TokenAlignment],
    set[str],
    set[str],
]:
    """
    Validate alignment and extract valid alignments and remaining tokens.
    Returns tuple of:
    - is_valid: bool
    - errors: list of (error_type, description, affected_tokens)
    - valid_alignments: list of valid TokenAlignment objects
    - remaining_source: set of unaligned source tokens
    - remaining_target: set of unaligned target tokens
    """
    if source_mapping is None:
        source_mapping = create_token_mapping(source_tokens, marker_generator)
    if target_mapping is None:
        target_mapping = create_token_mapping(target_tokens, marker_generator)

    valid_alignments = list(existing_alignments) if existing_alignments else []
    errors: List[Tuple[ValidationErrorType, str, List[str]]] = []
    explicitly_unaligned_source = set()
    explicitly_unaligned_target = set()
    invalid_source: List[str] = []
    invalid_target: List[str] = []

    special_tokens = {
        SpecialTokens.UNALIGNED.value,
        SpecialTokens.SOURCE_SPECIFIC.value,
        SpecialTokens.TARGET_SPECIFIC.value,
    }

    # Process all alignments in one pass
    for align in alignment.alignment:
        # Handle special alignments
        if align.source in special_tokens or align.target in special_tokens:
            valid_alignments.append(align)
            if align.source == UNALIGNED_MARKER:
                explicitly_unaligned_target.add(align.target)
            elif align.target == UNALIGNED_MARKER:
                explicitly_unaligned_source.add(align.source)
            continue

        # Validate format
        if not align.source or not align.source.strip():
            invalid_source.append("<empty>")
            continue
        if len(align.source.split()) > 1:
            invalid_source.append(repr(align.source))
            continue
        if not align.target or not align.target.strip():
            invalid_target.append("<empty>")
            continue
        if len(align.target.split()) > 1:
            invalid_target.append(repr(align.target))
            continue

        # Validate tokens exist in mappings
        s_valid = source_mapping.get_position(align.source) != -1
        t_valid = target_mapping.get_position(align.target) != -1

        if s_valid and t_valid:
            valid_alignments.append(align)
        else:
            if not s_valid:
                invalid_source.append(repr(align.source))
                logger.error(
                    f"❌ INVALID SOURCE TOKEN: {repr(align.source)} not in mapping. "
                    f"Available: {source_mapping.uniquified}"
                )
            if not t_valid:
                invalid_target.append(repr(align.target))
                logger.error(
                    f"❌ INVALID TARGET TOKEN: {repr(align.target)} not in mapping. "
                    f"Available: {target_mapping.uniquified}"
                )

    # Add error messages for invalid tokens
    if invalid_source:
        from collections import Counter

        counts = Counter(invalid_source)
        formatted = ", ".join(
            f"{token} (x{count})" if count > 1 else token
            for token, count in sorted(counts.items())
        )
        errors.append(
            (
                ValidationErrorType.INVALID_SOURCE_TOKEN,
                f"Invalid source tokens: {formatted}",
                invalid_source,
            )
        )
        logger.error(f"Validation found invalid source tokens: {formatted}")

    if invalid_target:
        from collections import Counter

        counts = Counter(invalid_target)
        formatted = ", ".join(
            f"{token} (x{count})" if count > 1 else token
            for token, count in sorted(counts.items())
        )
        errors.append(
            (
                ValidationErrorType.INVALID_TARGET_TOKEN,
                f"Invalid target tokens: {formatted}",
                invalid_target,
            )
        )
        logger.error(f"Validation found invalid target tokens: {formatted}")

    # Calculate remaining tokens
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

    if remaining_source:
        errors.append(
            (
                ValidationErrorType.MISSING_SOURCE_ALIGNMENTS,
                f"Unaligned source tokens: {', '.join(remaining_source)}",
                list(remaining_source),
            )
        )
        logger.warning(
            f"Validation found remaining source tokens: {sorted(remaining_source)}"
        )

    if remaining_target:
        errors.append(
            (
                ValidationErrorType.MISSING_TARGET_ALIGNMENTS,
                f"Unaligned target tokens: {', '.join(remaining_target)}",
                list(remaining_target),
            )
        )
        logger.warning(
            f"Validation found remaining target tokens: {sorted(remaining_target)}"
        )

    is_valid = bool(valid_alignments) and not remaining_source and not remaining_target

    return (is_valid, errors, valid_alignments, remaining_source, remaining_target)


def _create_retry_message(
    valid_alignments: List[TokenAlignment],
    remaining_source: set[str],
    remaining_target: set[str],
    source_tokens: List[str],
    target_tokens: List[str],
    marker_generator: Optional[MarkerGenerator] = None,
) -> UserMessage:
    """Create message for retry attempts with partial alignments."""
    message_parts = []

    # Cache uniquified tokens once to preserve order and avoid recomputation
    unique_source = make_unique(source_tokens, marker_generator)
    unique_target = make_unique(target_tokens, marker_generator)

    # First show the complete token lists
    message_parts.append(SOURCE_TOKENS_PREFIX + " ".join(unique_source))
    message_parts.append(TARGET_TOKENS_PREFIX + " ".join(unique_target))
    message_parts.append("")

    # Add partial alignments
    if valid_alignments:
        alignment_str = TextAlignmentSchema(alignment=valid_alignments).model_dump_json(
            indent=None
        )
        message_parts.append("Here are partial alignments:")
        message_parts.append(alignment_str)
        message_parts.append("")

    # Add remaining unaligned tokens
    message_parts.append("Please provide alignments for the remaining tokens:")
    message_parts.append(
        "Only output alignments for the remaining tokens; do not repeat alignments shown above."
    )
    message_parts.append(
        "If only one side has remaining tokens, restrict that side to the tokens listed under remaining_source_tokens/remaining_target_tokens."
    )
    if remaining_source:
        ordered_remaining_source = [t for t in unique_source if t in remaining_source]
        message_parts.append(
            REMAINING_SOURCE_PREFIX + " ".join(ordered_remaining_source)
        )
    if remaining_target:
        ordered_remaining_target = [t for t in unique_target if t in remaining_target]
        message_parts.append(
            REMAINING_TARGET_PREFIX + " ".join(ordered_remaining_target)
        )

    return UserMessage("\n".join(message_parts))


def _process_alignment_sync(
    llm_adapter: LLMAdapter,
    messages: List[Message],
    source_tokens: List[str],
    target_tokens: List[str],
    marker_generator: Optional[MarkerGenerator],
    max_retries: int,
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

        # Track prior remaining sets to determine whether this attempt made progress
        prev_remaining_source = set(remaining_source)
        prev_remaining_target = set(remaining_target)

        try:
            raw_response = llm_adapter(current_messages)
            raw_response = ensure_text_alignment(raw_response)
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
                source_mapping=source_mapping,
                target_mapping=target_mapping,
            )

            # Update unaligned token sets from new alignments
            for align in raw_response.alignment:
                if align.target == UNALIGNED_MARKER:
                    unaligned_source.add(align.source)
                if align.source == UNALIGNED_MARKER:
                    unaligned_target.add(align.target)

            # Filter out alignments containing UNALIGNED_MARKER
            new_valid_alignments = [
                align
                for align in new_valid_alignments
                if align.source != UNALIGNED_MARKER and align.target != UNALIGNED_MARKER
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
            progress = (
                len(remaining_source) < len(prev_remaining_source)
                or len(remaining_target) < len(prev_remaining_target)
                or bool(new_valid_alignments)
            )
            current_attempt.validation_passed = progress
            current_attempt.validation_errors = error_messages

            if is_complete:
                alignment = TextAlignment(
                    alignment=valid_alignments,
                    source_mapping=source_mapping,
                    target_mapping=target_mapping,
                )
                attempts.append(current_attempt)
                break

            # Prepare messages for the next retry attempt
            # Add the assistant's (failed) response to the history
            if current_attempt.raw_response:
                # Ensure raw_response content is suitable for AssistantMessage
                messages.append(AssistantMessage(current_attempt.raw_response))

            # Add the new user message asking for correction/completion
            messages.append(
                _create_retry_message(
                    valid_alignments,
                    remaining_source,
                    remaining_target,
                    source_tokens,  # Pass original tokens for context
                    target_tokens,  # Pass original tokens for context
                    marker_generator,
                )
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
    use_reasoning: bool = False,
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
        include_schema: Whether to include schema in system message
        use_reasoning: Whether to request reasoning before alignment

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

    if use_reasoning:
        from lexi_align.constants import REASONING_PROMPT

        system_msg_parts.append("\n" + REASONING_PROMPT)
        system_msg_parts.append(
            "\nIMPORTANT: You MUST provide reasoning in the 'reasoning' field. "
            "Do not leave it null or empty. Explain your alignment decisions step-by-step."
        )

    if include_schema:
        DynamicSchema = create_dynamic_alignment_schema(
            source_tokens, target_tokens, marker_generator
        )
        simplified = simplify_schema(DynamicSchema, include_metadata=True)
        schema_str = simplified.to_string()
        system_msg_parts.append(f"\nExpected schema:\n```text\n{schema_str}\n```")

    system_msg_parts.extend(
        [
            "Constraints:",
            "1) Use only tokens from the enumerated sets; do not invent or normalize tokens.",
            '2) Emit exactly one JSON object with top-level key "alignment"; no extra text or markdown.',
            "3) Articles and determiners (e.g., 'the', 'a', 'an') are often <unaligned>; output <unaligned> rather than forcing an incorrect pair.",
            "4) Align punctuation only if both sides contain the corresponding punctuation; otherwise use <unaligned>.",
            "5) Never align <unaligned> with <unaligned>.",
        ]
    )

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
        logger.debug(f"Processing {len(examples)} training examples")
        for idx, (
            example_source_tokens,
            example_target_tokens,
            example_alignment,
        ) in enumerate(examples):
            try:
                logger.debug(f"Processing training example {idx}")
                logger.debug(
                    f"  Source tokens: {example_source_tokens[:3] if len(example_source_tokens) > 3 else example_source_tokens}..."
                )
                logger.debug(
                    f"  Target tokens: {example_target_tokens[:3] if len(example_target_tokens) > 3 else example_target_tokens}..."
                )
                logger.debug(f"  Alignment type: {type(example_alignment)}")
                logger.debug(
                    f"  Alignment has {len(example_alignment.alignment)} pairs"
                )

                # Add defensive check for marker_generator
                if marker_generator is None:
                    logger.error(f"marker_generator is None at example {idx}!")
                    marker_generator = create_subscript_generator()

                # Check if marker_generator.generate is callable
                if not callable(marker_generator.generate):
                    logger.error(
                        f"marker_generator.generate is not callable at example {idx}!"
                    )
                    logger.error(f"  marker_generator: {marker_generator}")
                    logger.error(
                        f"  marker_generator.generate: {marker_generator.generate}"
                    )
                    raise ValueError(
                        f"marker_generator.generate is not callable: {marker_generator.generate}"
                    )

                # Try the make_unique calls with explicit error handling
                try:
                    unique_source = make_unique(example_source_tokens, marker_generator)
                    logger.debug(
                        f"  Uniquified source: {unique_source[:3] if len(unique_source) > 3 else unique_source}..."
                    )
                except Exception as e:
                    logger.error(
                        f"Error in make_unique for source at example {idx}: {e}",
                        exc_info=True,
                    )
                    raise

                try:
                    unique_target = make_unique(example_target_tokens, marker_generator)
                    logger.debug(
                        f"  Uniquified target: {unique_target[:3] if len(unique_target) > 3 else unique_target}..."
                    )
                except Exception as e:
                    logger.error(
                        f"Error in make_unique for target at example {idx}: {e}",
                        exc_info=True,
                    )
                    raise

                messages.append(
                    UserMessage(
                        SOURCE_TOKENS_PREFIX
                        + " ".join(unique_source)
                        + "\n"
                        + TARGET_TOKENS_PREFIX
                        + " ".join(unique_target)
                    )
                )

                # Check if alignment serialization works
                try:
                    logger.debug("  Serializing alignment...")
                    messages.append(AssistantMessage(example_alignment))
                    logger.debug("  Alignment serialized successfully")
                except Exception as e:
                    logger.error(
                        f"Error serializing alignment at example {idx}: {e}",
                        exc_info=True,
                    )
                    raise

            except Exception as e:
                logger.error(
                    f"Failed to process training example {idx}:\n"
                    f"  Source: {example_source_tokens}\n"
                    f"  Target: {example_target_tokens}\n"
                    f"  Error: {e}",
                    exc_info=True,
                )
                raise

    # Single final user message including both standard and snake_case blocks
    messages.append(
        UserMessage(
            SOURCE_TOKENS_PREFIX
            + " ".join(make_unique(source_tokens, marker_generator))
            + "\n"
            + TARGET_TOKENS_PREFIX
            + " ".join(make_unique(target_tokens, marker_generator))
        )
    )

    return messages


def build_alignment_messages(
    source_tokens: list[str],
    target_tokens: list[str],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    marker_generator: Optional[MarkerGenerator] = None,
    include_schema: bool = False,
    use_reasoning: bool = False,
) -> List[Message]:
    """
    Public wrapper to build alignment chat messages.
    Returns the same content as the internal builder.
    """
    return _create_alignment_messages(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        source_language=source_language,
        target_language=target_language,
        guidelines=guidelines,
        examples=examples,
        marker_generator=marker_generator,
        include_schema=include_schema,
        use_reasoning=use_reasoning,
    )


def normalize_examples(
    examples: Optional[
        List[
            Tuple[
                Sequence[str] | str,
                Sequence[str] | str,
                TextAlignment | Sequence[Tuple[str, str]],
            ]
        ]
    ],
    marker_generator: Optional[MarkerGenerator] = None,
    adapter: Optional[LLMAdapter] = None,
) -> Optional[List[Tuple[List[str], List[str], TextAlignment]]]:
    """
    Normalize example triples into (list[str], list[str], TextAlignment).
    Accepts strings or sequences for tokens, and a TextAlignment or list of (src,tgt) tuples.
    """
    if examples is None:
        return None
    out: List[Tuple[List[str], List[str], TextAlignment]] = []
    for src, tgt, aln in examples:
        src_tokens = src.split() if isinstance(src, str) else list(src)
        tgt_tokens = tgt.split() if isinstance(tgt, str) else list(tgt)
        if isinstance(aln, TextAlignment):
            ta = aln
        else:
            pairs = [TokenAlignment(source=s, target=t) for s, t in aln]
            ta = TextAlignment.from_token_alignments(
                pairs,
                src_tokens,
                tgt_tokens,
                marker_generator=marker_generator,
                adapter=adapter,
            )
        out.append((src_tokens, tgt_tokens, ta))
    return out


def summarize_result(alignment_result: AlignmentResult) -> dict[str, Any]:
    """
    Summarize attempts and validation errors for one AlignmentResult.
    Returns: dict with keys total_attempts, total_validation_errors,
    exception_counts (by type), validation_error_stats (by ValidationErrorType).
    """
    total_attempts = len(alignment_result.attempts)
    total_validation_errors = sum(
        1 for a in alignment_result.attempts if not a.validation_passed
    )
    from collections import Counter

    exc_counter: Counter[str] = Counter()
    for a in alignment_result.attempts:
        if a.exception:
            et = a.exception.split(":", 1)[0].strip()
            exc_counter[et] += 1
    all_errors = [err for a in alignment_result.attempts for err in a.validation_errors]
    val_err_stats = categorize_validation_errors(all_errors)
    return {
        "total_attempts": total_attempts,
        "total_validation_errors": total_validation_errors,
        "exception_counts": dict(exc_counter),
        "validation_error_stats": val_err_stats,
    }


def align_many(
    llm_adapter: LLMAdapter,
    pairs: Sequence[Tuple[Sequence[str] | str, Sequence[str] | str]],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> list[AlignmentResult]:
    """
    Convenience: align many (src,tgt) pairs.
    - Uses true batching if supported and batch_size given.
    - Otherwise runs sequentially (use align_many_async for async concurrency).
    """

    def _tok(seq: Sequence[str] | str) -> list[str]:
        return seq.split() if isinstance(seq, str) else list(seq)

    src_seqs = [_tok(s) for s, _ in pairs]
    tgt_seqs = [_tok(t) for _, t in pairs]
    if batch_size and llm_adapter.supports_true_batching() and len(src_seqs) > 0:
        return list(
            align_tokens_batched(
                llm_adapter,
                src_seqs,
                tgt_seqs,
                source_language=source_language,
                target_language=target_language,
                guidelines=guidelines,
                examples=examples,
                max_retries=max_retries,
                marker_generator=marker_generator,
                batch_size=batch_size,
            )
        )
    if concurrency and concurrency > 1:
        logger.info(
            "align_many: concurrency requested; use align_many_async for non-blocking concurrency."
        )
    return [
        align_tokens(
            llm_adapter,
            s,
            t,
            source_language=source_language,
            target_language=target_language,
            guidelines=guidelines,
            examples=examples,
            max_retries=max_retries,
            marker_generator=marker_generator,
        )
        for s, t in zip(src_seqs, tgt_seqs)
    ]


async def align_many_async(
    llm_adapter: LLMAdapter,
    pairs: Sequence[Tuple[Sequence[str] | str, Sequence[str] | str]],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    show_progress: bool = True,
) -> list[AlignmentResult]:
    """
    Async convenience: align many (src,tgt) pairs with bounded concurrency.
    """
    import asyncio as _asyncio

    def _tok(seq: Sequence[str] | str) -> list[str]:
        return seq.split() if isinstance(seq, str) else list(seq)

    src_seqs = [_tok(s) for s, _ in pairs]
    tgt_seqs = [_tok(t) for _, t in pairs]
    sem = _asyncio.Semaphore(max(1, concurrency))

    async def _one(s: list[str], t: list[str]) -> AlignmentResult:
        async with sem:
            return await align_tokens_async(
                llm_adapter,
                s,
                t,
                source_language=source_language,
                target_language=target_language,
                guidelines=guidelines,
                examples=examples,
                max_retries=max_retries,
                marker_generator=marker_generator,
            )

    # Create tasks indexed to preserve order
    tasks = [_asyncio.create_task(_one(s, t)) for s, t in zip(src_seqs, tgt_seqs)]

    if show_progress:
        # Track results by original index to preserve order
        results: list[Optional[AlignmentResult]] = [None] * len(tasks)
        pending = {task: idx for idx, task in enumerate(tasks)}

        with tqdm(
            total=len(tasks), desc="Aligning (async)", disable=not show_progress
        ) as pbar:
            while pending:
                done, _ = await _asyncio.wait(
                    pending.keys(), return_when=_asyncio.FIRST_COMPLETED
                )
                for task in done:
                    idx = pending.pop(task)
                    results[idx] = await task
                    pbar.update(1)

        # Return results in original order (filter None just in case of errors)
        return [r for r in results if r is not None]  # type: ignore[misc]
    else:
        # gather() preserves order naturally
        return list(await _asyncio.gather(*tasks))


def build_micro_metrics(
    predictions_and_gold: Sequence[Tuple[TextAlignment, TextAlignment]],
    f_alpha: float = 0.5,
) -> Dict[str, float | int]:
    """
    Compute micro-averaged metrics across multiple (predicted, gold) alignment pairs.
    Returns a dict with keys: precision, recall, f_measure, aer, total_true_positives,
    total_predicted, total_gold.
    """
    tp_sum = 0
    pred_sum = 0
    gold_sum = 0
    for pred, gold in predictions_and_gold:
        A = {(a.source, a.target) for a in pred.alignment}
        G = {(a.source, a.target) for a in gold.alignment}
        tp_sum += len(A & G)
        pred_sum += len(A)
        gold_sum += len(G)
    precision = tp_sum / pred_sum if pred_sum else 0.0
    recall = tp_sum / gold_sum if gold_sum else 0.0
    aer = 1.0 - ((tp_sum * 2) / (pred_sum + gold_sum)) if (pred_sum + gold_sum) else 1.0
    if precision > 0 and recall > 0:
        f_divident = (f_alpha / precision) + ((1.0 - f_alpha) / recall)
        f_measure = 1.0 / f_divident
    else:
        f_measure = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "aer": aer,
        "total_true_positives": tp_sum,
        "total_predicted": pred_sum,
        "total_gold": gold_sum,
    }


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
        >>> adapter = OutlinesAdapter("Qwen/Qwen3-0.6B")
        >>> source = ["The", "cat", "sat"]
        >>> target = ["Le", "chat", "assis"]
        >>> result = align_tokens(adapter, source, target, "English", "French")
        >>> result.alignment.alignment  # doctest: +NORMALIZE_WHITESPACE
        [TokenAlignment(source='The', target='Le'), TokenAlignment(source='cat', target='chat'), TokenAlignment(source='sat', target='assis')]
    """
    # Create mappings before processing
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
        include_schema=llm_adapter.include_schema,
        use_reasoning=llm_adapter.use_reasoning,
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
) -> AlignmentResult:
    """
    Async version of align_tokens with retry/accumulation parity to sync path.
    """
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    source_mapping = create_token_mapping(source_tokens, marker_generator)
    target_mapping = create_token_mapping(target_tokens, marker_generator)

    messages: List[Message] = _create_alignment_messages(
        source_tokens,
        target_tokens,
        source_language,
        target_language,
        guidelines,
        examples,
        marker_generator,
        include_schema=llm_adapter.include_schema,  # parity with sync
        use_reasoning=llm_adapter.use_reasoning,
    )

    attempts: List[AlignmentAttempt] = []
    valid_alignments: List[TokenAlignment] = []
    alignment: Optional[TextAlignment] = None

    # Track explicitly unaligned tokens and remaining sets
    unaligned_source: set[str] = set()
    unaligned_target: set[str] = set()
    remaining_source: set[str] = set(source_mapping.uniquified)
    remaining_target: set[str] = set(target_mapping.uniquified)

    for attempt in range(max_retries):
        current_messages = format_messages(messages)
        current_attempt = AlignmentAttempt(
            attempt_number=attempt + 1,
            messages_sent=current_messages.copy(),
            raw_response=None,
            validation_passed=False,
            validation_errors=[],
        )

        # Track prior remaining sets to determine whether this attempt made progress
        prev_remaining_source = set(remaining_source)
        prev_remaining_target = set(remaining_target)
        try:
            raw = await llm_adapter.acall(current_messages)
            ta = ensure_text_alignment(raw)
            current_attempt.raw_response = ta

            (
                _,
                error_messages,
                new_valid_alignments,
                rem_src,
                rem_tgt,
            ) = _validate_alignment(
                ta,
                source_tokens,
                target_tokens,
                marker_generator,
                existing_alignments=valid_alignments,
                source_mapping=source_mapping,
                target_mapping=target_mapping,
            )

            # Track explicit <unaligned> entries
            for align in ta.alignment:
                if align.target == UNALIGNED_MARKER:
                    unaligned_source.add(align.source)
                if align.source == UNALIGNED_MARKER:
                    unaligned_target.add(align.target)

            # Filter out <unaligned> pairs
            new_valid_alignments = [
                a
                for a in new_valid_alignments
                if a.source != UNALIGNED_MARKER and a.target != UNALIGNED_MARKER
            ]

            # Deduplicate and re-sort via TextAlignment
            if new_valid_alignments:
                existing_pairs = {(a.source, a.target) for a in valid_alignments}
                unique_pairs = {
                    (a.source, a.target) for a in new_valid_alignments
                } - existing_pairs
                if unique_pairs:
                    valid_alignments.extend(
                        TokenAlignment(source=s, target=t) for s, t in unique_pairs
                    )
                    # Rebuild to enforce sorting/dedup
                    temp = TextAlignment(
                        alignment=valid_alignments,
                        source_mapping=source_mapping,
                        target_mapping=target_mapping,
                    )
                    valid_alignments = temp.alignment

            # Update remaining after excluding explicit unaligned
            remaining_source = rem_src - unaligned_source
            remaining_target = rem_tgt - unaligned_target

            is_complete = not (remaining_source or remaining_target)
            progress = (
                len(remaining_source) < len(prev_remaining_source)
                or len(remaining_target) < len(prev_remaining_target)
                or bool(new_valid_alignments)
            )
            current_attempt.validation_passed = progress
            current_attempt.validation_errors = error_messages

            if is_complete:
                alignment = TextAlignment(
                    alignment=valid_alignments,
                    source_mapping=source_mapping,
                    target_mapping=target_mapping,
                )
                attempts.append(current_attempt)
                break

            # Prepare retry: retain assistant response and ask for remaining
            messages.append(AssistantMessage(ta))
            messages.append(
                _create_retry_message(
                    valid_alignments,
                    remaining_source,
                    remaining_target,
                    source_tokens,
                    target_tokens,
                    marker_generator,
                )
            )

        except Exception as e:
            current_attempt.exception = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

        attempts.append(current_attempt)

    # Partial alignment if incomplete but something valid exists
    if not alignment and valid_alignments:
        alignment = TextAlignment(
            alignment=valid_alignments,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
        )

    # Sort final alignment by position
    if alignment:
        alignment = alignment.sort_by_position(source_mapping, target_mapping)

    return AlignmentResult(alignment=alignment, attempts=attempts)


def batch_sequences(sequences: list, chunk_size: int) -> list[list]:
    """Split sequences into chunks of specified size."""
    return batch_iterable(sequences, chunk_size)


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
    batch_size: Optional[int] = None,
    show_progress: bool = True,
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

    # Precompute mappings
    source_mappings = [
        create_token_mapping(src, marker_generator) for src in source_sequences
    ]
    target_mappings = [
        create_token_mapping(tgt, marker_generator) for tgt in target_sequences
    ]

    # Initialize per-sequence message histories and state
    sequence_messages: list[list[Message]] = [
        _create_alignment_messages(
            src,
            tgt,
            source_language,
            target_language,
            guidelines,
            examples,
            marker_generator,
            include_schema=llm_adapter.include_schema,
            use_reasoning=llm_adapter.use_reasoning,
        )
        for src, tgt in zip(source_sequences, target_sequences)
    ]
    sequence_attempts: list[list[AlignmentAttempt]] = [[] for _ in source_sequences]
    final_results: list[Optional[TextAlignment]] = [None] * len(source_sequences)
    existing_valid_alignments: list[list[TokenAlignment]] = [
        [] for _ in source_sequences
    ]
    retry_indices = list(range(len(source_sequences)))

    # Create progress bar to track completed sequences
    pbar = tqdm(
        total=len(source_sequences),
        desc="Aligning (batched)",
        disable=not show_progress,
        unit="seq",
    )

    # Resolve effective batch size when not provided by caller
    if batch_size is None:
        pref = getattr(llm_adapter, "preferred_batch_size", None)
        val = llm_adapter.preferred_batch_size() if callable(pref) else None
        batch_size = val if isinstance(val, int) and val > 0 else DEFAULT_BATCH_SIZE

    # Track remaining/unaligned state per sequence for correct progress detection
    remaining_sources_state: list[set[str]] = [
        set(m.uniquified) for m in source_mappings
    ]
    remaining_targets_state: list[set[str]] = [
        set(m.uniquified) for m in target_mappings
    ]
    unaligned_sources_state: list[set[str]] = [set() for _ in source_sequences]
    unaligned_targets_state: list[set[str]] = [set() for _ in target_sequences]

    for attempt in range(max_retries):
        if not retry_indices:
            break

        new_retry_indices_total: list[int] = []

        # process in chunks of batch_size
        for chunk in batch_sequences(retry_indices, batch_size):
            batch_to_run = [format_messages(sequence_messages[i]) for i in chunk]
            try:
                batch_results = llm_adapter.batch(batch_to_run)
            except Exception as e:
                logger.warning(f"Batch attempt {attempt + 1} failed for chunk: {e}")
                for bi, seq_idx in enumerate(chunk):
                    sequence_attempts[seq_idx].append(
                        AlignmentAttempt(
                            attempt_number=attempt + 1,
                            messages_sent=batch_to_run[bi],
                            raw_response=None,
                            validation_passed=False,
                            validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                            exception=str(e),
                        )
                    )
                    new_retry_indices_total.append(seq_idx)
                continue

            for bi, seq_idx in enumerate(chunk):
                result = batch_results[bi]
                msgs_sent = batch_to_run[bi]

                if result is None:
                    sequence_attempts[seq_idx].append(
                        AlignmentAttempt(
                            attempt_number=attempt + 1,
                            messages_sent=msgs_sent,
                            raw_response=None,
                            validation_passed=False,
                            validation_errors=[
                                (ValidationErrorType.OTHER, "Generation failed", [])
                            ],
                        )
                    )
                    new_retry_indices_total.append(seq_idx)
                    continue

                # Normalize and validate
                try:
                    ta = ensure_text_alignment(result)
                except Exception as e:
                    sequence_attempts[seq_idx].append(
                        AlignmentAttempt(
                            attempt_number=attempt + 1,
                            messages_sent=msgs_sent,
                            raw_response=None,
                            validation_passed=False,
                            validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                            exception=str(e),
                        )
                    )
                    new_retry_indices_total.append(seq_idx)
                    continue

                (
                    is_valid,
                    error_msg,
                    valid_aligns,
                    remaining_source,
                    remaining_target,
                ) = _validate_alignment(
                    ta,
                    source_sequences[seq_idx],
                    target_sequences[seq_idx],
                    marker_generator,
                    existing_alignments=existing_valid_alignments[seq_idx],
                    source_mapping=source_mappings[seq_idx],
                    target_mapping=target_mappings[seq_idx],
                )

                # Track explicit <unaligned> tokens from this generation and update remaining sets
                unaligned_source_cur = {
                    a.source for a in ta.alignment if a.target == UNALIGNED_MARKER
                }
                unaligned_target_cur = {
                    a.target for a in ta.alignment if a.source == UNALIGNED_MARKER
                }
                unaligned_sources_state[seq_idx].update(unaligned_source_cur)
                unaligned_targets_state[seq_idx].update(unaligned_target_cur)
                remaining_source = remaining_source - unaligned_sources_state[seq_idx]
                remaining_target = remaining_target - unaligned_targets_state[seq_idx]

                # Determine progress compared to previous remaining sets
                prev_rs = remaining_sources_state[seq_idx]
                prev_rt = remaining_targets_state[seq_idx]

                # Filter out UNALIGNED pairs from stored valid alignments
                valid_aligns_filtered = [
                    a
                    for a in valid_aligns
                    if a.source != UNALIGNED_MARKER and a.target != UNALIGNED_MARKER
                ]

                # Correctly detect progress by checking if we got NEW alignments
                prev_alignment_count = len(existing_valid_alignments[seq_idx])
                new_alignment_count = len(valid_aligns_filtered)
                has_new_alignments = new_alignment_count > prev_alignment_count

                progress = (
                    len(remaining_source) < len(prev_rs)
                    or len(remaining_target) < len(prev_rt)
                    or has_new_alignments
                )

                remaining_sources_state[seq_idx] = set(remaining_source)
                remaining_targets_state[seq_idx] = set(remaining_target)

                # Update stored alignments AFTER checking progress
                existing_valid_alignments[seq_idx] = valid_aligns_filtered

                is_complete = not (remaining_source or remaining_target)

                sequence_attempts[seq_idx].append(
                    AlignmentAttempt(
                        attempt_number=attempt + 1,
                        messages_sent=msgs_sent,
                        raw_response=ta,
                        validation_passed=progress,
                        validation_errors=error_msg if not is_complete else [],
                    )
                )

                if is_complete:
                    final_results[seq_idx] = TextAlignment(
                        alignment=valid_aligns_filtered,
                        source_mapping=source_mappings[seq_idx],
                        target_mapping=target_mappings[seq_idx],
                    ).sort_by_position(
                        source_mappings[seq_idx], target_mappings[seq_idx]
                    )
                else:
                    # Keep conversation history and add retry prompts (assistant -> user)
                    sequence_messages[seq_idx].append(AssistantMessage(ta))
                    sequence_messages[seq_idx].append(
                        _create_retry_message(
                            valid_aligns_filtered,
                            remaining_source,
                            remaining_target,
                            source_sequences[seq_idx],
                            target_sequences[seq_idx],
                            marker_generator,
                        )
                    )
                    new_retry_indices_total.append(seq_idx)

        # Update progress: sequences that are no longer in retry list are complete
        newly_completed = len(retry_indices) - len(new_retry_indices_total)
        pbar.update(newly_completed)

        retry_indices = new_retry_indices_total

    # Update progress bar for any remaining incomplete sequences and close
    pbar.update(len(retry_indices))  # Count remaining incomplete as "done" (failed)
    pbar.close()

    # Create final AlignmentResults
    final_alignment_results: list[AlignmentResult] = []
    for i in range(len(source_sequences)):
        attempts = sequence_attempts[i]
        result = final_results[i]
        # Fallback to partial alignments if no complete result
        if result is None and existing_valid_alignments[i]:
            result = TextAlignment(
                alignment=existing_valid_alignments[i],
                source_mapping=source_mappings[i],
                target_mapping=target_mappings[i],
            )
        sorted_result = (
            result.sort_by_position(source_mappings[i], target_mappings[i])
            if isinstance(result, TextAlignment)
            else None
        )
        final_alignment_results.append(
            AlignmentResult(alignment=sorted_result, attempts=attempts)
        )
    return final_alignment_results


def align_dataset(
    adapter: LLMAdapter,
    source_sequences: Sequence[Sequence[str] | str],
    target_sequences: Sequence[Sequence[str] | str],
    *,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
    concurrency: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> list[AlignmentResult]:
    """
    Align multiple sequence pairs with automatic optimization.

    Automatically selects the best processing strategy:
    - True batching if adapter supports it and batch_size specified
    - Async concurrent processing if adapter supports async
    - Sequential processing otherwise

    Args:
        adapter: LLM adapter to use
        source_sequences: List of source token sequences (strings or lists)
        target_sequences: List of target token sequences (strings or lists)
        source_language: Source language name
        target_language: Target language name
        guidelines: Optional alignment guidelines
        examples: Optional few-shot examples
        max_retries: Max retries per alignment (default: 3)
        marker_generator: Optional marker generator
        concurrency: Max concurrent requests for async (default: from adapter)
        batch_size: Batch size for batching (default: from adapter)
        show_progress: Show progress bar (default: True)

    Returns:
        List of AlignmentResult objects

    Example:
        >>> from lexi_align.adapters.outlines_adapter import OutlinesAdapter
        >>> adapter = OutlinesAdapter("Qwen/Qwen3-0.6B")
        >>> source = [["The", "cat"], ["A", "dog"]]
        >>> target = [["Le", "chat"], ["Un", "chien"]]
        >>> results = align_dataset(adapter, source, target, show_progress=False)
        >>> len(results)
        2
        >>> all(r.alignment for r in results)
        True
    """
    if len(source_sequences) != len(target_sequences):
        raise ValueError(
            f"Number of source and target sequences must match "
            f"(got {len(source_sequences)} vs {len(target_sequences)})"
        )

    # Normalize sequences to lists of strings
    def normalize_seq(seq: Sequence[str] | str) -> list[str]:
        return seq.split() if isinstance(seq, str) else list(seq)

    sources = [normalize_seq(s) for s in source_sequences]
    targets = [normalize_seq(t) for t in target_sequences]

    # Get adapter capabilities
    caps = adapter.capabilities

    # Determine batch size
    effective_batch_size = batch_size or caps.get("preferred_batch_size")

    # Determine concurrency
    effective_concurrency = (
        concurrency or caps.get("max_concurrency") or DEFAULT_CONCURRENCY
    )

    # Select processing strategy
    if effective_batch_size and caps["supports_batching"]:
        logger.info(f"Using batch processing (batch_size={effective_batch_size})")
        return list(
            align_tokens_batched(
                adapter,
                sources,
                targets,
                source_language=source_language,
                target_language=target_language,
                guidelines=guidelines,
                examples=examples,
                max_retries=max_retries,
                marker_generator=marker_generator,
                batch_size=effective_batch_size,
                show_progress=show_progress,
            )
        )
    elif caps["supports_async"]:
        logger.info(f"Using async processing (concurrency={effective_concurrency})")
        return asyncio_run(
            align_many_async(
                adapter,
                list(zip(sources, targets)),
                source_language=source_language,
                target_language=target_language,
                guidelines=guidelines,
                examples=examples,
                max_retries=max_retries,
                marker_generator=marker_generator,
                concurrency=effective_concurrency,
                show_progress=show_progress,
            )
        )
    else:
        logger.info("Using sequential processing")

        # Add progress wrapper for sequential processing
        if show_progress:
            results = []
            for s, t in tqdm(
                zip(sources, targets),
                total=len(sources),
                desc="Aligning",
                disable=not show_progress,
            ):
                result = align_tokens(
                    adapter,
                    s,
                    t,
                    source_language=source_language,
                    target_language=target_language,
                    guidelines=guidelines,
                    examples=examples,
                    max_retries=max_retries,
                    marker_generator=marker_generator,
                )
                results.append(result)
            return results
        else:
            return align_many(
                adapter,
                list(zip(sources, targets)),
                source_language=source_language,
                target_language=target_language,
                guidelines=guidelines,
                examples=examples,
                max_retries=max_retries,
                marker_generator=marker_generator,
            )


def align_and_evaluate_dataset(
    adapter: LLMAdapter,
    source_sequences: Sequence[Sequence[str] | str],
    target_sequences: Sequence[Sequence[str] | str],
    gold_alignments: Sequence[TextAlignment],
    *,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
    marker_generator: Optional[MarkerGenerator] = None,
    concurrency: Optional[int] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = True,
) -> tuple[list[AlignmentResult], dict[str, Any]]:
    """Align dataset and evaluate against gold standard.

    Convenience function that combines alignment and evaluation in one call.

    Args:
        adapter: LLM adapter to use
        source_sequences: List of source token sequences
        target_sequences: List of target token sequences
        gold_alignments: Gold standard alignments for evaluation
        source_language: Source language name
        target_language: Target language name
        guidelines: Optional alignment guidelines
        examples: Optional few-shot examples
        max_retries: Max retries per alignment
        marker_generator: Optional marker generator
        concurrency: Max concurrent requests for async
        batch_size: Batch size for batching
        show_progress: Show progress bar

    Returns:
        Tuple of (alignment_results, metrics_dict)

    Example:
        >>> from lexi_align.adapters.outlines_adapter import OutlinesAdapter
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> adapter = OutlinesAdapter("Qwen/Qwen3-0.6B")
        >>> sources = [["The", "cat"], ["A", "dog"]]
        >>> targets = [["Le", "chat"], ["Un", "chien"]]
        >>> gold = [
        ...     TextAlignment(alignment=[
        ...         TokenAlignment(source="The", target="Le"),
        ...         TokenAlignment(source="cat", target="chat")
        ...     ]),
        ...     TextAlignment(alignment=[
        ...         TokenAlignment(source="A", target="Un"),
        ...         TokenAlignment(source="dog", target="chien")
        ...     ])
        ... ]
        >>> results, metrics = align_and_evaluate_dataset(
        ...     adapter, sources, targets, gold, show_progress=False
        ... )
        >>> len(results)
        2
        >>> "micro" in metrics
        True
    """
    if len(gold_alignments) != len(source_sequences):
        raise ValueError(
            f"Number of gold alignments must match number of sequences "
            f"(got {len(gold_alignments)} vs {len(source_sequences)})"
        )

    # Run alignment
    results = align_dataset(
        adapter,
        source_sequences,
        target_sequences,
        source_language=source_language,
        target_language=target_language,
        guidelines=guidelines,
        examples=examples,
        max_retries=max_retries,
        marker_generator=marker_generator,
        concurrency=concurrency,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    # Filter successful alignments and corresponding gold
    predicted = [r.alignment for r in results if r.alignment]
    gold = [g for r, g in zip(results, gold_alignments) if r.alignment]

    if not predicted:
        logger.warning("No successful alignments to evaluate")
        return results, {
            "micro": {
                "precision": 0.0,
                "recall": 0.0,
                "f_measure": 0.0,
                "aer": 1.0,
                "total_predicted": 0,
                "total_gold": 0,
                "total_true_positives": 0,
            },
            "macro": {
                "precision": 0.0,
                "recall": 0.0,
                "f_measure": 0.0,
                "aer": 1.0,
            },
            "per_example": [],
            "token_stats": {"source": {}, "target": {}},
            "summary": {
                "total_examples": len(results),
                "total_predictions": 0,
                "total_gold": sum(len(g.alignment) for g in gold_alignments),
            },
        }

    # Evaluate
    metrics = evaluate_alignments(predicted, gold)

    return results, metrics


def evaluate_alignments(
    predictions: Sequence[TextAlignment],
    gold: Sequence[TextAlignment],
) -> dict[str, Any]:
    """
    Evaluate predicted alignments against gold standard.

    Returns comprehensive metrics including:
    - Micro-averaged precision, recall, F1, AER
    - Per-alignment metrics
    - Token-level statistics

    Args:
        predictions: List of predicted TextAlignment objects
        gold: List of gold standard TextAlignment objects

    Returns:
        Dictionary with evaluation metrics

    Example:
        >>> from lexi_align.models import TextAlignment, TokenAlignment
        >>> pred = [TextAlignment(alignment=[
        ...     TokenAlignment(source="the", target="le"),
        ...     TokenAlignment(source="cat", target="chat")
        ... ])]
        >>> gold = [TextAlignment(alignment=[
        ...     TokenAlignment(source="the", target="le"),
        ...     TokenAlignment(source="cat", target="chat")
        ... ])]
        >>> metrics = evaluate_alignments(pred, gold)
        >>> metrics['micro']['precision']
        1.0
        >>> metrics['micro']['f_measure']
        1.0
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"Number of predictions and gold alignments must match "
            f"(got {len(predictions)} vs {len(gold)})"
        )

    from lexi_align.metrics import calculate_metrics

    # Calculate per-example metrics
    per_example_metrics = []
    for pred, gold_align in zip(predictions, gold):
        metrics = calculate_metrics(pred, gold_align)
        per_example_metrics.append(metrics)

    # Calculate micro-averaged metrics
    micro_metrics = build_micro_metrics(list(zip(predictions, gold)))

    # Calculate macro-averaged metrics
    macro_precision = sum(m["precision"] for m in per_example_metrics) / len(
        per_example_metrics
    )
    macro_recall = sum(m["recall"] for m in per_example_metrics) / len(
        per_example_metrics
    )
    macro_f = sum(m["f_measure"] for m in per_example_metrics) / len(
        per_example_metrics
    )
    macro_aer = sum(m["aer"] for m in per_example_metrics) / len(per_example_metrics)

    # Token-level analysis
    token_stats = _analyze_token_level_accuracy(predictions, gold)

    return {
        "micro": micro_metrics,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f_measure": macro_f,
            "aer": macro_aer,
        },
        "per_example": per_example_metrics,
        "token_stats": token_stats,
        "summary": {
            "total_examples": len(predictions),
            "total_predictions": sum(len(p.alignment) for p in predictions),
            "total_gold": sum(len(g.alignment) for g in gold),
        },
    }


def _analyze_token_level_accuracy(
    predictions: Sequence[TextAlignment],
    gold: Sequence[TextAlignment],
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Analyze per-token alignment accuracy across dataset."""

    # Track statistics per unique token
    source_stats: dict[str, dict[str, int]] = {}
    target_stats: dict[str, dict[str, int]] = {}

    for pred, gold_align in zip(predictions, gold):
        pred_pairs = {(a.source, a.target) for a in pred.alignment}
        gold_pairs = {(a.source, a.target) for a in gold_align.alignment}
        correct_pairs = pred_pairs & gold_pairs

        # Track source tokens
        for src, tgt in gold_pairs:
            if src not in source_stats:
                source_stats[src] = {"correct": 0, "total": 0}
            source_stats[src]["total"] += 1
            if (src, tgt) in correct_pairs:
                source_stats[src]["correct"] += 1

        # Track target tokens
        for src, tgt in gold_pairs:
            if tgt not in target_stats:
                target_stats[tgt] = {"correct": 0, "total": 0}
            target_stats[tgt]["total"] += 1
            if (src, tgt) in correct_pairs:
                target_stats[tgt]["correct"] += 1

    # Calculate accuracy for each token
    def compute_accuracy(
        stats: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int | float]]:
        return {
            token: {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": counts["correct"] / counts["total"]
                if counts["total"] > 0
                else 0.0,
            }
            for token, counts in stats.items()
            if counts["total"] >= 3  # Only include tokens with 3+ occurrences
        }

    return {
        "source": compute_accuracy(source_stats),
        "target": compute_accuracy(target_stats),
    }


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
    messages_dicts = custom_messages.copy()  # Make a copy to not modify the input
    messages_dicts.append(
        {
            "role": "user",
            "content": SOURCE_TOKENS_PREFIX
            + " ".join(make_unique(source_tokens))
            + "\n"
            + TARGET_TOKENS_PREFIX
            + " ".join(make_unique(target_tokens)),
        }
    )
    formatted_messages = format_messages(messages_dicts)

    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)
    try:
        result = llm_adapter(formatted_messages)

        # Normalize result to TextAlignment
        result = ensure_text_alignment(result)

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
            source_mapping=source_mapping,
            target_mapping=target_mapping,
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
            alignment = alignment.sort_by_position(source_mapping, target_mapping)

        return AlignmentResult(
            alignment=alignment,
            attempts=[
                AlignmentAttempt(
                    attempt_number=1,
                    messages_sent=formatted_messages,
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
                    messages_sent=formatted_messages,
                    raw_response=None,
                    validation_passed=False,
                    validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                    exception=str(e),
                )
            ],
        )


async def align_tokens_raw_async(
    llm_adapter: LLMAdapter,
    source_tokens: List[str],
    target_tokens: List[str],
    custom_messages: List[Dict[str, Any]],
) -> AlignmentResult:
    """
    Async version of align_tokens_raw. Awaits the adapter and never calls asyncio.run.
    """
    messages_dicts = custom_messages.copy()
    messages_dicts.append(
        {
            "role": "user",
            "content": SOURCE_TOKENS_PREFIX
            + " ".join(make_unique(source_tokens))
            + "\n"
            + TARGET_TOKENS_PREFIX
            + " ".join(make_unique(target_tokens)),
        }
    )
    formatted_messages = format_messages(messages_dicts)

    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)
    try:
        result = await llm_adapter.acall(formatted_messages)
        result = ensure_text_alignment(result)
        (
            is_valid,
            error_messages,
            valid_alignments,
            _,
            _,
        ) = _validate_alignment(
            result,
            source_tokens,
            target_tokens,
            marker_generator=None,
            existing_alignments=None,
            source_mapping=source_mapping,
            target_mapping=target_mapping,
        )

        alignment = (
            TextAlignment(
                alignment=valid_alignments,
                source_mapping=source_mapping,
                target_mapping=target_mapping,
            )
            if valid_alignments
            else None
        )

        if alignment:
            alignment = alignment.sort_by_position(source_mapping, target_mapping)

        return AlignmentResult(
            alignment=alignment,
            attempts=[
                AlignmentAttempt(
                    attempt_number=1,
                    messages_sent=formatted_messages,
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
                    messages_sent=formatted_messages,
                    raw_response=None,
                    validation_passed=False,
                    validation_errors=[(ValidationErrorType.OTHER, str(e), [])],
                    exception=str(e),
                )
            ],
        )
