import json
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

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    UNALIGNED_MARKER,
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
    to_text_alignment,
)

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
    Now handles explicit unaligned tokens and improved error reporting.
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
        from collections import Counter

        counts = Counter(tokens)
        return ", ".join(
            f"{token} (x{count})" if count > 1 else token
            for token, count in sorted(counts.items())
        )

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
    marker_generator: Optional[MarkerGenerator] = None,
) -> UserMessage:
    """Create message for retry attempts with partial alignments."""
    message_parts = []

    # First show the complete token lists (snake_case only)
    message_parts.append(
        "source_tokens: " + " ".join(make_unique(source_tokens, marker_generator))
    )
    message_parts.append(
        "target_tokens: " + " ".join(make_unique(target_tokens, marker_generator))
    )
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
    if remaining_source:
        message_parts.append(
            "remaining_source_tokens: " + " ".join(sorted(remaining_source))
        )
    if remaining_target:
        message_parts.append(
            "remaining_target_tokens: " + " ".join(sorted(remaining_target))
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

        try:
            raw_response = llm_adapter(current_messages)
            raw_response = to_text_alignment(raw_response)
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
        schema_obj = create_dynamic_alignment_schema(
            source_tokens, target_tokens, marker_generator
        ).model_json_schema()
        system_msg_parts.append(
            f"\nExpected JSON format:\n```json\n{json.dumps(schema_obj, ensure_ascii=False)}\n```"
        )

    system_msg_parts.extend(
        [
            "Constraints:",
            "1) Use only tokens from the enumerated sets; do not invent or normalize tokens.",
            '2) Emit exactly one JSON object with top-level key "alignment"; no extra text or markdown.',
            "3) Articles and determiners (e.g., 'the', 'a', 'an') are often <unaligned>; output <unaligned> rather than forcing an incorrect pair.",
            "4) Align punctuation only if both sides contain the corresponding punctuation; otherwise use <unaligned>.",
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
        for example_source_tokens, example_target_tokens, example_alignment in examples:
            messages.append(
                UserMessage(
                    "source_tokens: "
                    + " ".join(make_unique(example_source_tokens, marker_generator))
                    + "\n"
                    + "target_tokens: "
                    + " ".join(make_unique(example_target_tokens, marker_generator))
                )
            )
            messages.append(AssistantMessage(example_alignment))

    # Single final user message including both standard and snake_case blocks
    messages.append(
        UserMessage(
            "source_tokens: "
            + " ".join(make_unique(source_tokens, marker_generator))
            + "\n"
            + "target_tokens: "
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
    concurrency: int = 8,
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

    return list(
        await _asyncio.gather(*[_one(s, t) for s, t in zip(src_seqs, tgt_seqs)])
    )


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
        include_schema=llm_adapter.include_schema,  # NEW
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
        try:
            raw = await llm_adapter.acall(current_messages)
            ta = to_text_alignment(raw)
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
            include_schema=llm_adapter.include_schema,  # NEW
        )
        for src, tgt in zip(source_sequences, target_sequences)
    ]
    sequence_attempts: list[list[AlignmentAttempt]] = [[] for _ in source_sequences]
    final_results: list[Optional[TextAlignment]] = [None] * len(source_sequences)
    existing_valid_alignments: list[list[TokenAlignment]] = [
        [] for _ in source_sequences
    ]  # NEW
    retry_indices = list(range(len(source_sequences)))

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
                    ta = to_text_alignment(result)
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

                # Update existing valid alignments
                existing_valid_alignments[seq_idx] = valid_aligns

                sequence_attempts[seq_idx].append(
                    AlignmentAttempt(
                        attempt_number=attempt + 1,
                        messages_sent=msgs_sent,
                        raw_response=ta,
                        validation_passed=is_valid,
                        validation_errors=error_msg if not is_valid else [],
                    )
                )

                if is_valid:
                    final_results[seq_idx] = TextAlignment(
                        alignment=valid_aligns,
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
                            valid_aligns,
                            remaining_source,
                            remaining_target,
                            source_sequences[seq_idx],
                            target_sequences[seq_idx],
                            marker_generator,
                        )
                    )
                    new_retry_indices_total.append(seq_idx)

        retry_indices = new_retry_indices_total

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
            "content": "source_tokens: "
            + " ".join(make_unique(source_tokens))
            + "\n"
            + "target_tokens: "
            + " ".join(make_unique(target_tokens)),
        }
    )
    formatted_messages = format_messages(messages_dicts)

    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)
    try:
        result = llm_adapter(formatted_messages)

        # Normalize result to TextAlignment
        result = to_text_alignment(result)

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
            "content": "source_tokens: "
            + " ".join(make_unique(source_tokens))
            + "\n"
            + "target_tokens: "
            + " ".join(make_unique(target_tokens)),
        }
    )
    formatted_messages = format_messages(messages_dicts)

    source_mapping = create_token_mapping(source_tokens)
    target_mapping = create_token_mapping(target_tokens)
    try:
        result = await llm_adapter.acall(formatted_messages)
        result = to_text_alignment(result)
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
