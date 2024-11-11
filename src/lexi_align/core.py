from typing import Optional, List, Tuple, Union, Dict, Any, Sequence
from logging import getLogger
import asyncio
from lexi_align.models import TextAlignment
from lexi_align.text_processing import remove_unique_one, create_subscript_generator
from lexi_align.utils import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    Message,
    format_messages,
    make_unique,
    format_tokens,
)
from lexi_align.text_processing import MarkerGenerator
from lexi_align.models import (
    TokenAlignment,
    AlignmentResult,
    AlignmentAttempt,
)
from lexi_align.adapters import LLMAdapter

logger = getLogger(__name__)


def _sort_alignment(
    alignment: TextAlignment, source_tokens: list[str], target_tokens: list[str]
) -> TextAlignment:
    """Sort alignment pairs by original token positions in source and target sequences."""
    if not alignment or not alignment.alignment:
        return alignment

    # Get default marker generator pattern for removing markers
    marker_pattern = create_subscript_generator().pattern

    # Sort the alignment pairs by original positions
    sorted_pairs = sorted(
        alignment.alignment,
        key=lambda x: (
            source_tokens.index(remove_unique_one(x.source_token, marker_pattern)),
            target_tokens.index(remove_unique_one(x.target_token, marker_pattern)),
        ),
    )

    return TextAlignment(alignment=sorted_pairs)


def _validate_alignment(
    alignment: TextAlignment,
    source_tokens: list[str],
    target_tokens: list[str],
    marker_generator: Optional[MarkerGenerator] = None,
    existing_alignments: Optional[List[TokenAlignment]] = None,
) -> tuple[bool, list[str], list[TokenAlignment], set[str], set[str]]:
    """
    Validate alignment and extract valid alignments and remaining tokens.

    Args:
        alignment: The alignment to validate
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        marker_generator: Optional generator for unique markers
        existing_alignments: Optional list of existing valid alignments

    Returns:
        Tuple containing:
        - is_valid: Whether the alignment contains any valid alignments
        - error_messages: List of validation error messages
        - valid_alignments: List of valid TokenAlignment objects
        - remaining_source: Set of unaligned source tokens
        - remaining_target: Set of unaligned target tokens
    """
    unique_source = set(make_unique(source_tokens, marker_generator))
    unique_target = set(make_unique(target_tokens, marker_generator))

    valid_alignments = list(existing_alignments) if existing_alignments else []
    error_messages = []

    # Validate and collect new valid alignments
    new_valid_alignments = []
    for align in alignment.alignment:
        if align.source_token not in unique_source:
            error_messages.append(
                f'Invalid source token "{align.source_token}" in alignment: {align.model_dump_json()}'
            )
        elif align.target_token not in unique_target:
            error_messages.append(
                f'Invalid target token "{align.target_token}" in alignment: {align.model_dump_json()}'
            )
        else:
            new_valid_alignments.append(align)

    valid_alignments.extend(new_valid_alignments)

    # Calculate remaining tokens that haven't been aligned yet
    aligned_sources = {a.source_token for a in valid_alignments}
    aligned_targets = {a.target_token for a in valid_alignments}
    remaining_source = unique_source - aligned_sources
    remaining_target = unique_target - aligned_targets

    if remaining_source:
        error_messages.append(
            f"Still need alignments for source tokens: {', '.join(sorted(remaining_source))}"
        )
    if remaining_target:
        error_messages.append(
            f"Still need alignments for target tokens: {', '.join(sorted(remaining_target))}"
        )

    return (
        bool(valid_alignments),
        error_messages,
        valid_alignments,
        remaining_source,
        remaining_target,
    )


def _create_retry_message(
    error_messages: List[str],
    valid_alignments: List[TokenAlignment],
    source_tokens: List[str],
    target_tokens: List[str],
    marker_generator: Optional[MarkerGenerator],
) -> UserMessage:
    """Create message for retry attempts."""
    return UserMessage(
        f"The previous alignment was partially valid. Please provide alignments for the remaining tokens:\n\n"
        f"{chr(10).join(error_messages)}\n\n"
        f"Already valid alignments: {valid_alignments}\n\n"
        f"Source tokens: {make_unique(source_tokens, marker_generator)}\n"
        f"Target tokens: {make_unique(target_tokens, marker_generator)}"
    )


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
            raw_response = llm_adapter(current_messages)
            current_attempt.raw_response = raw_response

            (
                _,  # is_valid not needed
                error_messages,
                valid_alignments,
                remaining_source,
                remaining_target,
            ) = _validate_alignment(
                raw_response,
                source_tokens,
                target_tokens,
                marker_generator,
                valid_alignments,
            )

            is_complete = not (remaining_source or remaining_target)
            current_attempt.validation_passed = bool(valid_alignments)
            current_attempt.validation_errors = error_messages

            if is_complete:
                alignment = TextAlignment(alignment=valid_alignments)
                attempts.append(current_attempt)
                break

            messages.append(AssistantMessage(raw_response))
            messages.append(
                _create_retry_message(
                    error_messages,
                    valid_alignments,
                    source_tokens,
                    target_tokens,
                    marker_generator,
                )
            )

        except Exception as e:
            current_attempt.exception = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

        attempts.append(current_attempt)

    if not alignment and valid_alignments:
        alignment = TextAlignment(alignment=valid_alignments)

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
    messages: List[Message] = []

    # Use default subscript generator if none provided
    if marker_generator is None:
        marker_generator = create_subscript_generator()

    # Create example with duplicates to show marker usage
    example_source = ["a", "a", "b", "a"]
    example_target = ["c", "b", "c"]
    unique_source = make_unique(example_source, marker_generator)
    unique_target = make_unique(example_target, marker_generator)

    system_msg = (
        "You are an expert translator and linguistic annotator"
        + (
            f" from {source_language} to {target_language}."
            if source_language and target_language
            else "."
        )
        + "\nGiven a list of tokens in the source and target, your task is to align them. Do not further split or merge the tokens and use the exact case/form of the tokens provided as-is."
        + f"\nFor duplicate tokens, unique markers will be added like this: source='{' '.join(unique_source)}', target='{' '.join(unique_target)}'"
    )

    if guidelines:
        system_msg += f"\nHere are annotation guidelines you should strictly follow:\n\n{guidelines}"
    if examples:
        system_msg += (
            "\nReturn alignments in the same format as the following examples:"
        )

    messages.append(SystemMessage(system_msg))

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
        [TokenAlignment(source_token='The', target_token='Le'),
         TokenAlignment(source_token='cat', target_token='chat'),
         TokenAlignment(source_token='sat', target_token='assis')]
    """
    messages = _create_alignment_messages(
        source_tokens,
        target_tokens,
        source_language,
        target_language,
        guidelines,
        examples,
        marker_generator,
    )

    result = _process_alignment_sync(
        llm_adapter,
        messages,
        source_tokens,
        target_tokens,
        marker_generator,
        max_retries,
    )

    # Sort alignment if present
    if result.alignment:
        result.alignment = _sort_alignment(
            result.alignment, source_tokens, target_tokens
        )

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
        [TokenAlignment(source_token='The', target_token='Le'),
         TokenAlignment(source_token='cat', target_token='chat'),
         TokenAlignment(source_token='sat', target_token='assis')]
    """
    messages = _create_alignment_messages(
        source_tokens,
        target_tokens,
        source_language,
        target_language,
        guidelines,
        examples,
        marker_generator,
    )

    result = None
    error_msg: list[str] = []

    for attempt in range(max_retries):
        try:
            # Use acall if available, otherwise fall back to sync call
            if hasattr(llm_adapter, "acall"):
                result = await llm_adapter.acall(format_messages(messages))
            else:
                result = llm_adapter(format_messages(messages))

            # Validate the alignment
            (
                is_valid,
                error_msg,
                _,  # valid_alignments not used
                _,  # remaining_source not used
                _,  # remaining_target not used
            ) = _validate_alignment(
                result,
                source_tokens,
                target_tokens,
                marker_generator,
                existing_alignments=None,  # FIXME We need to pass the exisiting alignments here!
            )

            if is_valid:
                return result

            # Add the failed response to the message history
            messages.append(AssistantMessage(result))

            messages.append(
                UserMessage(
                    f"The previous alignment was invalid. The following issues were found:\n\n"
                    f"{chr(10).join(error_msg)}\n\n"
                    f"Please provide a new alignment using only these exact tokens:\n"
                    f"Source tokens: {make_unique(source_tokens)}\n"
                    f"Target tokens: {make_unique(target_tokens)}"
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
                            validation_errors=error_msg,
                            exception=str(e),
                        )
                    ],
                )

    return AlignmentResult(
        alignment=None,
        attempts=[
            AlignmentAttempt(
                attempt_number=max_retries,
                messages_sent=format_messages(messages),
                raw_response=None,
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

    # Track attempts and results for each sequence
    sequence_attempts: list[list[AlignmentAttempt]] = [[] for _ in source_sequences]
    final_results: list[Optional[TextAlignment]] = [None] * len(source_sequences)  # type: ignore

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
                            validation_errors=["Generation failed"],
                        )
                    )
                    new_retry_indices.append(seq_idx)
                    continue

                exising_alignments = None
                if (
                    seq_idx < len(final_results)
                    and final_results[seq_idx] is not None
                    and isinstance(final_results[seq_idx], TextAlignment)
                ):
                    exising_alignments = final_results[seq_idx].alignment

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
                    existing_alignments=exising_alignments,
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
                            alignment=valid_alignments
                        )
                    else:
                        # Only access alignment if we know it's a TextAlignment
                        final_results[seq_idx].alignment.extend(valid_alignments)

                if not is_valid:
                    # Add retry message with information about remaining tokens
                    retry_messages[batch_idx].append(
                        UserMessage(
                            "The previous alignment was partially valid. Please provide alignments for the remaining tokens:\n\n"
                            + (
                                f"Remaining source tokens: {', '.join(remaining_source)}\n"
                                if remaining_source
                                else ""
                            )
                            + (
                                f"Remaining target tokens: {', '.join(remaining_target)}"
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
                        validation_errors=[],
                        exception=str(e),
                    )
                )

    # Create final AlignmentResults
    final_alignment_results = []
    for i, (result, attempts) in enumerate(zip(final_results, sequence_attempts)):
        sorted_result = None
        if isinstance(result, TextAlignment):
            # Sort the alignment if present
            sorted_result = _sort_alignment(
                result, source_sequences[i], target_sequences[i]
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
        >>> adapter = OutlinesAdapter("meta-llama/Llama-3.2-3B-Instruct")
        >>> source = ["The", "cat", "sat"]
        >>> target = ["Le", "chat", "assis"]
        >>> messages = [
        ...     {"role": "system", "content": "You are a translator aligning English to French."},
        ...     {"role": "user", "content": "Align these tokens:\\n"
        ...         f"English: {' '.join(source)}\\n"
        ...         f"French: {' '.join(target)}"}
        ... ]
        >>> result = align_tokens_raw(adapter, source, target, messages)
        >>> result.alignment.alignment  # doctest: +NORMALIZE_WHITESPACE
        [TokenAlignment(source_token='The', target_token='Le'),
         TokenAlignment(source_token='cat', target_token='chat'),
         TokenAlignment(source_token='sat', target_token='assis')]
    """
    messages = custom_messages
    messages.append(
        {
            "role": "user",
            "content": (
                f"source_tokens: {make_unique(source_tokens)}\n"
                f"target_tokens: {make_unique(target_tokens)}"
            ),
        }
    )

    try:
        if asyncio.iscoroutinefunction(getattr(llm_adapter, "acall", None)):
            result = asyncio.run(llm_adapter.acall(messages))
        else:
            result = llm_adapter(messages)

        # Sort alignment if present
        if isinstance(result, TextAlignment):
            result = _sort_alignment(result, source_tokens, target_tokens)

        return AlignmentResult(
            alignment=result if isinstance(result, TextAlignment) else None,
            attempts=[
                AlignmentAttempt(
                    attempt_number=1,
                    messages_sent=messages,
                    raw_response=result,
                    validation_passed=isinstance(result, TextAlignment),
                    validation_errors=[],
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
                    validation_errors=[],
                    exception=str(e),
                )
            ],
        )
