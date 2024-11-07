from lexi_align.utils import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    format_messages,
    make_unique,
)
from lexi_align.models import TextAlignment
from lexi_align.adapters import LLMAdapter
from typing import Optional, List, Tuple, Union
from logging import getLogger

logger = getLogger(__name__)

Message = Union[SystemMessage, UserMessage, AssistantMessage]


def align_tokens(
    llm_adapter: LLMAdapter,
    source_tokens: list[str],
    target_tokens: list[str],
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    guidelines: Optional[str] = None,
    examples: Optional[List[Tuple[List[str], List[str], TextAlignment]]] = None,
    max_retries: int = 3,
) -> TextAlignment:
    """
    Align tokens from source language to target language using a language model.
    Handles token uniqueness internally.

    Args:
        llm_adapter (LLMAdapter): An adapter instance for running the language model
        source_tokens (list[str]): List of tokens in the source language
        target_tokens (list[str]): List of tokens in the target language
        source_language (str, optional): The source language name
        target_language (str, optional): The target language name
        guidelines (str, optional): Specific guidelines for the alignment task
        examples (list[tuple], optional): List of example alignments
        max_retries (int, optional): Maximum number of retries for invalid alignments

    Returns:
        TextAlignment: An object containing the aligned tokens

    Raises:
        ValueError: If unable to get valid alignment after max_retries attempts

    Example:
        >>> from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
        >>> from lexi_align.core import align_tokens
        >>>
        >>> # Set up the language model adapter (must support JSON output)
        >>> adapter = LiteLLMAdapter(model_params={"model": "gpt-4o"})
        >>>
        >>> source_tokens = ["The", "cat", "is", "on", "the", "mat"]
        >>> target_tokens = ["Le", "chat", "est", "sur", "le", "tapis"]
        >>>
        >>> alignment = align_tokens(
        ...     adapter,
        ...     source_tokens,
        ...     target_tokens,
        ...     source_language="English",
        ...     target_language="French"
        ... )
        >>>
        >>> print(alignment)
        TextAlignment(alignment=[
            TokenAlignment(source_token='The', target_token='Le'),
            TokenAlignment(source_token='cat', target_token='chat'),
            TokenAlignment(source_token='is', target_token='est'),
            TokenAlignment(source_token='on', target_token='sur'),
            TokenAlignment(source_token='the', target_token='le'),
            TokenAlignment(source_token='mat', target_token='tapis')
        ])
    """

    def validate_alignment(
        alignment: TextAlignment, source_tokens: list[str], target_tokens: list[str]
    ) -> tuple[bool, Optional[str]]:
        """Validate alignment against source and target tokens."""
        # Create sets of unique tokens for comparison
        unique_source = set(make_unique(source_tokens))
        unique_target = set(make_unique(target_tokens))

        # Check each alignment pair
        for align in alignment.alignment:
            # Check if source token exists in uniquified source tokens
            if align.source_token not in unique_source:
                return False, f"Invalid source token: {align.source_token}"

            # Check if target token exists in uniquified target tokens
            if align.target_token not in unique_target:
                return False, f"Invalid target token: {align.target_token}"

        return True, None

    messages: List[Message] = []
    messages.append(
        SystemMessage(
            (
                (
                    f"You are an expert translator and linguistic annotator from {source_language} to {target_language}."
                    if source_language and target_language
                    else "You are an expert translator and linguistic annotator."
                )
                + "\nGiven a list of tokens in the source and target, your task is to align them."
            )
            + (
                f"\nHere are annotation guidelines you should strictly follow:\n\n{guidelines}"
                if guidelines
                else ""
            )
            + (
                "\nReturn alignments in the same format as the given examples."
                if examples
                else ""
            )
        )
    )

    def format_tokens(source_tokens: list[str], target_tokens: list[str]) -> str:
        # Create unique token mappings
        unique_source = make_unique(source_tokens)
        unique_target = make_unique(target_tokens)
        return f"source_tokens: {unique_source}\n" f"target_tokens: {unique_target}"

    if examples:
        for example_source_tokens, example_target_tokens, example_alignment in examples:
            messages.append(
                UserMessage(format_tokens(example_source_tokens, example_target_tokens))
            )
            messages.append(AssistantMessage(example_alignment))

    messages.append(UserMessage(format_tokens(source_tokens, target_tokens)))

    for attempt in range(max_retries):
        try:
            result = llm_adapter(format_messages(messages))

            # Validate the alignment
            is_valid, error_msg = validate_alignment(
                result, source_tokens, target_tokens
            )

            if is_valid:
                return result

            # Add the failed response to the message history
            messages.append(AssistantMessage(result))

            # If invalid, add detailed error feedback and retry
            logger.warning(f"Attempt {attempt + 1}: Invalid alignment: {error_msg}")

            # Build list of problematic alignments
            invalid_alignments = []
            unique_source = set(make_unique(source_tokens))
            unique_target = set(make_unique(target_tokens))

            for align in result.alignment:
                if align.source_token not in unique_source:
                    invalid_alignments.append(
                        f"TokenAlignment(source_token='{align.source_token}', target_token='{align.target_token}') - Invalid source token '{align.source_token}'"
                    )
                if align.target_token not in unique_target:
                    invalid_alignments.append(
                        f"TokenAlignment(source_token='{align.source_token}', target_token='{align.target_token}') - Invalid target token '{align.target_token}'"
                    )

            messages.append(
                UserMessage(
                    f"The previous alignment was invalid. The following alignments contain invalid tokens:\n\n"
                    f"{chr(10).join(invalid_alignments)}\n\n"
                    f"Please provide a new alignment using only these exact tokens:\n"
                    f"Source tokens: {make_unique(source_tokens)}\n"
                    f"Target tokens: {make_unique(target_tokens)}"
                )
            )

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            # If we're on the last attempt, extract valid alignments
            if attempt == max_retries - 1:
                unique_source = set(make_unique(source_tokens))
                unique_target = set(make_unique(target_tokens))

                valid_alignments = [
                    align
                    for align in result.alignment
                    if align.source_token in unique_source
                    and align.target_token in unique_target
                ]

                if valid_alignments:
                    logger.warning(
                        f"Returning {len(valid_alignments)} valid alignments after {max_retries} failed attempts"
                    )
                    return TextAlignment(alignment=valid_alignments)
                raise

    # If we get here and have no valid alignments, raise the error
    raise ValueError(
        f"Failed to get any valid alignments after {max_retries} attempts. "
        f"Last error: {error_msg}"
    )


def align_tokens_raw(
    llm_adapter: LLMAdapter,
    source_tokens: list[str],
    target_tokens: list[str],
    custom_messages: list[dict],
) -> TextAlignment:
    """
    Align tokens using custom messages instead of the default system/guidelines/examples template.

    Args:
        llm_adapter (LLMAdapter): An adapter instance for running the language model
        source_tokens (list[str]): List of tokens in the source language
        target_tokens (list[str]): List of tokens in the target language
        custom_messages (list[dict]): List of custom message dictionaries

    Returns:
        TextAlignment: An object containing the aligned tokens

    Example:
        >>> from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
        >>> from lexi_align.core import align_tokens_raw
        >>>
        >>> # Set up the language model adapter
        >>> adapter = LiteLLMAdapter(model_params={"model": "gpt-4o-mini"})
        >>>
        >>> source_tokens = ["The", "cat", "is", "on", "the", "mat"]
        >>> target_tokens = ["Le", "chat", "est", "sur", "le", "tapis"]
        >>>
        >>> custom_messages = [
        ...     {"role": "system", "content": "You are a translator aligning English to French."},
        ...     {"role": "user", "content": f"Align these tokens:\nEnglish: {' '.join(source_tokens)}\nFrench: {' '.join(target_tokens)}"}
        ... ]
        >>>
        >>> alignment = align_tokens_raw(
        ...     model_fn,
        ...     source_tokens,
        ...     target_tokens,
        ...     custom_messages
        ... )
        >>>
        >>> print(alignment)
        TextAlignment(alignment=[
            TokenAlignment(source_token='The', target_token='Le'),
            TokenAlignment(source_token='cat', target_token='chat'),
            TokenAlignment(source_token='is', target_token='est'),
            TokenAlignment(source_token='on', target_token='sur'),
            TokenAlignment(source_token='the', target_token='le'),
            TokenAlignment(source_token='mat', target_token='tapis')
        ])
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
    return llm_adapter(messages)
