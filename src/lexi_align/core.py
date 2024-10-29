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

    Returns:
        TextAlignment: An object containing the aligned tokens

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
    messages: List[Message] = []
    messages.append(
        SystemMessage(
            (
                (
                    f"You are an expert translator and linguistic annotator from {source_language} to {target_language}."
                    if source_language and target_language
                    else "You are an expert translator and linguistic annotator."
                )
                + "\n Given a list of tokens in the source and target, your task is to align them."
            )
            + (
                f"\nHere are annotation guidelines you should strictly follow:\n\n{guidelines}"
                if guidelines
                else ""
            )
            + (
                "Return alignments in the same format as the given examples."
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

    return llm_adapter(format_messages(messages))


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
        >>> adapter = LiteLLMAdapter(model_params={"model": "gpt-3.5-turbo"})
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
