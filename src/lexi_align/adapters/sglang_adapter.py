from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, cast

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict

import openai
from outlines import Generator, from_sglang  # type: ignore

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    TextAlignment,
    TextAlignmentSchema,
    TokenAlignment,
)
from lexi_align.utils import (
    extract_existing_alignments_from_messages,
    extract_tokens_and_retry_flag,
    select_alignment_schema,
    to_text_alignment,
)

logger = getLogger(__name__)


class SGLangAdapter(LLMAdapter):
    """Adapter for using an SGLang server via Outlines' sglang backend.

    This adapter connects to a running SGLang OpenAI-compatible server using the
    OpenAI client and leverages Outlines structured generation.

    Example:
        >>> # Requires a running SGLang server; example is skipped.
        >>> from lexi_align.adapters.sglang_adapter import SGLangAdapter  # doctest: +SKIP
        >>> adapter = SGLangAdapter(base_url="http://localhost:11434")    # doctest: +SKIP
        >>> msgs = [
        ...     {"role": "system", "content": "Align tokens."},
        ...     {"role": "user", "content": "Source tokens: The cat\\nTarget tokens: Le chat"}
        ... ]                                                             # doctest: +SKIP
        >>> _ = adapter(msgs)                                             # doctest: +SKIP
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        # Sampling / generation controls (forwarded to OpenAI chat API)
        temperature: float = 0.0,
        samples: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        # Low-level kwargs
        client_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        # Optional controls forwarded to SGLang / OpenAI-like API
        presence_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        min_alignments: Optional[int] = 0,
        json_retry_attempts: int = 3,
    ):
        """Initialize SGLang adapter.

        Args:
            base_url: URL of the SGLang server (OpenAI-compatible endpoint)
            api_key: Optional API key (SGLang often ignores, but OpenAI client requires a value)
            model: Optional model identifier to send to the SGLang server (OpenAI-compatible)
            temperature: Sampling temperature
            samples: Number of samples (not all backends use this)
            top_k: Optional top-k
            top_p: Optional top-p
            beam_size: Optional beam size
            max_tokens: Maximum tokens (forwarded as 'max_tokens')
            client_kwargs: Additional kwargs for OpenAI client
            generation_kwargs: Extra kwargs forwarded to the OpenAI chat API
            extra_body: SGLang-specific parameters passed under 'extra_body'
            presence_penalty: Optional presence penalty forwarded to SGLang/OpenAI
            min_p: Optional minimum probability mass forwarded to SGLang/OpenAI
        """
        self.base_url = base_url
        self.api_key = api_key or "not-needed"
        self.model_id = model or "gpt-4o-mini"
        self.temperature = temperature
        self.samples = samples
        self.top_k = top_k
        self.top_p = top_p
        self.beam_size = beam_size
        self.max_tokens: Optional[int] = max_tokens
        self.generation_kwargs = generation_kwargs or {}
        self.extra_body = extra_body
        self.presence_penalty = presence_penalty
        self.min_p = min_p

        self._client_kwargs = client_kwargs or {}

        # OpenAI clients (sync/async)
        self._sync_client = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key, **self._client_kwargs
        )
        self._async_client = openai.AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, **self._client_kwargs
        )

        # Outlines models (lazy init)
        self._model: Optional[Any] = None
        self._amodel: Optional[Any] = None

        self.include_schema = True  # Include JSON Schema in prompt by default
        self.min_alignments = min_alignments or 0
        self.json_retry_attempts = json_retry_attempts

    @property
    def model(self):
        """Lazy init of the Outlines SGLang model (sync)."""
        if self._model is None:
            self._model = from_sglang(self._sync_client, self.model_id)
        return self._model

    @property
    def amodel(self):
        """Lazy init of the Outlines SGLang model (async)."""
        if self._amodel is None:
            self._amodel = from_sglang(self._async_client, self.model_id)
        return self._amodel

    def supports_true_batching(self) -> bool:
        """Indicate batch support (implemented in adapter)."""
        return False

    def supports_length_constraints(self) -> bool:
        """SGLang supports dynamic schema length constraints via Outlines."""
        return True

    def _get_schema_class(
        self,
        source_tokens: list[str],
        target_tokens: list[str],
        is_retry: bool = False,
        existing_alignments: Optional[List[TokenAlignment]] = None,
    ) -> Type[TextAlignmentSchema]:
        """Select dynamic schema with length constraints (mirrors OutlinesAdapter)."""
        return select_alignment_schema(
            source_tokens,
            target_tokens,
            min_alignments=self.min_alignments or 0,
            is_retry=is_retry,
            existing_alignments=existing_alignments,
        )

    def _inference_kwargs(self) -> Dict[str, Any]:
        """Build kwargs forwarded to OpenAI Chat Completions (SGLang)."""
        kwargs: Dict[str, Any] = {}

        # OpenAI-supported arguments
        kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if getattr(self, "presence_penalty", None) is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        # Map samples -> n (OpenAI-compatible)
        if getattr(self, "samples", None) is not None and int(self.samples) > 1:
            kwargs["n"] = int(self.samples)

        # Collect SGLang-specific params into extra_body
        extra_body: Dict[str, Any] = {}
        if self.extra_body:
            extra_body.update(self.extra_body)

        if self.top_k is not None:
            extra_body.setdefault("top_k", self.top_k)
        if self.beam_size is not None:
            extra_body.setdefault("beam_size", self.beam_size)
        if getattr(self, "min_p", None) is not None:
            extra_body.setdefault("min_p", self.min_p)

        # Hoist known SGLang keys from generation_kwargs into extra_body
        gen = dict(self.generation_kwargs or {})
        for k in ("top_k", "beam_size", "min_p"):
            if k in gen and k not in extra_body:
                extra_body[k] = gen.pop(k)

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Remaining generation kwargs (OpenAI-recognized ones can stay top-level)
        kwargs.update(gen)
        return kwargs

    def __call__(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Generate alignments using the SGLang model through OpenAI chat.completions."""
        # Extract tokens and retry state
        source_tokens, target_tokens, is_retry = extract_tokens_and_retry_flag(
            cast(List[Dict[str, Any]], messages)
        )
        existing_alignments = extract_existing_alignments_from_messages(
            cast(List[Dict[str, Any]], messages)
        )
        schema_class = self._get_schema_class(
            source_tokens, target_tokens, is_retry, existing_alignments
        )
        schema_json = schema_class.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_json.get("title", "DynamicTextAlignmentSchema"),
                "schema": schema_json,
            },
        }
        base_seed = int((self.generation_kwargs or {}).get("seed", 0) or 0)

        def _gen(seed: Optional[int]) -> TextAlignment:
            client_args: Dict[str, Any] = {
                "model": self.model_id,
                "messages": cast(List[Dict[str, Any]], messages),
                "response_format": response_format,
                **self._inference_kwargs(),
            }
            if self.max_tokens is not None:
                client_args["max_tokens"] = self.max_tokens
            if seed is not None:
                client_args["seed"] = seed
            resp = self._sync_client.chat.completions.create(**client_args)
            content = resp.choices[0].message.content
            return to_text_alignment(content)

        try:
            return self._retry_on_invalid_json(
                _gen,
                max_retries=self.json_retry_attempts,
                base_seed=base_seed,
            )
        except Exception as e:
            logger.error(f"SGLangAdapter call failed: {type(e).__name__}: {e}")
            raise

    async def acall(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Async generation using OpenAI Async client with SGLang."""
        source_tokens, target_tokens, is_retry = extract_tokens_and_retry_flag(
            cast(List[Dict[str, Any]], messages)
        )
        existing_alignments = extract_existing_alignments_from_messages(
            cast(List[Dict[str, Any]], messages)
        )
        schema_class = self._get_schema_class(
            source_tokens, target_tokens, is_retry, existing_alignments
        )
        schema_json = schema_class.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_json.get("title", "DynamicTextAlignmentSchema"),
                "schema": schema_json,
            },
        }
        base_seed = int((self.generation_kwargs or {}).get("seed", 0) or 0)

        async def _agen(seed: Optional[int]) -> TextAlignment:
            client_args: Dict[str, Any] = {
                "model": self.model_id,
                "messages": cast(List[Dict[str, Any]], messages),
                "response_format": response_format,
                **self._inference_kwargs(),
            }
            if self.max_tokens is not None:
                client_args["max_tokens"] = self.max_tokens
            if seed is not None:
                client_args["seed"] = seed
            resp = await self._async_client.chat.completions.create(**client_args)
            content = resp.choices[0].message.content
            return to_text_alignment(content)

        try:
            return await self._retry_on_invalid_json_async(
                _agen,
                max_retries=self.json_retry_attempts,
                base_seed=base_seed,
            )
        except Exception as e:
            logger.error(f"SGLangAdapter async call failed: {type(e).__name__}: {e}")
            raise

    def batch(
        self,
        batch_messages: list[list["ChatMessageDict"]],
        max_retries: int = 3,
    ) -> list[Optional[TextAlignment]]:
        """Generate alignments for a batch of message sequences.

        Returns:
            A list where each element is a TextAlignment or None if an item failed.
        """
        # Format inputs and select schemas per example
        inputs: list[list["ChatMessageDict"]] = []
        schema_classes: list[Type[TextAlignmentSchema]] = []

        for messages in batch_messages:
            try:
                source_tokens, target_tokens, is_retry = extract_tokens_and_retry_flag(
                    cast(List[Dict[str, Any]], messages)
                )
                existing_alignments = extract_existing_alignments_from_messages(
                    cast(List[Dict[str, Any]], messages)
                )
                schema_class = self._get_schema_class(
                    source_tokens, target_tokens, is_retry, existing_alignments
                )
            except ValueError:
                logger.warning(f"Could not find tokens in messages: {messages}")
                schema_class = TextAlignmentSchema

            schema_classes.append(schema_class)
            inputs.append(messages)

        # Process inputs with their corresponding schemas
        batch_results: list[Optional[TextAlignment]] = []
        for i, (msgs, schema_class) in enumerate(zip(inputs, schema_classes)):
            try:
                gen = Generator(self.model, schema_class)
                gen_kwargs = self._inference_kwargs()
                if self.max_tokens is not None:
                    gen_kwargs["max_tokens"] = self.max_tokens
                result = gen(msgs, **gen_kwargs)
                ta = to_text_alignment(result)
                if ta.alignment:
                    batch_results.append(ta)
                else:
                    logger.error("Received empty alignment from SGLangAdapter in batch")
                    batch_results.append(None)
            except Exception as e:
                logger.error(
                    f"Error processing input {i}:\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}\n",
                    exc_info=True,
                )
                batch_results.append(None)

        return batch_results
