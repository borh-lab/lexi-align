from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, cast

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict

import openai
from outlines import Generator, from_sglang  # type: ignore

from lexi_align.adapters import LLMAdapter
from lexi_align.constants import (
    DEFAULT_MAX_LOG_CHARS,
    TEMPERATURE_INCREMENT_PER_RETRY,
)
from lexi_align.models import (
    TextAlignment,
    TextAlignmentSchema,
)
from lexi_align.utils.common import filter_none_values, redact_for_logging

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
        use_dynamic_schema: bool = True,
        use_reasoning: bool = False,
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
            use_reasoning: Whether to include reasoning field in responses
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

        # Set default timeout to 15 minutes if not specified
        if "timeout" not in self._client_kwargs:
            import httpx

            self._client_kwargs["timeout"] = httpx.Timeout(900.0)  # 15 minutes

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
        self._init_common_params(
            min_alignments, use_dynamic_schema, json_retry_attempts, use_reasoning
        )

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
        return self.use_dynamic_schema

    def _build_client_args(
        self,
        messages: list["ChatMessageDict"],
        schema_class: Type[TextAlignmentSchema],
        seed: Optional[int],
        base_seed: int,
    ) -> tuple[Dict[str, Any], int]:
        """Build request arguments for OpenAI client (shared by sync/async).

        Returns:
            Tuple of (client_args dict, attempt_idx)
        """
        schema_json = schema_class.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_json.get("title", "DynamicTextAlignmentSchema"),
                "schema": schema_json,
                "strict": True,
            },
        }

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

        # Increase temperature by TEMPERATURE_INCREMENT_PER_RETRY per retry attempt
        attempt_idx = 0
        if seed is not None:
            try:
                attempt_idx = max(0, int(seed) - int(base_seed) - 1)
            except Exception:
                attempt_idx = 0
        base_temp = float(client_args.get("temperature", self.temperature or 0.0))
        client_args["temperature"] = (
            base_temp + TEMPERATURE_INCREMENT_PER_RETRY * attempt_idx
        )

        # Debug logging for max_tokens
        logger.info(
            f"SGLang client_args max_tokens: {client_args.get('max_tokens', 'NOT SET')}, "
            f"self.max_tokens: {self.max_tokens}"
        )
        logger.debug(f"Full SGLang client_args keys: {list(client_args.keys())}")
        logger.debug(
            f"Full SGLang client_args (redacted): {redact_for_logging(client_args)}"
        )

        return client_args, attempt_idx

    def _inference_kwargs(self) -> Dict[str, Any]:
        """Build kwargs forwarded to OpenAI Chat Completions (SGLang)."""
        # OpenAI-supported arguments with None filtering
        kwargs = filter_none_values(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": getattr(self, "presence_penalty", None),
            }
        )

        # Map samples -> n (OpenAI-compatible)
        if getattr(self, "samples", None) is not None and int(self.samples) > 1:
            kwargs["n"] = int(self.samples)

        # Collect SGLang-specific params into extra_body
        extra_body: Dict[str, Any] = dict(self.extra_body) if self.extra_body else {}

        for key in ("top_k", "beam_size", "min_p"):
            if (val := getattr(self, key, None)) is not None:
                extra_body.setdefault(key, val)

        # Hoist known SGLang keys from generation_kwargs
        gen = dict(self.generation_kwargs or {})
        for k in ("top_k", "beam_size", "min_p"):
            if k in gen and k not in extra_body:
                extra_body[k] = gen.pop(k)

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Remaining generation kwargs
        kwargs.update(gen)
        return kwargs

    def __call__(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Generate alignments using the SGLang model through OpenAI chat.completions."""
        schema_class = self._select_schema_for_messages(messages)
        base_seed = int((self.generation_kwargs or {}).get("seed", 0) or 0)

        def _gen(seed: Optional[int]) -> TextAlignment:
            client_args, attempt_idx = self._build_client_args(
                messages, schema_class, seed, base_seed
            )
            logger.debug(
                "SGLangAdapter sync call",
                extra={
                    "attempt_idx": attempt_idx,
                    "seed": seed,
                    "temperature": client_args["temperature"],
                },
            )
            resp = self._sync_client.chat.completions.create(**client_args)
            content = resp.choices[0].message.content
            content = content.strip() if isinstance(content, str) else str(content)
            logger.debug(f"SGLang response content: {content}")
            logger.debug(f"SGLang response content length: {len(content)}")
            return self._validate_response_content(
                content,
                schema_class,
                context={
                    "seed": seed,
                    "mode": "sync",
                    "model": self.model_id,
                    "max_log_chars": int(
                        (self.generation_kwargs or {}).get(
                            "log_max_chars", DEFAULT_MAX_LOG_CHARS
                        )
                    ),
                },
            )

        try:
            return self._retry_on_invalid_json(
                _gen,
                max_retries=self.json_retry_attempts,
                base_seed=base_seed,
            )
        except Exception as e:
            logger.error(
                "SGLangAdapter call failed",
                extra={"error_type": type(e).__name__, "model": self.model_id},
                exc_info=True,
            )
            logger.error(f"Exception details: {str(e)}")
            if hasattr(e, "__cause__") and e.__cause__:
                logger.error(f"Caused by: {str(e.__cause__)}")
            raise

    async def acall(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Async generation using OpenAI Async client with SGLang."""
        schema_class = self._select_schema_for_messages(messages)
        base_seed = int((self.generation_kwargs or {}).get("seed", 0) or 0)

        async def _agen(seed: Optional[int]) -> TextAlignment:
            client_args, attempt_idx = self._build_client_args(
                messages, schema_class, seed, base_seed
            )
            logger.debug(
                "SGLangAdapter async call",
                extra={
                    "attempt_idx": attempt_idx,
                    "seed": seed,
                    "temperature": client_args["temperature"],
                },
            )
            resp = await self._async_client.chat.completions.create(**client_args)
            content = resp.choices[0].message.content
            content = content.strip() if isinstance(content, str) else str(content)
            logger.debug(f"SGLang async response content: {content}")
            logger.debug(f"SGLang async response content length: {len(content)}")
            return self._validate_response_content(
                content,
                schema_class,
                context={
                    "seed": seed,
                    "mode": "async",
                    "model": self.model_id,
                    "max_log_chars": int(
                        (self.generation_kwargs or {}).get(
                            "log_max_chars", DEFAULT_MAX_LOG_CHARS
                        )
                    ),
                },
            )

        try:
            return await self._retry_on_invalid_json_async(
                _agen,
                max_retries=self.json_retry_attempts,
                base_seed=base_seed,
            )
        except Exception as e:
            logger.error(
                "SGLangAdapter async call failed",
                extra={"error_type": type(e).__name__, "model": self.model_id},
                exc_info=True,
            )
            logger.error(f"Async exception details: {str(e)}")
            if hasattr(e, "__cause__") and e.__cause__:
                logger.error(f"Caused by: {str(e.__cause__)}")
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

        def process_one(
            messages: list["ChatMessageDict"], schema_class: Type[TextAlignmentSchema]
        ) -> Any:
            """Process a single message sequence with its schema."""
            gen = Generator(self.model, schema_class)
            gen_kwargs = self._inference_kwargs()
            if self.max_tokens is not None:
                gen_kwargs["max_tokens"] = self.max_tokens
            return gen(messages, **gen_kwargs)

        return self._batch_with_per_item_schema(batch_messages, process_one)
