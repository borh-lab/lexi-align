try:
    from litellm import acompletion, completion
except ImportError:
    raise ImportError(
        "litellm not installed. Install directly or using 'pip install lexi-align[litellm]'"
    )

from logging import getLogger
from typing import Any, Optional, cast

import litellm

from lexi_align.adapters.base import LLMAdapter
from lexi_align.models import ChatMessageDict, TextAlignment, TextAlignmentSchema
from lexi_align.utils import (
    extract_existing_alignments_from_messages,
    extract_tokens_and_retry_flag,
    select_alignment_schema,
)

logger = getLogger(__name__)


class LiteLLMAdapter(LLMAdapter):
    """Adapter for running models via litellm."""

    def __init__(
        self,
        model_params: Optional[dict[str, Any]] = None,
        use_dynamic_schema: bool = False,
        min_alignments: int = 0,
    ):
        """Initialize the adapter with model parameters."""
        self.model_params = model_params or {}
        self.use_dynamic_schema = bool(use_dynamic_schema)
        self.min_alignments = int(min_alignments or 0)
        # Always include the schema in prompts for parity with other adapters
        self.include_schema = True
        self.json_retry_attempts = 3

    def supports_length_constraints(self) -> bool:
        return self.use_dynamic_schema

    def _select_response_schema(
        self, messages: list[ChatMessageDict]
    ) -> type[TextAlignmentSchema]:
        if not self.use_dynamic_schema:
            return TextAlignmentSchema
        src_tokens, tgt_tokens, is_retry = extract_tokens_and_retry_flag(
            cast(list[dict], messages)
        )
        existing_aligns = extract_existing_alignments_from_messages(
            cast(list[dict], messages)
        )
        return select_alignment_schema(
            src_tokens,
            tgt_tokens,
            min_alignments=self.min_alignments,
            is_retry=is_retry,
            existing_alignments=existing_aligns,
        )

    async def acall(self, messages: list[ChatMessageDict]) -> TextAlignment:
        """Async version using acompletion with JSON-retry wrappers."""
        base_seed = int(self.model_params.get("seed", 0) or 0)

        schema_for_response = self._select_response_schema(messages)

        async def _agen(seed: Optional[int]) -> TextAlignment:
            params = dict(self.model_params)
            if seed is not None:
                params["seed"] = seed
            response = await acompletion(
                messages=cast(list[dict], messages),
                response_format=schema_for_response,
                **params,
            )
            content = response.choices[0].message.content
            return TextAlignment.model_validate_json(content)

        return await self._retry_on_invalid_json_async(
            _agen,
            max_retries=self.json_retry_attempts,
            base_seed=base_seed,
        )

    def __call__(self, messages: list[ChatMessageDict]) -> TextAlignment:
        """Synchronous version using completion with JSON-retry wrappers."""
        base_seed = int(self.model_params.get("seed", 0) or 0)

        schema_for_response = self._select_response_schema(messages)

        def _gen(seed: Optional[int]) -> TextAlignment:
            params = dict(self.model_params)
            if seed is not None:
                params["seed"] = seed
            response = completion(
                messages=cast(list[dict], messages),
                response_format=schema_for_response,
                **params,
            )
            content = response.choices[0].message.content
            return TextAlignment.model_validate_json(content)

        return self._retry_on_invalid_json(
            _gen,
            max_retries=self.json_retry_attempts,
            base_seed=base_seed,
        )


def custom_callback(kwargs, completion_response, start_time, end_time):
    """Callback for custom logging."""
    logger.debug(kwargs["litellm_params"]["metadata"])


def track_cost_callback(kwargs, completion_response, start_time, end_time):
    """Callback for cost tracking."""
    try:
        response_cost = kwargs["response_cost"]
        logger.info(f"regular response_cost: {response_cost}")
    except Exception:
        pass


def get_transformed_inputs(kwargs):
    """Callback for logging transformed inputs."""
    params_to_model = kwargs["additional_args"]["complete_input_dict"]
    logger.info(f"params to model: {params_to_model}")


# Set up litellm callbacks
litellm.input_callback = [get_transformed_inputs]
litellm.success_callback = [track_cost_callback, custom_callback]
