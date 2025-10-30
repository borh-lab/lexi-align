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
from lexi_align.models import ChatMessageDict, TextAlignment
from lexi_align.utils import (
    to_text_alignment,
)

logger = getLogger(__name__)


class LiteLLMAdapter(LLMAdapter):
    """Adapter for running models via litellm."""

    def __init__(
        self,
        model_params: Optional[dict[str, Any]] = None,
        use_dynamic_schema: bool = False,
        min_alignments: int = 0,
        use_reasoning: bool = False,
    ):
        """Initialize the adapter with model parameters."""
        self.model_params = model_params or {}

        # Set default timeout to 15 minutes if not specified
        if "timeout" not in self.model_params:
            self.model_params["timeout"] = 900.0  # 15 minutes

        self._init_common_params(
            min_alignments,
            use_dynamic_schema,
            json_retry_attempts=3,
            use_reasoning=use_reasoning,
        )
        # Always include the schema in prompts for parity with other adapters
        self.include_schema = True

    def supports_length_constraints(self) -> bool:
        return self.use_dynamic_schema

    async def acall(self, messages: list[ChatMessageDict]) -> TextAlignment:
        """Async version using acompletion with JSON-retry wrappers."""
        base_seed = int(self.model_params.get("seed", 0) or 0)

        schema_for_response = self._select_schema_for_messages(messages)

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
            content = content.strip() if isinstance(content, str) else str(content)
            schema_obj = schema_for_response.model_validate_json(content, strict=True)
            return to_text_alignment(schema_obj)

        return await self._retry_on_invalid_json_async(
            _agen,
            max_retries=self.json_retry_attempts,
            base_seed=base_seed,
        )

    def __call__(self, messages: list[ChatMessageDict]) -> TextAlignment:
        """Synchronous version using completion with JSON-retry wrappers."""
        base_seed = int(self.model_params.get("seed", 0) or 0)

        schema_for_response = self._select_schema_for_messages(messages)

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
            content = content.strip() if isinstance(content, str) else str(content)
            schema_obj = schema_for_response.model_validate_json(content, strict=True)
            return to_text_alignment(schema_obj)

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
