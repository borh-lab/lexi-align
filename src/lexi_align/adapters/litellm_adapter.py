try:
    from litellm import acompletion, completion
except ImportError:
    raise ImportError(
        "litellm not installed. Install directly or using 'pip install lexi-align[litellm]'"
    )

from logging import getLogger
from typing import Any, Optional

import litellm

from lexi_align.adapters.base import LLMAdapter
from lexi_align.models import TextAlignment, TextAlignmentSchema

logger = getLogger(__name__)


class LiteLLMAdapter(LLMAdapter):
    """Adapter for running models via litellm."""

    def __init__(self, model_params: Optional[dict[str, Any]] = None):
        """Initialize the adapter with model parameters."""
        self.model_params = model_params or {}
        self.include_schema = False  # Schema is passed directly to the model's structured generation feature

    async def acall(self, messages: list[dict]) -> TextAlignment:
        """Async version using acompletion."""
        try:
            response = await acompletion(
                messages=messages,
                response_format=TextAlignmentSchema,
                **self.model_params,
            )

            # Extract the response content
            content = response.choices[0].message.content

            # Parse into TextAlignment
            return TextAlignment.model_validate_json(content)

        except Exception as e:
            logger.error(f"Error in async LiteLLM call: {e}")
            raise

    def __call__(self, messages: list[dict]) -> TextAlignment:
        """Synchronous version using completion."""
        try:
            response = completion(
                messages=messages,
                response_format=TextAlignmentSchema,
                **self.model_params,
            )

            # Extract the response content
            content = response.choices[0].message.content

            # Parse into TextAlignment
            return TextAlignment.model_validate_json(content)

        except Exception as e:
            logger.error(f"Error in LiteLLM call: {e}")
            raise


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
