from typing import Any
from lexi_align.adapters.base import LLMAdapter
from lexi_align.models import TextAlignment
from logging import getLogger
import json

logger = getLogger(__name__)

try:
    from litellm import completion
    import litellm
except ImportError:
    raise ImportError(
        "litellm is not installed. Install with 'pip install lexi-align[litellm]'"
    )


class LiteLLMAdapter(LLMAdapter):
    """Adapter for running models via litellm."""

    def __init__(self, model_params: dict[str, Any]):
        self.model_params = model_params

    def __call__(self, messages: list[dict]) -> TextAlignment:
        content = None
        try:
            resp = completion(
                messages=messages,
                response_format=TextAlignment,
                **self.model_params,
            )
            content = resp.choices[0].message.content

            # String fallback:
            if isinstance(content, str):
                logger.info("Model response is a string; attempting to parse as JSON")
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse model response as JSON: {e}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError("Model response was not valid JSON")

            return TextAlignment.model_validate(content)
        except Exception as e:
            logger.error(f"Error running model: {e}")
            if content:
                logger.debug(f"Raw response: {content}")
            raise


def custom_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    logger.debug(kwargs["litellm_params"]["metadata"])


def track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    try:
        response_cost = kwargs[
            "response_cost"
        ]  # litellm calculates response cost for you
        logger.info(f"regular response_cost: {response_cost}")
    except Exception:
        pass


def get_transformed_inputs(
    kwargs,
):
    params_to_model = kwargs["additional_args"]["complete_input_dict"]
    logger.info(f"params to model: {params_to_model}")


litellm.input_callback = [get_transformed_inputs]
litellm.success_callback = [track_cost_callback, custom_callback]
