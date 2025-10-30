"""Mixins for common adapter functionality."""

from logging import getLogger
from typing import Any, Awaitable, Callable, Optional

from pydantic import ValidationError as PydanticValidationError

logger = getLogger(__name__)


class RetryMixin:
    """Mixin providing JSON retry logic for adapters."""

    def _is_json_invalid_error(self, e: Exception) -> bool:
        """Check if error is a JSON validation error that warrants retry."""
        if isinstance(e, PydanticValidationError):
            return True
        s = str(e)
        return any(
            phrase in s
            for phrase in ["Invalid JSON", "json_invalid", "EOF while parsing a string"]
        )

    def _retry_on_invalid_json(
        self,
        gen: Callable[[Optional[int]], Any],
        max_retries: int,
        base_seed: int,
    ) -> Any:
        """Retry generation on invalid JSON errors with incrementing seeds."""
        for i in range(max_retries):
            seed = base_seed + i + 1
            try:
                return gen(seed)
            except Exception as e:
                should_retry = self._is_json_invalid_error(e)
                if not should_retry or i == max_retries - 1:
                    raise
                logger.warning(
                    f"{self.__class__.__name__}: retrying due to invalid JSON "
                    f"(attempt {i + 1}/{max_retries}, seed={seed}): {e}"
                )
        raise RuntimeError("Unexpected fall-through in _retry_on_invalid_json")

    async def _retry_on_invalid_json_async(
        self,
        agen: Callable[[Optional[int]], Awaitable[Any]],
        max_retries: int,
        base_seed: int,
    ) -> Any:
        """Async variant of _retry_on_invalid_json."""
        for i in range(max_retries):
            seed = base_seed + i + 1
            try:
                return await agen(seed)
            except Exception as e:
                should_retry = self._is_json_invalid_error(e)
                if not should_retry or i == max_retries - 1:
                    raise
                logger.warning(
                    f"{self.__class__.__name__}: retrying due to invalid JSON "
                    f"(attempt {i + 1}/{max_retries}, seed={seed}): {e}"
                )
        raise RuntimeError("Unexpected fall-through in _retry_on_invalid_json_async")


class InferenceKwargsMixin:
    """Mixin for building inference kwargs with None filtering."""

    def _build_inference_kwargs(self, **params: Any) -> dict[str, Any]:
        """Build inference kwargs, filtering out None values."""
        from lexi_align.utils.common import filter_none_values

        return filter_none_values(params)
