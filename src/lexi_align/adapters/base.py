import asyncio
from abc import ABC, abstractmethod
from logging import getLogger
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional

from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict, TextAlignment

logger = getLogger(__name__)


class LLMAdapter(ABC):
    """Base class for LLM adapters."""

    include_schema: bool = False

    @abstractmethod
    def __call__(self, messages: list["ChatMessageDict"]) -> "TextAlignment":
        """Synchronous call to generate alignments."""
        pass

    async def acall(self, messages: list["ChatMessageDict"]) -> "TextAlignment":
        """
        Async call to generate alignments.
        Default implementation runs the sync call in a worker thread to avoid
        blocking the event loop. Adapters with true async can override this.
        """
        return await asyncio.to_thread(self.__call__, messages)

    def supports_true_batching(self) -> bool:
        """
        Check if the adapter supports true batched processing.
        Override this method to return True in adapters that implement efficient batching.
        """
        return False

    def supports_length_constraints(self) -> bool:
        """
        Check if the adapter supports alignment length constraints.
        Override this method to return True in adapters that support min/max alignment lengths.
        """
        return False

    def batch(
        self,
        batch_messages: List[List["ChatMessageDict"]],
        max_retries: int = 3,
    ) -> List[Optional["TextAlignment"]]:
        """
        Process multiple message sequences in batch.
        Default implementation processes sequences sequentially - override for true batch support.

        Args:
            batch_messages: List of message sequences to process
            max_retries: Maximum number of retries per sequence

        Returns:
            List of TextAlignment objects or None for failed generations
        """
        logger.warning(
            f"{self.__class__.__name__} does not support true batching - falling back to sequential processing"
        )
        results: List[Optional["TextAlignment"]] = []
        for messages in batch_messages:
            try:
                result = self(messages)
                results.append(result)
            except Exception as e:
                logger.warning(f"Sequential processing failed: {str(e)}")
                results.append(None)
        return results

    def _is_json_invalid_error(self, e: Exception) -> bool:
        """Detect Pydantic 'json_invalid' or similar invalid-JSON errors."""
        try:
            if isinstance(e, PydanticValidationError):
                return any(err.get("type") == "json_invalid" for err in e.errors())
        except Exception:
            pass
        s = str(e)
        return (
            "Invalid JSON" in s
            or "json_invalid" in s
            or "EOF while parsing a string" in s
        )

    def _retry_on_invalid_json(
        self,
        gen: Callable[[Optional[int]], "TextAlignment"],
        max_retries: int = 3,
        base_seed: int = 0,
    ) -> "TextAlignment":
        """Retry gen(seed) on invalid JSON errors. Use seeds base_seed+1, base_seed+2, ... per attempt."""
        for i in range(max_retries):
            seed = base_seed + i + 1
            try:
                return gen(seed)
            except Exception as e:
                should_retry = self._is_json_invalid_error(e)
                last = i == max_retries - 1
                if not should_retry or last:
                    raise
                logger.warning(
                    f"{self.__class__.__name__}: retrying due to invalid JSON "
                    f"(attempt {i + 1}/{max_retries}, seed={seed})"
                )
        raise RuntimeError("Unexpected fall-through in _retry_on_invalid_json")

    async def _retry_on_invalid_json_async(
        self,
        agen: Callable[[Optional[int]], Awaitable["TextAlignment"]],
        max_retries: int = 3,
        base_seed: int = 0,
    ) -> "TextAlignment":
        """Async variant of _retry_on_invalid_json. Use seeds base_seed+1, base_seed+2, ... per attempt."""
        for i in range(max_retries):
            seed = base_seed + i + 1
            try:
                return await agen(seed)
            except Exception as e:
                should_retry = self._is_json_invalid_error(e)
                last = i == max_retries - 1
                if not should_retry or last:
                    raise
                logger.warning(
                    f"{self.__class__.__name__}: retrying due to invalid JSON "
                    f"(attempt {i + 1}/{max_retries}, seed={seed})"
                )
        raise RuntimeError("Unexpected fall-through in _retry_on_invalid_json_async")
