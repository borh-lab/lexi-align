from abc import ABC, abstractmethod
from typing import List, Optional
from lexi_align.models import TextAlignment
from logging import getLogger

logger = getLogger(__name__)


class LLMAdapter(ABC):
    """Base class for LLM adapters."""

    @abstractmethod
    def __call__(self, messages: list[dict]) -> TextAlignment:
        """Synchronous call to generate alignments."""
        pass

    async def acall(self, messages: list[dict]) -> TextAlignment:
        """
        Async call to generate alignments.
        Default implementation calls sync version - override for true async support.
        """
        return self(messages)

    def supports_true_batching(self) -> bool:
        """
        Check if the adapter supports true batched processing.
        Override this method to return True in adapters that implement efficient batching.
        """
        return False

    def batch(
        self,
        batch_messages: List[List[dict]],
        max_retries: int = 3,
    ) -> List[Optional[TextAlignment]]:
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
        results: List[Optional[TextAlignment]] = []
        for messages in batch_messages:
            try:
                result = self(messages)
                results.append(result)
            except Exception as e:
                logger.warning(f"Sequential processing failed: {str(e)}")
                results.append(None)
        return results
