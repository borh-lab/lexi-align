import asyncio
import json
from abc import ABC, abstractmethod
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypedDict,
    cast,
)

from pydantic import ValidationError as PydanticValidationError

from lexi_align.adapters.mixins import RetryMixin

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict, TextAlignment, TextAlignmentSchema

logger = getLogger(__name__)


class AdapterCapabilities(TypedDict):
    """Adapter capability flags for automatic optimization."""

    supports_async: bool
    supports_batching: bool
    supports_length_constraints: bool
    preferred_batch_size: Optional[int]
    max_concurrency: Optional[int]


class LLMAdapter(RetryMixin, ABC):
    """Base class for LLM adapters."""

    include_schema: bool = False
    min_alignments: int = 0
    use_dynamic_schema: bool = False
    json_retry_attempts: int = 3
    use_reasoning: bool = False

    def _init_common_params(
        self,
        min_alignments: Optional[int] = 0,
        use_dynamic_schema: bool = False,
        json_retry_attempts: int = 3,
        use_reasoning: bool = False,
    ):
        """Initialize common adapter parameters.

        Args:
            min_alignments: Minimum number of alignments to generate
            use_dynamic_schema: Whether to use dynamic schema with length constraints
            json_retry_attempts: Number of retry attempts for invalid JSON
            use_reasoning: Whether to include reasoning field in responses
        """
        self.min_alignments = int(min_alignments or 0)
        self.use_dynamic_schema = bool(use_dynamic_schema)
        self.json_retry_attempts = json_retry_attempts
        self.use_reasoning = bool(use_reasoning)

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Get adapter capabilities for automatic optimization.

        Returns:
            Dictionary with capability flags

        Example:
            >>> from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
            >>> adapter = LiteLLMAdapter(model_params={"model": "gpt-4o-mini"})
            >>> caps = adapter.capabilities
            >>> caps['supports_async']
            True
        """
        # Check if adapter has overridden acall (not just the base implementation)
        has_custom_acall = (
            hasattr(self.__class__, "acall")
            and self.__class__.acall != LLMAdapter.acall
        )

        return {
            "supports_async": has_custom_acall,
            "supports_batching": self.supports_true_batching(),
            "supports_length_constraints": self.supports_length_constraints(),
            "preferred_batch_size": self.preferred_batch_size(),
            "max_concurrency": getattr(self, "max_concurrency", 8),
        }

    def _select_schema_for_messages(
        self,
        messages: List["ChatMessageDict"],
    ) -> Type["TextAlignmentSchema"]:
        """Extract tokens from messages and select appropriate schema.

        This method handles the common pattern of:
        1. Extracting tokens and retry state from messages
        2. Extracting existing alignments and remaining tokens
        3. Selecting the appropriate schema (static or dynamic)

        Args:
            messages: Chat messages containing token information

        Returns:
            Schema class to use for response validation
        """
        if not self.supports_length_constraints():
            from lexi_align.models import TextAlignmentSchema

            return TextAlignmentSchema

        from lexi_align.utils import (
            extract_existing_alignments_from_messages,
            extract_remaining_tokens_from_messages,
            extract_tokens_and_retry_flag,
            select_alignment_schema,
        )

        source_tokens, target_tokens, is_retry = extract_tokens_and_retry_flag(
            cast(List[Dict[str, Any]], messages)
        )
        existing_alignments = extract_existing_alignments_from_messages(
            cast(List[Dict[str, Any]], messages)
        )
        rem_src, rem_tgt = extract_remaining_tokens_from_messages(
            cast(List[Dict[str, Any]], messages)
        )

        return select_alignment_schema(
            source_tokens,
            target_tokens,
            min_alignments=self.min_alignments,
            is_retry=is_retry,
            existing_alignments=existing_alignments,
            remaining_source_tokens=rem_src,
            remaining_target_tokens=rem_tgt,
            use_reasoning=self.use_reasoning,
        )

    def _extract_partial_alignment_from_error(
        self,
        content: str,
        error: Exception,
        schema_class: Type["TextAlignmentSchema"],
    ) -> Optional["TextAlignment"]:
        """Extract valid alignment items from a validation error response.

        When Pydantic rejects a response due to length constraints (too_short/too_long),
        but the individual items are valid, extract and return what we have.

        Args:
            content: Response content string
            error: The validation error that was raised
            schema_class: Schema class that failed validation

        Returns:
            Partial TextAlignment if extractable, None otherwise
        """
        if not isinstance(error, PydanticValidationError):
            return None

        # Check if this is a length constraint error
        errors = error.errors()
        is_length_error = any(
            e.get("type") in ("too_short", "too_long")
            and e.get("loc") == ("alignment",)
            for e in errors
        )

        if not is_length_error:
            return None

        try:
            from llm_schema_lite import loads as llm_loads

            from lexi_align.models import TextAlignmentSchema
            from lexi_align.utils import to_text_alignment

            # Parse the raw content
            data = llm_loads(content, mode="json")
            alignment_items = data.get("alignment", [])

            if not alignment_items:
                return None

            # Validate individual items without length constraints
            # Use the base schema which doesn't have length constraints
            partial_data = {"alignment": alignment_items}
            partial_schema = TextAlignmentSchema.model_validate(partial_data)

            logger.info(
                f"{self.__class__.__name__}: Extracted {len(alignment_items)} valid alignments "
                f"from length-constraint validation error"
            )
            return to_text_alignment(partial_schema)

        except Exception as e:
            logger.debug(f"Failed to extract partial alignment: {e}")
            return None

    def _validate_response_content(
        self,
        content: str,
        schema_class: Type["TextAlignmentSchema"],
        context: Optional[Dict[str, Any]] = None,
    ) -> "TextAlignment":
        """Validate and parse response content with structured logging.

        Args:
            content: Response content string (expected to be JSON)
            schema_class: Schema class to validate against
            context: Optional context dict for logging (e.g., seed, attempt_idx)

        Returns:
            Validated TextAlignment object

        Raises:
            Exception: If validation fails
        """
        from llm_schema_lite import loads as llm_loads

        from lexi_align.utils import to_text_alignment

        context = context or {}
        try:
            # Use lenient parser first to handle malformed JSON, then validate with Pydantic
            data = llm_loads(content, mode="json")
            schema_obj = schema_class.model_validate(data)
            return to_text_alignment(schema_obj)
        except Exception as e:
            # Try to extract partial results on length constraint errors
            partial = self._extract_partial_alignment_from_error(
                content, e, schema_class
            )
            if partial:
                logger.warning(
                    f"{self.__class__.__name__}: Returning partial alignment, "
                    f"will retry for remaining items"
                )
                return partial

            # Original error logging
            max_chars = context.get("max_log_chars", 4000)
            snippet = (
                content[:max_chars]
                + f"... [truncated {len(content) - max_chars} chars]"
                if len(content) > max_chars
                else content
            )
            logger.error(
                f"{self.__class__.__name__} validation failed",
                extra={
                    "error_type": type(e).__name__,
                    "content_length": len(content),
                    **context,
                },
            )
            logger.debug(f"Failed content: {snippet}")
            logger.error(f"Full validation error: {str(e)}")
            logger.error(f"Schema class: {schema_class.__name__}")
            if isinstance(e, PydanticValidationError):
                logger.error("Pydantic validation errors", extra={"errors": e.errors()})
                logger.error(
                    f"Detailed Pydantic errors:\n{json.dumps(e.errors(), indent=2, default=str)}"
                )
            raise

    def _batch_with_per_item_schema(
        self,
        batch_messages: List[List["ChatMessageDict"]],
        process_fn: Callable[
            [List["ChatMessageDict"], Type["TextAlignmentSchema"]], Any
        ],
    ) -> List[Optional["TextAlignment"]]:
        """Common batch processing logic that selects schema per item.

        This method handles the common pattern of:
        1. Iterating over batch messages
        2. Selecting appropriate schema for each item
        3. Processing each item with the provided function
        4. Converting results to TextAlignment
        5. Handling errors gracefully

        Args:
            batch_messages: List of message sequences to process
            process_fn: Function that takes (messages, schema_class) and returns a result

        Returns:
            List of TextAlignment objects or None for failed items
        """
        from lexi_align.utils import to_text_alignment

        results: List[Optional["TextAlignment"]] = []

        for i, messages in enumerate(batch_messages):
            try:
                schema_class = self._select_schema_for_messages(messages)
                result = process_fn(messages, schema_class)
                ta = to_text_alignment(result)
                if ta.alignment:
                    results.append(ta)
                else:
                    logger.error(
                        f"Received empty alignment from {self.__class__.__name__} in batch item {i}"
                    )
                    results.append(None)
            except Exception as e:
                logger.error(
                    f"Error processing batch item {i}:\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}",
                    exc_info=True,
                )
                results.append(None)

        return results

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

    def preferred_batch_size(self) -> Optional[int]:
        """Return adapter's preferred batch size, if any."""
        return None

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
