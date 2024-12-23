import json
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Type, cast

import outlines
import torch
from outlines import generate, models
from outlines.samplers import Sampler
from transformers import AutoConfig, AutoTokenizer  # type: ignore

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    UNALIGNED_MARKER,
    TextAlignment,
    TextAlignmentSchema,
    TokenAlignment,
    calculate_max_alignments,
    create_dynamic_alignment_schema,
)

logger = getLogger(__name__)


class OutlinesAdapter(LLMAdapter):
    """Adapter for using Outlines models with lexi_align."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        # Sampling parameters
        temperature: float = 0.0,
        samples: int = 1,
        batch_size: int = 5,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: Optional[int] = None,
        max_tokens: int = 4096,
        # Model configuration
        device: Optional[str] = None,
        dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"] = "bfloat16",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **transformers_kwargs: Any,
    ):
        """Initialize the adapter with an Outlines model.

        Args:
            model_name: Name/path of the model to load
            temperature: Sampling temperature (0.0 for greedy)
            samples: Number of samples for multinomial sampling
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            beam_size: Number of beams for beam search
            max_tokens: Maximum number of new tokens to generate (passed as max_new_tokens to outlines)
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Model weight data type
            model_kwargs: Additional kwargs for model initialization
            transformers_kwargs: Additional kwargs for transformers.AutoModelForCausalLM.from_pretrained()
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_kwargs = model_kwargs or {}
        self.transformers_kwargs = transformers_kwargs
        self._batch_size = batch_size

        # Store sampling parameters
        self.samples = samples
        self.beam_size = beam_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Initialize other components lazily
        self._model = None
        self._sampler: Optional[Any] = None
        self.include_schema = True  # Default to True for local models

    def _get_model(self):
        """Initialize model with appropriate configuration."""
        import transformers

        logger.info(
            f"Loading model {self.model_name} ({self.dtype}) "
            f"(Transformers {transformers.__version__} / PyTorch {torch.__version__})"
        )

        config, unused_config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            return_unused_kwargs=True,
        )
        if unused_config:
            logger.warning(f"Unused Transformers config keys: {unused_config}")

        # Handle quantization for int8/int4
        if self.dtype in ["int8", "int4"]:
            try:
                from transformers import BitsAndBytesConfig

                logger.info(f"Using BitsAndBytesConfig for {self.dtype} quantization")
                config.init_device = "meta"
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=(self.dtype == "int8"),
                    load_in_4bit=(self.dtype == "int4"),
                )
                kwargs = {
                    "config": config,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,  # Changed from bfloat16 to float16
                    "device_map": {"": 0},
                    "quantization_config": quantization_config,
                }
                # Only add flash attention if available
                import importlib.util

                if importlib.util.find_spec("flash_attn"):
                    kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using flash attention 2")
                else:
                    logger.info(
                        "Flash attention package not found, using default attention"
                    )

                return models.transformers(
                    model_name=self.model_name,
                    device="cuda",
                    model_kwargs=kwargs,
                )
            except ImportError as e:
                logger.info(
                    f"BitsAndBytesConfig not available, falling back to bfloat16: {e}"
                )

        # Handle other dtype options
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        kwargs = {
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }

        # Only add flash attention if on CUDA and available
        if self.device == "cuda":
            import importlib.util

            if importlib.util.find_spec("flash_attn"):
                kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash attention 2")
            else:
                logger.info(
                    "Flash attention package not found, using default attention"
                )

        # Add any additional model kwargs
        kwargs.update(self.model_kwargs)

        model = models.transformers(
            self.model_name,
            device=self.device,
            model_kwargs=kwargs,
        )
        logger.debug(f"Model: {model} with config {config}")
        return model

    @property
    def model(self):
        """Lazy initialization of the Outlines model wrapper."""
        if self._model is None:
            self._model = self._get_model()
        return self._model

    @property
    def sampler(self) -> Any:
        """Lazy initialization of the sampler."""
        if self._sampler is None:
            # Choose sampler based on parameters
            if self.beam_size is not None:
                self._sampler = cast(
                    Sampler, outlines.samplers.beam_search(beams=self.beam_size)
                )
            elif self.temperature == 0.0:
                self._sampler = cast(Sampler, outlines.samplers.greedy())
            else:
                self._sampler = cast(
                    Sampler,
                    outlines.samplers.multinomial(
                        self.samples,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                    ),
                )

        return self._sampler

    def _extract_tokens_from_messages(
        self, messages: list[dict]
    ) -> tuple[list[str], list[str], bool]:
        """Extract source and target tokens from messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Tuple of (source_tokens, target_tokens, is_retry)
            The tokens returned are always the complete set, even for retry attempts,
            since tokens may participate in multiple alignments.

        Raises:
            ValueError: If tokens cannot be found in messages
        """
        source_tokens = []
        target_tokens = []
        is_retry = False

        # First find the initial complete token lists
        for message in reversed(messages):
            content = message.get("content", "")
            if not isinstance(content, str):
                continue

            lines = content.split("\n")
            if len(lines) >= 2:
                if lines[0].startswith("Source tokens: ") and lines[1].startswith(
                    "Target tokens: "
                ):
                    source_tokens = lines[0].replace("Source tokens: ", "").split()
                    target_tokens = lines[1].replace("Target tokens: ", "").split()
                    break

        if not source_tokens or not target_tokens:
            raise ValueError(
                "Could not find original source and target tokens in messages"
            )

        # Then check if this is a retry attempt
        for message in reversed(messages):
            content = message.get("content", "")
            if (
                isinstance(content, str)
                and "Please provide alignments for the remaining tokens:" in content
            ):
                is_retry = True
                break

        return source_tokens, target_tokens, is_retry

    def _get_schema_class(
        self,
        source_tokens: list[str],
        target_tokens: list[str],
        is_retry: bool = False,
        existing_alignments: Optional[List[TokenAlignment]] = None,
    ) -> Type[TextAlignmentSchema]:
        """Get appropriate schema class with length constraints.

        Args:
            source_tokens: List of source tokens
            target_tokens: List of target tokens
            is_retry: Whether this is a retry attempt
            existing_alignments: Optional list of existing valid alignments

        Returns:
            Schema class to use for validation
        """
        if not self.supports_length_constraints():
            return TextAlignmentSchema

        if is_retry and existing_alignments:
            # Calculate remaining unaligned tokens
            aligned_source = {
                align.source
                for align in existing_alignments
                if align.source != UNALIGNED_MARKER
            }
            aligned_target = {
                align.target
                for align in existing_alignments
                if align.target != UNALIGNED_MARKER
            }

            remaining_source = set(source_tokens) - aligned_source
            remaining_target = set(target_tokens) - aligned_target

            # Scale length constraints based on remaining tokens
            min_length = min(len(remaining_source), len(remaining_target))
            max_length = calculate_max_alignments(
                list(remaining_source), list(remaining_target)
            )
        else:
            # For initial attempts, use full token sets
            min_length = min(len(source_tokens), len(target_tokens))
            max_length = calculate_max_alignments(source_tokens, target_tokens)

        return create_dynamic_alignment_schema(
            source_tokens,  # Always pass full token lists for enums
            target_tokens,  # Always pass full token lists for enums
            min_length=min_length,
            max_length=max_length,
        )

    def batch(
        self,
        batch_messages: list[list[dict]],
        max_retries: int = 3,
    ) -> list[Optional[TextAlignment]]:
        """Generate alignments for a batch of message sequences."""
        try:
            # Format all prompts and create schemas
            prompts: list[str] = []
            schema_classes: list[Type[TextAlignmentSchema]] = []

            for messages in batch_messages:
                try:
                    source_tokens, target_tokens, is_retry = (
                        self._extract_tokens_from_messages(messages)
                    )

                    # Extract existing alignments if this is a retry
                    existing_alignments = None
                    if is_retry:
                        for message in reversed(messages):
                            if isinstance(
                                message.get("content"), str
                            ) and "Here are partial alignments:" in message.get(
                                "content", ""
                            ):
                                try:
                                    content = message["content"]
                                    alignment_start = content.find('{"alignment":')
                                    alignment_end = content.find("\n", alignment_start)
                                    if alignment_end == -1:
                                        alignment_end = len(content)
                                    alignment_json = content[
                                        alignment_start:alignment_end
                                    ]
                                    partial_alignment = TextAlignment.parse_raw(
                                        alignment_json
                                    )
                                    existing_alignments = partial_alignment.alignment
                                    break
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to parse existing alignments: {e}"
                                    )

                    schema_class = self._get_schema_class(
                        source_tokens, target_tokens, is_retry, existing_alignments
                    )
                except ValueError:
                    logger.warning(f"Could not find tokens in messages: {messages}")
                    schema_class = TextAlignmentSchema

                schema_classes.append(schema_class)

                # Format prompt
                prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                prompts.append(prompt)

            # Process prompts with their corresponding schemas
            batch_results: list[Optional[TextAlignment]] = []
            for i, (prompt, schema_class) in enumerate(zip(prompts, schema_classes)):
                try:
                    generator = generate.json(
                        self.model,
                        schema_class,
                        sampler=self.sampler,
                    )
                    result = generator(prompt, max_tokens=self.max_tokens)

                    # Add detailed logging of what we received
                    logger.debug(f"Result {i} type: {type(result)}")
                    logger.debug(f"Result {i} content: {result}")

                    # Convert result to proper TextAlignment
                    if isinstance(result, TextAlignmentSchema) and not isinstance(
                        result, TextAlignment
                    ):
                        batch_results.append(TextAlignment(alignment=result.alignment))
                    elif isinstance(result, TextAlignment):
                        batch_results.append(result)
                    else:
                        logger.error(
                            f"Invalid result type: {type(result)}, expected TextAlignment"
                        )
                        batch_results.append(None)
                except Exception as e:
                    logger.error(
                        f"Error processing prompt {i}:\n"
                        f"Error type: {type(e).__name__}\n"
                        f"Error message: {str(e)}\n"
                        f"Stack trace:",
                        exc_info=True,
                    )
                    batch_results.append(None)

            return batch_results

        except json.JSONDecodeError as e:
            context = e.doc[max(0, e.pos - 50) : min(len(e.doc), e.pos + 50)]
            logger.error(
                f"JSON decode error processing batch:\n"
                f"Error: {str(e)}\n"
                f"Position: {e.pos}\n"
                f"Context: '...{context}...'\n"
                f"Raw document: {e.doc}\n"
                f"Stack trace:",
                exc_info=True,
            )
            return [None] * len(batch_messages)

        except Exception as e:
            logger.error(
                f"Batch processing failed:\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"Number of messages in batch: {len(batch_messages)}\n"
                f"Prompts: {prompts}\n"
                f"Stack trace:",
                exc_info=True,
            )
            return [None] * len(batch_messages)

    def supports_true_batching(self) -> bool:
        """Indicate that this adapter supports efficient batching."""
        return True

    def supports_length_constraints(self) -> bool:
        """Indicate that this adapter supports alignment length constraints."""
        return True

    def __call__(self, messages: list[dict]) -> TextAlignment:
        """Generate alignments using the Outlines model."""
        source_tokens, target_tokens, is_retry = self._extract_tokens_from_messages(
            messages
        )

        # Extract existing alignments if this is a retry
        existing_alignments = None
        if is_retry:
            for message in reversed(messages):
                if isinstance(
                    message.get("content"), str
                ) and "Here are partial alignments:" in message.get("content", ""):
                    try:
                        content = message["content"]
                        alignment_start = content.find('{"alignment":')
                        alignment_end = content.find("\n", alignment_start)
                        if alignment_end == -1:
                            alignment_end = len(content)
                        alignment_json = content[alignment_start:alignment_end]
                        partial_alignment = TextAlignment.parse_raw(alignment_json)
                        existing_alignments = partial_alignment.alignment
                        break
                    except Exception as e:
                        logger.warning(f"Failed to parse existing alignments: {e}")

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        logger.debug(f"# Formatted prompt: {prompt}")

        schema_class = self._get_schema_class(
            source_tokens, target_tokens, is_retry, existing_alignments
        )
        logger.debug(f"# Schema class: {schema_class}")

        generator = generate.json(
            self.model,
            schema_class,
            sampler=self.sampler,
        )
        result = generator(prompt, max_tokens=self.max_tokens)

        # Convert to TextAlignment if needed
        if isinstance(result, TextAlignmentSchema) and not isinstance(
            result, TextAlignment
        ):
            return TextAlignment(alignment=result.alignment)
        return cast(TextAlignment, result)
