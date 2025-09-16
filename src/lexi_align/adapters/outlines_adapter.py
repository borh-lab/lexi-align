from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, cast

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict

import torch
from outlines import Generator, from_transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    TextAlignment,
    TextAlignmentSchema,
    TokenAlignment,
)
from lexi_align.utils import (
    extract_existing_alignments_from_messages,
    extract_tokens_and_retry_flag,
    select_alignment_schema,
    to_text_alignment,
)

logger = getLogger(__name__)


class OutlinesAdapter(LLMAdapter):
    """Adapter for using Outlines models with lexi_align."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
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
        # Optional approximations for sampling
        presence_penalty: Optional[float] = None,
        min_p: Optional[float] = None,
        min_alignments: Optional[int] = 0,
        json_retry_attempts: int = 3,
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
            presence_penalty: Optional presence penalty (approximated via repetition_penalty)
            min_p: Optional minimum probability mass threshold (approximated via top_p if not set)
            transformers_kwargs: Additional kwargs for transformers.AutoModelForCausalLM.from_pretrained()
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_kwargs = model_kwargs or {}
        self.transformers_kwargs = transformers_kwargs
        self._batch_size = batch_size
        self.min_alignments = min_alignments or 0
        self.json_retry_attempts = json_retry_attempts

        # Store sampling parameters
        self.samples = samples
        self.beam_size = beam_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Approximated/optional controls
        self.presence_penalty: Optional[float] = presence_penalty
        self.min_p: Optional[float] = min_p

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
            f"(Transformers {transformers.__version__} / PyTorch {torch.__version__}) using device {self.device}"
        )

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.float16,  # fallback dtype for quantized wrappers
            "int4": torch.float16,  # fallback dtype for quantized wrappers
        }.get(self.dtype, torch.float32)

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        # Merge kwargs for model loading (model_kwargs + transformers_kwargs)
        load_kwargs: Dict[str, Any] = dict(self.model_kwargs or {})
        load_kwargs.update(self.transformers_kwargs)
        # Ensure dtype is honored (Transformers recent versions accept 'dtype')
        load_kwargs.setdefault("dtype", torch_dtype)
        # If CUDA requested, set sane defaults unless user provided them
        if self.device == "cuda":
            load_kwargs.setdefault("device_map", "auto")
            load_kwargs.setdefault("low_cpu_mem_usage", True)

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            trust_remote_code=True,
            **load_kwargs,
        )
        # If no device_map was used, explicitly move to a single CUDA device
        if self.device == "cuda" and "device_map" not in load_kwargs:
            hf_model = hf_model.to("cuda")

        return cast(Any, from_transformers)(cast(Any, hf_model), self.tokenizer)

    @property
    def model(self):
        """Lazy initialization of the Outlines model wrapper."""
        if self._model is None:
            self._model = self._get_model()
        return self._model

    @property
    def sampler(self) -> Any:
        """Lazy initialization of the sampler."""
        # Deprecated: outlines.samplers and Sampler are no longer used in v1 API.
        # This property is retained for backward compatibility but will always return None.
        return None

    def _inference_kwargs(self) -> Dict[str, Any]:
        """Map adapter sampling params to HF/generator kwargs; approximate unsupported ones."""
        kwargs: Dict[str, Any] = {}

        # Beam search vs sampling
        if self.beam_size is not None:
            kwargs["num_beams"] = self.beam_size
            kwargs["do_sample"] = False
        else:
            # Enable sampling when temperature > 0
            kwargs["do_sample"] = bool(self.temperature and self.temperature > 0)

        # Common sampling params
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        # Approximate min_p via top_p if top_p not explicitly set
        if getattr(self, "min_p", None) is not None and "top_p" not in kwargs:
            kwargs["top_p"] = self.min_p
            logger.warning(
                "OutlinesAdapter: approximating min_p via top_p=%s", self.min_p
            )

        # Approximate presence_penalty via repetition_penalty (safe non-negative transform)
        if getattr(self, "presence_penalty", None) is not None:
            pp = self.presence_penalty
            rep = max(0.0, 1.0 + float(pp)) if pp is not None else 1.0
            kwargs["repetition_penalty"] = rep
            logger.warning(
                "OutlinesAdapter: approximating presence_penalty via repetition_penalty=%s",
                rep,
            )

        return kwargs

    def _get_schema_class(
        self,
        source_tokens: list[str],
        target_tokens: list[str],
        is_retry: bool = False,
        existing_alignments: Optional[List[TokenAlignment]] = None,
    ) -> Type[TextAlignmentSchema]:
        """Get appropriate schema class with length constraints."""
        return select_alignment_schema(
            source_tokens,
            target_tokens,
            min_alignments=self.min_alignments or 0,
            is_retry=is_retry,
            existing_alignments=existing_alignments,
        )

    def batch(
        self,
        batch_messages: list[list["ChatMessageDict"]],
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
                        extract_tokens_and_retry_flag(
                            cast(List[Dict[str, Any]], messages)
                        )
                    )

                    # Extract existing alignments if this is a retry
                    existing_alignments = None
                    existing_alignments = extract_existing_alignments_from_messages(
                        cast(List[Dict[str, Any]], messages)
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
                    cast(List[Dict[str, Any]], messages),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompts.append(prompt)

            # Process prompts with their corresponding schemas
            batch_results: list[Optional[TextAlignment]] = []
            for i, (prompt, schema_class) in enumerate(zip(prompts, schema_classes)):
                try:
                    gen = Generator(self.model, schema_class)
                    result = gen(
                        prompt,
                        max_new_tokens=self.max_tokens,
                        **self._inference_kwargs(),
                    )
                    ta = to_text_alignment(result)
                    if ta.alignment:
                        batch_results.append(ta)
                    else:
                        logger.error("Received empty alignment from OutlinesAdapter")
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

    def __call__(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Generate alignments using the Outlines model."""
        source_tokens, target_tokens, is_retry = extract_tokens_and_retry_flag(
            cast(List[Dict[str, Any]], messages)
        )

        # Extract existing alignments if this is a retry
        existing_alignments = extract_existing_alignments_from_messages(
            cast(List[Dict[str, Any]], messages)
        )

        prompt = self.tokenizer.apply_chat_template(
            cast(List[Dict[str, Any]], messages),
            add_generation_prompt=True,
            tokenize=False,
        )
        logger.debug(f"# Formatted prompt: {prompt}")

        schema_class = self._get_schema_class(
            source_tokens, target_tokens, is_retry, existing_alignments
        )
        logger.debug(f"# Schema class: {schema_class}")

        def _gen(seed: Optional[int]) -> TextAlignment:
            cpu_state = None
            cuda_states = None
            try:
                if seed is not None:
                    import torch as _torch

                    cpu_state = _torch.random.get_rng_state()
                    cuda_states = (
                        _torch.cuda.get_rng_state_all()
                        if _torch.cuda.is_available()
                        else None
                    )
                    _torch.manual_seed(seed)
                    if _torch.cuda.is_available():
                        _torch.cuda.manual_seed_all(seed)

                gen = Generator(self.model, schema_class)
                result = gen(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    **self._inference_kwargs(),
                )
                return to_text_alignment(result)
            finally:
                if seed is not None:
                    import torch as _torch

                    if cpu_state is not None:
                        _torch.random.set_rng_state(cpu_state)
                    if cuda_states is not None:
                        _torch.cuda.set_rng_state_all(cuda_states)

        ta = self._retry_on_invalid_json(
            _gen,
            max_retries=self.json_retry_attempts,
            base_seed=0,
        )
        if not ta.alignment:
            logger.error("Received empty alignment from OutlinesAdapter")
            raise ValueError("Empty TextAlignment from OutlinesAdapter")
        return ta
