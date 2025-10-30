from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    cast,
)

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict

import torch
from outlines import Generator, from_transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    TextAlignment,
    TextAlignmentSchema,
)
from lexi_align.utils import to_text_alignment
from lexi_align.utils.common import filter_none_values, temporary_torch_seed

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
        use_dynamic_schema: bool = True,
        use_reasoning: bool = False,
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
            use_reasoning: Whether to include reasoning field in responses
            transformers_kwargs: Additional kwargs for transformers.AutoModelForCausalLM.from_pretrained()
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_kwargs = model_kwargs or {}
        self.transformers_kwargs = transformers_kwargs
        self._batch_size = batch_size
        self._init_common_params(
            min_alignments, use_dynamic_schema, json_retry_attempts, use_reasoning
        )

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
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
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
        # Prefer SDPA on CPU to avoid attempting FlashAttention-2 on CPU backends.
        if self.device != "cuda" and "attn_implementation" not in load_kwargs:
            load_kwargs["attn_implementation"] = "sdpa"

        if self.device == "cuda" and torch.cuda.is_available():
            load_kwargs.setdefault("device_map", "auto")

        # Prefer FlashAttention-2 on CUDA if available and not explicitly overridden
        if self.device == "cuda" and "attn_implementation" not in load_kwargs:
            try:
                if torch.cuda.is_available():
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Requesting attn_implementation=flash_attention_2")
            except Exception:
                pass

        try:
            hf_model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                trust_remote_code=True,
                **load_kwargs,
            )
        except TypeError as e:
            # Older transformers/model may not accept 'attn_implementation'
            if "attn_implementation" in load_kwargs and (
                "attn_implementation" in str(e)
                or "unexpected keyword argument" in str(e)
            ):
                if self.device == "cuda":
                    # On CUDA, do not silently drop flash-attn; fail fast.
                    raise
                logger.warning(
                    "attn_implementation not supported by this model/transformers; retrying without it"
                )
                load_kwargs.pop("attn_implementation", None)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    config=config,
                    trust_remote_code=True,
                    **load_kwargs,
                )
            else:
                raise
        except Exception:
            # flash-attn might be unavailable; handle depending on device
            if load_kwargs.get("attn_implementation") == "flash_attention_2":
                if self.device == "cuda":
                    logger.error(
                        "FlashAttention-2 not available on CUDA; failing per configuration."
                    )
                    raise
                # CPU path: fall back to SDPA
                logger.warning(
                    "FlashAttention-2 not available; retrying with attn_implementation='sdpa'"
                )
                load_kwargs["attn_implementation"] = "sdpa"
                hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    config=config,
                    trust_remote_code=True,
                    **load_kwargs,
                )
            else:
                raise

        if self.device == "cuda":
            try:
                device_types = {p.device.type for p in hf_model.parameters()}
            except Exception:
                device_types = set()
            # Require full CUDA placement; do not proceed if any CPU params are present.
            if "cuda" not in device_types or ("cpu" in device_types):
                raise RuntimeError(
                    f"CUDA requested, but model parameters are not fully on GPU. "
                    f"Seen device types: {device_types}. Ensure sufficient GPU memory "
                    f"and proper CUDA/FlashAttention installation."
                )

        builder: Callable[[Any, Any], Any] = cast(Any, from_transformers)
        return builder(cast(Any, hf_model), cast(Any, self.tokenizer))  # type: ignore[arg-type]

    @property
    def model(self):
        """Lazy initialization of the Outlines model wrapper."""
        if self._model is None:
            self._model = self._get_model()
        return self._model

    def _inference_kwargs(self) -> Dict[str, Any]:
        """Map adapter sampling params to HF/generator kwargs; approximate unsupported ones."""
        # Base kwargs with None filtering
        kwargs = filter_none_values(
            {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
            }
        )

        # Beam search vs sampling
        if self.beam_size is not None:
            kwargs["num_beams"] = self.beam_size
            kwargs["do_sample"] = False
        else:
            # Enable sampling when temperature > 0
            kwargs["do_sample"] = bool(self.temperature and self.temperature > 0)

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

    def batch(
        self,
        batch_messages: list[list["ChatMessageDict"]],
        max_retries: int = 3,
    ) -> list[Optional[TextAlignment]]:
        """Generate alignments for a batch of message sequences."""

        def process_one(
            messages: list["ChatMessageDict"], schema_class: Type[TextAlignmentSchema]
        ) -> Any:
            """Process a single message sequence with its schema."""
            # Format prompt
            prompt = self.tokenizer.apply_chat_template(
                cast(List[Dict[str, Any]], messages),
                add_generation_prompt=True,
                tokenize=False,
            )
            gen = Generator(self.model, schema_class)
            return gen(
                prompt,
                max_new_tokens=self.max_tokens,
                **self._inference_kwargs(),
            )

        return self._batch_with_per_item_schema(batch_messages, process_one)

    def supports_true_batching(self) -> bool:
        """Indicate that this adapter supports efficient batching."""
        return True

    def supports_length_constraints(self) -> bool:
        """Indicate that this adapter supports alignment length constraints."""
        return self.use_dynamic_schema

    def preferred_batch_size(self) -> Optional[int]:
        return self._batch_size

    async def acall(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        raise NotImplementedError(
            "OutlinesAdapter does not support async; use the synchronous API."
        )

    def __call__(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Generate alignments using the Outlines model."""
        prompt = self.tokenizer.apply_chat_template(
            cast(List[Dict[str, Any]], messages),
            add_generation_prompt=True,
            tokenize=False,
        )
        logger.debug(f"# Formatted prompt: {prompt}")

        schema_class = self._select_schema_for_messages(messages)
        logger.debug(f"# Schema class: {schema_class}")

        def _gen(seed: Optional[int]) -> TextAlignment:
            with temporary_torch_seed(seed):
                gen = Generator(self.model, schema_class)
                result = gen(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    **self._inference_kwargs(),
                )
                return to_text_alignment(result)

        ta = self._retry_on_invalid_json(
            _gen,
            max_retries=self.json_retry_attempts,
            base_seed=0,
        )
        if not ta.alignment:
            logger.error("Received empty alignment from OutlinesAdapter")
            raise ValueError("Empty TextAlignment from OutlinesAdapter")
        return ta
