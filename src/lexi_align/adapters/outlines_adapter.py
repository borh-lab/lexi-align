from typing import Optional, Any, Dict, Literal, cast
from outlines.generate import SequenceGenerator
from outlines.samplers import Sampler
import torch
import outlines
from outlines import models, generate
from transformers import AutoTokenizer, AutoConfig  # type: ignore
from lexi_align.adapters import LLMAdapter
from lexi_align.models import TextAlignment
from logging import getLogger

logger = getLogger(__name__)


class OutlinesAdapter(LLMAdapter):
    """Adapter for using Outlines models with lexi_align."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        # Sampling parameters
        temperature: float = 0.0,
        samples: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: Optional[int] = None,
        # Model configuration
        device: Optional[str] = None,
        dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"] = "bfloat16",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **transformers_kwargs: Any,
    ):
        """Initialize the adapter with an Outlines model."""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_kwargs = model_kwargs or {}
        self.transformers_kwargs = transformers_kwargs

        # Store sampling parameters
        self.samples = samples
        self.beam_size = beam_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Initialize other components lazily
        self._model = None
        self._generator: Optional[Any] = None

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
                    "torch_dtype": torch.bfloat16,
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
    def generator(self) -> Any:
        """Lazy initialization of the generator with appropriate sampler."""
        if self._generator is None:
            # Choose sampler based on parameters
            sampler: Sampler
            if self.beam_size is not None:
                sampler = cast(
                    Sampler, outlines.samplers.beam_search(beams=self.beam_size)
                )
            elif self.temperature == 0.0:
                sampler = cast(Sampler, outlines.samplers.greedy())
            else:
                sampler = cast(
                    Sampler,
                    outlines.samplers.multinomial(
                        self.samples,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                    ),
                )

            self._generator = generate.json(
                self.model,
                TextAlignment,
                sampler=sampler,
            )
        assert self._generator is not None
        return self._generator

    def __call__(self, messages: list[dict]) -> TextAlignment:
        """Generate alignments using the Outlines model."""
        # Apply chat template to convert messages to the model's expected format
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        logger.debug(f"Formatted prompt: {prompt}")

        # Use cached generator
        # Use cached generator
        result = self.generator(prompt)
        return cast(TextAlignment, result)
