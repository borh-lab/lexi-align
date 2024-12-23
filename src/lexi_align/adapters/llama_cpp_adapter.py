import re
from logging import getLogger
from typing import Any, List, Optional, Tuple, cast

from llama_cpp import Llama

from lexi_align.adapters import LLMAdapter
from lexi_align.models import TextAlignment, TextAlignmentSchema

logger = getLogger(__name__)


def _get_model_files(model_path: str) -> Tuple[str, List[str]]:
    """Get list of model files for split models.

    Args:
        model_path: Path to first model file

    Returns:
        Tuple of (main_file, additional_files)
    """
    # Check if this is a split model
    match = re.match(r"(.+)-(\d{5})-of-(\d{5})\.gguf$", model_path)
    if not match:
        return model_path, []

    base, current, total = match.groups()
    current_num = int(current)
    total_num = int(total)

    # Generate list of all parts except the current one
    additional = []
    for i in range(1, total_num + 1):
        if i != current_num:
            part = f"{base}-{i:05d}-of-{total_num:05d}.gguf"
            additional.append(part)

    return model_path, additional


class LlamaCppAdapter(LLMAdapter):
    """Adapter for using llama.cpp models with lexi_align."""

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        split_mode: int = 1,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        n_ctx: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        repo_id: Optional[str] = None,
        tokenizer_repo_id: Optional[str] = None,
        enforce_length_constraints: bool = False,
        **kwargs: Any,
    ):
        """Initialize the adapter with a llama.cpp model.

        Args:
            model_path: Path to model file (or any split GGUF file)
            n_gpu_layers: Number of layers to offload to GPU (set to high number such as 99 to use all layers)
            split_mode: How to split model across GPUs (1=layer-wise, 2=row-wise)
            main_gpu: Main GPU to use
            tensor_split: How to distribute tensors across GPUs
            n_ctx: Text context (0 to infer from model)
            n_threads: Number of threads (None for all available)
            verbose: Print verbose output
            repo_id: Optional HuggingFace repo ID for downloading model
            tokenizer_repo_id: Optional HuggingFace repo ID for tokenizer
            **kwargs: Additional kwargs passed to Llama
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.split_mode = split_mode
        self.main_gpu = main_gpu
        self.tensor_split = tensor_split
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.repo_id = repo_id
        self.tokenizer_repo_id = (
            tokenizer_repo_id or repo_id
        )  # Default to model repo_id
        self.kwargs = kwargs

        # Initialize components lazily
        self._model: Optional[Llama] = None
        self.include_schema = True  # Default to True for local models

    @property
    def model(self) -> Llama:
        """Lazy initialization of the model."""
        if self._model is None:
            logger.info(f"Loading model {self.model_path} ({self.repo_id})")

            # Set up base parameters including tokenizer
            model_params = {
                "n_gpu_layers": self.n_gpu_layers,
                "split_mode": self.split_mode,
                "main_gpu": self.main_gpu,
                "tensor_split": self.tensor_split,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "verbose": self.verbose,
                **self.kwargs,
            }

            # Add tokenizer if repo ID is provided
            if self.tokenizer_repo_id:
                from llama_cpp.llama_tokenizer import LlamaHFTokenizer

                model_params["tokenizer"] = LlamaHFTokenizer.from_pretrained(
                    self.tokenizer_repo_id
                )

            # Handle split models and HF downloads
            if self.repo_id:
                main_file, additional_files = _get_model_files(self.model_path)
                self._model = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=main_file,
                    additional_files=additional_files,
                    **model_params,
                )
            else:
                self._model = Llama(
                    model_path=self.model_path,
                    **model_params,
                )
        return self._model

    def format_messages(self, messages: list[dict]) -> str:
        """Format chat messages into a prompt string."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                formatted.append(f"[INST] {content} [/INST]\n")
            elif role == "assistant":
                formatted.append(f"{content}\n")

        return "".join(formatted)

    def supports_length_constraints(self) -> bool:
        """Indicate that this adapter supports alignment length constraints."""
        return True

    def __call__(self, messages: list[dict]) -> TextAlignment:
        """Generate alignments using the llama.cpp model."""
        from outlines import generate
        from outlines.models.llamacpp import LlamaCpp

        # Format messages into prompt
        prompt = self.format_messages(messages)
        logger.debug(f"Formatted prompt: {prompt}")

        # Use Outlines JSON generator
        model = LlamaCpp(self.model)
        generator = generate.json(
            model,
            TextAlignmentSchema,
        )

        return cast(TextAlignment, generator(prompt))
