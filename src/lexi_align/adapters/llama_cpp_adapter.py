import re
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from lexi_align.models import ChatMessageDict

from llama_cpp import Llama

from lexi_align.adapters import LLMAdapter
from lexi_align.models import (
    TextAlignment,
    TokenAlignment,
)
from lexi_align.utils import (
    extract_existing_alignments_from_messages,
    extract_tokens_and_retry_flag,
    select_alignment_schema,
    to_text_alignment,
)

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
        model_path: str = "gemma-3n-E2B-it-UD-Q4_K_XL.gguf",
        n_gpu_layers: int = 0,
        split_mode: int = 1,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        n_ctx: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        repo_id: Optional[str] = "unsloth/gemma-3n-E2B-it-GGUF",
        enforce_length_constraints: bool = False,
        max_tokens: int = 4096,  # NEW
        min_alignments: Optional[int] = 0,
        json_retry_attempts: int = 3,
        **kwargs: Any,
    ):
        """Initialize the adapter with a llama.cpp model.

        Args:
            model_path: HF filename within the repo when using repo_id (default: "gemma-3n-E2B-it-UD-Q4_K_XL.gguf"). If repo_id is None, treated as a local path.
            n_gpu_layers: Number of layers to offload to GPU (set to high number such as 99 to use all layers)
            split_mode: How to split model across GPUs (1=layer-wise, 2=row-wise)
            main_gpu: Main GPU to use
            tensor_split: How to distribute tensors across GPUs
            n_ctx: Text context (0 to infer from model)
            n_threads: Number of threads (None for all available)
            verbose: Print verbose output
            repo_id: Hugging Face repo ID (default: "unsloth/gemma-3n-E2B-it-GGUF")
            max_tokens: Maximum number of new tokens to generate (passed to the Outlines generator)
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
        self.kwargs = kwargs
        self.max_tokens = max_tokens  # NEW
        self.min_alignments = min_alignments or 0

        # Initialize components lazily
        self._model: Optional[Llama] = None
        self.include_schema = True  # Default to True for local models
        self.json_retry_attempts = json_retry_attempts

    @property
    def model(self) -> Llama:
        """Lazy initialization of the model."""
        if self._model is None:
            logger.info(
                f"Loading llama.cpp model from "
                f"{'HF ' + self.repo_id if self.repo_id else 'local file'} "
                f"filename={self.model_path}"
            )

            # Set up base parameters
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

            # Handle split models and HF downloads
            if self.repo_id:
                # HF models: let llama.cpp handle sharded files; do not precompute parts
                self._model = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.model_path,
                    **model_params,
                )
            else:
                # Local filesystem: detect split models via filename pattern
                main_file, _ = _get_model_files(self.model_path)
                self._model = Llama(
                    model_path=main_file,
                    **model_params,
                )
        return self._model

    def format_messages(self, messages: list["ChatMessageDict"]) -> str:
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

    def __call__(self, messages: list["ChatMessageDict"]) -> TextAlignment:
        """Generate alignments using the llama.cpp model."""

        # Format messages into prompt
        prompt = self.format_messages(messages)
        logger.debug(f"Formatted prompt: {prompt}")

        try:
            base_seed = int(self.kwargs.get("seed", 0) or 0)

            def _gen(seed: Optional[int]) -> TextAlignment:
                from outlines import Generator, from_llamacpp

                outlines_model = from_llamacpp(self.model)
                src_tokens, tgt_tokens, is_retry = extract_tokens_and_retry_flag(
                    cast(List[Dict[str, Any]], messages)
                )
                existing_alignments = extract_existing_alignments_from_messages(
                    cast(List[Dict[str, Any]], messages)
                )
                dyn_schema = select_alignment_schema(
                    src_tokens,
                    tgt_tokens,
                    min_alignments=self.min_alignments or 0,
                    is_retry=is_retry,
                    existing_alignments=existing_alignments,
                )

                # Try to set seed deterministically when provided
                if seed is not None:
                    try:
                        if hasattr(self.model, "set_seed"):
                            self.model.set_seed(seed)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                generator = Generator(outlines_model, dyn_schema)
                gen_kwargs: Dict[str, Any] = {"max_tokens": self.max_tokens}
                if seed is not None:
                    gen_kwargs["seed"] = (
                        seed  # accepted by some backends; harmless otherwise
                    )
                out = generator(prompt, **gen_kwargs)
                return to_text_alignment(out)

            return self._retry_on_invalid_json(
                _gen,
                max_retries=self.json_retry_attempts,
                base_seed=base_seed,
            )
        except Exception as e:
            # Original heuristic fallback preserved
            logger.warning(
                f"LlamaCppAdapter generation failed, using heuristic fallback: {e}"
            )
            try:
                src_tokens, tgt_tokens, _ = extract_tokens_and_retry_flag(
                    cast(List[Dict[str, Any]], messages)
                )
                pairs = [
                    TokenAlignment(source=s, target=t)
                    for s, t in zip(src_tokens, tgt_tokens)
                ]
                return TextAlignment.from_token_alignments(
                    pairs, src_tokens, tgt_tokens, adapter=self
                )
            except Exception as e2:
                logger.error(f"Fallback alignment failed: {e2}")
                raise e
