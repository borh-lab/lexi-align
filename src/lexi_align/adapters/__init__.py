from typing import TYPE_CHECKING, Any, Dict, Union

from .base import LLMAdapter

__all__ = ["LLMAdapter", "SGLangAdapter", "create_adapter"]


def __getattr__(name: str):
    if name == "SGLangAdapter":
        # Lazy import to avoid importing heavy deps unless actually requested
        from .sglang_adapter import SGLangAdapter

        return SGLangAdapter
    raise AttributeError(f"module {__name__} has no attribute {name}")


def _create_litellm_adapter(model: str, params: dict[str, Any]) -> LLMAdapter:
    """Create LiteLLM adapter."""
    from .litellm_adapter import LiteLLMAdapter

    model_params = params.pop("model_params", {})
    if model and "model" not in model_params:
        model_params["model"] = model
    if not model_params.get("model"):
        raise ValueError("'model' must be specified for litellm")

    return LiteLLMAdapter(
        model_params=model_params,
        use_dynamic_schema=params.get("use_dynamic_schema", False),
        min_alignments=params.get("min_alignments", 0),
        use_reasoning=params.get("use_reasoning", False),
    )


def _create_outlines_adapter(model_name: str, params: dict[str, Any]) -> LLMAdapter:
    """Create Outlines adapter."""
    from lexi_align.utils.common import filter_none_values

    from .outlines_adapter import OutlinesAdapter

    if not model_name:
        raise ValueError("model name required for transformers adapter")

    allowed = [
        "temperature",
        "samples",
        "batch_size",
        "top_k",
        "top_p",
        "beam_size",
        "max_tokens",
        "device",
        "dtype",
        "model_kwargs",
        "presence_penalty",
        "min_p",
        "min_alignments",
        "use_dynamic_schema",
        "use_reasoning",
    ]
    init_kwargs = filter_none_values({k: params.get(k) for k in allowed})
    transformers_kwargs = params.get("transformers_kwargs", {})

    return OutlinesAdapter(model_name=model_name, **init_kwargs, **transformers_kwargs)


def _create_llama_adapter(model_path: str, params: dict[str, Any]) -> LLMAdapter:
    """Create LlamaCpp adapter."""
    from lexi_align.utils.common import filter_none_values

    from .llama_cpp_adapter import LlamaCppAdapter

    if not model_path:
        raise ValueError("'model_path' required for llama-cpp adapter")

    allowed = [
        "n_gpu_layers",
        "split_mode",
        "main_gpu",
        "tensor_split",
        "n_ctx",
        "n_threads",
        "verbose",
        "repo_id",
        "max_tokens",
        "min_alignments",
        "use_dynamic_schema",
        "use_reasoning",
    ]
    init_kwargs = filter_none_values({k: params.get(k) for k in allowed})

    return LlamaCppAdapter(model_path=model_path, **init_kwargs)


def _create_sglang_adapter(model: str, params: dict[str, Any]) -> LLMAdapter:
    """Create SGLang adapter."""
    from lexi_align.utils.common import filter_none_values

    from .sglang_adapter import SGLangAdapter

    if not model:
        raise ValueError("'model' required for sglang adapter")

    allowed = [
        "base_url",
        "api_key",
        "temperature",
        "samples",
        "top_k",
        "top_p",
        "beam_size",
        "max_tokens",
        "client_kwargs",
        "generation_kwargs",
        "extra_body",
        "presence_penalty",
        "min_p",
        "min_alignments",
        "use_dynamic_schema",
        "use_reasoning",
    ]
    init_kwargs = filter_none_values({k: params.get(k) for k in allowed})
    init_kwargs["model"] = model

    return SGLangAdapter(**init_kwargs)


def create_adapter(
    spec: Union[str, Dict[str, Any], None] = None, **kwargs: Any
) -> LLMAdapter:
    """
    Factory for creating adapters from a simple spec.

    Examples:
      create_adapter("litellm:gpt-4o")
      create_adapter("transformers:Qwen/Qwen3-0.6B", temperature=0.0)
      create_adapter("llama:path/to/model.gguf", n_gpu_layers=-1)
      create_adapter({"backend": "transformers", "model": "Qwen/Qwen3-0.6B"})
    """
    if isinstance(spec, str):
        backend, _, model_spec = spec.partition(":")
        backend = backend.strip().lower()
        model_spec = model_spec.strip()

        match backend:
            case "litellm" | "openai":
                return _create_litellm_adapter(model_spec, kwargs)
            case "transformers" | "hf":
                if not model_spec:
                    raise ValueError(
                        "create_adapter(transformers): model name required (e.g., 'transformers:Qwen/Qwen3-0.6B')"
                    )
                return _create_outlines_adapter(model_spec, kwargs)
            case "llama" | "llama-cpp":
                model_path = model_spec or kwargs.get("model_path")
                if not model_path:
                    raise ValueError(
                        "create_adapter(llama): 'model_path' required (e.g., 'llama:path/to/model.gguf')"
                    )
                return _create_llama_adapter(model_path, kwargs)
            case "sglang":
                return _create_sglang_adapter(model_spec, kwargs)
            case _:
                raise ValueError(f"Unknown adapter backend: {backend}")

    elif isinstance(spec, dict):
        backend_val = spec.get("backend") or spec.get("type")
        if not isinstance(backend_val, str):
            raise ValueError(
                "Adapter spec dict must include 'backend' or 'type' as a string"
            )
        backend = backend_val.lower()

        match backend:
            case "litellm" | "openai":
                model = spec.get("model", "")
                return _create_litellm_adapter(model, spec)
            case "transformers" | "hf":
                model = spec.get("model") or spec.get("model_name")
                if not model:
                    raise ValueError(
                        "'model' or 'model_name' required for transformers"
                    )
                return _create_outlines_adapter(model, spec)
            case "llama" | "llama-cpp":
                model_path = spec.get("model_path")
                if not model_path:
                    raise ValueError("'model_path' required for llama-cpp")
                return _create_llama_adapter(model_path, spec)
            case "sglang":
                model = spec.get("model")
                if not model:
                    raise ValueError("'model' required for sglang")
                return _create_sglang_adapter(model, spec)
            case _:
                raise ValueError(f"Unknown adapter backend: {backend}")

    else:
        raise ValueError(
            "create_adapter: 'spec' must be provided with backend and model "
            "(e.g., 'transformers:Qwen/...', 'litellm:gpt-4o', 'sglang:...')"
        )


if TYPE_CHECKING:
    # For type checkers only
    from .sglang_adapter import SGLangAdapter  # noqa: F401
