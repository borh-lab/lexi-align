from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

from .base import LLMAdapter

__all__ = ["LLMAdapter", "SGLangAdapter", "create_adapter"]


def __getattr__(name: str):
    if name == "SGLangAdapter":
        # Lazy import to avoid importing heavy deps unless actually requested
        from .sglang_adapter import SGLangAdapter

        return SGLangAdapter
    raise AttributeError(f"module {__name__} has no attribute {name}")


def create_adapter(
    spec: Union[str, Dict[str, Any], None] = None, **kwargs: Any
) -> LLMAdapter:
    """
    Factory for creating adapters from a simple spec.

    Examples:
      create_adapter("litellm:gpt-4o")
      create_adapter("transformers:Qwen/Qwen3-0.6B", temperature=0.0)
      create_adapter("llama:path/to/model.gguf", n_gpu_layers=-1)
      create_adapter({"backend": "transformers", "model": "Qwen/Qwen3-0.6B", "temperature": 0.0})
    """
    # String-based spec: "backend:rest"
    if isinstance(spec, str):
        backend, _, model_spec = spec.partition(":")
        backend = backend.strip().lower()
        model_spec = model_spec.strip()

        if backend in ("litellm", "openai"):
            from .litellm_adapter import LiteLLMAdapter

            params = {"model": model_spec} if model_spec else {}
            params.update(kwargs.pop("model_params", {}))
            if not params.get("model"):
                raise ValueError(
                    "create_adapter(litellm): 'model' must be specified (e.g., 'litellm:gpt-4o')."
                )
            use_dynamic_schema = bool(kwargs.pop("use_dynamic_schema", False))
            min_alignments = (
                int(kwargs.pop("min_alignments", 0))
                if "min_alignments" in kwargs
                else 0
            )
            return LiteLLMAdapter(
                model_params=params,
                use_dynamic_schema=use_dynamic_schema,
                min_alignments=min_alignments,
            )

        elif backend in ("transformers", "hf"):
            from .outlines_adapter import OutlinesAdapter

            if not model_spec:
                raise ValueError(
                    "create_adapter(transformers): model name is required in spec (e.g., 'transformers:Qwen/Qwen3-0.6B')."
                )
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
            ]
            init_kwargs = {k: kwargs[k] for k in allowed if k in kwargs}
            transformers_kwargs = kwargs.get("transformers_kwargs", {})
            return OutlinesAdapter(
                model_name=model_spec, **init_kwargs, **transformers_kwargs
            )

        elif backend in ("llama", "llama-cpp"):
            from .llama_cpp_adapter import LlamaCppAdapter

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
            ]
            init_kwargs = {k: kwargs[k] for k in allowed if k in kwargs}
            model_path = model_spec or kwargs.get("model_path")
            if not model_path:
                raise ValueError(
                    "create_adapter(llama): 'model_path' must be specified (e.g., 'llama:path/to/model.gguf' or model_path=...)."
                )
            return LlamaCppAdapter(model_path=model_path, **init_kwargs)

        elif backend == "sglang":
            from .sglang_adapter import SGLangAdapter

            allowed = [
                "model",
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
            ]
            init_kwargs = {k: kwargs[k] for k in allowed if k in kwargs}
            model_name: Optional[str] = model_spec or cast(
                Optional[str], init_kwargs.get("model")
            )
            if not model_name:
                raise ValueError(
                    "create_adapter(sglang): 'model' must be specified (e.g., 'sglang:Qwen/Qwen3-0.6B' or model='...')."
                )
            init_kwargs["model"] = model_name
            return SGLangAdapter(**init_kwargs)

        else:
            raise ValueError(f"Unknown adapter backend: {backend}")

    # Dict-based spec
    elif isinstance(spec, dict):
        backend_val = spec.get("backend") or spec.get("type")
        if not isinstance(backend_val, str):
            raise ValueError(
                "Adapter spec dict must include 'backend' or 'type' as a string"
            )
        backend = backend_val.lower()

        if backend in ("litellm", "openai"):
            from .litellm_adapter import LiteLLMAdapter

            model_params = spec.get("model_params") or {}
            if "model" in spec and "model" not in model_params:
                model_params["model"] = spec["model"]
            if not model_params.get("model"):
                raise ValueError(
                    "create_adapter(litellm): 'model' must be provided in 'model_params' or as 'model'."
                )
            use_dynamic_schema = bool(spec.get("use_dynamic_schema", False))
            min_alignments = int(spec.get("min_alignments", 0))
            return LiteLLMAdapter(
                model_params=model_params,
                use_dynamic_schema=use_dynamic_schema,
                min_alignments=min_alignments,
            )

        elif backend in ("transformers", "hf"):
            from .outlines_adapter import OutlinesAdapter

            model_name = spec.get("model") or spec.get("model_name")
            if not model_name:
                raise ValueError(
                    "create_adapter(transformers): 'model' (or 'model_name') is required."
                )
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
            ]
            init_kwargs = {k: spec[k] for k in allowed if k in spec}
            transformers_kwargs = spec.get("transformers_kwargs") or {}
            return OutlinesAdapter(
                model_name=model_name, **init_kwargs, **transformers_kwargs
            )

        elif backend in ("llama", "llama-cpp"):
            from .llama_cpp_adapter import LlamaCppAdapter

            model_path = spec.get("model_path")
            if not model_path:
                raise ValueError("create_adapter(llama): 'model_path' is required.")
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
            ]
            init_kwargs = {k: spec[k] for k in allowed if k in spec}
            return LlamaCppAdapter(model_path=model_path, **init_kwargs)

        elif backend == "sglang":
            from .sglang_adapter import SGLangAdapter

            allowed = [
                "model",
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
            ]
            init_kwargs = {k: spec[k] for k in allowed if k in spec}
            if not init_kwargs.get("model"):
                raise ValueError("create_adapter(sglang): 'model' is required.")
            return SGLangAdapter(**init_kwargs)

        else:
            raise ValueError(f"Unknown adapter backend: {backend}")

    # Default: return a small OutlinesAdapter instance
    else:
        raise ValueError(
            "create_adapter: 'spec' must be provided and include a backend and model (e.g., 'transformers:Qwen/...', 'litellm:gpt-4o', 'sglang:...')."
        )


if TYPE_CHECKING:
    # For type checkers only
    from .sglang_adapter import SGLangAdapter  # noqa: F401
