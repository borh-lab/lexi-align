"""Test adapter factory creation."""

import pytest

from lexi_align.adapters import LLMAdapter, create_adapter


def test_create_adapter_litellm_string():
    """Test creating LiteLLM adapter from string spec."""
    adapter = create_adapter("litellm:gpt-4o-mini")
    assert adapter is not None
    assert isinstance(adapter, LLMAdapter)
    from lexi_align.adapters.litellm_adapter import LiteLLMAdapter

    assert isinstance(adapter, LiteLLMAdapter)


def test_create_adapter_litellm_dict():
    """Test creating LiteLLM adapter from dict spec."""
    spec = {"backend": "litellm", "model": "gpt-4o-mini"}
    adapter = create_adapter(spec)
    from lexi_align.adapters.litellm_adapter import LiteLLMAdapter

    assert isinstance(adapter, LiteLLMAdapter)


def test_create_adapter_openai_alias():
    """Test that 'openai' is an alias for 'litellm'."""
    adapter = create_adapter("openai:gpt-4o-mini")
    from lexi_align.adapters.litellm_adapter import LiteLLMAdapter

    assert isinstance(adapter, LiteLLMAdapter)


def test_create_adapter_transformers_string():
    """Test creating Transformers adapter from string spec."""
    adapter = create_adapter("transformers:Qwen/Qwen3-0.6B")
    from lexi_align.adapters.outlines_adapter import OutlinesAdapter

    assert isinstance(adapter, OutlinesAdapter)


def test_create_adapter_hf_alias():
    """Test that 'hf' is an alias for 'transformers'."""
    adapter = create_adapter("hf:Qwen/Qwen3-0.6B")
    from lexi_align.adapters.outlines_adapter import OutlinesAdapter

    assert isinstance(adapter, OutlinesAdapter)


def test_create_adapter_llama_string():
    """Test creating Llama adapter from string spec."""
    adapter = create_adapter("llama:path/to/model.gguf")
    from lexi_align.adapters.llama_cpp_adapter import LlamaCppAdapter

    assert isinstance(adapter, LlamaCppAdapter)


def test_create_adapter_llama_dict():
    """Test creating Llama adapter from dict spec."""
    spec = {"backend": "llama-cpp", "model_path": "path/to/model.gguf"}
    adapter = create_adapter(spec)
    from lexi_align.adapters.llama_cpp_adapter import LlamaCppAdapter

    assert isinstance(adapter, LlamaCppAdapter)


def test_create_adapter_sglang_string():
    """Test creating SGLang adapter from string spec."""
    adapter = create_adapter("sglang:Qwen/Qwen3-0.6B")
    from lexi_align.adapters.sglang_adapter import SGLangAdapter

    assert isinstance(adapter, SGLangAdapter)


def test_create_adapter_kwargs_propagation():
    """Test that kwargs are propagated to adapter."""
    adapter = create_adapter(
        "transformers:Qwen/Qwen3-0.6B", temperature=0.7, max_tokens=2048
    )
    from lexi_align.adapters.outlines_adapter import OutlinesAdapter

    assert isinstance(adapter, OutlinesAdapter)
    assert adapter.temperature == 0.7
    assert adapter.max_tokens == 2048


def test_create_adapter_invalid_backend():
    """Test error on invalid backend."""
    with pytest.raises(ValueError, match="Unknown adapter backend"):
        create_adapter("invalid:model")


def test_create_adapter_missing_model_litellm():
    """Test error when model is missing for LiteLLM."""
    with pytest.raises(ValueError, match="model.*must be specified"):
        create_adapter("litellm:")


def test_create_adapter_missing_model_transformers():
    """Test error when model is missing for Transformers."""
    with pytest.raises(ValueError, match="model name required"):
        create_adapter("transformers:")


def test_create_adapter_missing_model_llama():
    """Test error when model_path is missing for Llama."""
    with pytest.raises(ValueError, match="model_path.*required"):
        create_adapter("llama:")


def test_create_adapter_no_spec():
    """Test error when no spec is provided."""
    with pytest.raises(ValueError, match="spec.*must be provided"):
        create_adapter()  # type: ignore


def test_create_adapter_invalid_spec_type():
    """Test error when spec is neither string nor dict."""
    with pytest.raises(ValueError, match="spec.*must be provided"):
        create_adapter(123)  # type: ignore


def test_create_adapter_dict_missing_backend():
    """Test error when dict spec missing backend."""
    with pytest.raises(ValueError, match="must include.*backend"):
        create_adapter({"model": "gpt-4"})


def test_create_adapter_dict_transformers_missing_model():
    """Test error when transformers dict missing model."""
    with pytest.raises(ValueError, match="model.*required"):
        create_adapter({"backend": "transformers"})


def test_create_adapter_dict_llama_missing_path():
    """Test error when llama dict missing model_path."""
    with pytest.raises(ValueError, match="model_path.*required"):
        create_adapter({"backend": "llama"})
