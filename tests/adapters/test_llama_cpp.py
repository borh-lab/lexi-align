import pytest

from lexi_align.adapters.llama_cpp_adapter import LlamaCppAdapter, _get_model_files
from lexi_align.core import align_tokens
from lexi_align.models import AlignmentResult, TextAlignment


def test_format_messages(mocker):
    """Test chat message formatting."""
    mocker.patch("os.path.exists").return_value = True  # Mock file existence check
    adapter = LlamaCppAdapter(model_path="dummy.gguf", _testing=True)
    messages = [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Align these tokens:"},
        {"role": "assistant", "content": "Here's the alignment:"},
    ]

    formatted = adapter.format_messages(messages)
    expected = (
        "[INST] <<SYS>>\nYou are a translator.\n<</SYS>>\n\n"
        "[INST] Align these tokens: [/INST]\n"
        "Here's the alignment:\n"
    )
    assert formatted == expected


def test_split_model_detection():
    """Test detection of split model filenames."""
    # Test non-split model
    main, additional = _get_model_files("model.gguf")
    assert main == "model.gguf"
    assert additional == []

    # Test split model
    main, additional = _get_model_files("model-00001-of-00003.gguf")
    assert main == "model-00001-of-00003.gguf"
    assert additional == ["model-00002-of-00003.gguf", "model-00003-of-00003.gguf"]


@pytest.mark.llm
@pytest.mark.slow
def test_llama_cpp_alignment():
    """Test end-to-end alignment with llama.cpp using the default model."""
    adapter = LlamaCppAdapter(
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=4,
    )

    # Test simple English-French alignment with ASCII-only text
    source = "the cat"
    target = "le chat"

    source_tokens = source.split()
    target_tokens = target.split()

    alignment = align_tokens(
        adapter,
        source_tokens,
        target_tokens,
        source_language="English",
        target_language="French",
    )

    # Basic sanity checks
    assert isinstance(alignment, AlignmentResult)
    # must always return a TextAlignment with at least one pair
    assert isinstance(alignment.alignment, TextAlignment), "Expected a TextAlignment"
    assert alignment.alignment.alignment, "Alignment was empty"

    # Check some expected alignments
    aligned_pairs = {(a.source, a.target) for a in alignment.alignment.alignment}
    assert ("the", "le") in aligned_pairs or ("cat", "chat") in aligned_pairs
