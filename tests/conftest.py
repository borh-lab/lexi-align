import pytest
from lexi_align.models import TextAlignment, TokenAlignment


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "llm: mark test as needing local llm inference (deselect with '-m \"not llm\"')",
    )


@pytest.fixture
def sample_alignment():
    return TextAlignment(
        alignment=[
            TokenAlignment(source_token="The", target_token="Le"),
            TokenAlignment(source_token="cat", target_token="chat"),
            TokenAlignment(source_token="sat", target_token="assis"),
        ]
    )


@pytest.fixture
def sample_alignments():
    return {
        "model1": TextAlignment(
            alignment=[
                TokenAlignment(source_token="The", target_token="Le"),
                TokenAlignment(source_token="cat", target_token="chat"),
            ]
        ),
        "model2": TextAlignment(
            alignment=[
                TokenAlignment(source_token="The", target_token="Le"),
                TokenAlignment(source_token="cat", target_token="chat"),
                TokenAlignment(source_token="sat", target_token="assis"),
            ]
        ),
    }


@pytest.fixture
def sample_tokens():
    return {"source": ["The", "cat", "sat"], "target": ["Le", "chat", "assis"]}
