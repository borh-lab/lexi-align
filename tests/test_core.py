import pytest
from lexi_align.core import (
    align_tokens,
    align_tokens_raw,
)
from lexi_align.adapters import LLMAdapter
from lexi_align.models import TextAlignment, TokenAlignment
from logging import getLogger

logger = getLogger(__name__)


@pytest.mark.parametrize(
    "source, target, source_lang, target_lang, expected, mock_result",
    [
        (
            "The cat",
            "Le chat",
            "English",
            "French",
            """
        TextAlignment(alignment=[
            TokenAlignment(source_token='The', target_token='Le'),
            TokenAlignment(source_token='cat', target_token='chat')
        ])
        """,
            TextAlignment(
                alignment=[
                    TokenAlignment(source_token="The", target_token="Le"),
                    TokenAlignment(source_token="cat", target_token="chat"),
                ]
            ),
        ),
        (
            "Good morning",
            "Bonjour",
            "English",
            "French",
            """
        TextAlignment(alignment=[
            TokenAlignment(source_token='Good', target_token='Bonjour'),
            TokenAlignment(source_token='morning', target_token='Bonjour')
        ])
        """,
            TextAlignment(
                alignment=[
                    TokenAlignment(source_token="Good", target_token="Bonjour"),
                    TokenAlignment(source_token="morning", target_token="Bonjour"),
                ]
            ),
        ),
    ],
)
def test_align_tokens(source, target, source_lang, target_lang, expected, mock_result):
    source_tokens = source.strip().split()
    target_tokens = target.strip().split()
    expected_alignment = eval(expected.strip())

    class TestMockLLMAdapter(LLMAdapter):
        def __call__(self, messages: list[dict]) -> TextAlignment:
            return mock_result

    adapter = TestMockLLMAdapter()
    result = align_tokens(
        adapter, source_tokens, target_tokens, source_lang, target_lang
    )
    assert result == expected_alignment


@pytest.mark.parametrize(
    "source, target, custom_messages, expected, mock_result",
    [
        (
            "The cat is on the mat",
            "Le chat est sur le tapis",
            """
        [
            {"role": "system", "content": "You are a translator aligning English to French."},
            {"role": "user", "content": "Align these tokens:"}
        ]
        """,
            """
        TextAlignment(alignment=[
            TokenAlignment(source_token='The', target_token='Le'),
            TokenAlignment(source_token='cat', target_token='chat'),
            TokenAlignment(source_token='is', target_token='est'),
            TokenAlignment(source_token='on', target_token='sur'),
            TokenAlignment(source_token='the', target_token='le'),
            TokenAlignment(source_token='mat', target_token='tapis')
        ])
        """,
            TextAlignment(
                alignment=[
                    TokenAlignment(source_token="The", target_token="Le"),
                    TokenAlignment(source_token="cat", target_token="chat"),
                    TokenAlignment(source_token="is", target_token="est"),
                    TokenAlignment(source_token="on", target_token="sur"),
                    TokenAlignment(source_token="the", target_token="le"),
                    TokenAlignment(source_token="mat", target_token="tapis"),
                ]
            ),
        ),
        (
            "I love sushi",
            "私 は 寿司 が 大好き です",
            """
        [
            {"role": "system", "content": "You are a translator aligning English to 日本語."},
            {"role": "user", "content": "Align these tokens:"}
        ]
        """,
            """
        TextAlignment(alignment=[
            TokenAlignment(source_token='I', target_token='私'),
            TokenAlignment(source_token='I', target_token='は'),
            TokenAlignment(source_token='love', target_token='大好き'),
            TokenAlignment(source_token='love', target_token='です'),
            TokenAlignment(source_token='sushi', target_token='寿司'),
            TokenAlignment(source_token='sushi', target_token='が')
        ])
        """,
            TextAlignment(
                alignment=[
                    TokenAlignment(source_token="I", target_token="私"),
                    TokenAlignment(source_token="I", target_token="は"),
                    TokenAlignment(source_token="love", target_token="大好き"),
                    TokenAlignment(source_token="love", target_token="です"),
                    TokenAlignment(source_token="sushi", target_token="寿司"),
                    TokenAlignment(source_token="sushi", target_token="が"),
                ]
            ),
        ),
    ],
)
def test_align_tokens_raw(source, target, custom_messages, expected, mock_result):
    source_tokens = source.strip().split()
    target_tokens = target.strip().split()
    custom_messages = eval(custom_messages.strip())
    expected_alignment = eval(expected.strip())

    class TestMockLLMAdapter(LLMAdapter):
        def __call__(self, messages):
            return mock_result

    adapter = TestMockLLMAdapter()
    result = align_tokens_raw(adapter, source_tokens, target_tokens, custom_messages)
    assert result == expected_alignment


def test_align_tokens_error_handling():
    """Test error handling in alignment functions."""

    class ErrorAdapter(LLMAdapter):
        def __call__(self, messages: list[dict]) -> TextAlignment:
            raise ValueError("Test error")

    error_adapter = ErrorAdapter()

    with pytest.raises(ValueError):
        align_tokens(error_adapter, ["test"], ["test"])

    with pytest.raises(ValueError):
        align_tokens_raw(error_adapter, ["test"], ["test"], [])
