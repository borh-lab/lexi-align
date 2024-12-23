from logging import getLogger
from typing import Optional

import pytest

from lexi_align.adapters import LLMAdapter
from lexi_align.core import (
    align_tokens,
    align_tokens_batched,
    align_tokens_raw,
)
from lexi_align.models import TextAlignment, TokenAlignment

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
        TextAlignment.from_token_alignments([
            TokenAlignment(source='The', target='Le'),
            TokenAlignment(source='cat', target='chat')
        ], ['The', 'cat'], ['Le', 'chat'])
        """,
            TextAlignment.from_token_alignments(
                [
                    TokenAlignment(source="The", target="Le"),
                    TokenAlignment(source="cat", target="chat"),
                ],
                ["The", "cat"],
                ["Le", "chat"],
            ),
        ),
        (
            "Good morning",
            "Bonjour",
            "English",
            "French",
            """
        TextAlignment.from_token_alignments([
            TokenAlignment(source='Good', target='Bonjour'),
            TokenAlignment(source='morning', target='Bonjour')
        ], ['Good', 'morning'], ['Bonjour'])
        """,
            TextAlignment.from_token_alignments(
                [
                    TokenAlignment(source="Good", target="Bonjour"),
                    TokenAlignment(source="morning", target="Bonjour"),
                ],
                ["Good", "morning"],
                ["Bonjour"],
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
    assert result.alignment == expected_alignment


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
        TextAlignment.from_token_alignments([
            TokenAlignment(source='The', target='Le'),
            TokenAlignment(source='cat', target='chat'),
            TokenAlignment(source='is',  target='est'),
            TokenAlignment(source='on',  target='sur'),
            TokenAlignment(source='the', target='le'),
            TokenAlignment(source='mat', target='tapis')
        ], ['The', 'cat', 'is', 'on', 'the', 'mat'],
           ['Le', 'chat', 'est', 'sur', 'le', 'tapis'])
        """,
            TextAlignment.from_token_alignments(
                [
                    TokenAlignment(source="The", target="Le"),
                    TokenAlignment(source="cat", target="chat"),
                    TokenAlignment(source="is", target="est"),
                    TokenAlignment(source="on", target="sur"),
                    TokenAlignment(source="the", target="le"),
                    TokenAlignment(source="mat", target="tapis"),
                ],
                ["The", "cat", "is", "on", "the", "mat"],
                ["Le", "chat", "est", "sur", "le", "tapis"],
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
        TextAlignment.from_token_alignments([
            TokenAlignment(source='I',     target='私'),
            TokenAlignment(source='I',     target='は'),
            TokenAlignment(source='love',  target='大好き'),
            TokenAlignment(source='love',  target='です'),
            TokenAlignment(source='sushi', target='寿司'),
            TokenAlignment(source='sushi', target='が')
        ], ['I', 'love', 'sushi'],
           ['私', 'は', '寿司', 'が', '大好き', 'です'])
        """,
            TextAlignment.from_token_alignments(
                [
                    TokenAlignment(source="I", target="私"),
                    TokenAlignment(source="I", target="は"),
                    TokenAlignment(source="love", target="大好き"),
                    TokenAlignment(source="love", target="です"),
                    TokenAlignment(source="sushi", target="寿司"),
                    TokenAlignment(source="sushi", target="が"),
                ],
                ["I", "love", "sushi"],
                ["私", "は", "寿司", "が", "大好き", "です"],
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
    assert result.alignment == expected_alignment


def test_align_tokens_error_handling():
    """Test error handling in alignment functions."""


def test_batch_vs_sequential_alignment():
    """Test that batch processing gives same results as sequential processing."""

    class DeterministicAdapter(LLMAdapter):
        """Mock adapter that returns deterministic results based on input."""

        def __init__(self):
            self.call_count = 0

        def __call__(self, messages: list[dict]) -> TextAlignment:
            self.call_count += 1
            # Extract tokens from the last message
            content = messages[-1]["content"]
            source = content.split("source_tokens: ")[1].split("\n")[0].split()
            target = content.split("target_tokens: ")[1].split("\n")[0].split()
            # Create deterministic alignment based on token lengths
            return TextAlignment.from_token_alignments(
                [
                    TokenAlignment(source=s, target=t)
                    for s, t in zip(source, target)
                    if len(s) == len(t)  # Simple deterministic rule
                ],
                source,
                target,
            )

        def supports_true_batching(self) -> bool:
            return True

        def batch(
            self, batch_messages: list[list[dict]], max_retries: int = 3
        ) -> list[Optional[TextAlignment]]:
            return [self(msgs) for msgs in batch_messages]

    # Test data
    source_sequences = [
        ["cat", "dog", "bird"],
        ["run", "walk", "jump"],
        ["red", "blue", "green"],
    ]
    target_sequences = [
        ["cat", "pet", "fly"],
        ["run", "move", "leap"],
        ["red", "cyan", "vert"],
    ]

    # Process sequentially
    adapter_seq = DeterministicAdapter()
    sequential_results = [
        align_tokens(adapter_seq, src, tgt, "en", "fr")
        for src, tgt in zip(source_sequences, target_sequences)
    ]

    # Process with batching
    adapter_batch = DeterministicAdapter()
    batch_results = align_tokens_batched(
        adapter_batch, source_sequences, target_sequences, "en", "fr", batch_size=2
    )

    # Compare results
    assert len(sequential_results) == len(batch_results)
    for seq_res, batch_res in zip(sequential_results, batch_results):
        assert seq_res.alignment == batch_res.alignment


def test_retry_mechanism():
    """Test retry mechanism with a simpler, more deterministic adapter."""

    class RetryAdapter(LLMAdapter):
        """Adapter that fails N times then succeeds with predetermined alignments."""

        def __init__(self, fail_count: int, final_alignment: TextAlignment):
            self.fail_count = fail_count
            self.final_alignment = final_alignment
            self.call_count = 0

        def __call__(self, messages: list[dict]) -> TextAlignment:
            self.call_count += 1
            if self.call_count <= self.fail_count:
                raise ValueError(f"Simulated failure #{self.call_count}")
            return self.final_alignment

        def supports_true_batching(self) -> bool:
            return False

    # Test case: should fail twice then succeed
    expected_alignment = TextAlignment.from_token_alignments(
        [TokenAlignment(source="test", target="test")],
        source_tokens=["test"],
        target_tokens=["test"],
    )
    adapter = RetryAdapter(fail_count=2, final_alignment=expected_alignment)

    result = align_tokens(adapter, ["test"], ["test"], max_retries=3)

    assert result.alignment == expected_alignment
    assert len(result.attempts) == 3  # Two failures + success
    assert result.attempts[0].exception == "Simulated failure #1"
    assert result.attempts[1].exception == "Simulated failure #2"
    assert result.attempts[2].exception is None
    assert result.attempts[2].raw_response == expected_alignment

    # Test case: should fail all attempts
    adapter = RetryAdapter(fail_count=3, final_alignment=expected_alignment)
    result = align_tokens(adapter, ["test"], ["test"], max_retries=3)

    assert result.alignment is None
    assert len(result.attempts) == 3
    assert all(a.exception is not None for a in result.attempts)


def test_batch_partial_failures():
    """Test batch processing with deterministic partial failures."""

    class BatchPartialFailureAdapter(LLMAdapter):
        """Adapter that succeeds/fails based on predetermined patterns."""

        def __init__(self, success_pattern: list[bool]):
            self.success_pattern = success_pattern
            self.current_batch = 0

        def __call__(self, messages: list[dict]) -> TextAlignment:
            return TextAlignment.from_token_alignments(
                [TokenAlignment(source="test", target="test")], ["test"], ["test"]
            )

        def supports_true_batching(self) -> bool:
            return True

        def batch(
            self, batch_messages: list[list[dict]], max_retries: int = 3
        ) -> list[Optional[TextAlignment]]:
            results: list[Optional[TextAlignment]] = []
            for i in range(len(batch_messages)):
                if self.success_pattern[i]:
                    results.append(
                        TextAlignment.from_token_alignments(
                            [TokenAlignment(source=f"test{i}", target=f"test{i}")],
                            [f"test{i}"],
                            [f"test{i}"],
                        )
                    )
                else:
                    results.append(None)
            return results

    # Test with mix of successes and failures
    success_pattern = [True, False, True]  # First and third succeed, second fails
    adapter = BatchPartialFailureAdapter(success_pattern)

    source_sequences = [["test0"], ["test1"], ["test2"]]
    target_sequences = [["test0"], ["test1"], ["test2"]]

    results = align_tokens_batched(
        adapter, source_sequences, target_sequences, batch_size=3
    )

    assert len(results) == 3
    assert results[0].alignment is not None
    assert results[1].alignment is None
    assert results[2].alignment is not None

    # Verify successful alignments
    assert results[0].alignment.alignment[0].source_token == "test0"
    assert results[2].alignment.alignment[0].source_token == "test2"

    # Verify failed alignment attempts
    assert len(results[1].attempts) > 0
    assert results[1].attempts[-1].validation_passed is False

    class ErrorAdapter(LLMAdapter):
        def __call__(self, messages: list[dict]) -> TextAlignment:
            raise ValueError("Test error")

        async def acall(self, messages: list[dict]) -> TextAlignment:
            raise ValueError("Test error")

    error_adapter = ErrorAdapter()

    # Test that errors are captured in AlignmentResult
    result = align_tokens(error_adapter, ["test"], ["test"])
    assert result.alignment is None
    assert len(result.attempts) > 0
    assert result.attempts[0].exception == "Test error"

    result = align_tokens_raw(error_adapter, ["test"], ["test"], [])
    assert result.alignment is None
    assert len(result.attempts) > 0
    assert result.attempts[0].exception == "Test error"
