"""Centralized constants for lexi_align."""

# Token prefixes for message formatting
SOURCE_TOKENS_PREFIX = "source_tokens: "
TARGET_TOKENS_PREFIX = "target_tokens: "
REMAINING_SOURCE_PREFIX = "remaining_source_tokens: "
REMAINING_TARGET_PREFIX = "remaining_target_tokens: "

# Special tokens
UNALIGNED_MARKER = "<unaligned>"

# Default configuration
DEFAULT_CONCURRENCY = 8
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_RETRIES = 3

# Logging
DEFAULT_MAX_LOG_CHARS = 4000

# Retry/temperature
TEMPERATURE_INCREMENT_PER_RETRY = 0.1

# Reasoning
REASONING_PROMPT = """Before providing alignments, explain your reasoning in the 'reasoning' field:
1. Identify semantic correspondences between tokens
2. Consider syntactic roles and grammatical structures
3. Handle idiomatic expressions and multi-word units
4. Note structural differences between languages
5. Explain why tokens should remain unaligned if applicable
Then provide the complete alignment in the 'alignment' field."""
