# lexi-align

[![PyPI version](https://badge.fury.io/py/lexi-align.svg)](https://badge.fury.io/py/lexi-align)
[![CI](https://github.com/borh-lab/lexi-align/actions/workflows/ci.yaml/badge.svg)](https://github.com/borh-lab/lexi-align/actions/workflows/ci.yaml)

Token alignment using structured generation with Large Language Models.
This library does not do any tokenization, so supports different types of units (words, phrases, etc. arbitrary tokens), as well as language pairs (or monolingual alignment).

The library is API-backend agnostic and only directly depends on [Pydantic](https://docs.pydantic.dev/latest/) and [llm-schema-lite](https://github.com/rohitgarud/llm-schema-lite), so you will need to bring your own API code or use the provided [litellm](https://github.com/BerriAI/litellm) integration.

## Quick Start

### Installation

```bash
# For API-based models (OpenAI, Anthropic, Bedrock, Azure, etc.)
pip install lexi-align[litellm]

# OR for local models (HuggingFace Transformers)
pip install lexi-align[outlines]

# OR for quantized models (llama.cpp)
pip install lexi-align[llama]
```

There are other extras for using CUDA/ROCM/CPU PyTorch versions, as well as the `viz` visualization extra.
Most of these are also available in the `dev` dependency group.

### Align Tokens in Two Sentences

```python
from lexi_align import create_adapter, align_tokens

# Create adapter (uses GPT-4o-mini via LiteLLM)
adapter = create_adapter("litellm:gpt-4o-mini")

# Your pre-tokenized input
source = ["The", "cat", "sat"]
target = ["Le", "chat", "était", "assis"]

# Align
result = align_tokens(
    adapter, source, target,
    source_language="English",
    target_language="French"
)

# Use the alignment
for align in result.alignment.alignment:
    print(f"{align.source} → {align.target}")
# Output:
# The → Le
# cat → chat
# sat → était
# sat → assis
```

### Process Multiple Pairs

```python
from lexi_align import align_many, align_many_async

# Align multiple (source, target) pairs sequentially
pairs = [
    (["The", "cat"], ["Le", "chat"]),
    (["A", "dog"], ["Un", "chien"]),
]

results = align_many(
    adapter, pairs,
    source_language="English",
    target_language="French"
)

# Or use async version for concurrent processing
results = await align_many_async(
    adapter, pairs,
    source_language="English",
    target_language="French",
    concurrency=8,  # Max concurrent requests
    show_progress=True
)
```

### Align Multiple Sentences (Dataset)

```python
from lexi_align import create_adapter, align_dataset

adapter = create_adapter("litellm:gpt-4o-mini")

# Your dataset
sources = [
    ["The", "cat", "sat"],
    ["A", "dog", "runs"],
    # ... more examples
]
targets = [
    ["Le", "chat", "était", "assis"],
    ["Un", "chien", "court"],
    # ... more examples
]

# Automatically uses best strategy (batch/async/sequential)
results = align_dataset(
    adapter, sources, targets,
    source_language="English",
    target_language="French",
    show_progress=True  # Shows progress bar
)

# Check results
successful = [r for r in results if r.alignment]
print(f"✓ Aligned {len(successful)}/{len(results)} pairs")
```

## Choosing an Adapter

| Adapter | Best For | Speed | Example |
|---------|----------|-------|---------|
| **litellm** | API models (OpenAI, Anthropic, Azure) | Fast (async) | `create_adapter("litellm:openai/gpt-5-mini")` |
| **outlines** | Local HuggingFace models | Medium (batched) | `create_adapter("transformers:Qwen/Qwen3-4B-Instruct-2507")` |
| **llama-cpp** | Quantized GGUF models | Slower (sequential) | `create_adapter("llama:model.gguf")` |
| **sglang** | SGLang inference server | Fast (batched) | `create_adapter("sglang:Qwen/Qwen3-4B-Instruct-2507")` |

**Note:** LiteLLM and SGLang adapters use a default 15-minute timeout for requests. This can be customized via adapter parameters if needed.

Processing strategies:
- **Async adapters** (litellm, sglang): Process multiple pairs concurrently for faster throughput
- **Batch adapters** (outlines, sglang): Process multiple pairs in GPU batches for efficiency
- **Sequential adapters** (llama-cpp): Process pairs one at a time

## Common Patterns

### Evaluate Against Gold Standard

```python
from lexi_align import align_and_evaluate_dataset

# Combined alignment and evaluation in one call
results, metrics = align_and_evaluate_dataset(
    adapter, sources, targets, gold_alignments,
    source_language="English",
    target_language="French",
    show_progress=True
)

print(f"Precision: {metrics['micro']['precision']:.3f}")
print(f"Recall:    {metrics['micro']['recall']:.3f}")
print(f"F1:        {metrics['micro']['f_measure']:.3f}")
print(f"AER:       {metrics['micro']['aer']:.3f}")

# Or use separate functions for more control
from lexi_align import align_dataset, evaluate_alignments

results = align_dataset(adapter, sources, targets)
predicted = [r.alignment for r in results if r.alignment]
metrics = evaluate_alignments(predicted, gold_alignments)
```

### Handle Failures Gracefully

Since performing alignments is side-effecting IO we are not guaranteed to succeed every time, given GPU memory limitation or API outages.
Each alignment returns information on the successful or unsuccesful state of the alignment.

```python
for i, result in enumerate(results):
    if result.alignment:
        # Success!
        print(f"✓ Pair {i}: {len(result.alignment.alignment)} alignments")
    else:
        # Check diagnostics
        print(f"✗ Pair {i} failed after {len(result.attempts)} attempts")
        for attempt in result.attempts:
            if attempt.validation_errors:
                print(f"  Validation errors: {attempt.validation_errors[:3]}")
            if attempt.exception:
                print(f"  Exception: {attempt.exception}")
```

### Few-Shot Examples

```python
from lexi_align.models import TextAlignment, TokenAlignment

# Provide examples
examples = [
    (
        ["The", "cat"],
        ["Le", "chat"],
        TextAlignment(alignment=[
            TokenAlignment(source="The", target="Le"),
            TokenAlignment(source="cat", target="chat")
        ])
    )
]

# Examples are automatically included in prompts
results = align_dataset(
    adapter, sources, targets,
    examples=examples,
    source_language="English",
    target_language="French"
)
```

### Custom Guidelines

```python
guidelines = """
1. Align content words (nouns, verbs, adjectives) first
2. Function words should be aligned when they have clear correspondences
3. Handle idiomatic expressions by aligning all components
"""

results = align_dataset(
    adapter, sources, targets,
    guidelines=guidelines
)
```

## Advanced Usage

### Basic Usage

The library expects pre-tokenized input--it does not perform any tokenization. You must provide tokens as lists of strings:

```python
from lexi_align import create_adapter, align_tokens

# Initialize the LLM adapter using the factory
llm_adapter = create_adapter(
    "litellm:bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.0
)

# Provide pre-tokenized input with repeated tokens
source_tokens = ["the", "big", "cat", "saw", "the", "cat"]  # Note: "the" and "cat" appear twice,
# but this is handled automatically by the library
target_tokens = ["le", "gros", "chat", "a", "vu", "le", "chat"]

result = align_tokens(
    llm_adapter,
    source_tokens,
    target_tokens,
    source_language="English",
    target_language="French"
)

# Access the alignment result
if result.alignment:
    print("Successful alignment:")
    for align in result.alignment.alignment:
        print(f"{align.source} -> {align.target}")
else:
    print("Alignment failed. Check result.attempts for details.")

# Example output (will vary based on model and input):
# Successful alignment:
# the₁ -> le₁
# big -> gros
# cat₁ -> chat₁
# saw -> a
# saw -> vu
# the₂ -> le₂
# cat₂ -> chat₂
```

By default, subscript numbers are added to duplicates to disambiguate alignments.

### Batched Processing

For processing multiple sequences efficiently using adapters that support native batching:

```python
from lexi_align import create_adapter, align_tokens_batched

# Initialize adapter with a local model
llm_adapter = create_adapter(
    "transformers:Qwen/Qwen3-0.6B",
    dtype="bfloat16",  # optional: choose quantization
    device="cuda"      # optional: specify device
)

# Multiple sequences to align
source_sequences = [
    ["The", "cat", "sat"],
    ["I", "love", "coding"],
]
target_sequences = [
    ["Le", "chat", "assis"],
    ["J'", "aime", "coder"],
]

# Process in batches
results = align_tokens_batched(
    llm_adapter,
    source_sequences,
    target_sequences,
    source_language="English",
    target_language="French",
    batch_size=2  # Process 2 sequences at a time
)

# Each result contains alignment and diagnostic information
for result in results:
    if result.alignment:
        print(result.alignment.alignment)
    else:
        print("Failed attempts:", len(result.attempts))
```

### Async Processing

For asynchronous processing:

```python
import asyncio
from lexi_align import create_adapter, align_tokens_async

async def align_async():
    llm_adapter = create_adapter(
        "litellm:gpt-5-nano",
        temperature=0.3
    )

    source = ["The", "cat", "sat"]
    target = ["Le", "chat", "assis"]

    result = await align_tokens_async(
        llm_adapter,
        source,
        target,
        source_language="English",
        target_language="French"
    )

    return result

# Run async alignment
result = asyncio.run(align_async())

# Access the alignment result (similar to sync version)
if result.alignment:
    print("Successful async alignment:")
    for align in result.alignment.alignment:
        print(f"{align.source} -> {align.target}")
else:
    print("Async alignment failed. Check result.attempts for details.")
```

### Diagnostic Information

The alignment functions return an `AlignmentResult` object containing both the alignment and diagnostic information:

```python
result = align_tokens(
    llm_adapter,
    source_tokens,
    target_tokens,
    source_language="English",
    target_language="French"
)

# Access the alignment
if result.alignment:
    print("Successful alignment:", result.alignment.alignment)

# Access attempt history
for attempt in result.attempts:
    print(f"Attempt {attempt.attempt_number}:")
    print("Messages sent:", attempt.messages_sent)
    print("Validation passed:", attempt.validation_passed)
    if attempt.validation_errors:
        print("Validation errors:", attempt.validation_errors)
    if attempt.exception:
        print("Exception:", attempt.exception)
```

Note that `AlignmentResult` is returned even if the alignment failed (due to external or internal factors).
Use the above code as a guide to examine the errors.

### Using Custom Guidelines and Examples

You can provide custom alignment guidelines and examples to improve alignment quality:

```python
from lexi_align import create_adapter, align_tokens
from lexi_align.models import TextAlignment, TokenAlignment

# Initialize adapter using the factory
llm_adapter = create_adapter(
    "litellm:gpt-5-mini",
    temperature=0.0
)

# Define custom guidelines
guidelines = """
1. Align content words (nouns, verbs, adjectives) first
2. Function words should be aligned when they have clear correspondences
3. Handle idiomatic expressions by aligning all components
4. One source token can align to multiple target tokens and vice versa
"""

# Provide examples to demonstrate desired alignments
examples = [
    (
        "The cat".split(),  # source tokens
        "Le chat".split(),  # target tokens
        TextAlignment(      # gold alignment
            alignment=[
                TokenAlignment(source="The", target="Le"),
                TokenAlignment(source="cat", target="chat"),
            ]
        )
    ),
    # Add more examples as needed
]

# Use guidelines and examples in alignment
alignment = align_tokens(
    llm_adapter,
    source_tokens,
    target_tokens,
    source_language="English",
    target_language="French",
    guidelines=guidelines,
    examples=examples
)
```

### Raw Message Control

For more control over the prompt, you can use `align_tokens_raw` to provide custom messages:

```python
from lexi_align.core import align_tokens_raw

custom_messages = [
    {"role": "system", "content": "You are an expert translator aligning English to French."},
    {"role": "user", "content": "Follow these guidelines:\n" + guidelines},
    # Add any other custom messages
]

alignment = align_tokens_raw(
    llm_adapter,
    source_tokens,
    target_tokens,
    custom_messages
)
```

### Token Uniquification

The library automatically handles repeated tokens by adding unique markers:

```python
from lexi_align.utils import make_unique, remove_unique

# Tokens with repeats
tokens = ["the", "cat", "the", "mat"]

# Add unique markers
unique_tokens = make_unique(tokens)
print(unique_tokens)  # ['the₁', 'cat', 'the₂', 'mat']

# Remove markers
original_tokens = remove_unique(unique_tokens)
print(original_tokens)  # ['the', 'cat', 'the', 'mat']
```

You can also customize the marker style:

```python
from lexi_align.text_processing import create_underscore_generator

# Use underscore markers instead of subscripts
marker_gen = create_underscore_generator()
unique_tokens = make_unique(tokens, marker_gen)
print(unique_tokens)  # ['the_1', 'cat', 'the_2', 'mat']
```

### Dynamic Schema Generation

The library supports dynamic JSON schema generation and uses it both as an enforcement mechanism and as explicit prompt guidance. Adapters that can pass a schema to the backend/generator (Outlines, SGLang, LlamaCpp via Outlines integration) use a dynamic schema to enforce token enums and alignment-length constraints server-side.
Remote/backends that cannot enforce schemas themselves (e.g., litellm) can optionally request schema-validated responses by enabling `use_dynamic_schema=True`.
Additionally, some adapters (LiteLLMAdapter and SGLangAdapter) embed the schema JSON in the system prompt by default so the model receives explicit, machine-readable guidance even when the API cannot validate responses.

```python
from lexi_align import create_adapter, align_tokens

# Initialize adapter with dynamic schema enabled
llm_adapter = create_adapter(
    "transformers:Qwen/Qwen3-4B-Instruct-2507",
    dtype="bfloat16",
    device="cuda",
    batch_size=4,  # Enable efficient batching
    use_dynamic_schema=True  # Enable schema validation
)

# For LiteLLM (API-based), enable schema validation:
llm_adapter = create_adapter(
    "litellm:gpt-5-nano",
    use_dynamic_schema=True  # Request schema-validated responses
)

# The library automatically:
# 1. Generates a schema specific to your token sets
# 2. Validates token existence and uniqueness
# 3. Enforces alignment length constraints
# 4. Provides detailed error messages for invalid alignments

result = align_tokens(
    llm_adapter,
    source_tokens,
    target_tokens,
    source_language="English",
    target_language="French"
)

# Check validation results
if result.alignment:
    print("Valid alignment achieved")
else:
    for attempt in result.attempts:
        if attempt.validation_errors:
            print(f"Attempt {attempt.attempt_number} errors:")
            for error_type, msg, tokens in attempt.validation_errors:
                print(f"- {error_type}: {msg}")
```

The dynamic schema:
- Ensures tokens exist in the source/target sets
- Handles repeated tokens with unique markers
- Sets minimum/maximum alignment lengths
- Provides clear error messages for invalid alignments
- Supports partial alignments with retries

**Adapter-specific behavior:**
- **Outlines/SGLang/LlamaCpp**: Enforce schema server-side for guaranteed valid JSON
- **LiteLLM**: By default embeds schema in prompt for guidance; pass `use_dynamic_schema=True` to request validation
- All adapters: Include schema in system prompt by default for explicit model guidance

### Logging

The library uses Python's standard `logging` module. To see detailed logs, including debug messages, configure the logger for the `lexi_align` namespace:

```python
import logging

# Configure basic logging (e.g., to console)
logging.basicConfig(level=logging.INFO) # Set your desired overall level

# Enable DEBUG level specifically for lexi-align
logging.getLogger("lexi_align").setLevel(logging.DEBUG)

# Now import and use lexi-align
from lexi_align.core import align_tokens
# ... rest of your code
```

### Using Local Models with llama.cpp

For running local models with llama.cpp:

```python
from lexi_align import create_adapter, align_tokens

# Initialize the llama.cpp adapter with a local model
llm_adapter = create_adapter(
    "llama:path/to/model.gguf",
    n_gpu_layers=-1  # Use GPU acceleration
)

# Note that for some GGUF models the pre-tokenizer might fail,
# in which case you can specify the tokenizer_repo_id parameter:
llm_adapter = create_adapter(
    "llama:path/to/model.gguf",
    n_gpu_layers=-1,
    repo_id="base-model-repo-id"  # HuggingFace repo for tokenizer
)

# Use the same API as with other adapters
alignment = align_tokens(
    llm_adapter,
    source_tokens,
    target_tokens,
    source_language="English",
    target_language="French"
)
```

### Performance

Here are some preliminary results on the test EN-SL subset of XL-WA (obtained with library version 0.1.0):

#### gpt-4o-2024-08-06 (1shot) (seed=42)

| Language Pair | Precision | Recall | F1 |
| --- | --- | --- | --- |
| EN-SL | 0.863 | 0.829 | 0.846 |
| **Average** | **0.863** | **0.829** | **0.846** |

#### claude-3-haiku-20240307 (1shot)

| Language Pair | Precision | Recall | F1 |
| --- | --- | --- | --- |
| EN-SL | 0.651 | 0.630 | 0.640 |
| **Average** | **0.651** | **0.630** | **0.640** |

#### meta-llama/Llama-3.2-3B-Instruct (1shot)

| Language Pair | Precision | Recall | F1 |
| --- | --- | --- | --- |
| EN-SL | 0.606 | 0.581 | 0.593 |
| **Average** | **0.606** | **0.581** | **0.593** |

For reference, the 1-shot (1 example) `gpt-4o-2024-08-06` results for EN-SL outperform all systems presented in the [paper](https://ceur-ws.org/Vol-3596/paper32.pdf) (Table 2).
Smaller LLMs perform below SOTA.

### Pharaoh Format Export

While the core alignment functions work with pre-tokenized input, the Pharaoh format utilities currently assume space-separated tokens when parsing/exporting. If your tokens contain spaces or require special tokenization, you'll need to handle this separately.

```python
from lexi_align.utils import export_pharaoh_format

# Note: Pharaoh format assumes space-separated tokens
# Default separator is tab
pharaoh_format = export_pharaoh_format(
    source_tokens,  # Pre-tokenized list of strings
    target_tokens,  # Pre-tokenized list of strings
    alignment
)

print(pharaoh_format)
# Output (will differ depending on chosen model):
# The cat sat on the mat    Le chat était assis sur le tapis    0-0 1-1 2-2 2-3 3-4 4-5 5-6

# Use custom separator
pharaoh_format = export_pharaoh_format(
    source_tokens,
    target_tokens,
    alignment,
    sep=" ||| "  # Custom separator
)

print(pharaoh_format)
# Output:
# The cat sat on the mat ||| Le chat était assis sur le tapis ||| 0-0 1-1 2-2 2-3 3-4 4-5 5-6
```

The Pharaoh format consists of three tab-separated fields:
1. Source sentence (space-separated tokens)
2. Target sentence (space-separated tokens)
3. Alignments as space-separated pairs of indices (source-target)

### Running Evaluations

The package includes scripts to evaluate alignment performance on the [XL-WA dataset](https://github.com/SapienzaNLP/XL-WA) (CC BY-NC-SA 4.0):

```bash
# Install dependencies
pip install lexi-align[litellm]

# Basic evaluation on a single language pair
python evaluations/xl-wa.py --lang-pairs EN-SL

# Evaluate on all language pairs
python evaluations/xl-wa.py --lang-pairs all

# Full evaluation with custom parameters using an API model
python evaluations/xl-wa.py \
    --lang-pairs EN-FR EN-DE \
    --model litellm:openai/gpt-5-mini \
    --async \
    --temperature 0.1 \
    --seed 42 \
    --model-seed 42 \
    --num-train-examples 3 \
    --output results.json

# Full evaluation with custom parameters using a local Transformers model
python evaluations/xl-wa.py \
    --lang-pairs EN-FR EN-DE \
    --model transformers:Qwen/Qwen3-0.6B \
    --temperature 0.0 \
    --seed 42 \
    --num-train-examples 3 \
    --output results.json
```

Available command-line arguments:

- `--lang-pairs`: Language pairs to evaluate (e.g., EN-SL EN-DE) or "all"
- `--model`: LLM model to use (default: gpt-4o)
- `--temperature`: Temperature for LLM sampling (default: 0.0)
- `--seed`: Random seed for example selection (default: 42)
- `--model-seed`: Seed for LLM sampling (optional)
- `--num-train-examples`: Number of training examples for few-shot learning
- `--sample-size`: Number of test examples to evaluate per language pair
- `--output`: Path to save results JSON file
- `--verbose`: Enable verbose logging

## Changelog

### v0.4.0 (September 2025)

Key highlights:

- New SGLang adapter: an Outlines-backed adapter using OpenAI-compatible SGLang servers (sync/async and batch (which is just async) support).
- Dynamic schema: dynamic JSON schemas are used for enforcement where supported (Outlines, SGLang, LlamaCpp). LiteLLMAdapter can optionally request schema-validated responses via use_dynamic_schema=True and (like SGLangAdapter) embeds the schema JSON in the system prompt by default (include_schema=True).
- Batching & retry robustness: align_tokens_batched now maintains per-sequence message histories, supports partial retries/fallbacks, and uses an adapter.batch() interface for true batched backends.
- LlamaCppAdapter: improved Outlines integration, split-model detection/tokenizer handling, max_tokens/seed/json-retry controls and a deterministic heuristic fallback.
- LLMAdapter base/API refinements: async fallback (acall runs sync call in a thread), a default batch() implementation (adapters may override), and improved JSON-retry detection/logging (_is_json_invalid_error and _retry_on_invalid_json/_async variants).
- Validation & diagnostics: better handling of explicit <unaligned> entries, richer AlignmentAttempt/AlignmentResult diagnostics, and helpers for categorizing/normalizing validation errors (categorize_validation_errors, summarize_result).
- API note: TokenAlignment/TextAlignment field naming and validators have been tightened — use TokenAlignment(source=..., target=...) in examples (source_token/target_token names are no longer used).
- Visualization & utilities: improved matplotlib/Altair visualizations (dynamic sizing and HTML export) and utility improvements (TokenMapping position handling, normalize/remove-unique helpers, Pharaoh format round‑trip verification).

### v0.3.0 (March 2024)
- Added support for batched processing with `align_tokens_batched`
- Added async support via `align_tokens_async`
- Added enhanced diagnostics and error reporting
- Added alignment visualization tools
- Added token-level analysis and metrics
- Added support for custom marker types (subscript/underscore)
- Added support for custom separators in Pharaoh format
- Improved retry logic and validation
- Added CI and evaluation scripts

### v0.2.x (2024-03-07)
- Added support for local models via Outlines and llama.cpp
- Added retries on errors or invalid alignments
- Added async completion support for litellm
- Added support for model weight quantization
- Added improved error messages and validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{lexi_align,
  title = {lexi-align: Word Alignment via Structured Generation},
  author = {Hodošček, Bor},
  year = {2024},
  url = {https://github.com/borh-lab/lexi-align}
}
```

## References

We use the XL-WA dataset ([repository](https://github.com/SapienzaNLP/XL-WA)) to perform evaluations:

```bibtex
@InProceedings{martelli-EtAl:2023:clicit,
  author    = {Martelli, Federico  and  Bejgu, Andrei Stefan  and  Campagnano, Cesare  and  Čibej, Jaka  and  Costa, Rute  and  Gantar, Apolonija  and  Kallas, Jelena  and  Koeva, Svetla  and  Koppel, Kristina  and  Krek, Simon  and  Langemets, Margit  and  Lipp, Veronika  and  Nimb, Sanni  and  Olsen, Sussi  and  Pedersen, Bolette Sandford  and  Quochi, Valeria  and  Salgado, Ana  and  Simon, László  and  Tiberius, Carole  and  Ureña-Ruiz, Rafael-J  and  Navigli, Roberto},
  title     = {XL-WA: a Gold Evaluation Benchmark for Word Alignment in 14 Language Pairs},
  booktitle      = {Procedings of the Ninth Italian Conference on Computational Linguistics (CLiC-it 2023)},
  month          = {November},
  year           = {2023}
}
```

This code was spun out of the [hachidaishu-translation](https://github.com/borh/hachidaishu-translation) project, presented at  [JADH2024](https://jadh2024.l.u-tokyo.ac.jp/).

## Development

Contributions are welcome! Please feel free to submit a Pull Request.

To set up the development environment:

```bash
git clone https://github.com/borh-lab/lexi-align.git
cd lexi-align
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Scripts are provided in the `scripts/` directory for linting and running tests in one go.
