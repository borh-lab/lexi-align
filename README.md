# lexi-align

Word alignment of multilingual sentences using structured generation with Large Language Models.

## Installation

Install from PyPI:

```bash
pip install lexi-align
```

(or your favorite method)

The library is API-backend agnostic and only directly depends on [Pydantic](https://docs.pydantic.dev/latest/), so you will need to bring your own API code or use the provided [litellm](https://github.com/BerriAI/litellm) integration.

For LLM support via litellm (recommended), install with the optional dependency:

```bash
pip install lexi-align[litellm]
```

Using uv:

```bash
uv add lexi-align --extra litellm
```

## Usage

### Basic Usage

The library expects pre-tokenized input--it does not perform any tokenization. You must provide tokens as lists of strings:

```python
from lexi_align.adapters.litellm_adapter import LiteLLMAdapter
from lexi_align.core import align_tokens

# Initialize the LLM adapter
llm_adapter = LiteLLMAdapter(model_params={
    "model": "gpt-4",
    "temperature": 0.0
})

# Provide pre-tokenized input with repeated tokens
source_tokens = ["the", "big", "cat", "saw", "the", "cat"]  # Note: "the" and "cat" appear twice
target_tokens = ["le", "gros", "chat", "a", "vu", "le", "chat"]

alignment = align_tokens(
    llm_adapter,
    source_tokens,
    target_tokens,
    source_language="English",
    target_language="French"
)

# Example output will show the uniquified tokens:
# the₁ -> le₁
# big -> gros
# cat₁ -> chat₁
# saw -> a
# saw -> vu
# the₂ -> le₂
# cat₂ -> chat₂
```

### Performance

Here are some preliminary results on the test EN-SL subset of XL-WA:

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

For reference, the 1-shot (1 example) `gpt-4o-2024-08-06` results are bettern than all systems presented in the [paper](https://ceur-ws.org/Vol-3596/paper32.pdf) (Table 2).

### Pharaoh Format Export

While the core alignment functions work with pre-tokenized input, the Pharaoh format utilities currently assume space-separated tokens when parsing/exporting. If your tokens contain spaces or require special tokenization, you'll need to handle this separately.

```python
from lexi_align.utils import export_pharaoh_format

# Note: Pharaoh format assumes space-separated tokens
pharaoh_format = export_pharaoh_format(
    source_tokens,  # Pre-tokenized list of strings
    target_tokens,  # Pre-tokenized list of strings
    alignment
)

print(pharaoh_format)
# Output (will differ depending on chosen model):
# The cat sat on the mat    Le chat était assis sur le tapis    0-0 1-1 2-2 2-3 3-4 4-5 5-6
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

# Full evaluation with custom parameters
python evaluations/xl-wa.py \
    --lang-pairs EN-FR EN-DE \
    --model gpt-4 \
    --temperature 0.0 \
    --seed 42 \
    --num-train-examples 3 \
    --output results.json
```

Available command-line arguments:

- `--lang-pairs`: Language pairs to evaluate (e.g., EN-SL EN-DE) or "all"
- `--model`: LLM model to use (default: gpt-4)
- `--temperature`: Temperature for LLM sampling (default: 0.0)
- `--seed`: Random seed for example selection (default: 42)
- `--model-seed`: Seed for LLM sampling (optional)
- `--num-train-examples`: Number of training examples for few-shot learning
- `--sample-size`: Number of test examples to evaluate per language pair
- `--output`: Path to save results JSON file
- `--verbose`: Enable verbose logging

## Planned improvements

- [ ] structured generation support (adapter additions) for local models via [Outlines](https://github.com/dottxt-ai/outlines) and [llama.cpp GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)

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
