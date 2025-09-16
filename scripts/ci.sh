#!/usr/bin/env bash

uv run ruff format
uv run ruff check
uv run mypy --show-error-context --check-untyped-defs src tests evaluations
uv run pytest -m "not llm" -vvv --log-level=DEBUG
