#!/usr/bin/env bash

uv run ruff format
uv run ruff check --fix
uv run ruff check --select I --fix
uv run mypy --show-error-context --check-untyped-defs src tests evaluations
uv run --dev --extra $ACCELERATOR pytest -vvv --log-level=DEBUG
