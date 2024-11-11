#!/usr/bin/env bash

uv run ruff format
uv run ruff check
uv run mypy --check-untyped-defs src tests evaluations
uv run pytest
