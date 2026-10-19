.PHONY: setup check test

setup:
	uv sync --locked --all-extras

check:
	uv lock --check
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy
	uv run --locked lint-imports

test:
	uv run --locked pytest -m "not live"
