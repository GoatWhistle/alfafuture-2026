.PHONY: setup check test web-setup web-demo web-api web-ui web-check

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

# Локальная панель AutoResearch (не входит в конкурсный архив).
web-setup:
	cd web && npm ci
	uv sync --project web_api

# Интерфейс с демонстрационными данными: http://127.0.0.1:5173
web-demo:
	cd web && npm run dev

# API на 127.0.0.1:8765 (отдельный процесс, запускает solution.py)
web-api:
	cd web_api && uv run python -m app.main

# Интерфейс, подключённый к API; запускайте вместе с `make web-api`
web-ui:
	cd web && VITE_DATA_SOURCE=api npm run dev

web-check:
	cd web && npm run build && npm test
	uv run --project web_api pytest web_api
