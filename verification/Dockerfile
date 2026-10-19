FROM python:3.12.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.17
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/opt/harness-venv
RUN --mount=type=cache,id=alfa-harness-uv,target=/root/.cache/uv uv sync --locked --no-install-project
RUN --mount=type=cache,id=alfa-harness-uv,target=/root/.cache/uv uv sync --locked --all-extras --no-install-project
ENV PATH="/opt/harness-venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
COPY ml_harness ./ml_harness
COPY tests ./tests
COPY scripts ./scripts
COPY verification ./verification
COPY solution.py ./solution.py
CMD ["python", "-m", "pytest", "-m", "not live", "-p", "no:cacheprovider", "-q"]
