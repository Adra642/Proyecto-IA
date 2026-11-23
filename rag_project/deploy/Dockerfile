# Cloud Run image. Build context: the rag_project/ folder.
#   docker build -f deploy/Dockerfile -t rag-churn .   (podman build works the same)
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first (cached layer); torch comes from the CPU index set in pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config.yaml factory.py ./
COPY core ./core
COPY rag ./rag
COPY app ./app
COPY eval/questions.json ./eval/questions.json
COPY data/processed ./data/processed

# Bake the embedding model and the Chroma index into the image: Cloud Run disk is ephemeral,
# so building them at start-up would repeat on every cold start.
RUN python -m rag.index

ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "app.app"]
