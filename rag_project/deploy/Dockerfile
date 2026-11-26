# Cloud Run image. Build context: the rag_project/ folder.
#   DOCKER_BUILDKIT=1 docker build -f deploy/Dockerfile -t rag-churn .   (podman build works the same)
#
# Layers are ordered by how often they change, so a rebuild (with --cache-from, see cloudbuild.yaml)
# only redoes what changed:
#   1. dependencies        change with pyproject.toml / uv.lock
#   2. e5 model + index    change with rag/, core/, config.yaml or data/processed
#   3. the app (UI)        changes most often and does not touch the index
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# 1. Runtime dependencies only: no dev tools and not the data-pipeline group (pymupdf, kagglehub,
#    bs4, lxml, markdownify). --no-cache keeps uv's wheel cache (about 1.7 GB) out of the layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project --no-cache

# 2. Bake the embedding model and the Chroma index into the image: Cloud Run's disk is ephemeral,
#    so building them at start-up would repeat on every cold start.
COPY config.yaml factory.py ./
COPY core ./core
COPY rag ./rag
COPY data/processed ./data/processed
RUN python -m rag.index
# The FlashRank cross-encoder (a few MB) is fetched on first use: do it now, not on a cold start.
RUN python -c "from core.settings import load_settings; from factory import Container; Container(load_settings()).retriever('advanced').retrieve('warm-up', 1)"

# 3. The app and the example questions (UI edits do not invalidate the index layer).
COPY app ./app
COPY eval/questions.json ./eval/questions.json

ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "app.app"]
