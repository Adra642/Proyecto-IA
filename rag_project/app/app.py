"""Entry point: wires the container, makes sure the index exists and launches Gradio.

Run locally with ``uv run python -m app.app``. On Cloud Run the port comes from ``PORT``.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.handlers import AppService
from app.ui import build_demo, load_example_questions
from core.corpus import load_documents
from core.logging_setup import get_logger, setup_logging
from core.settings import Settings, load_settings
from factory import Container
from rag.index import Indexer

log = get_logger(__name__)

DEFAULT_PORT = 8080
EXAMPLES_FILE = Path("eval/questions.json")  # the evaluation questions double as examples
HOST = "0.0.0.0"  # noqa: S104  (a container must listen on all interfaces)


def ensure_index(container: Container, settings: Settings) -> None:
    """Sync the index with the committed documents (a no-op when the image already has it)."""
    repo = container.vector_store()
    docs = load_documents(settings.resolve(settings.paths.processed_dir))
    report = Indexer(repo, container.chunker(), settings.chunking, settings.embeddings.model).sync(
        docs
    )
    log.info("Index ready: %d chunks (%d docs added)", report.chunks_total, report.added)


def main() -> None:
    """Build everything from settings and serve the app."""
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    container = Container(settings)
    ensure_index(container, settings)
    examples = load_example_questions(settings.resolve(EXAMPLES_FILE))
    demo = build_demo(AppService(container.chain()), examples)
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name=HOST, server_port=int(os.environ.get("PORT", DEFAULT_PORT)))


if __name__ == "__main__":
    main()
