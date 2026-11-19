import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from core.exceptions import ProviderError, RateLimitError
from core.logging_setup import get_logger, setup_logging
from core.models import Document, DocumentMetadata
from rag.llm import LangChainAdapter


class Chat:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result, self.error = result, error

    def invoke(self, prompt: str) -> Any:
        if self.error:
            raise self.error
        return self.result


def test_adapter_returns_text() -> None:
    adapter = LangChainAdapter("x", Chat(SimpleNamespace(text="hello", content="ignored")))
    assert adapter.invoke("q") == "hello"


def test_adapter_maps_rate_limit() -> None:
    err = Exception("429 Too Many Requests: quota exceeded")
    with pytest.raises(RateLimitError):
        LangChainAdapter("x", Chat(error=err)).invoke("q")


def test_adapter_maps_other_errors() -> None:
    with pytest.raises(ProviderError):
        LangChainAdapter("x", Chat(error=RuntimeError("boom"))).invoke("q")


def _meta(**over: Any) -> DocumentMetadata:
    base: dict[str, Any] = {
        "doc_id": "d1",
        "title": "T",
        "source": "wikipedia",
        "source_type": "encyclopedia",
        "url": "https://x",
        "language": "en",
        "geography": "global",
        "topic": "churn",
        "retrieved_at": date(2026, 11, 19),
        "license": "CC BY-SA",
    }
    return DocumentMetadata(**{**base, **over})


def test_document_metadata_valid() -> None:
    doc = Document(text="body", metadata=_meta())
    assert doc.doc_id == "d1"


@pytest.mark.parametrize("field", [{"language": "pt"}, {"geography": "PT"}, {"doc_id": ""}])
def test_document_metadata_rejects_invalid(field: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        _meta(**field)


def test_logging_setup_is_idempotent(tmp_path: Path) -> None:
    root = logging.getLogger("rag")
    for h in list(root.handlers):
        root.removeHandler(h)
    if hasattr(root, "_configured"):
        del root._configured
    setup_logging(tmp_path)
    setup_logging(tmp_path)
    get_logger("t").info("hello")
    assert len(root.handlers) == 2
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)
    del root._configured  # type: ignore[attr-defined]
    assert "hello" in (tmp_path / "pipeline.log").read_text(encoding="utf-8")
