"""Live smoke test: every source returns at least one valid document (1 item each)."""

from pathlib import Path
from typing import Any

import pytest

from core.settings import load_settings
from data_pipeline.fetchers import load_all
from data_pipeline.fetchers.base import FETCHER_REGISTRY, FetchContext
from data_pipeline.http import HttpClient

pytestmark = pytest.mark.integration

LIST_KEYS = ("pages", "series", "datasets", "documents", "indicators", "queries")


def trimmed(options: dict[str, Any]) -> dict[str, Any]:
    out = dict(options)
    for key in LIST_KEYS:
        if key in out:
            out[key] = out[key][:1]
    if "max_papers" in out:
        out["max_papers"] = 1
        out["per_query"] = 1
    return out


@pytest.mark.parametrize(
    "source_id", ["a1", "a3", "a4", "a5", "a6", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
)
def test_source_returns_documents(source_id: str, tmp_path: Path) -> None:
    load_all()
    settings = load_settings()
    http = HttpClient(settings.http)
    try:
        ctx = FetchContext(settings, http, tmp_path)
        fetcher = FETCHER_REGISTRY[source_id](ctx, trimmed(settings.sources[source_id]))
        result = fetcher.fetch()
    finally:
        http.close()
    assert result.documents, result.errors
    assert all(len(d.text) > 100 for d in result.documents)
