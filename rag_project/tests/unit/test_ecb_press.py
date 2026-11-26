from datetime import date
from pathlib import Path
from typing import Any

import httpx

from core.settings import Settings
from data_pipeline.clean import clean_documents
from data_pipeline.fetchers.base import FetchContext
from data_pipeline.fetchers.ecb_press import EcbPressFetcher, parse_feed
from data_pipeline.http import HttpClient
from data_pipeline.resilience import RateLimiter

FIX = Path(__file__).parent.parent / "fixtures"
PAGE = (
    "<html><body><nav>menu</nav><main><h1>Press release</h1>"
    + "<p>The Governing Council decided to keep interest rates unchanged today.</p>" * 10
    + "</main></body></html>"
)


def ctx(settings: Settings, tmp_path: Path, handler: Any) -> FetchContext:
    limiter = RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda s: None)
    http = HttpClient(
        settings.http, transport=httpx.MockTransport(handler), limiter=limiter, sleep=lambda s: None
    )
    return FetchContext(settings, http, tmp_path)


def test_parse_real_feed_fixture() -> None:
    items = parse_feed((FIX / "ecb_press.xml").read_text(encoding="utf-8"))
    assert len(items) >= 5
    first = items[0]
    assert first["date"].count("-") == 2 and "//" not in first["link"].replace("https://", "")
    assert first["title"] == first["title"].strip()


def test_fetcher_filters_press_releases_and_builds_docs(settings: Settings, tmp_path: Path) -> None:
    feed = (FIX / "ecb_press.xml").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path.endswith("press.xml"):
            return httpx.Response(200, content=feed)
        return httpx.Response(200, content=PAGE.encode())

    opts = {**settings.sources["b4"], "max_articles": 2}
    result = EcbPressFetcher(ctx(settings, tmp_path, handler), opts).fetch()
    assert not result.errors and 1 <= len(result.documents) <= 2
    meta = result.documents[0].metadata
    assert meta.source == "ecb_press" and meta.source_type == "press" and meta.geography == "EU"
    assert meta.doc_id.startswith("ecb_press_ecb_pr") and isinstance(meta.date, date)
    (clean,) = clean_documents(result.documents[:1])
    assert "menu" not in clean.text and "interest rates unchanged" in clean.text
