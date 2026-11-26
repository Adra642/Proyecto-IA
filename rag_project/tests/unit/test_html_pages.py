from pathlib import Path
from typing import Any

import httpx
import pymupdf

from core.settings import Settings
from data_pipeline.clean import clean_documents
from data_pipeline.fetchers.base import FETCHER_REGISTRY, FetchContext
from data_pipeline.fetchers.html_pages import (
    BdeFetcher,
    CreditRegistersFetcher,
    HfcsFetcher,
    RegulatorsFetcher,
)
from data_pipeline.http import HttpClient
from data_pipeline.resilience import RateLimiter

PAGE = (
    "<html><body><nav>menu</nav><main><h1>Como realizar una reclamacion</h1>"
    + "<p>Puede presentar su reclamacion ante el Banco de Espana por via electronica.</p>" * 8
    + "</main><footer>pie</footer></body></html>"
)


def ctx(settings: Settings, tmp_path: Path, handler: Any) -> FetchContext:
    limiter = RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda s: None)
    http = HttpClient(
        settings.http, transport=httpx.MockTransport(handler), limiter=limiter, sleep=lambda s: None
    )
    return FetchContext(settings, http, tmp_path)


def test_registered_ids() -> None:
    assert FETCHER_REGISTRY["b1"] is BdeFetcher and FETCHER_REGISTRY["b5"] is HfcsFetcher
    assert FETCHER_REGISTRY["b7"] is CreditRegistersFetcher
    assert FETCHER_REGISTRY["b8"] is RegulatorsFetcher


def test_html_page_end_to_end_with_cleaning(settings: Settings, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if "bad" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, content=PAGE.encode())

    pages = [
        {
            "url": "https://x.test/ok",
            "title": "Reclamacion",
            "language": "es",
            "geography": "ES",
            "topic": "complaints",
            "columns": ["Complain"],
        },
        {
            "url": "https://x.test/bad",
            "title": "Missing",
            "language": "es",
            "geography": "ES",
            "topic": "complaints",
        },
    ]
    result = BdeFetcher(ctx(settings, tmp_path, handler), {"pages": pages}).fetch()
    assert len(result.documents) == 1 and len(result.errors) == 1
    meta = result.documents[0].metadata
    assert meta.doc_id == "bde_reclamacion" and meta.language == "es" and meta.format == "html"
    assert meta.source_type == "regulator_guide" and meta.columns == ["Complain"]
    (clean,) = clean_documents(result.documents)
    assert clean.text.startswith("# Como realizar una reclamacion")
    assert "menu" not in clean.text and "pie" not in clean.text


def test_pdf_pages_are_truncated(settings: Settings, tmp_path: Path) -> None:
    pdf = pymupdf.open()
    for n in range(3):
        page = pdf.new_page()
        page.insert_text(
            (30, 40), "\n".join(f"Wealth by age group line {n}-{i}." for i in range(40)), fontsize=8
        )
    data: bytes = pdf.tobytes()
    handler = lambda r: httpx.Response(404 if r.url.path == "/robots.txt" else 200, content=data)  # noqa: E731
    pages = [
        {
            "url": "https://x.test/r.pdf",
            "format": "pdf",
            "max_chars": 300,
            "title": "HFCS report",
            "language": "en",
            "geography": "EU",
            "topic": "wealth",
        }
    ]
    result = HfcsFetcher(ctx(settings, tmp_path, handler), {"pages": pages}).fetch()
    doc = result.documents[0]
    assert doc.metadata.format == "pdf" and doc.metadata.source_type == "macro_data"
    assert len(doc.text) < 400 and "Wealth by age group" in doc.text


def test_config_pages_have_required_metadata(settings: Settings) -> None:
    langs = set()
    for key in ("b1", "b5", "b7", "b8"):
        for page in settings.sources[key]["pages"]:
            assert {"url", "title", "language", "geography", "topic"} <= set(page)
            langs.add(page["language"])
    assert {"es", "fr", "de", "en"} <= langs


def test_eba_is_registered() -> None:
    from data_pipeline.fetchers.html_pages import EbaFetcher

    assert FETCHER_REGISTRY["b6"] is EbaFetcher
