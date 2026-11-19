from pathlib import Path
from typing import Any

import httpx
import pymupdf

from core.settings import Settings
from data_pipeline.fetchers.base import FetchContext
from data_pipeline.fetchers.eurlex import EurLexFetcher, eurlex_to_markdown, is_challenge
from data_pipeline.http import HttpClient
from data_pipeline.resilience import RateLimiter

FILLER = "".join(
    f'<p class="oj-normal">Filler paragraph {i} about covered deposits and guarantee schemes.</p>'
    for i in range(120)
)
XHTML = f"""<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>
<p class="oj-doc-ti">DIRECTIVE 2014/49/EU OF THE EUROPEAN PARLIAMENT</p>
<p class="oj-doc-ti">on deposit guarantee schemes</p>
<p class="oj-note">(1) OJ L 176, footnote that must be skipped</p>
<p class="oj-ti-section-1">CHAPTER I</p><p class="oj-ti-section-2">General provisions</p>
<p class="oj-ti-art">Article 1</p><p class="oj-sti-art">Subject matter and scope</p>
<p class="oj-normal">This Directive lays down rules.</p>
<table><tbody><tr><td><p class="oj-normal">1.</p></td><td><p class="oj-normal">The coverage level
 shall be EUR 100 000.</p></td></tr></tbody></table>
{FILLER}
</body></html>"""
CHALLENGE = b"<html><script src='https://x/awswaf.js'></script></html>"


def ctx(settings: Settings, tmp_path: Path, handler: Any) -> FetchContext:
    limiter = RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda s: None)
    http = HttpClient(
        settings.http, transport=httpx.MockTransport(handler), limiter=limiter, sleep=lambda s: None
    )
    return FetchContext(settings, http, tmp_path)


def opts() -> dict[str, Any]:
    return {
        "documents": [
            {
                "celex": "32014L0049",
                "title": "DGS Directive",
                "topic": "deposit_guarantee",
                "columns": ["Balance"],
            }
        ]
    }


def test_is_challenge() -> None:
    assert is_challenge(202, b"")
    assert is_challenge(200, CHALLENGE)
    assert not is_challenge(200, XHTML.encode())


def test_markdown_structure() -> None:
    title, md = eurlex_to_markdown(XHTML)
    assert title.startswith("DIRECTIVE 2014/49/EU") and "deposit guarantee schemes" in title
    assert "## CHAPTER I - General provisions" in md
    assert "### Article 1 - Subject matter and scope" in md
    assert "1. The coverage level shall be EUR 100 000." in md
    assert "footnote that must be skipped" not in md


def test_falls_back_to_cellar_xhtml_on_challenge(settings: Settings, tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls.append(request.url.host)
        if request.url.host == "eur-lex.europa.eu":
            return httpx.Response(202, content=b"")
        return httpx.Response(
            200, content=XHTML.encode(), headers={"content-type": "application/xhtml+xml"}
        )

    result = EurLexFetcher(ctx(settings, tmp_path, handler), opts()).fetch()
    assert not result.errors and calls == ["eur-lex.europa.eu", "publications.europa.eu"]
    doc = result.documents[0]
    assert doc.doc_id == "eurlex_32014l0049" and doc.metadata.geography == "EU"
    assert doc.metadata.source_type == "regulation" and doc.metadata.format == "html"
    assert "EUR 100 000" in doc.text


def test_falls_back_to_pdf(settings: Settings, tmp_path: Path) -> None:
    pdf = pymupdf.open()
    for n in range(3):
        page = pdf.new_page()
        lines = "\n".join(
            f"Coverage level EUR 100 000 article {n}-{i} of the directive." for i in range(40)
        )
        page.insert_text((30, 40), lines, fontsize=8)
    data: bytes = pdf.tobytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "eur-lex.europa.eu":
            return httpx.Response(202)
        accept = request.headers.get("accept", "")
        return (
            httpx.Response(200, content=data)
            if accept == "application/pdf"
            else httpx.Response(404)
        )

    result = EurLexFetcher(ctx(settings, tmp_path, handler), opts()).fetch()
    assert not result.errors and result.documents[0].metadata.format == "pdf"
    assert "Coverage level" in result.documents[0].text


def test_all_sources_fail_is_reported(settings: Settings, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404 if request.url.path == "/robots.txt" else 202)

    result = EurLexFetcher(ctx(settings, tmp_path, handler), opts()).fetch()
    assert result.documents == [] and "all sources failed" in result.errors[0]
