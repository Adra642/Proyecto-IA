import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest

from core.settings import Settings
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.arxiv import (
    ArxivEntry,
    ArxivFetcher,
    parse_atom,
    pdf_text,
    round_robin,
    select_sections,
)
from data_pipeline.fetchers.base import FetchContext, FetchItem
from data_pipeline.fetchers.wikipedia import WikipediaFetcher, wiki_to_markdown
from data_pipeline.http import HttpClient
from data_pipeline.resilience import RateLimiter

FIX = Path(__file__).parent.parent / "fixtures"

PAPER_TEXT = """Title of the paper
Abstract
Short abstract text.
1 Introduction
Churn is costly for banks. Retention is
cheaper than acquisition and hyphen-
ated words are joined.
2 Related Work
Lots of unrelated stuff.
3 Results and Discussion
Gradient boosting reached an AUC of 0.86 on the data.
4 Conclusion
Age and number of products drive churn.
References
[1] Someone. Not wanted.
"""


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), text, fontsize=8)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def ctx(settings: Settings, tmp_path: Path, handler: Any) -> FetchContext:
    limiter = RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda s: None)
    http = HttpClient(
        settings.http, transport=httpx.MockTransport(handler), limiter=limiter, sleep=lambda s: None
    )
    return FetchContext(settings, http, tmp_path)


def test_parse_atom() -> None:
    entries = parse_atom((FIX / "arxiv_sample.xml").read_text(encoding="utf-8"))
    assert [e.arxiv_id for e in entries] == ["2101.00001", "2102.00002"]
    assert entries[0].title == "Explainable Churn Prediction for Banks"
    assert entries[0].authors == ["Ana Perez", "Bo Chen"]
    assert entries[0].published == "2021-01-02"
    assert entries[0].abstract.startswith("We study churn")


def test_round_robin_dedupes_and_caps() -> None:
    def e(i: str) -> ArxivEntry:
        return ArxivEntry(i, i, [], "2021-01-01", "")

    out = round_robin([[e("a"), e("b"), e("c")], [e("a"), e("d")]], cap=3)
    assert [x.arxiv_id for x in out] == ["a", "b", "d"]


def test_select_sections_keeps_wanted_only() -> None:
    sections = dict(select_sections(PAPER_TEXT, 500))
    assert set(sections) == {"Introduction", "Results and Discussion", "Conclusion"}
    assert "hyphenated words are joined" in sections["Introduction"]
    assert "Not wanted" not in " ".join(sections.values())


def test_select_sections_truncates() -> None:
    text = "1 Introduction\n" + "word " * 200
    ((_, body),) = select_sections(text, 50)
    assert body.endswith("[...]") and len(body) < 70


def test_pdf_text_roundtrip() -> None:
    assert "Churn is costly" in pdf_text(make_pdf(PAPER_TEXT))


def test_arxiv_fetcher_end_to_end(settings: Settings, tmp_path: Path) -> None:
    atom = (FIX / "arxiv_sample.xml").read_bytes()
    pdf = make_pdf(PAPER_TEXT)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/query":
            return httpx.Response(200, content=atom)
        if request.url.path.startswith("/pdf/2101"):
            return httpx.Response(200, content=pdf)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(404)  # second paper's PDF is missing

    options = {**settings.sources["a3"], "queries": ["q1"], "max_papers": 5}
    result = ArxivFetcher(ctx(settings, tmp_path, handler), options).fetch()
    assert len(result.documents) == 1 and len(result.errors) == 1
    doc = result.documents[0]
    assert doc.doc_id == "arxiv_2101_00001"
    assert doc.metadata.source_type == "paper" and doc.metadata.date == date(2021, 1, 2)
    assert "## Abstract" in doc.text and "## Conclusion (excerpt)" in doc.text
    assert "https://arxiv.org/abs/2101.00001" in doc.text


def test_arxiv_pdf_failure_keeps_abstract(settings: Settings, tmp_path: Path) -> None:
    item = FetchItem(
        "arxiv:x",
        "pdf",
        params={
            "arxiv_id": "2101.00001",
            "title": "T",
            "authors": ["A"],
            "published": "2021-01-02",
            "abstract": "Abstract text.",
        },
    )
    fetcher = ArxivFetcher(ctx(settings, tmp_path, lambda r: httpx.Response(404)), {})
    docs = fetcher._parse(item, CachedPayload(tmp_path / "x", b"not a pdf", date(2026, 1, 1)))
    assert "Abstract text." in docs[0].text and "(excerpt)" not in docs[0].text


def test_wiki_to_markdown() -> None:
    md = wiki_to_markdown("T", "Intro.\n\n== History ==\nText\n=== Sub ===\nMore")
    assert md.startswith("# T\n\nIntro.") and "\n## History\n" in md and "\n### Sub\n" in md


def test_wikipedia_fetcher(settings: Settings, tmp_path: Path) -> None:
    fixture = (FIX / "wikipedia_churn_rate.json").read_bytes()
    missing = json.dumps({"query": {"pages": [{"title": "Nope", "missing": True}]}}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        title = request.url.params.get("titles")
        return httpx.Response(200, content=fixture if title == "Churn rate" else missing)

    pages = [
        {"title": "Churn rate", "topic": "churn", "columns": ["Exited"], "geography": "EU"},
        {"title": "Nope", "topic": "churn"},
    ]
    result = WikipediaFetcher(ctx(settings, tmp_path, handler), {"pages": pages}).fetch()
    assert len(result.documents) == 1 and len(result.errors) == 1
    meta = result.documents[0].metadata
    assert meta.doc_id == "wiki_churn_rate" and meta.license == "CC BY-SA 4.0"
    assert meta.url == "https://en.wikipedia.org/wiki/Churn_rate" and meta.geography == "EU"
    assert result.documents[0].text.startswith("# Churn rate")


@pytest.mark.parametrize("key", ["a3", "a4"])
def test_config_has_source_blocks(settings: Settings, key: str) -> None:
    assert settings.sources[key]["enabled"] is True
