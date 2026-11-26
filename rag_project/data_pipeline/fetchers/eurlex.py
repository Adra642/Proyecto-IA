"""A6: EU legal texts. EUR-Lex HTML first; Cellar XHTML/PDF when EUR-Lex serves a bot challenge."""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from core.exceptions import FetchError
from core.logging_setup import get_logger
from core.models import Document, DocumentMetadata
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.util import pdf_text

log = get_logger(__name__)

EURLEX = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
CELLAR = "https://publications.europa.eu/resource/celex/{celex}"
LICENSE = "EU law: reuse authorised with attribution (Commission Decision 2011/833/EU)"
MIN_BODY_BYTES = 5_000
MIN_TEXT_BYTES = 1_000
CHALLENGE_MARKERS = (b"awswaf", b"aws-waf", b"challenge", b"captcha")


def is_challenge(status: int, body: bytes) -> bool:
    """True when EUR-Lex answered with a bot-challenge instead of the legal text."""
    if status == 202 or len(body) < MIN_BODY_BYTES:
        return True
    head = body[:4000].lower()
    return any(marker in head for marker in CHALLENGE_MARKERS) and b"oj-" not in body[:200000]


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def eurlex_to_markdown(markup: str) -> tuple[str, str]:
    """Convert an Official Journal XHTML/HTML text into Markdown with article headings.

    Args:
        markup: EUR-Lex HTML or Cellar XHTML.

    Returns:
        ``(title, markdown)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # Cellar serves XHTML
        soup = BeautifulSoup(markup, "lxml")
    body = soup.body or soup
    title_parts: list[str] = []
    lines: list[str] = []
    pending: tuple[str, str] | None = None  # (markdown prefix, heading text)

    def flush(sub: str = "") -> None:
        nonlocal pending
        if pending is not None:
            prefix, text = pending
            lines.append(f"{prefix} {text}{' - ' + sub if sub else ''}")
            pending = None

    for el in body.find_all(["p", "table"]):
        if not isinstance(el, Tag) or el.find_parent("table") is not None:
            continue
        classes = set(el.get("class") or [])
        if el.name == "table":
            flush()
            for row in el.find_all("tr"):
                cells = [_text(c) for c in row.find_all("td", recursive=False)]
                text = " ".join(c for c in cells if c)
                if text:
                    lines.append(text)
            continue
        text = _text(el)
        if not text or "oj-note" in classes:
            continue
        if "oj-doc-ti" in classes:
            title_parts.append(text)
        elif "oj-ti-art" in classes:
            flush()
            pending = ("###", text)
        elif "oj-sti-art" in classes:
            flush(text)
        elif "oj-ti-section-1" in classes:
            flush()
            pending = ("##", text)
        elif "oj-ti-section-2" in classes:
            flush(text)
        else:
            flush()
            lines.append(text)
    flush()
    title = " ".join(title_parts[:2]) or "EU legal text"
    return title, f"# {title}\n\n" + "\n\n".join(lines) + "\n"


def select_articles(text: str, articles: list[int], keep_matching: str = "") -> str:
    """Reduce a long act to the articles that matter, plus paragraphs matching a pattern.

    Args:
        text: Markdown from :func:`eurlex_to_markdown` (articles are ``### Article N - ...``).
        articles: Article numbers to keep whole.
        keep_matching: Optional regex; paragraphs outside the kept articles that match it
            (a recital or an annex item) are appended under an excerpt heading.

    Returns:
        The title block, the kept articles in order, and the matching excerpts.
    """
    blocks = re.split(r"(?m)^(?=### Article \d+ )", text)
    head, body = blocks[0], blocks[1:]
    head = head.split("\n\n", 2)
    kept = [head[0] + "\n"]  # the "# title" line only: recitals are not kept
    wanted = {str(n) for n in articles}
    for block in body:
        match = re.match(r"### Article (\d+) ", block)
        if match and match.group(1) in wanted:
            # drop the next chapter/section headings that trail the article
            block = re.sub(r"(?s)\n\n## .*$", "", block)
            kept.append(block.rstrip() + "\n")
    if keep_matching:
        pattern = re.compile(keep_matching, re.IGNORECASE)
        extras = [
            para
            for para in re.split(r"\n\n", text)
            if pattern.search(para) and not para.startswith("#") and para not in "".join(kept)
        ]
        if extras:
            kept.append("### Excerpts (recitals and annexes)\n\n" + "\n\n".join(extras) + "\n")
    return "\n".join(kept)


@register_fetcher("a6")
class EurLexFetcher(BaseFetcher):
    """One document per CELEX id, with a chain of fallbacks."""

    source_name = "eurlex"

    def _items(self) -> list[FetchItem]:
        return [
            FetchItem(f"eurlex:{d['celex']}", "html", params=dict(d))
            for d in self.option("documents")
        ]

    def _download(self, item: FetchItem) -> bytes:
        celex = item.params["celex"]
        attempts = (
            ("EUR-Lex HTML", EURLEX, {"params": {"uri": f"CELEX:{celex}"}}),
            (
                "Cellar XHTML",
                CELLAR.format(celex=celex),
                {"headers": {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}},
            ),
            (
                "Cellar PDF",
                CELLAR.format(celex=celex),
                {"headers": {"Accept": "application/pdf", "Accept-Language": "eng"}},
            ),
        )
        errors: list[str] = []
        for label, url, kwargs in attempts:
            try:
                resp = self.get(url, **kwargs)
            except FetchError as exc:
                errors.append(f"{label}: {exc}")
                continue
            if label == "EUR-Lex HTML" and is_challenge(resp.status_code, resp.content):
                log.warning("EUR-Lex bot challenge for %s; falling back to Cellar", celex)
                errors.append(f"{label}: bot challenge")
                continue
            if resp.status_code != 200 or len(resp.content) < MIN_TEXT_BYTES:
                errors.append(f"{label}: HTTP {resp.status_code}, {len(resp.content)} bytes")
                continue
            if label == "Cellar PDF" and not resp.content.startswith(b"%PDF"):
                errors.append(f"{label}: not a PDF")
                continue
            return resp.content
        raise FetchError(f"all sources failed for {celex}: {'; '.join(errors)}")

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p = item.params
        if payload.data.startswith(b"%PDF"):
            raw = pdf_text(payload.data)
            title = p.get("title", p["celex"])
            text = f"# {title}\n\n{raw}\n"
            fmt = "pdf"
        else:
            title, text = eurlex_to_markdown(payload.text())
            fmt = "html"
        if p.get("keep_articles"):
            text = select_articles(text, list(p["keep_articles"]), str(p.get("keep_matching", "")))
        meta = DocumentMetadata(
            doc_id=f"eurlex_{p['celex'].lower()}",
            title=str(p.get("title") or title),
            source="eurlex",
            source_type="regulation",
            url=f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{p['celex']}",
            language="en",
            geography="EU",
            topic=p["topic"],
            columns=list(p.get("columns", [])),
            retrieved_at=payload.retrieved_at,
            license=LICENSE,
            format=fmt,
        )
        if len(text) < MIN_TEXT_BYTES:
            raise FetchError(f"EUR-Lex text for {p['celex']} is suspiciously short")
        return [Document(text=text, metadata=meta)]
