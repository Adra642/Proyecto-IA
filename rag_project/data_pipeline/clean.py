"""Cleaning: PII removal for the CSV and the ordered text-cleaning pipeline.

The text pipeline is a list of small steps applied in order:
``strip_html -> to_markdown -> normalize -> dedupe -> min_length``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings
from collections.abc import Callable

import pandas as pd
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from markdownify import markdownify

from core.logging_setup import get_logger
from core.models import Document

log = get_logger(__name__)

CleaningStep = Callable[[list[Document]], list[Document]]
MIN_CHARS = 200

_DROP_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe",
    "svg", "button", "select", "dialog", "template", "figure",
)  # fmt: skip
_CONTAINERS = {"html", "body", "main", "article"}
_DROP_ROLES = {"navigation", "banner", "contentinfo", "search", "dialog", "complementary"}
_DROP_CLASS = re.compile(
    r"cookie|consent|breadcrumb|social|newsletter|sidebar|skip-?link|toolbar|pagination|"
    r"\bcmp\b|gdpr|popup|modal|navbar|navigation|mega-?menu|share-|print-",
    re.IGNORECASE,
)
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))

PII_COLUMNS: tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` without the personal-identifier columns.

    Called before any document is created, so identifiers never reach the corpus.

    Args:
        df: Raw customer table.

    Returns:
        The table without ``RowNumber``, ``CustomerId`` and ``Surname``.
    """
    # errors="ignore": a table that already lacks a column is fine. .copy() leaves the input intact.
    return df.drop(columns=list(PII_COLUMNS), errors="ignore").copy()


# ----------------------------------------------------------------------------- text steps
def looks_like_html(text: str) -> bool:
    """True when ``text`` is markup rather than Markdown/plain text."""
    head = text.lstrip()[:2000].lower()
    return head.startswith("<") and any(t in head for t in ("<html", "<body", "<div", "<p", "<h1"))


def _is_boilerplate(tag: Tag) -> bool:
    if tag.attrs is None or tag.name in _CONTAINERS or tag.find(["main", "article"]) is not None:
        return False  # e.g. <body class="layout-no-sidebars"> must not be dropped
    if tag.get("role") in _DROP_ROLES:
        return True
    marker = " ".join([*(tag.get("class") or []), str(tag.get("id") or "")])
    return bool(marker and _DROP_CLASS.search(marker))


def _main_content(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role=main]"):
        found = soup.select_one(selector)
        if found is not None and len(found.get_text(strip=True)) > MIN_CHARS:
            return found
    return soup.body or soup


def html_to_markdown(markup: str, title: str) -> str:
    """Strip navigation/boilerplate from HTML and convert it to Markdown (ATX headings)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(markup, "lxml")
    for tag in soup.find_all(list(_DROP_TAGS)):
        tag.decompose()
    for tag in [t for t in soup.find_all(True) if _is_boilerplate(t)]:
        tag.decompose()
    body = markdownify(str(_main_content(soup)), heading_style="ATX", strip=["a", "img"])
    body = body.strip()
    return body if body.startswith("# ") else f"# {title}\n\n{body}"


def strip_html_and_convert(doc: Document) -> Document:
    """Step 1+2: HTML documents become clean Markdown; other documents pass through."""
    if not looks_like_html(doc.text):
        return doc
    return doc.model_copy(update={"text": html_to_markdown(doc.text, doc.metadata.title)})


# PDF tables of contents: runs of dot leaders, sometimes extracted as ".\ufffd.\ufffd" (replacement
# characters). They carry no information and waste embedding tokens.
_LEADERS = re.compile(r"(?:[.\u2026\u00b7\ufffd][ ]?){5,}")


def normalize_text(text: str) -> str:
    """Normalize unicode and whitespace (idempotent), keeping Markdown structure."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _LEADERS.sub(" ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def normalize_step(docs: list[Document]) -> list[Document]:
    """Step: unicode and whitespace normalization."""
    return [d.model_copy(update={"text": normalize_text(d.text)}) for d in docs]


def html_step(docs: list[Document]) -> list[Document]:
    """Step: strip HTML boilerplate and convert to Markdown."""
    return [strip_html_and_convert(d) for d in docs]


def text_hash(text: str) -> str:
    """Hash of the case- and whitespace-insensitive text (for duplicate detection)."""
    squashed = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(squashed.encode("utf-8")).hexdigest()


def dedupe_step(docs: list[Document]) -> list[Document]:
    """Step: drop documents with identical text (first by ``doc_id`` wins)."""
    seen: dict[str, str] = {}
    kept: list[Document] = []
    for doc in sorted(docs, key=lambda d: d.doc_id):
        digest = text_hash(doc.text)
        if digest in seen:
            log.info("Dropping duplicate %s (same text as %s)", doc.doc_id, seen[digest])
            continue
        seen[digest] = doc.doc_id
        kept.append(doc)
    return kept


def min_length_step(docs: list[Document], min_chars: int = MIN_CHARS) -> list[Document]:
    """Step: drop empty or tiny documents."""
    kept = [d for d in docs if len(d.text.strip()) >= min_chars]
    for dropped in sorted({d.doc_id for d in docs} - {d.doc_id for d in kept}):
        log.warning("Dropping too-short document %s", dropped)
    return kept


DEFAULT_STEPS: tuple[CleaningStep, ...] = (
    html_step,
    normalize_step,
    dedupe_step,
    min_length_step,
)


def clean_documents(
    docs: list[Document], steps: tuple[CleaningStep, ...] = DEFAULT_STEPS
) -> list[Document]:
    """Run the cleaning steps in order and return documents sorted by ``doc_id``.

    Args:
        docs: Documents from the fetchers.
        steps: Ordered cleaning steps (each maps a list of documents to a list).

    Returns:
        Cleaned, de-duplicated documents in a deterministic order.
    """
    # Each step is a small pure function (strip_html, normalize, dedupe, ...), applied in order.
    for step in steps:
        docs = step(docs)
    # Sorting makes the output deterministic, so re-running gives identical files.
    return sorted(docs, key=lambda d: d.doc_id)
