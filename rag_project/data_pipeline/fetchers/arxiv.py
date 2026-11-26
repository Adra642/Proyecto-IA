"""A3: arXiv papers on churn (API metadata + PDF excerpts, not full text)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from functools import partial
from itertools import zip_longest
from typing import Any

from bs4 import BeautifulSoup

from core.logging_setup import get_logger
from core.models import Document, DocumentMetadata
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.util import pdf_text

log = get_logger(__name__)

API = "https://export.arxiv.org/api/query"
# export.arxiv.org disallows /pdf in robots.txt; arxiv.org allows it (Crawl-delay 15 s)
PDF = "https://arxiv.org/pdf/{id}"
WANTED = ("introduction", "result", "discussion", "conclusion")
KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "methodology", "methods",
    "method", "experiments", "results", "results and discussion", "discussion",
    "conclusion", "conclusions", "references", "acknowledgements", "acknowledgments",
    "appendix", "literature review", "data", "dataset", "model", "future work",
}  # fmt: skip
_NUMBERED = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+([A-Z][A-Za-z0-9 ,&:/'-]{2,60})$")


@dataclass(frozen=True)
class ArxivEntry:
    """Metadata of one arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _child_text(tag: Any, name: str) -> str:
    child = tag.find(name)
    return str(child.get_text(strip=True)) if child is not None else ""


def parse_atom(xml: str) -> list[ArxivEntry]:
    """Parse an arXiv API Atom response."""
    soup = BeautifulSoup(xml, "xml")
    entries: list[ArxivEntry] = []
    for entry in soup.find_all("entry"):
        raw_id = _child_text(entry, "id")
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=_squash(_child_text(entry, "title")),
                authors=[_squash(_child_text(a, "name")) for a in entry.find_all("author")],
                published=_child_text(entry, "published")[:10],
                abstract=_squash(_child_text(entry, "summary")),
            )
        )
    return entries


def round_robin(lists: list[list[ArxivEntry]], cap: int) -> list[ArxivEntry]:
    """Interleave result lists, drop duplicate ids, keep at most ``cap`` papers."""
    chosen: dict[str, ArxivEntry] = {}
    for group in zip_longest(*lists):
        for entry in group:
            if entry is not None and entry.arxiv_id not in chosen and len(chosen) < cap:
                chosen[entry.arxiv_id] = entry
    return list(chosen.values())


def _heading_title(line: str) -> str | None:
    stripped = line.strip().rstrip(".:")
    if stripped.lower() in KNOWN_HEADINGS:
        return stripped
    match = _NUMBERED.match(line)
    if match and not line.rstrip().endswith("."):
        return match.group(1).strip()
    return None


def _unwrap(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newline -> space
    return re.sub(r"[ \t]+", " ", text).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut} [...]"


def select_sections(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Return ``(title, excerpt)`` for introduction, results/discussion and conclusion."""
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in text.splitlines():
        title = _heading_title(line)
        if title is not None:
            if title.lower() == "references":
                break
            current = (title, [])
            sections.append(current)
        elif current is not None:
            current[1].append(line)
    kept = []
    for title, lines in sections:
        if any(w in title.lower() for w in WANTED):
            body = _truncate(_unwrap("\n".join(lines)), max_chars)
            if body:
                kept.append((title, body))
    return kept


@register_fetcher("a3")
class ArxivFetcher(BaseFetcher):
    """Searches arXiv, downloads PDFs politely, keeps abstract + key sections."""

    source_name = "arxiv"

    def _search(self, query: str, per_query: int) -> bytes:
        params: dict[str, str | int]
        if query.startswith("id:"):  # explicit papers chosen by the team (``include_ids``)
            params = {"id_list": query[3:], "max_results": per_query}
        else:
            params = {
                "search_query": query,
                "max_results": per_query,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        interval = float(self.options.get("api_interval_seconds", 3.0))
        return self.get(API, params=params, check_robots=False, min_interval=interval).content

    def _items(self) -> list[FetchItem]:
        per_query = int(self.option("per_query"))
        groups: list[list[ArxivEntry]] = []
        extra = (
            [f"id:{','.join(self.options['include_ids'])}"]
            if self.options.get("include_ids")
            else []
        )
        for query in [*self.option("queries"), *extra]:
            payload = self.cache.get_or_fetch(
                f"arxiv-search:{query}:{per_query}", "xml", partial(self._search, query, per_query)
            )
            groups.append(parse_atom(payload.text()))
        excluded = {str(i) for i in self.options.get("exclude_ids", [])}
        groups = [[e for e in g if e.arxiv_id not in excluded] for g in groups]
        papers = round_robin(groups, int(self.option("max_papers")))
        return [FetchItem(f"arxiv:{p.arxiv_id}", "pdf", params=asdict(p)) for p in papers]

    def _download(self, item: FetchItem) -> bytes:
        interval = float(self.options.get("pdf_interval_seconds", 15.0))
        url = PDF.format(id=item.params["arxiv_id"])
        return self.get(url, min_interval=interval).content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p: dict[str, Any] = item.params
        max_chars = int(self.options.get("section_max_chars", 3000))
        try:
            sections = select_sections(pdf_text(payload.data), max_chars)
        except Exception as exc:
            log.warning("PDF text extraction failed for %s: %s", p["arxiv_id"], exc)
            sections = []
        authors = ", ".join(p["authors"])
        parts = [
            f"# {p['title']}",
            f"Authors: {authors}. Published: {p['published']}. arXiv:{p['arxiv_id']}.",
            f"## Abstract\n\n{p['abstract']}",
        ]
        parts += [f"## {title} (excerpt)\n\n{body}" for title, body in sections]
        parts.append(f"Full paper: https://arxiv.org/abs/{p['arxiv_id']}")
        meta = DocumentMetadata(
            doc_id=f"arxiv_{p['arxiv_id'].replace('/', '_').replace('.', '_')}",
            title=p["title"],
            source="arxiv",
            source_type="paper",
            url=f"https://arxiv.org/abs/{p['arxiv_id']}",
            language="en",
            geography="global",
            topic="churn",
            columns=["Exited"],
            date=date.fromisoformat(p["published"]),
            retrieved_at=payload.retrieved_at,
            license="arXiv: authors' copyright; metadata, abstract and short excerpts with link",
            format="pdf",
        )
        return [Document(text="\n\n".join(parts) + "\n", metadata=meta)]
