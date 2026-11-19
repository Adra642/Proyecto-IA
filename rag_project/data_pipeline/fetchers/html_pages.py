"""B1, B5, B7, B8: regulator / central-bank / bureau pages (HTML, some PDF), ES/FR/DE/EN.

All four sources share one template: a list of pages from ``config.yaml``. The raw HTML
is returned as-is; ``clean.py`` strips navigation and converts it to Markdown.
"""

from __future__ import annotations

from typing import Any, ClassVar

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.util import pdf_text, slugify

DEFAULT_PDF_CHARS = 60_000
MIN_CHARS = 200


class HtmlPageFetcher(BaseFetcher):
    """Template for page-list sources; subclasses only set names and defaults."""

    default_source_type: ClassVar[str] = "regulator_guide"
    default_license: ClassVar[str] = (
        "Public institution website content; excerpted for research with citation "
        "(reuse terms not verified)"
    )

    def _items(self) -> list[FetchItem]:
        items = []
        for page in self.option("pages"):
            ext = "pdf" if page.get("format") == "pdf" else "html"
            items.append(FetchItem(f"{self.source_name}:{page['url']}", ext, params=dict(page)))
        return items

    def _download(self, item: FetchItem) -> bytes:
        resp = self.get(item.params["url"])
        if len(resp.content) < MIN_CHARS:
            raise FetchError(
                f"empty response ({len(resp.content)} bytes) from {item.params['url']}"
            )
        return resp.content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p: dict[str, Any] = item.params
        is_pdf = payload.data.startswith(b"%PDF")
        if is_pdf:
            limit = int(p.get("max_chars", DEFAULT_PDF_CHARS))
            text = f"# {p['title']}\n\n{pdf_text(payload.data)[:limit]}\n"
        else:
            text = payload.text()
        meta: dict[str, Any] = {
            "doc_id": f"{self.source_name}_{slugify(p['title'])}",
            "title": p["title"],
            "source": self.source_name,
            "source_type": p.get("source_type", self.default_source_type),
            "url": p["url"],
            "language": p["language"],
            "geography": p["geography"],
            "topic": p["topic"],
            "columns": list(p.get("columns", [])),
            "retrieved_at": payload.retrieved_at,
            "license": p.get("license", self.default_license),
            "format": "pdf" if is_pdf else "html",
        }
        return [Document(text=text, metadata=DocumentMetadata(**meta))]


@register_fetcher("b1")
class BdeFetcher(HtmlPageFetcher):
    """Banco de España, Portal del Cliente Bancario (Spanish)."""

    source_name = "bde"


@register_fetcher("b5")
class HfcsFetcher(HtmlPageFetcher):
    """ECB Household Finance and Consumption Survey pages and reports."""

    source_name = "ecb_hfcs"
    default_source_type = "macro_data"
    default_license = "ECB publications: reuse allowed with source acknowledgement"


@register_fetcher("b7")
class CreditRegistersFetcher(HtmlPageFetcher):
    """Credit reporting systems: CIRBE (ES), SCHUFA (DE), FICP (FR)."""

    source_name = "credit_registers"


@register_fetcher("b8")
class RegulatorsFetcher(HtmlPageFetcher):
    """Banque de France (FR) and BaFin / Bundesbank (DE) consumer pages."""

    source_name = "regulators"
