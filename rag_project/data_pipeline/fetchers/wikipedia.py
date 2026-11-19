"""A4: Wikipedia concept pages through the MediaWiki API (CC BY-SA)."""

from __future__ import annotations

import json
import re

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.util import slugify

API = "https://en.wikipedia.org/w/api.php"
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def wiki_to_markdown(title: str, extract: str) -> str:
    """Convert a plain-text extract (``== Heading ==``) to Markdown headings."""
    body = _HEADING.sub(lambda m: f"{'#' * len(m.group(1))} {m.group(2)}", extract.strip())
    return f"# {title}\n\n{body}\n"


@register_fetcher("a4")
class WikipediaFetcher(BaseFetcher):
    """One document per configured page; headings kept as Markdown."""

    source_name = "wikipedia"

    def _items(self) -> list[FetchItem]:
        return [
            FetchItem(f"wikipedia:{page['title']}", "json", params=dict(page))
            for page in self.option("pages")
        ]

    def _download(self, item: FetchItem) -> bytes:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": item.params["title"],
            "format": "json",
            "formatversion": 2,
        }
        return self.get(API, params=params, check_robots=False).content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        page = json.loads(payload.text())["query"]["pages"][0]
        if page.get("missing") or not page.get("extract"):
            raise FetchError(f"Wikipedia page not found or empty: {item.params['title']}")
        title = str(page["title"])
        meta = DocumentMetadata(
            doc_id=f"wiki_{slugify(title)}",
            title=title,
            source="wikipedia",
            source_type="encyclopedia",
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            language="en",
            geography=item.params.get("geography", "global"),
            topic=item.params["topic"],
            columns=list(item.params.get("columns", [])),
            retrieved_at=payload.retrieved_at,
            license="CC BY-SA 4.0",
            format="json",
        )
        return [Document(text=wiki_to_markdown(title, str(page["extract"])), metadata=meta)]
