"""B4: ECB press releases from the RSS feed (time-stamped, so recent news is RAG-only)."""

from __future__ import annotations

import re
from datetime import UTC, date
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.util import slugify

LICENSE = "ECB publications: reuse allowed with source acknowledgement"
_ECB_ID = re.compile(r"/(ecb\.[a-z]+\d+)")


def parse_feed(xml: str) -> list[dict[str, str]]:
    """Parse the RSS feed into ``{title, link, date}`` dicts (``date`` is ISO, UTC)."""
    soup = BeautifulSoup(xml, "xml")
    items = []
    for item in soup.find_all("item"):
        link = item.find("link")
        title = item.find("title")
        stamp = item.find("pubDate")
        if link is None or title is None or stamp is None:
            continue
        published = parsedate_to_datetime(stamp.get_text(strip=True)).astimezone(UTC)
        items.append(
            {
                "title": " ".join(title.get_text().split()),
                "link": link.get_text(strip=True).replace("eu//", "eu/"),
                "date": published.date().isoformat(),
            }
        )
    return items


@register_fetcher("b4")
class EcbPressFetcher(BaseFetcher):
    """Latest press releases (HTML pages only; speeches and PDFs are skipped)."""

    source_name = "ecb_press"

    def _items(self) -> list[FetchItem]:
        feed_url = str(self.option("feed_url"))
        payload = self.cache.get_or_fetch(
            f"rss:{feed_url}", "xml", lambda: self.get(feed_url, check_robots=False).content
        )
        needles = [str(n) for n in self.option("link_contains")]
        pattern = re.compile(str(self.options.get("title_regex", "")), re.IGNORECASE)
        entries = [
            e
            for e in parse_feed(payload.text())
            if e["link"].endswith(".html")
            and any(n in e["link"] for n in needles)
            and pattern.search(e["title"])
        ]
        limit = int(self.options.get("max_articles", 10))
        return [FetchItem(e["link"], "html", params=e) for e in entries[:limit]]

    def _download(self, item: FetchItem) -> bytes:
        return self.get(item.params["link"]).content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p = item.params
        match = _ECB_ID.search(p["link"])
        if match is None:
            raise FetchError(f"cannot derive an id from {p['link']}")
        monetary = re.search(r"monetary|interest rate|inflation", p["title"], re.IGNORECASE)
        meta = DocumentMetadata(
            doc_id=f"ecb_press_{slugify(match.group(1))}",
            title=p["title"],
            source="ecb_press",
            source_type="press",
            url=p["link"],
            language="en",
            geography="EU",
            topic="deposits_rates" if monetary else "macro",
            columns=["Balance"],
            date=date.fromisoformat(p["date"]),
            retrieved_at=payload.retrieved_at,
            license=LICENSE,
            format="html",
        )
        return [Document(text=payload.text(), metadata=meta)]
