"""B3: World Bank indicators for France, Germany and Spain as short notes."""

from __future__ import annotations

import json
from typing import Any

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata, Geography
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.fetchers.notes import series_note
from data_pipeline.util import slugify

BASE = "https://api.worldbank.org/v2/country/FRA;DEU;ESP/indicator"
LICENSE = "World Bank Open Data: CC BY 4.0"
ISO3_TO_GEO: dict[str, Geography] = {"FRA": "FR", "DEU": "DE", "ESP": "ES"}


@register_fetcher("b3")
class WorldBankFetcher(BaseFetcher):
    """One note per indicator and country."""

    source_name = "worldbank"

    def _items(self) -> list[FetchItem]:
        return [
            FetchItem(f"worldbank:{i['id']}", "json", params=dict(i))
            for i in self.option("indicators")
        ]

    def _download(self, item: FetchItem) -> bytes:
        params = {"format": "json", "mrv": item.params.get("last", 5), "per_page": 100}
        return self.get(f"{BASE}/{item.params['id']}", params=params, check_robots=False).content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p = item.params
        data: Any = json.loads(payload.text())
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            raise FetchError(f"World Bank indicator {p['id']} returned no data: {str(data)[:120]}")
        by_country: dict[str, list[tuple[str, float]]] = {}
        names: dict[str, str] = {}
        indicator_label = p["name"]
        for rec in data[1]:
            if rec.get("value") is None:
                continue
            iso3 = rec["countryiso3code"]
            names[iso3] = rec["country"]["value"]
            indicator_label = rec["indicator"]["value"]
            by_country.setdefault(iso3, []).append((str(rec["date"]), float(rec["value"])))
        docs: list[Document] = []
        for iso3, series in sorted(by_country.items()):
            series.sort()
            title = f"{p['name']} in {names[iso3]} (World Bank)"
            text = series_note(
                title=title,
                source_line=f"Source: World Bank indicator {p['id']} ({indicator_label}).",
                series=series,
                unit=p.get("unit", "%"),
                subject=f"{p['name'].lower()} in {names[iso3]}",
            )
            meta = DocumentMetadata(
                doc_id=f"worldbank_{slugify(p['id'])}_{iso3.lower()}",
                title=title,
                source="worldbank",
                source_type="macro_data",
                url=f"https://data.worldbank.org/indicator/{p['id']}?locations={iso3[:2]}",
                language="en",
                geography=ISO3_TO_GEO.get(iso3, "EU"),
                topic=p.get("topic", "macro"),
                columns=list(p.get("columns", [])),
                retrieved_at=payload.retrieved_at,
                license=LICENSE,
                format="json",
            )
            docs.append(Document(text=text, metadata=meta))
        if not docs:
            raise FetchError(f"World Bank indicator {p['id']} had only empty values")
        return docs
