"""A5: ECB Data Portal series (policy rates, inflation, bank rates) as short notes."""

from __future__ import annotations

import csv
import io

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.fetchers.notes import period_to_date, series_note
from data_pipeline.util import slugify

BASE = "https://data-api.ecb.europa.eu/service/data"
LICENSE = "ECB statistical data: reuse allowed with source acknowledgement"


@register_fetcher("a5")
class EcbFetcher(BaseFetcher):
    """One note per configured SDMX series; latest values carry their date."""

    source_name = "ecb_api"

    def _items(self) -> list[FetchItem]:
        return [
            FetchItem(f"ecb:{s['series']}", "csv", params=dict(s)) for s in self.option("series")
        ]

    def _download(self, item: FetchItem) -> bytes:
        url = f"{BASE}/{item.params['series']}"
        params = {"format": "csvdata", "lastNObservations": item.params.get("last", 6)}
        return self.get(url, params=params, check_robots=False).content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p = item.params
        rows = [r for r in csv.DictReader(io.StringIO(payload.text())) if r.get("OBS_VALUE")]
        if not rows:
            raise FetchError(f"ECB series {p['series']} returned no observations")
        rows.sort(key=lambda r: r["TIME_PERIOD"])
        series = [(r["TIME_PERIOD"], float(r["OBS_VALUE"])) for r in rows]
        source_line = (
            f"Source: ECB Data Portal, series {p['series']} "
            f"({rows[-1].get('TITLE', '').strip(chr(34))})."
        )
        text = series_note(
            title=p["name"],
            source_line=source_line,
            series=series,
            unit=p.get("unit", "%"),
            subject=p["subject"],
            as_of=payload.retrieved_at,
            change_dates=p.get("kind") == "change_dates",
        )
        meta = DocumentMetadata(
            doc_id=f"ecb_{slugify(p['name'])}",
            title=p["name"],
            source="ecb_api",
            source_type="macro_data",
            url=f"https://data.ecb.europa.eu/data/datasets/{p['series'].split('/')[0]}",
            language="en",
            geography=p.get("geography", "EU"),
            topic=p.get("topic", "deposits_rates"),
            columns=list(p.get("columns", [])),
            date=period_to_date(series[-1][0]),
            retrieved_at=payload.retrieved_at,
            license=LICENSE,
            format="csv",
        )
        return [Document(text=text, metadata=meta)]
