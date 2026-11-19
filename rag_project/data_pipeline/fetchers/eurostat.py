"""B2: Eurostat dissemination API (JSON-stat) as short per-country notes."""

from __future__ import annotations

import json
from itertools import product
from typing import Any

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata, Geography
from data_pipeline.cache import CachedPayload
from data_pipeline.fetchers.base import BaseFetcher, FetchItem, register_fetcher
from data_pipeline.fetchers.notes import period_to_date, series_note
from data_pipeline.util import slugify

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
LICENSE = "Eurostat: CC BY 4.0 (reuse with source acknowledgement)"
GEO_CODES: dict[str, Geography] = {"FR": "FR", "DE": "DE", "ES": "ES"}


def parse_jsonstat(data: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """Flatten a JSON-stat cube into ``{geo: [(time, value), ...]}`` (oldest first).

    All dimensions other than ``geo`` and ``time`` must have a single category
    (the fetcher fixes them with query filters).
    """
    ids: list[str] = data["id"]
    sizes: list[int] = data["size"]
    if "geo" not in ids or "time" not in ids:
        raise FetchError("JSON-stat cube without geo/time dimensions")
    categories = {
        k: sorted(data["dimension"][k]["category"]["index"].items(), key=lambda kv: kv[1])
        for k in ids
    }
    values = data.get("value", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    out: dict[str, list[tuple[str, float]]] = {}
    positions = product(*[range(n) for n in sizes])
    for flat, pos in enumerate(positions):
        value = values.get(str(flat))
        if value is None:
            continue
        labels = {k: categories[k][i][0] for k, i in zip(ids, pos, strict=True)}
        out.setdefault(labels["geo"], []).append((labels["time"], float(value)))
    return {geo: sorted(series) for geo, series in out.items()}


@register_fetcher("b2")
class EurostatFetcher(BaseFetcher):
    """One note per dataset and country (FR, DE, ES)."""

    source_name = "eurostat"

    def _items(self) -> list[FetchItem]:
        return [
            FetchItem(f"eurostat:{d['dataset']}", "json", params=dict(d))
            for d in self.option("datasets")
        ]

    def _download(self, item: FetchItem) -> bytes:
        p = item.params
        params: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN")]
        params += [("geo", g) for g in self.option("countries")]
        params += [(k, str(v)) for k, v in p.get("filters", {}).items()]
        params.append(("lastTimePeriod", str(p.get("last", 4))))
        url = f"{BASE}/{p['dataset']}"
        return self.get(url, params=params, check_robots=False).content

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        p = item.params
        data = json.loads(payload.text())
        if "error" in data or "id" not in data:
            raise FetchError(f"Eurostat {p['dataset']} returned an error: {str(data)[:120]}")
        country_names: dict[str, str] = data["dimension"]["geo"]["category"].get("label", {})
        docs: list[Document] = []
        for geo, series in parse_jsonstat(data).items():
            country = country_names.get(geo, geo)
            title = f"{p['name']} in {country} (Eurostat)"
            text = series_note(
                title=title,
                source_line=f"Source: Eurostat dataset {p['dataset']} ({data.get('label', '')}).",
                series=series,
                unit=p.get("unit", "%"),
                subject=f"{p['name'].lower()} in {country}",
            )
            meta = DocumentMetadata(
                doc_id=f"eurostat_{p['dataset']}_{slugify(geo)}",
                title=title,
                source="eurostat",
                source_type="macro_data",
                url=f"https://ec.europa.eu/eurostat/databrowser/view/{p['dataset']}/default/table",
                language="en",
                geography=GEO_CODES.get(geo, "EU"),
                topic=p.get("topic", "macro"),
                columns=list(p.get("columns", [])),
                date=period_to_date(series[-1][0]),
                retrieved_at=payload.retrieved_at,
                license=LICENSE,
                format="json",
            )
            docs.append(Document(text=text, metadata=meta))
        if not docs:
            raise FetchError(f"Eurostat {p['dataset']} returned no observations")
        return docs
