import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.exceptions import FetchError
from core.settings import Settings
from data_pipeline.fetchers.base import FetchContext
from data_pipeline.fetchers.ecb import EcbFetcher
from data_pipeline.fetchers.eurostat import EurostatFetcher, parse_jsonstat
from data_pipeline.fetchers.notes import fmt_value, period_to_date, series_note
from data_pipeline.fetchers.worldbank import WorldBankFetcher
from data_pipeline.http import HttpClient
from data_pipeline.resilience import RateLimiter

FIX = Path(__file__).parent.parent / "fixtures"


def ctx(settings: Settings, tmp_path: Path, handler: Any) -> FetchContext:
    limiter = RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda s: None)
    http = HttpClient(
        settings.http, transport=httpx.MockTransport(handler), limiter=limiter, sleep=lambda s: None
    )
    return FetchContext(settings, http, tmp_path)


def test_period_to_date() -> None:
    assert period_to_date("2025") == date(2025, 1, 1)
    assert period_to_date("2025-12") == date(2025, 12, 1)
    assert period_to_date("2026-09-16") == date(2026, 9, 16)
    with pytest.raises(FetchError):
        period_to_date("Q3-2025")


def test_fmt_value() -> None:
    assert fmt_value(2.5, "%") == "2.50%"
    assert fmt_value(33324, "EUR") == "33,324 EUR"
    assert fmt_value(66.92, "% of adults") == "66.92 % of adults"


def test_series_note_change_dates() -> None:
    text = series_note(
        "T", "Src.", [("2025-06-11", 2.0), ("2026-09-16", 2.5)], "%", "the rate",
        as_of=date(2026, 11, 19), change_dates=True,
    )  # fmt: skip
    assert "As of 2026-11-19, the rate is 2.50% (in effect since 2026-09-16)." in text
    assert text.index("2026-09-16: 2.50%") < text.index("2025-06-11: 2.00%")


def test_series_note_empty_fails() -> None:
    with pytest.raises(FetchError):
        series_note("T", "S", [], "%", "x")


def test_ecb_fetcher_parses_real_fixture(settings: Settings, tmp_path: Path) -> None:
    body = (FIX / "ecb_dfr.csv").read_bytes()
    opts = {
        "series": [
            {
                "series": "FM/B.U2.EUR.4F.KR.DFR.LEV",
                "name": "ECB deposit facility rate",
                "subject": "the ECB deposit facility rate",
                "kind": "change_dates",
                "columns": ["Balance"],
            }
        ]
    }
    result = EcbFetcher(
        ctx(settings, tmp_path, lambda r: httpx.Response(200, content=body)), opts
    ).fetch()
    assert not result.errors
    doc = result.documents[0]
    assert doc.doc_id == "ecb_ecb_deposit_facility_rate"
    assert doc.metadata.source_type == "macro_data" and doc.metadata.geography == "EU"
    assert "the ECB deposit facility rate is 2.50% (in effect since 2026-09-16)" in doc.text
    assert doc.metadata.date == date(2026, 9, 16)


def test_ecb_empty_series_is_an_error(settings: Settings, tmp_path: Path) -> None:
    opts = {"series": [{"series": "X/Y", "name": "n", "subject": "s"}]}
    result = EcbFetcher(
        ctx(settings, tmp_path, lambda r: httpx.Response(200, content=b"KEY,OBS_VALUE\n")), opts
    ).fetch()
    assert result.documents == [] and "no observations" in result.errors[0]


def test_parse_jsonstat_fixture() -> None:
    data = json.loads((FIX / "eurostat_unemployment.json").read_text(encoding="utf-8"))
    out = parse_jsonstat(data)
    assert set(out) == {"DE", "ES", "FR"}
    assert len(out["DE"]) == 3 and out["DE"] == sorted(out["DE"])


def test_eurostat_fetcher(settings: Settings, tmp_path: Path) -> None:
    body = (FIX / "eurostat_unemployment.json").read_bytes()
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["geo"] = request.url.params.get_list("geo")
        seen["unit"] = request.url.params.get("unit")
        return httpx.Response(200, content=body)

    opts = {
        "countries": ["FR", "DE", "ES"],
        "datasets": [
            {
                "dataset": "une_rt_a",
                "name": "Unemployment rate",
                "unit": "%",
                "filters": {"unit": "PC_ACT"},
            }
        ],
    }
    result = EurostatFetcher(ctx(settings, tmp_path, handler), opts).fetch()
    assert seen == {"geo": ["FR", "DE", "ES"], "unit": "PC_ACT"}
    ids = {d.doc_id: d for d in result.documents}
    assert set(ids) == {"eurostat_une_rt_a_de", "eurostat_une_rt_a_es", "eurostat_une_rt_a_fr"}
    assert ids["eurostat_une_rt_a_de"].metadata.geography == "DE"
    assert "Unemployment rate in Germany (Eurostat)" in ids["eurostat_une_rt_a_de"].text


def test_eurostat_error_payload(settings: Settings, tmp_path: Path) -> None:
    err = json.dumps({"error": [{"status": 404}]}).encode()
    opts = {"countries": ["FR"], "datasets": [{"dataset": "zzz", "name": "n"}]}
    result = EurostatFetcher(
        ctx(settings, tmp_path, lambda r: httpx.Response(200, content=err)), opts
    ).fetch()
    assert result.documents == [] and result.errors


def test_worldbank_fetcher(settings: Settings, tmp_path: Path) -> None:
    body = (FIX / "worldbank_inflation.json").read_bytes()
    opts = {
        "indicators": [{"id": "FP.CPI.TOTL.ZG", "name": "Inflation, consumer prices", "unit": "%"}]
    }
    result = WorldBankFetcher(
        ctx(settings, tmp_path, lambda r: httpx.Response(200, content=body)), opts
    ).fetch()
    assert not result.errors
    ids = {d.doc_id for d in result.documents}
    assert ids == {
        "worldbank_fp_cpi_totl_zg_deu",
        "worldbank_fp_cpi_totl_zg_esp",
        "worldbank_fp_cpi_totl_zg_fra",
    }
    de = next(d for d in result.documents if d.doc_id.endswith("deu"))
    assert de.metadata.geography == "DE" and "Germany" in de.text and "2025" in de.text


def test_worldbank_no_data(settings: Settings, tmp_path: Path) -> None:
    body = json.dumps([{"message": [{"id": "120"}]}]).encode()
    opts = {"indicators": [{"id": "BAD", "name": "x"}]}
    result = WorldBankFetcher(
        ctx(settings, tmp_path, lambda r: httpx.Response(200, content=body)), opts
    ).fetch()
    assert result.documents == [] and "no data" in result.errors[0]
