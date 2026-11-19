from pathlib import Path

import httpx
import pytest

from core.settings import Settings
from data_pipeline.fetchers.base import FETCHER_REGISTRY, FetchContext
from data_pipeline.fetchers.kaggle import KaggleFetcher
from data_pipeline.http import HttpClient

CSV = Path(__file__).parent.parent / "fixtures" / "tiny_churn.csv"


def make(tmp_path: Path, settings: Settings) -> KaggleFetcher:
    http = HttpClient(settings.http, transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    return KaggleFetcher(FetchContext(settings, http, tmp_path), settings.sources["a1"])


def test_registered() -> None:
    assert FETCHER_REGISTRY["a1"] is KaggleFetcher


def test_fetch_builds_documents_and_stats(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(KaggleFetcher, "_fetch_csv", lambda self: CSV.read_bytes())
    result = make(tmp_path, settings).fetch()
    assert not result.errors
    assert len(result.documents) > 40
    assert result.artifacts["stats.json"]["overall.rows"] == 12
    assert list((tmp_path / "kaggle").glob("*.csv"))  # raw CSV cached


def test_second_run_uses_cache(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake(self: KaggleFetcher) -> bytes:
        calls["n"] += 1
        return CSV.read_bytes()

    monkeypatch.setattr(KaggleFetcher, "_fetch_csv", fake)
    first = make(tmp_path, settings).fetch()
    second = make(tmp_path, settings).fetch()
    assert calls["n"] == 1
    assert [d.model_dump() for d in first.documents] == [d.model_dump() for d in second.documents]


def test_failure_is_reported_not_raised(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kagglehub

    def boom(*a: object, **k: object) -> str:
        raise RuntimeError("auth required")

    monkeypatch.setattr(kagglehub, "dataset_download", boom)
    result = make(tmp_path, settings).fetch()
    assert result.documents == [] and "a1" in result.errors[0]
