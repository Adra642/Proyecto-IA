import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.models import Document, DocumentMetadata
from core.settings import PathsSettings, Settings
from data_pipeline import run_all
from data_pipeline.fetchers import load_all
from data_pipeline.fetchers.base import FETCHER_REGISTRY, BaseFetcher, FetchItem, FetchResult
from data_pipeline.http import HttpClient
from data_pipeline.pii_check import check_corpus, scan
from data_pipeline.writer import build_sources_md, render_document, write_documents, write_stats

LONG = "Plain markdown content about banking customers and churn drivers. " * 12


def mk(doc_id: str, source: str = "wikipedia", text: str = LONG, lang: str = "en") -> Document:
    meta = DocumentMetadata(
        doc_id=doc_id, title=doc_id, source=source, source_type="encyclopedia",
        url=f"https://x.test/{doc_id}", language=lang, geography="EU", topic="churn",
        columns=["Exited"], retrieved_at=date(2026, 11, 19), license="CC BY-SA 4.0", format="json",
    )  # fmt: skip
    return Document(text=text, metadata=meta)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_render_has_front_matter() -> None:
    out = render_document(mk("d1"))
    assert out.startswith("---\ndoc_id: d1\n") and "\n---\n\nPlain markdown" in out
    assert "retrieved_at: '2026-11-19'" in out or "retrieved_at: 2026-11-19" in out


def test_write_is_idempotent_and_removes_stale_only_in_active_sources(tmp_path: Path) -> None:
    docs = [mk("a"), mk("b"), mk("c", source="eurlex")]
    write_documents(docs, tmp_path)
    (tmp_path / "wikipedia" / "old.md").write_text("stale", encoding="utf-8")
    (tmp_path / "bde").mkdir()
    (tmp_path / "bde" / "keep.md").write_text("previous run of a failed source", encoding="utf-8")
    write_documents(docs, tmp_path)
    after_cleanup = snapshot(tmp_path)
    assert "wikipedia/old.md" not in after_cleanup and "bde/keep.md" in after_cleanup
    write_documents(docs, tmp_path)
    assert snapshot(tmp_path) == after_cleanup


def test_stats_nan_becomes_null(tmp_path: Path) -> None:
    write_stats({"b": 1.5, "a": float("nan")}, tmp_path / "stats.json")
    assert json.loads((tmp_path / "stats.json").read_text(encoding="utf-8")) == {
        "a": None,
        "b": 1.5,
    }


def test_sources_md(tmp_path: Path) -> None:
    write_documents(
        [mk("a"), mk("c", source="eurlex", lang="en"), mk("z", source="bde", lang="es")], tmp_path
    )
    md = build_sources_md(tmp_path)
    assert "| A4 | Wikipedia concept pages | 1 |" in md and "| B1 |" in md and "| A6 |" in md
    assert "CC BY-SA 4.0" in md and "2026-11-19" in md and "es" in md


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    (d / "x.csv").write_text(
        "RowNumber,CustomerId,Surname,Age\n1,15634602,Hargrave,42\n2,15647311,Hill,41\n",
        encoding="utf-8",
    )
    return d


def test_pii_clean_corpus_and_collisions(tmp_path: Path, csv_dir: Path) -> None:
    write_documents(
        [
            mk("a", text=LONG + " Written by Mark Hill, a researcher."),
            mk("d", source="kaggle_derived"),
        ],
        tmp_path,
    )
    report = check_corpus(tmp_path, csv_dir)
    assert report.ok and report.checked_files == 2
    assert report.collisions == {"hill": ["wikipedia/a.md"]}


def test_pii_violations(tmp_path: Path, csv_dir: Path) -> None:
    write_documents(
        [mk("d", source="kaggle_derived", text=LONG + " Customer Hargrave churned."),
         mk("e", text=LONG + " id 15634602 appears")],
        tmp_path,
    )  # fmt: skip
    report = check_corpus(tmp_path, csv_dir)
    assert not report.ok and len(report.violations) == 2
    assert any("derived" in v for v in report.violations) and any(
        "CustomerId" in v for v in report.violations
    )


def test_pii_scan_whole_words_only(tmp_path: Path) -> None:
    write_documents(
        [mk("d", source="kaggle_derived", text=LONG + " Hillary and hillside")], tmp_path
    )
    assert scan(tmp_path, {"Hill"}, set()).ok


def test_pii_skipped_without_csv(tmp_path: Path) -> None:
    report = check_corpus(tmp_path, tmp_path / "nothing")
    assert report.ok and report.skipped_reason


class Good(BaseFetcher):
    source_id = "f1"
    source_name = "wikipedia"

    def _items(self) -> list[FetchItem]:
        return [FetchItem("one")]

    def _download(self, item: FetchItem) -> bytes:
        return b"x"

    def _parse(self, item: FetchItem, payload: Any) -> list[Document]:
        return [mk("good_doc"), mk("dup_doc")]  # identical text -> deduped


class BadItems(Good):
    source_id = "f2"
    source_name = "eurlex"

    def _items(self) -> list[FetchItem]:
        raise RuntimeError("listing failed")


class Crashes(Good):
    source_id = "f3"
    source_name = "bde"

    def fetch(self) -> FetchResult:
        raise RuntimeError("boom")


@pytest.fixture
def pipeline_settings(settings: Settings, tmp_path: Path) -> Settings:
    s = settings.model_copy(deep=True)
    s.paths = PathsSettings(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "data" / "processed",
        logs_dir=tmp_path / "logs",
    )
    s.sources = {"f1": {}, "f2": {}, "f3": {}}
    return s


def test_run_isolates_failures_and_is_idempotent(
    pipeline_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(FETCHER_REGISTRY, "f1", Good)
    monkeypatch.setitem(FETCHER_REGISTRY, "f2", BadItems)
    monkeypatch.setitem(FETCHER_REGISTRY, "f3", Crashes)
    http = HttpClient(
        pipeline_settings.http, transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    report = run_all.run(pipeline_settings, ["f1", "f2", "f3"], refresh=False, http=http)
    assert report["failed_sources"] == ["f2", "f3"]
    assert report["totals"]["sources_ok"] == 1 and report["totals"]["documents"] == 1
    errors = {s["source_id"]: s["errors"] for s in report["sources"]}
    assert "listing failed" in errors["f2"][0] and "crashed" in errors["f3"][0]
    processed = pipeline_settings.paths.processed_dir
    first = snapshot(processed)
    assert sorted(first) == ["wikipedia/dup_doc.md"]  # duplicate text: first doc_id wins
    http2 = HttpClient(
        pipeline_settings.http, transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    run_all.run(pipeline_settings, ["f1", "f2", "f3"], refresh=False, http=http2)
    assert snapshot(processed) == first
    assert (pipeline_settings.paths.logs_dir / "run_report.json").exists()
    assert (processed.parent / "SOURCES.md").exists()


def test_parse_source_ids(settings: Settings) -> None:
    load_all()
    assert run_all.parse_source_ids("a2, A3", settings) == ["a1", "a3"]
    assert "a1" in run_all.parse_source_ids(None, settings)
    with pytest.raises(SystemExit, match="Unknown"):
        run_all.parse_source_ids("zz", settings)
