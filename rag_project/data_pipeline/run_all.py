"""Orchestrator: fetch -> clean -> build documents -> write, with a run report.

Usage::

    uv run python -m data_pipeline.run_all [--sources a1,a3] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.logging_setup import get_logger, setup_logging
from core.models import Document
from core.settings import Settings, load_settings
from data_pipeline.clean import clean_documents
from data_pipeline.fetchers import load_all
from data_pipeline.fetchers.base import FETCHER_REGISTRY, FetchContext, FetchResult
from data_pipeline.http import HttpClient
from data_pipeline.pii_check import PiiReport, check_corpus
from data_pipeline.writer import write_documents, write_sources_md, write_stats

log = get_logger(__name__)

ALIASES = {"a2": "a1"}  # A2 (derived documents) is produced by the A1 fetcher


@dataclass
class SourceSummary:
    """Per-source figures for the run report."""

    source_id: str
    raw_documents: int = 0
    documents: int = 0
    words: int = 0
    formats: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_source_ids(arg: str | None, settings: Settings) -> list[str]:
    """Resolve ``--sources`` (or all enabled sources) to registered fetcher ids."""
    if arg:
        ids = [
            ALIASES.get(s.strip().lower(), s.strip().lower()) for s in arg.split(",") if s.strip()
        ]
    else:
        ids = [k for k, v in settings.sources.items() if v.get("enabled", True)]
    unknown = [i for i in ids if i not in FETCHER_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown source id(s): {', '.join(unknown)}. Known: {sorted(FETCHER_REGISTRY)}"
        )
    return sorted(dict.fromkeys(ids))


def run_fetchers(ids: Sequence[str], ctx: FetchContext) -> list[FetchResult]:
    """Run each fetcher in isolation; a crash in one never stops the others."""
    results: list[FetchResult] = []
    for source_id in ids:
        cls = FETCHER_REGISTRY[source_id]
        log.info("=== %s (%s) ===", source_id, cls.source_name)
        try:
            results.append(cls(ctx, ctx.settings.sources.get(source_id, {})).fetch())
        except Exception as exc:  # fetch() already isolates items; this is the last guard
            log.exception("Fetcher %s crashed", source_id)
            results.append(FetchResult(source_id, errors=[f"[{source_id}] crashed: {exc}"]))
    return results


def summarize(
    ids: Sequence[str], results: list[FetchResult], cleaned: list[Document]
) -> list[SourceSummary]:
    """Per-source summary after cleaning."""
    by_source = {r.source_id: r for r in results}
    names = {sid: FETCHER_REGISTRY[sid].source_name for sid in ids}
    summaries = []
    for sid in ids:
        res = by_source[sid]
        wanted = {names[sid], "kaggle_derived"} if sid == "a1" else {names[sid]}
        docs = [d for d in cleaned if d.metadata.source in wanted]
        summaries.append(
            SourceSummary(
                source_id=sid,
                raw_documents=len(res.documents),
                documents=len(docs),
                words=sum(len(d.text.split()) for d in docs),
                formats=sorted({d.metadata.format for d in docs}),
                languages=sorted({d.metadata.language for d in docs}),
                errors=res.errors,
            )
        )
    return summaries


def write_report(
    path: Path, summaries: list[SourceSummary], pii: PiiReport, refresh: bool
) -> dict[str, Any]:
    """Write ``logs/run_report.json`` and return it."""
    ok = [s for s in summaries if s.documents]
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "refresh": refresh,
        "sources": [asdict(s) for s in summaries],
        "failed_sources": [s.source_id for s in summaries if not s.documents],
        "sources_with_errors": [s.source_id for s in summaries if s.errors],
        "totals": {
            "sources_ok": len(ok),
            "documents": sum(s.documents for s in summaries),
            "words": sum(s.words for s in summaries),
            "formats": sorted({f for s in ok for f in s.formats}),
            "languages": sorted({lang for s in ok for lang in s.languages}),
        },
        "pii_check": {
            "ok": pii.ok,
            "checked_files": pii.checked_files,
            "violations": pii.violations,
            "external_surname_collisions": {
                k: sorted(set(v)) for k, v in sorted(pii.collisions.items())
            },
            "skipped_reason": pii.skipped_reason,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def log_summary(report: dict[str, Any]) -> None:
    """Log a readable table of the run report."""
    log.info(
        "%-4s %6s %6s %9s  %-14s %-12s %s",
        "id",
        "raw",
        "docs",
        "words",
        "formats",
        "languages",
        "errors",
    )
    for s in report["sources"]:
        log.info(
            "%-4s %6d %6d %9d  %-14s %-12s %d",
            s["source_id"], s["raw_documents"], s["documents"], s["words"],
            ",".join(s["formats"]), ",".join(s["languages"]), len(s["errors"]),
        )  # fmt: skip
    t = report["totals"]
    log.info(
        "TOTAL: %d sources ok, %d documents, %d words, formats=%s, languages=%s",
        t["sources_ok"], t["documents"], t["words"], t["formats"], t["languages"],
    )  # fmt: skip
    if report["failed_sources"]:
        log.warning("FAILED sources (no documents): %s", report["failed_sources"])
    for s in report["sources"]:
        for err in s["errors"]:
            log.warning("  %s", err)
    log.info(
        "PII check ok=%s violations=%s",
        report["pii_check"]["ok"],
        report["pii_check"]["violations"],
    )


def run(
    settings: Settings, ids: Sequence[str], refresh: bool, http: HttpClient | None = None
) -> dict[str, Any]:
    """Run the pipeline for ``ids`` and return the run report."""
    raw_dir = settings.resolve(settings.paths.raw_dir)
    processed_dir = settings.resolve(settings.paths.processed_dir)
    client = http or HttpClient(settings.http)
    ctx = FetchContext(settings, client, raw_dir, refresh=refresh)
    try:
        results = run_fetchers(ids, ctx)
    finally:
        client.close()
    raw_docs = [d for r in results for d in r.documents]
    cleaned = clean_documents(raw_docs)
    write_documents(cleaned, processed_dir)
    for result in results:
        if "stats.json" in result.artifacts:
            write_stats(result.artifacts["stats.json"], processed_dir / "stats.json")
    write_sources_md(processed_dir, processed_dir.parent / "SOURCES.md")
    pii = check_corpus(processed_dir, raw_dir / "kaggle")
    report = write_report(
        settings.resolve(settings.paths.logs_dir) / "run_report.json",
        summarize(ids, results, cleaned),
        pii,
        refresh,
    )
    log_summary(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 unless nothing at all could be produced."""
    parser = argparse.ArgumentParser(description="Build the RAG corpus (fetch, clean, write).")
    parser.add_argument("--sources", help="comma-separated ids, e.g. a1,a3 (default: all enabled)")
    parser.add_argument("--refresh", action="store_true", help="bypass the raw download cache")
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    load_all()
    ids = parse_source_ids(args.sources, settings)
    log.info("Running sources: %s (refresh=%s)", ids, args.refresh)
    report = run(settings, ids, args.refresh)
    if not report["totals"]["documents"]:
        return 1
    return 2 if not report["pii_check"]["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
