"""Privacy check: customer identifiers of the CSV must not reach the corpus.

Scoped rules (decided with the team):
* no ``CustomerId`` anywhere in ``data/processed``;
* no surname in documents derived from the dataset (``kaggle_derived``);
* surnames found in *external* documents are reported as collisions (many surnames are
  ordinary words or authors' names) and logged, not treated as failures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.logging_setup import get_logger

log = get_logger(__name__)

DERIVED_SOURCE = "kaggle_derived"


@dataclass
class PiiReport:
    """Result of the privacy scan."""

    violations: list[str] = field(default_factory=list)
    collisions: dict[str, list[str]] = field(default_factory=dict)  # surname -> doc paths
    checked_files: int = 0
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when there are no violations."""
        return not self.violations


def load_identifiers(csv_path: Path) -> tuple[set[str], set[str]]:
    """Read surnames and customer ids from the raw CSV (kept in memory only)."""
    df = pd.read_csv(csv_path, usecols=["CustomerId", "Surname"])
    surnames = {str(s).strip() for s in df["Surname"].dropna() if len(str(s).strip()) >= 3}
    ids = {str(i) for i in df["CustomerId"].dropna()}
    return surnames, ids


def scan(processed_dir: Path, surnames: set[str], ids: set[str]) -> PiiReport:
    """Scan every processed Markdown file against the identifiers."""
    report = PiiReport()
    surname_re = (
        re.compile(
            r"\b(" + "|".join(re.escape(s) for s in sorted(surnames)) + r")\b", re.IGNORECASE
        )
        if surnames
        else None
    )
    id_re = re.compile(r"\b(" + "|".join(sorted(ids)) + r")\b") if ids else None
    for path in sorted(processed_dir.glob("*/*.md")):
        report.checked_files += 1
        text = path.read_text(encoding="utf-8")
        rel = f"{path.parent.name}/{path.name}"
        if id_re and (m := id_re.search(text)):
            report.violations.append(f"CustomerId {m.group(1)} in {rel}")
        if surname_re is None:
            continue
        hits = {m.group(1).lower() for m in surname_re.finditer(text)}
        if not hits:
            continue
        if path.parent.name == DERIVED_SOURCE:
            report.violations.append(f"surname(s) {sorted(hits)[:3]} in derived document {rel}")
        else:
            for hit in hits:
                report.collisions.setdefault(hit, []).append(rel)
    return report


def check_corpus(processed_dir: Path, raw_csv_dir: Path) -> PiiReport:
    """Run the scan using the cached raw CSV; skip (with a reason) if it is not cached."""
    csvs = sorted(raw_csv_dir.glob("*.csv"))
    if not csvs:
        report = PiiReport(skipped_reason=f"no raw CSV cached in {raw_csv_dir}")
        log.warning("PII check skipped: %s", report.skipped_reason)
        return report
    surnames, ids = load_identifiers(csvs[0])
    report = scan(processed_dir, surnames, ids)
    log.info(
        "PII check: %d files, %d violations, %d external surname collisions",
        report.checked_files,
        len(report.violations),
        len(report.collisions),
    )
    return report
