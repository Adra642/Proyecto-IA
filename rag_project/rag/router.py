"""Rule-based router for the advanced mode: retrieve or answer directly, and which filter to use.

Rules instead of an LLM call: deterministic, free, instant and unit-testable. The filter is a
*hint* (the chain falls back to an unfiltered search when it finds nothing).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SMALL_TALK_MAX_CHARS = 60
_SMALL_TALK = re.compile(
    r"^\W*(hi|hello|hey|hola|buenas|bonjour|salut|hallo|thanks|thank you|gracias|merci|danke|"
    r"who are you|qui es-tu|qu[ié]n eres|what can you do|qu[ée] puedes hacer)\b",
    re.IGNORECASE,
)

# (source types to search, pattern). First match wins, so the order matters.
_SOURCE_RULES: tuple[tuple[tuple[str, ...], re.Pattern[str]], ...] = (
    (("dataset_stats",), re.compile(r"kaggle|bank customer churn|\bdataset\b", re.I)),
    (("paper",), re.compile(r"arxiv|\bpapers?\b|\bauthors?\b|transformer", re.I)),
    (
        ("regulation", "regulator_guide"),
        re.compile(
            r"regulation|directive|article\s+\d|\(eu\)|reglamento|directiva|r[eè]glement|"
            r"verordnung|\beba\b|gdpr|psd2",
            re.I,
        ),
    ),
    (
        ("macro_data", "press"),
        re.compile(
            r"eurostat|\becb\b|deposit facility|interest rate|world bank|\bgdp\b|inflation|"
            r"internet banking|tipo de inter[eé]s",
            re.I,
        ),
    ),
)
_COUNTRIES: dict[str, re.Pattern[str]] = {
    "DE": re.compile(r"germany|german\b|deutschland|alemania|allemagne|alem[aá]n", re.I),
    "FR": re.compile(r"france|french|francia|franc[eé]s|frankreich", re.I),
    "ES": re.compile(r"spain|spanish|espa[nñ]a|espagne|spanien", re.I),
}
_GEO_FILTERED = {"dataset_stats", "macro_data"}  # types whose documents are per country


@dataclass(frozen=True)
class Route:
    """The router's decision."""

    retrieve: bool
    where: dict[str, Any] | None = None
    reason: str = ""


def all_of(conditions: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine Chroma conditions with ``$and`` (a single condition stays as it is)."""
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": list(conditions)}


def where_from_filters(
    source_types: Sequence[str] = (),
    geographies: Sequence[str] = (),
    languages: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Build a Chroma filter from the UI selections (empty selections are ignored)."""
    conds = [
        {key: {"$in": list(values)}}
        for key, values in (
            ("source_type", source_types),
            ("geography", geographies),
            ("language", languages),
        )
        if values
    ]
    return all_of(conds)


def _single_country(question: str) -> str | None:
    found = [code for code, pat in _COUNTRIES.items() if pat.search(question)]
    return found[0] if len(found) == 1 else None


class RuleRouter:
    """Decides ``retrieve`` vs ``answer directly`` and picks a metadata filter."""

    def route(self, question: str) -> Route:
        """Route a question: small talk is answered directly, the rest gets a filtered search."""
        text = question.strip()
        if len(text) <= SMALL_TALK_MAX_CHARS and _SMALL_TALK.search(text):
            return Route(False, None, "small talk: answered without retrieval")
        for source_types, pattern in _SOURCE_RULES:
            if not pattern.search(text):
                continue
            conds: list[dict[str, Any]] = [{"source_type": {"$in": list(source_types)}}]
            country = _single_country(text)
            if country and _GEO_FILTERED.intersection(source_types):
                conds.append({"geography": {"$in": [country, "EU"]}})
            where = all_of(conds)
            return Route(True, where, f"filter {where}")
        return Route(True, None, "no filter: searching the whole corpus")
