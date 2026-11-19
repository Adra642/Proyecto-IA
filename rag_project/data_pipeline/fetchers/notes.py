"""Shared helpers for macro-data fetchers: period parsing and short text notes."""

from __future__ import annotations

from datetime import date

from core.exceptions import FetchError


def period_to_date(period: str) -> date:
    """Convert ``2025``, ``2025-12`` or ``2026-09-16`` to a date (first day when partial)."""
    parts = period.strip().split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError) as exc:
        raise FetchError(f"Unrecognised time period: {period!r}") from exc


def fmt_value(value: float, unit: str) -> str:
    """Format an observation: percentages with two decimals, others with separators."""
    if unit == "%":
        return f"{value:.2f}%"
    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"
    return f"{text} {unit}".strip()


def series_note(
    title: str,
    source_line: str,
    series: list[tuple[str, float]],
    unit: str,
    subject: str,
    as_of: date | None = None,
    change_dates: bool = False,
) -> str:
    """Render a short Markdown note for one time series.

    Args:
        title: Heading of the note.
        source_line: Provenance sentence (dataset or series id).
        series: ``(period, value)`` pairs, oldest first, non-empty.
        unit: Unit label (``%``, ``EUR``, ...).
        subject: Noun phrase used in the sentence, e.g. ``the ECB deposit facility rate``.
        as_of: Retrieval date used in the "As of" sentence for change-date series.
        change_dates: True when periods are dates on which the value changed.

    Returns:
        Markdown text.
    """
    if not series:
        raise FetchError(f"No observations for {title}")
    period, value = series[-1]
    latest = fmt_value(value, unit)
    if change_dates and as_of is not None:
        sentence = f"As of {as_of.isoformat()}, {subject} is {latest} (in effect since {period})."
        label = "Recent changes"
    else:
        sentence = f"The latest available value of {subject}, for {period}, is {latest}."
        label = "Recent observations"
    history = "\n".join(f"- {p}: {fmt_value(v, unit)}" for p, v in reversed(series))
    return f"# {title}\n\n{source_line}\n\n{sentence}\n\n{label}:\n{history}\n"
