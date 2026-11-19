"""Cleaning: PII removal for the CSV and (below) the text cleaning pipeline."""

from __future__ import annotations

import pandas as pd

PII_COLUMNS: tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` without the personal-identifier columns.

    Called before any document is created, so identifiers never reach the corpus.

    Args:
        df: Raw customer table.

    Returns:
        The table without ``RowNumber``, ``CustomerId`` and ``Surname``.
    """
    return df.drop(columns=list(PII_COLUMNS), errors="ignore").copy()
