"""Fetchers: one module per source, registered with ``@register_fetcher``."""

from __future__ import annotations

import importlib

MODULES = (
    "kaggle",
    "arxiv",
    "wikipedia",
    "ecb",
    "eurlex",
    "eurostat",
    "worldbank",
    "html_pages",
)


def load_all() -> None:
    """Import every fetcher module so its ``@register_fetcher`` decorator runs."""
    for name in MODULES:
        importlib.import_module(f"{__name__}.{name}")
