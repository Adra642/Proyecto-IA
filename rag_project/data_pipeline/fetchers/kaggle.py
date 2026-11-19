"""A1/A2: Kaggle Bank Customer Churn CSV and the documents derived from it."""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from core.exceptions import FetchError
from core.logging_setup import get_logger
from core.models import Document
from data_pipeline.build_docs import build_documents
from data_pipeline.cache import CachedPayload
from data_pipeline.clean import drop_pii
from data_pipeline.fetchers.base import (
    BaseFetcher,
    FetchItem,
    FetchResult,
    register_fetcher,
)

log = get_logger(__name__)


@contextmanager
def _env(values: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables (kagglehub reads its config from the env)."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, previous in old.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


@register_fetcher("a1")
class KaggleFetcher(BaseFetcher):
    """Downloads the CSV, drops PII, and builds the derived documents and stats."""

    source_name = "kaggle"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """See :class:`BaseFetcher`."""
        super().__init__(*args, **kwargs)
        self._stats: dict[str, float] = {}

    def _items(self) -> list[FetchItem]:
        return [FetchItem(f"kaggle:{self.option('dataset')}", ext="csv")]

    def _fetch_csv(self) -> bytes:
        """Download via kagglehub (anonymous works for public data), else the fallback URL."""
        env = {"KAGGLEHUB_CACHE": str(self.ctx.raw_dir / "kaggle" / "hub")}
        settings = self.ctx.settings
        if settings.kaggle_username and settings.kaggle_key:
            env["KAGGLE_USERNAME"] = settings.kaggle_username
            env["KAGGLE_KEY"] = settings.kaggle_key.get_secret_value()
        try:
            import kagglehub

            with _env(env):
                folder = kagglehub.dataset_download(self.option("dataset"))
            files = sorted(Path(folder).glob("*.csv"))
            if not files:
                raise FetchError(f"No CSV found in {folder}")
            return files[0].read_bytes()
        except Exception as exc:
            fallback = str(self.options.get("fallback_url") or "")
            if not fallback:
                raise FetchError(f"kagglehub failed and no fallback_url is set: {exc}") from exc
            log.warning("kagglehub failed (%s); using fallback_url", exc)
            return self.get(fallback, check_robots=False).content

    def _download(self, item: FetchItem) -> bytes:
        return self._fetch_csv()

    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        df = drop_pii(pd.read_csv(io.BytesIO(payload.data)))
        docs, self._stats = build_documents(
            df,
            url=str(self.option("url")),
            retrieved_at=payload.retrieved_at,
            license_=str(self.option("license")),
        )
        return docs

    def _finalize(self, result: FetchResult) -> None:
        if self._stats:
            result.artifacts["stats.json"] = self._stats
