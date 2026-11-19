"""Fetcher base class (Template Method) and the fetcher registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import httpx

from core.exceptions import FetchError
from core.logging_setup import get_logger
from core.models import Document
from core.settings import Settings
from data_pipeline.cache import CachedPayload, RawCache
from data_pipeline.http import HttpClient

log = get_logger(__name__)


@dataclass(frozen=True)
class FetchItem:
    """One unit of work of a fetcher (a page, a series, a paper...)."""

    key: str
    ext: str = "html"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchResult:
    """Outcome of one fetcher: documents, per-item errors and extra artifacts."""

    source_id: str
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchContext:
    """Shared dependencies injected into every fetcher."""

    settings: Settings
    http: HttpClient
    raw_dir: Path
    refresh: bool = False
    today: date | None = None


class BaseFetcher(ABC):
    """Implements cache, error isolation and logging once.

    Subclasses define ``_items()``, ``_download()`` and ``_parse()``.
    """

    source_id: ClassVar[str]  # plan id, e.g. "a4"
    source_name: ClassVar[str]  # metadata ``source`` value and raw cache folder

    def __init__(self, ctx: FetchContext, options: dict[str, Any]) -> None:
        """Args: ctx: shared dependencies; options: this source's ``config.yaml`` block."""
        self.ctx = ctx
        self.options = options
        self.cache = RawCache(ctx.raw_dir, self.source_name, ctx.refresh)

    @abstractmethod
    def _items(self) -> list[FetchItem]:
        """List the items to download."""

    @abstractmethod
    def _download(self, item: FetchItem) -> bytes:
        """Download one item and return its raw bytes."""

    @abstractmethod
    def _parse(self, item: FetchItem, payload: CachedPayload) -> list[Document]:
        """Turn one cached payload into documents."""

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET through the shared polite client."""
        return self.ctx.http.get(url, **kwargs)

    def _finalize(self, result: FetchResult) -> None:  # noqa: B027 - optional hook
        """Optional hook to add documents/artifacts after all items were processed."""

    def fetch(self) -> FetchResult:
        """Run the source; a failing item is logged and never aborts the run."""
        result = FetchResult(self.source_id)
        try:
            items = self._items()
        except Exception as exc:
            msg = f"[{self.source_id}] could not list items: {exc}"
            log.error(msg)
            result.errors.append(msg)
            return result
        for item in items:
            try:
                payload = self.cache.get_or_fetch(item.key, item.ext, partial(self._download, item))
                docs = self._parse(item, payload)
                result.documents.extend(docs)
                log.info("[%s] %s -> %d document(s)", self.source_id, item.key, len(docs))
            except Exception as exc:
                msg = f"[{self.source_id}] {item.key}: {exc}"
                log.error(msg)
                result.errors.append(msg)
        try:
            self._finalize(result)
        except Exception as exc:
            msg = f"[{self.source_id}] finalize failed: {exc}"
            log.error(msg)
            result.errors.append(msg)
        return result

    def option(self, name: str) -> Any:
        """Return a required option or raise a clear error."""
        if name not in self.options:
            raise FetchError(f"config.yaml: sources.{self.source_id}.{name} is required")
        return self.options[name]


FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {}


def register_fetcher(
    source_id: str,
) -> Callable[[type[BaseFetcher]], type[BaseFetcher]]:
    """Class decorator registering a fetcher under its plan id (e.g. ``"a4"``)."""

    def decorator(cls: type[BaseFetcher]) -> type[BaseFetcher]:
        cls.source_id = source_id
        FETCHER_REGISTRY[source_id] = cls
        return cls

    return decorator
