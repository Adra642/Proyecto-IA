"""Polite HTTP client: User-Agent, robots.txt, rate limit, retries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from core.exceptions import FetchError
from core.logging_setup import get_logger
from core.settings import HttpSettings
from data_pipeline.resilience import RateLimiter, RetryableError, call_with_retry

log = get_logger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpClient:
    """Thin wrapper around ``httpx.Client`` used by every fetcher."""

    def __init__(
        self,
        settings: HttpSettings,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Args: settings: HTTP options; transport/limiter/sleep: injectable for tests."""
        self._settings = settings
        self._client = httpx.Client(
            headers={"User-Agent": settings.user_agent},
            timeout=settings.timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = limiter or RateLimiter(settings.min_interval_seconds)
        self._sleep = sleep
        self._robots: dict[str, RobotFileParser | None] = {}

    def _robots_for(self, host_url: str) -> RobotFileParser | None:
        if host_url not in self._robots:
            parser: RobotFileParser | None = None
            try:
                resp = self._client.get(f"{host_url}/robots.txt")
                if resp.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(resp.text.splitlines())
            except httpx.HTTPError as exc:
                log.warning("Could not read robots.txt of %s: %s", host_url, exc)
            self._robots[host_url] = parser  # None = no usable rules, allow
        return self._robots[host_url]

    def _check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        parser = self._robots_for(f"{parts.scheme}://{parts.netloc}")
        if parser is not None and not parser.can_fetch(self._settings.user_agent, url):
            raise FetchError(f"Blocked by robots.txt: {url}")

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        check_robots: bool = True,
        min_interval: float | None = None,
    ) -> httpx.Response:
        """GET ``url`` politely.

        Args:
            url: Target URL.
            params: Query parameters.
            headers: Extra headers (e.g. ``Accept``).
            check_robots: Respect robots.txt. Documented public APIs pass ``False``.
            min_interval: Larger per-call minimum interval (e.g. 3 s for arXiv).

        Returns:
            The successful response.

        Raises:
            FetchError: On robots block, 4xx status, or after exhausting retries.
        """
        if check_robots and self._settings.respect_robots:
            self._check_robots(url)
        host = urlsplit(url).netloc

        def attempt() -> httpx.Response:
            self._limiter.wait(host, min_interval)
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
            if resp.status_code in _RETRY_STATUS:
                raise RetryableError(f"HTTP {resp.status_code}")
            return resp

        kwargs: dict[str, Any] = {}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        try:
            resp = call_with_retry(
                attempt,
                self._settings.max_retries,
                self._settings.backoff_base_seconds,
                **kwargs,
            )
        except RetryableError as exc:
            raise FetchError(f"GET {url} failed after retries: {exc}") from exc
        if resp.status_code >= 400:
            raise FetchError(f"GET {url} returned HTTP {resp.status_code}")
        return resp

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()
