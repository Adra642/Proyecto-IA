"""Reusable resilience helpers: per-host rate limiting and retry with backoff."""

from __future__ import annotations

import time
from collections.abc import Callable

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RetryableError(Exception):
    """Raised for failures worth retrying (timeouts, 429, 5xx)."""


class RateLimiter:
    """Enforces a minimum interval between calls to the same key (host)."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args: min_interval: default seconds between calls; clock/sleep: injectable for tests."""
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def wait(self, key: str, interval: float | None = None) -> None:
        """Block until ``interval`` seconds passed since the last call for ``key``."""
        required = self._min_interval if interval is None else max(interval, self._min_interval)
        last = self._last.get(key)
        if last is not None:
            remaining = required - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last[key] = self._clock()


def call_with_retry[T](
    func: Callable[[], T],
    max_retries: int,
    backoff_base: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` retrying on :class:`RetryableError` with exponential backoff.

    Args:
        func: Zero-argument callable.
        max_retries: Extra attempts after the first one.
        backoff_base: Multiplier (seconds) of the exponential wait.
        sleep: Injectable sleep for tests.

    Returns:
        Whatever ``func`` returns.

    Raises:
        RetryableError: If every attempt fails.
    """
    retrying = Retrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=backoff_base, max=30),
        retry=retry_if_exception_type(RetryableError),
        sleep=sleep,
        reraise=True,
    )
    return retrying(func)
