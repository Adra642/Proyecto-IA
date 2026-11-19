import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.exceptions import FetchError
from core.models import Document, DocumentMetadata
from core.settings import HttpSettings, Settings
from data_pipeline.cache import RawCache
from data_pipeline.fetchers.base import (
    FETCHER_REGISTRY,
    BaseFetcher,
    FetchContext,
    FetchItem,
    register_fetcher,
)
from data_pipeline.http import HttpClient
from data_pipeline.resilience import RateLimiter, RetryableError, call_with_retry


def http_settings(**over: Any) -> HttpSettings:
    base: dict[str, Any] = {"user_agent": "test-bot", "max_retries": 2, "respect_robots": True}
    return HttpSettings(**{**base, **over})


def client(handler: Any, **over: Any) -> tuple[HttpClient, list[float]]:
    sleeps: list[float] = []
    clock = {"t": 0.0}

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock["t"] += s

    limiter = RateLimiter(1.0, clock=lambda: clock["t"], sleep=fake_sleep)
    c = HttpClient(
        http_settings(**over),
        transport=httpx.MockTransport(handler),
        limiter=limiter,
        sleep=fake_sleep,
    )
    return c, sleeps


def test_rate_limiter_waits_between_calls_to_same_host() -> None:
    now = {"t": 0.0}
    sleeps: list[float] = []
    rl = RateLimiter(1.0, clock=lambda: now["t"], sleep=sleeps.append)
    rl.wait("h")
    now["t"] = 0.4
    rl.wait("h")
    rl.wait("other")
    assert sleeps == [pytest.approx(0.6)]


def test_rate_limiter_honours_larger_interval() -> None:
    sleeps: list[float] = []
    rl = RateLimiter(1.0, clock=lambda: 0.0, sleep=sleeps.append)
    rl.wait("h", 3.0)
    rl.wait("h", 3.0)
    assert sleeps == [3.0]


def test_retry_then_success() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("boom")
        return "ok"

    assert call_with_retry(flaky, 3, 0.1, sleep=lambda s: None) == "ok"


def test_retry_exhausted_raises() -> None:
    def always() -> str:
        raise RetryableError("boom")

    with pytest.raises(RetryableError):
        call_with_retry(always, 1, 0.1, sleep=lambda s: None)


def test_http_retries_on_500_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls["n"] += 1
        return httpx.Response(500 if calls["n"] == 1 else 200, text="hi")

    c, _ = client(handler)
    assert c.get("https://x.test/page").text == "hi"
    assert calls["n"] == 2


def test_http_gives_up_with_fetch_error() -> None:
    c, _ = client(lambda r: httpx.Response(404 if r.url.path == "/robots.txt" else 503))
    with pytest.raises(FetchError, match="after retries"):
        c.get("https://x.test/page")


def test_http_4xx_is_fetch_error_without_retry() -> None:
    n = {"c": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/robots.txt":
            n["c"] += 1
        return httpx.Response(404)

    c, _ = client(handler)
    with pytest.raises(FetchError, match="HTTP 404"):
        c.get("https://x.test/page")
    assert n["c"] == 1


def test_robots_disallow_is_honoured_and_can_be_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private")
        return httpx.Response(200, text="ok")

    c, _ = client(handler)
    with pytest.raises(FetchError, match="robots"):
        c.get("https://x.test/private/a")
    assert c.get("https://x.test/public").text == "ok"
    assert c.get("https://x.test/private/a", check_robots=False).text == "ok"


def test_cache_hit_skips_download_and_refresh_redownloads(tmp_path: Path) -> None:
    calls = {"n": 0}

    def dl() -> bytes:
        calls["n"] += 1
        return b"data"

    RawCache(tmp_path, "src").get_or_fetch("k", "html", dl)
    p = RawCache(tmp_path, "src").get_or_fetch("k", "html", dl)
    assert calls["n"] == 1 and p.data == b"data"
    RawCache(tmp_path, "src", refresh=True).get_or_fetch("k", "html", dl)
    assert calls["n"] == 2
    manifest = json.loads((tmp_path / "src" / "manifest.json").read_text(encoding="utf-8"))
    assert "sha256" in manifest["k"]


def _doc(doc_id: str) -> Document:
    meta = DocumentMetadata(
        doc_id=doc_id,
        title=doc_id,
        source="t",
        source_type="encyclopedia",
        url="u",
        language="en",
        geography="global",
        topic="x",
        retrieved_at=date(2026, 1, 1),
        license="l",
    )
    return Document(text="body " * 50, metadata=meta)


class FlakyFetcher(BaseFetcher):
    source_id = "flaky"
    source_name = "flaky"

    def _items(self) -> list[FetchItem]:
        return [FetchItem("good"), FetchItem("bad"), FetchItem("good2")]

    def _download(self, item: FetchItem) -> bytes:
        if item.key == "bad":
            raise FetchError("nope")
        return item.key.encode()

    def _parse(self, item: FetchItem, payload: Any) -> list[Document]:
        return [_doc(item.key)]


def test_failing_item_does_not_stop_the_fetcher(tmp_path: Path, settings: Settings) -> None:
    ctx = FetchContext(settings, client(lambda r: httpx.Response(200))[0], tmp_path)
    result = FlakyFetcher(ctx, {}).fetch()
    assert [d.doc_id for d in result.documents] == ["good", "good2"]
    assert len(result.errors) == 1 and "bad" in result.errors[0]


def test_registry_registers_by_id() -> None:
    @register_fetcher("zz")
    class Z(FlakyFetcher):
        pass

    assert FETCHER_REGISTRY["zz"] is Z and Z.source_id == "zz"
    del FETCHER_REGISTRY["zz"]
