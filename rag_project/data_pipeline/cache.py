"""On-disk cache of raw downloads (``data/raw/<source>/``) with a manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path


@dataclass(frozen=True)
class CachedPayload:
    """A raw download stored on disk."""

    path: Path
    data: bytes
    retrieved_at: date

    def text(self, encoding: str = "utf-8") -> str:
        """Decode the payload as text."""
        return self.data.decode(encoding, errors="replace")


class RawCache:
    """Caches downloads by key; ``refresh=True`` bypasses existing entries."""

    def __init__(self, root: Path, source: str, refresh: bool = False) -> None:
        """Args: root: ``data/raw``; source: sub-folder name; refresh: ignore cached entries."""
        self._dir = root / source
        self._manifest_path = self._dir / "manifest.json"
        self._refresh = refresh

    def _load(self) -> dict[str, dict[str, str]]:
        if self._manifest_path.exists():
            data: dict[str, dict[str, str]] = json.loads(
                self._manifest_path.read_text(encoding="utf-8")
            )
            return data
        return {}

    def _save(self, manifest: dict[str, dict[str, str]]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def get_or_fetch(self, key: str, ext: str, download: Callable[[], bytes]) -> CachedPayload:
        """Return the cached payload for ``key`` or download and store it.

        Args:
            key: Unique identifier (usually the URL plus parameters).
            ext: File extension without dot (``html``, ``json``, ...).
            download: Called only on a cache miss or refresh.

        Returns:
            The payload; ``retrieved_at`` is the date it was first downloaded.
        """
        manifest = self._load()
        entry = manifest.get(key)
        if entry is not None and not self._refresh:
            path = self._dir / entry["file"]
            if path.exists():
                return CachedPayload(
                    path, path.read_bytes(), date.fromisoformat(entry["retrieved_at"])
                )
        data = download()
        name = f"{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}.{ext}"
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / name).write_bytes(data)
        today = datetime.now(UTC).date()
        manifest[key] = {
            "file": name,
            "retrieved_at": today.isoformat(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        self._save(manifest)
        return CachedPayload(self._dir / name, data, today)
