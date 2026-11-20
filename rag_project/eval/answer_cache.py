"""On-disk cache of evaluation answers (reproducible runs, no repeated LLM calls)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.models import RagAnswer

CACHE_DIR = Path(__file__).parent / "cache"


def cache_key(parts: dict[str, Any]) -> str:
    """Stable hash of everything that determines an answer."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class AnswerCache:
    """One JSON file per answer; the key includes provider, model, prompts and index."""

    def __init__(self, directory: Path = CACHE_DIR, refresh: bool = False) -> None:
        """Args: directory: where answers are stored; refresh: ignore existing entries."""
        self._dir = directory
        self._refresh = refresh

    def get(self, parts: dict[str, Any]) -> RagAnswer | None:
        """Return the cached answer for ``parts`` (``None`` on a miss or when refreshing)."""
        path = self._dir / f"{cache_key(parts)}.json"
        if self._refresh or not path.exists():
            return None
        return RagAnswer(**json.loads(path.read_text(encoding="utf-8"))["answer"])

    def put(self, parts: dict[str, Any], answer: RagAnswer) -> None:
        """Store ``answer`` together with the key parts (for human inspection)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {"key_parts": parts, "answer": answer.model_dump(mode="json")}
        path = self._dir / f"{cache_key(parts)}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
