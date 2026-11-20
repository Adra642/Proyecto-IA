"""Deterministic fakes shared by the unit tests."""

from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words are close (cosine)."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim  # noqa: S324
            v[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
