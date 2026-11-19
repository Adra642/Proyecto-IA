from typing import Any

from rag.embeddings import E5Embedder


class FakeModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(len(t)), 0.0] for t in texts]


def test_prefixes_are_added() -> None:
    model = FakeModel()
    emb = E5Embedder(model, "query: ", "passage: ")
    emb.embed_passages(["a", "b"])
    emb.embed_query("q")
    assert model.seen == ["passage: a", "passage: b", "query: q"]


def test_returns_plain_float_lists() -> None:
    vec = E5Embedder(FakeModel(), "query: ", "passage: ").embed_query("x")
    assert vec == [float(len("query: x")), 0.0]
