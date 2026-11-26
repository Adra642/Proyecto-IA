import csv
from collections import Counter
from pathlib import Path

import pytest

from core.models import Document
from core.settings import ChunkingSettings
from eval.chunking_experiment import (
    CachedEmbedder,
    build_variants,
    detail_rows,
    evaluate_variant,
    plot,
    write_csv,
)
from eval.questions import Question, load_questions, retrieval_questions
from eval.retrieval_metrics import first_gold_rank, hit_at_k, mean, reciprocal_rank
from tests.fakes import FakeEmbedder
from tests.unit.test_index import FRANCE, GERMANY, LAW


def test_hit_at_k_and_mrr() -> None:
    ranked = ["a", "b", "c", "b"]
    gold = {"b"}
    assert first_gold_rank(ranked, gold) == 2
    assert [hit_at_k(ranked, gold, k) for k in (1, 2, 5)] == [False, True, True]
    assert reciprocal_rank(ranked, gold) == 0.5
    assert reciprocal_rank(ranked, {"z"}) == 0.0 and first_gold_rank([], gold) is None
    assert mean([1, 0, 0.5]) == pytest.approx(0.5) and mean([]) == 0.0


def test_gold_may_be_any_of_several_documents() -> None:
    assert first_gold_rank(["x", "d2", "d1"], {"d1", "d2"}) == 2


def test_question_file_is_valid() -> None:
    qs = load_questions()
    assert Counter(q.category for q in qs) == {"D": 4, "L": 2, "R": 7, "U": 2}
    assert len({q.id for q in qs}) == 15
    assert all(q.behavior == "answer" for q in qs if q.category != "U")
    assert [q.id for q in qs if not q.gold_source_ids] == ["U1", "U2"]
    assert len(retrieval_questions(qs)) == 13


def test_gold_sources_exist_in_the_committed_corpus() -> None:
    corpus = Path(__file__).resolve().parents[2] / "data" / "processed"
    ids = {p.stem for p in corpus.glob("*/*.md")}
    for q in load_questions():
        assert set(q.gold_source_ids) <= ids, q.id


def test_gold_number_keys_exist_in_stats() -> None:
    import json

    stats = json.loads(
        (Path(__file__).resolve().parents[2] / "data/processed/stats.json").read_text(
            encoding="utf-8"
        )
    )
    for q in load_questions():
        for name, key in q.gold_numbers_from.items():
            assert key in stats, f"{q.id}.{name}: {key}"


def test_build_variants() -> None:
    variants = build_variants([300, 600, 1000], 0.15, with_fixed=True)
    assert [v.label for v in variants] == [
        "markdown_recursive_300",
        "markdown_recursive_600",
        "markdown_recursive_1000",
        "fixed_600",
    ]
    assert variants[0].cfg.chunk_overlap == 45 and variants[2].cfg.chunk_overlap == 150


def test_cached_embedder_encodes_each_text_once() -> None:
    class Counting(FakeEmbedder):
        calls = 0

        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            Counting.calls += len(texts)
            return super().embed_passages(texts)

    emb = CachedEmbedder(Counting())
    first = emb.embed_passages(["a b", "c d", "a b"])
    again = emb.embed_passages(["a b", "e f"])
    assert Counting.calls == 3 and first[0] == first[2] == again[0]


def test_experiment_end_to_end_with_fakes(tmp_path: Path) -> None:
    docs: list[Document] = [GERMANY, FRANCE, LAW]
    q = Question(
        id="T1", category="D", behavior="answer", question="churn rate in Germany",
        gold_answer="x", gold_source_ids=["card_de"],
    )  # fmt: skip
    variants = build_variants([200, 400], 0.15, with_fixed=False)
    rows, details = [], []
    for v in variants:
        summary, results = evaluate_variant(v, docs, [q], CachedEmbedder(FakeEmbedder()), tmp_path)
        rows.append(summary)
        details.extend(results)
    assert all(r["hit@1"] == 1.0 and r["mrr"] == 1.0 for r in rows)
    assert rows[0]["chunks_total"] >= rows[1]["chunks_total"] >= 3
    write_csv(tmp_path / "r.csv", rows)
    write_csv(tmp_path / "d.csv", detail_rows(details))
    plot(rows, tmp_path / "p.png")
    with (tmp_path / "r.csv").open(encoding="utf-8") as fh:
        assert [r["chunk_size"] for r in csv.DictReader(fh)] == ["200", "400"]
    assert (tmp_path / "p.png").stat().st_size > 5000
    assert ChunkingSettings(chunk_size=200, chunk_overlap=30).chunk_size == 200
