"""Stage 5: BM25, filter matching, fusion, reranking, router and the advanced chain (offline)."""

from __future__ import annotations

from typing import Any

import pytest

from core.models import RetrievedChunk
from core.settings import RetrievalSettings
from eval.ablation import Config, evaluate, filtered
from eval.questions import Question
from rag.chain import RagChain
from rag.lexical import BM25Retriever, matches_where, tokenize
from rag.prompts import NO_CONTEXT, REFUSAL_PII
from rag.rerank import rerank_text
from rag.retrievers import HybridRetriever, RerankingRetriever
from rag.router import Route, RuleRouter, all_of, where_from_filters


def chunk(cid: str, text: str, **meta: str) -> RetrievedChunk:
    base = {
        "title": f"T {cid}", "header_path": "", "source": "test", "source_type": "paper",
        "geography": "EU",
    }  # fmt: skip
    return RetrievedChunk(
        chunk_id=cid, doc_id=cid.split("#")[0], text=text, score=0.0, metadata={**base, **meta}
    )


CORPUS = [
    chunk("a#0", "directive 2015/751 interchange fee caps", source_type="regulation"),
    chunk("b#0", "churn rate in Germany is high", source_type="dataset_stats", geography="DE"),
    chunk("c#0", "wikipedia article about customer churn", source_type="encyclopedia"),
    chunk("d#0", "tarifas de intercambio en España", source_type="regulator_guide", geography="ES"),
]


class ListRetriever:
    """Returns a fixed ranking (after applying ``where``), like a real retriever."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, Any]] = []

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, where))
        return [c for c in self.chunks if matches_where(c.metadata, where)][:k]


# --- lexical -----------------------------------------------------------------------------------


def test_tokenize_keeps_accents_and_lowercases() -> None:
    assert tokenize("Tarifas de España, 2015/751!") == ["tarifas", "de", "españa", "2015", "751"]


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        (None, True),
        ({"geography": "DE"}, True),
        ({"geography": "FR"}, False),
        ({"geography": {"$in": ["FR", "DE"]}}, True),
        ({"$and": [{"geography": "DE"}, {"source_type": {"$in": ["regulation"]}}]}, False),
        ({"$or": [{"geography": "FR"}, {"source_type": "paper"}]}, True),
    ],
)
def test_matches_where(where: dict[str, Any] | None, expected: bool) -> None:
    meta = {"geography": "DE", "source_type": "paper"}
    assert matches_where(meta, where) is expected


def test_bm25_finds_exact_terms_and_applies_filters() -> None:
    bm25 = BM25Retriever(lambda: CORPUS)
    assert bm25.retrieve("2015/751 interchange", 3)[0].chunk_id == "a#0"
    only_de = bm25.retrieve("germany", 5, {"geography": "DE"})
    assert [c.chunk_id for c in only_de] == ["b#0"] and only_de[0].score > 0


def test_bm25_empty_cases() -> None:
    assert BM25Retriever(lambda: []).retrieve("anything", 3) == []
    assert BM25Retriever(lambda: CORPUS).retrieve("zzzz", 3) == []  # no matching token
    assert BM25Retriever(lambda: CORPUS).retrieve("???", 3) == []  # no tokens


# --- hybrid and rerank -------------------------------------------------------------------------


def test_hybrid_fuses_ranks_and_a_chunk_in_both_lists_wins() -> None:
    dense = ListRetriever([CORPUS[2], CORPUS[1], CORPUS[0]])
    lexical = ListRetriever([CORPUS[1], CORPUS[0], CORPUS[3]])
    hybrid = HybridRetriever(dense, lexical, candidates=10)
    ids = [c.chunk_id for c in hybrid.retrieve("q", 4)]
    assert ids[0] == "b#0"  # rank 2 in dense + rank 1 in lexical beats a dense-only rank 1
    assert set(ids) == {"a#0", "b#0", "c#0", "d#0"}
    assert hybrid.retrieve("q", 4)[0].score > hybrid.retrieve("q", 4)[-1].score


def test_hybrid_weights_and_depth_and_filter_are_forwarded() -> None:
    dense, lexical = ListRetriever(CORPUS), ListRetriever(CORPUS)
    lexical_only = HybridRetriever(dense, lexical, 20, vector_weight=0.0, bm25_weight=1.0)
    where = {"geography": "DE"}
    lexical_only.retrieve("q", 2, where)
    assert dense.calls == [("q", 20, where)] and lexical.calls == [("q", 20, where)]


class ReverseReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [c.model_copy(update={"score": float(i)}) for i, c in enumerate(reversed(chunks))]


def test_reranking_retriever_reorders_a_wider_pool_and_keeps_top_k() -> None:
    base = ListRetriever(CORPUS)
    out = RerankingRetriever(base, ReverseReranker(), candidates=4).retrieve("q", 2)
    assert [c.chunk_id for c in out] == ["d#0", "c#0"]
    assert base.calls[0][1] == 4  # asked the base for the candidate pool, not just k


def test_rerank_text_includes_title_and_header() -> None:
    c = chunk("x#0", "body", title="Title", header_path="H1 > H2")
    assert rerank_text(c) == "Title > H1 > H2\nbody"


# --- router ------------------------------------------------------------------------------------


def test_router_small_talk_skips_retrieval() -> None:
    route = RuleRouter().route("Hola, ¿qué puedes hacer?")
    assert route.retrieve is False and route.where is None


@pytest.mark.parametrize(
    ("question", "types", "geo"),
    [
        (
            "What is the churn rate in Germany in the Kaggle dataset?",
            ["dataset_stats"],
            ["DE", "EU"],
        ),
        ("What caps does Regulation (EU) 2015/751 set?", ["regulation", "regulator_guide"], None),
        ("Which classifier did the arXiv study find best?", ["paper"], None),
        ("Eurostat internet banking share in Germany", ["macro_data", "press"], ["DE", "EU"]),
    ],
)
def test_router_picks_source_type_and_single_country(
    question: str, types: list[str], geo: list[str] | None
) -> None:
    route = RuleRouter().route(question)
    assert route.retrieve is True
    conds = route.where["$and"] if geo else [route.where]  # type: ignore[index]
    assert {"source_type": {"$in": types}} in conds
    assert ({"geography": {"$in": geo}} in conds) is bool(geo)


def test_router_no_geography_for_several_countries_and_no_rule_means_no_filter() -> None:
    both = RuleRouter().route("Kaggle dataset: churn in France versus Germany")
    assert both.where == {"source_type": {"$in": ["dataset_stats"]}}
    assert RuleRouter().route("Tell me about retention strategies") == Route(
        True, None, "no filter: searching the whole corpus"
    )


def test_where_helpers() -> None:
    assert all_of([]) is None
    assert all_of([{"a": 1}]) == {"a": 1}
    assert where_from_filters() is None
    assert where_from_filters(["paper"], [], ["en"]) == {
        "$and": [{"source_type": {"$in": ["paper"]}}, {"language": {"$in": ["en"]}}]
    }


# --- advanced chain ----------------------------------------------------------------------------


class FakeLLM:
    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Answer [b]."


def make_chain(
    retriever: ListRetriever, llm: FakeLLM | None = None
) -> tuple[RagChain, FakeLLM, ListRetriever]:
    llm = llm or FakeLLM()
    plain = ListRetriever([])
    chain = RagChain(llm, plain, 4, "fake:m", advanced_retriever=retriever, router=RuleRouter())
    return chain, llm, retriever


def test_advanced_uses_the_router_filter_and_reports_the_route() -> None:
    chain, llm, retriever = make_chain(ListRetriever(CORPUS))
    answer = chain.answer_advanced("Kaggle dataset churn in Germany")
    assert answer.mode == "rag_advanced" and "dataset_stats" in answer.route
    assert [c.doc_id for c in answer.chunks] == ["b"] and answer.cited_ids == ["b"]
    assert retriever.calls[0][2]["$and"][0] == {"source_type": {"$in": ["dataset_stats"]}}
    assert "churn rate in Germany" in llm.prompts[0]


def test_advanced_user_filter_overrides_the_router() -> None:
    chain, _, retriever = make_chain(ListRetriever(CORPUS))
    where = {"geography": "ES"}
    answer = chain.answer_advanced("Kaggle dataset churn in Germany", where=where)
    assert retriever.calls[0][2] == where and answer.route == "user filter"
    assert [c.doc_id for c in answer.chunks] == ["d"]


def test_advanced_repeats_without_filter_when_the_filter_finds_nothing() -> None:
    chain, _, retriever = make_chain(ListRetriever([CORPUS[2]]))  # only an encyclopedia chunk
    answer = chain.answer_advanced("Which classifier did the arXiv study find best?")
    assert [c[2] is None for c in retriever.calls] == [False, True]
    assert "repeated without filter" in answer.route and len(answer.chunks) == 1


def test_advanced_small_talk_pii_and_empty_do_not_retrieve_or_hallucinate() -> None:
    chain, llm, retriever = make_chain(ListRetriever(CORPUS))
    hello = chain.answer_advanced("hola")
    assert hello.chunks == [] and retriever.calls == [] and len(llm.prompts) == 1
    pii = chain.answer_advanced("What is the surname of customer 15634602?")
    assert pii.refused and pii.answer == REFUSAL_PII and len(llm.prompts) == 1
    empty_chain, empty_llm, _ = make_chain(ListRetriever([]))
    assert empty_chain.answer_advanced("some question").answer == NO_CONTEXT
    assert empty_llm.prompts == []


def test_advanced_requires_configuration() -> None:
    chain = RagChain(FakeLLM(), ListRetriever([]), 4, "m")
    with pytest.raises(RuntimeError, match="not configured"):
        chain.answer_advanced("q")


# --- settings and ablation ---------------------------------------------------------------------


def test_retrieval_weights_cannot_both_be_zero() -> None:
    with pytest.raises(ValueError, match="cannot both be zero"):
        RetrievalSettings(vector_weight=0.0, bm25_weight=0.0)


def test_ablation_scores_each_configuration_and_filtered_falls_back() -> None:
    q = Question(
        id="Q1", category="D", behavior="answer", question="Kaggle dataset churn Germany",
        gold_answer="x", gold_source_ids=["b"],
    )  # fmt: skip
    good = Config("good", lambda text: [CORPUS[1], CORPUS[0]])
    bad = Config("bad", lambda text: [CORPUS[0], CORPUS[2], CORPUS[3], CORPUS[1]])
    summary, detail = evaluate([good, bad], [q])
    assert (summary[0]["hit@1"], summary[0]["mrr"]) == (1.0, 1.0)
    assert (summary[1]["hit@1"], summary[1]["hit@4"], summary[1]["mrr"]) == (0.0, 1.0, 0.25)
    assert detail == [
        {"config": "good", "id": "Q1", "gold_rank": 1},
        {"config": "bad", "id": "Q1", "gold_rank": 4},
    ]
    only_other = ListRetriever([CORPUS[2]])
    run = filtered(only_other, RuleRouter())
    assert [c.doc_id for c in run("Kaggle dataset churn")] == ["c"]  # filter empty -> unfiltered
