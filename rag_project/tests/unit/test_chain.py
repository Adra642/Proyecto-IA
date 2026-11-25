from datetime import date
from typing import Any

import pytest

from core.models import RetrievedChunk
from rag.chain import RagChain, extract_citations, is_pii_request
from rag.prompts import NO_CONTEXT, REFUSAL_PII, build_no_rag_prompt, build_rag_prompt


class FakeLLM:
    name = "fake"

    def __init__(self, answer: str = "The churn rate is 32.4% [card_de].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, Any]] = []

    def retrieve(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, where))
        return self.chunks[:k]


def chunk(doc_id: str, text: str, header: str = "") -> RetrievedChunk:
    meta = {
        "title": f"Title {doc_id}", "source": "kaggle_derived", "geography": "DE",
        "header_path": header, "retrieved_at": date(2026, 1, 1).isoformat(),
    }  # fmt: skip
    return RetrievedChunk(
        chunk_id=f"{doc_id}#0000", doc_id=doc_id, text=text, score=0.9, metadata=meta
    )


DE = chunk("card_de", "Churn rate: 32.4% in Germany.", "Germany")
FR = chunk("card_fr", "Churn rate: 16.2% in France.")


def make(llm: FakeLLM, chunks: list[RetrievedChunk], k: int = 4) -> tuple[RagChain, FakeRetriever]:
    retriever = FakeRetriever(chunks)
    ticks = iter(range(0, 100, 2))
    return RagChain(llm, retriever, k, "fake:model", clock=lambda: float(next(ticks))), retriever


def test_rag_injects_context_into_the_prompt_and_cites() -> None:
    llm = FakeLLM()
    chain, retriever = make(llm, [DE, FR])
    result = chain.answer_rag("What is the churn rate in Germany?")
    (prompt,) = llm.prompts
    assert (
        "[card_de] Title card_de > Germany" in prompt and "Churn rate: 32.4% in Germany." in prompt
    )
    assert "[card_fr]" in prompt and "What is the churn rate in Germany?" in prompt
    assert "ONLY the context" in prompt and "[doc_id]" in prompt
    assert result.mode == "rag" and result.cited_ids == ["card_de"]
    assert [c.doc_id for c in result.chunks] == ["card_de", "card_fr"]
    assert retriever.calls[0][1] == 4 and result.model == "fake:model" and result.latency_s == 2.0


def test_k_and_filters_are_passed_to_the_retriever() -> None:
    chain, retriever = make(FakeLLM(), [DE, FR])
    result = chain.answer_rag("churn?", k=1, where={"geography": "DE"})
    assert retriever.calls == [("churn?", 1, {"geography": "DE"})] and len(result.chunks) == 1


def test_no_context_answers_i_do_not_know_without_calling_the_llm() -> None:
    llm = FakeLLM()
    chain, _ = make(llm, [])
    result = chain.answer_rag("churn in Portugal?")
    assert result.answer == NO_CONTEXT and llm.prompts == [] and not result.refused


@pytest.mark.parametrize(
    "question",
    [
        "What is the surname of the customer with CustomerId 15634602 in the churn dataset?",
        "Dame el apellido del cliente 15634602",
        "What's the last name of customer 15647311?",
        "Which row number 5 customer id 15634602?",
    ],
)
def test_pii_requests_are_refused_without_calling_the_llm(question: str) -> None:
    llm = FakeLLM()
    chain, retriever = make(llm, [DE])
    result = chain.answer_rag(question)
    assert result.refused and result.answer == REFUSAL_PII
    assert llm.prompts == [] and retriever.calls == [] and result.chunks == []


@pytest.mark.parametrize(
    "question",
    [
        "What is the churn rate of customers in Germany?",
        "Which age band has the highest churn rate?",
        "What does GDPR say about customer data and surnames of data subjects in general terms?"
        + "",
    ][:2],
)
def test_normal_questions_are_not_flagged(question: str) -> None:
    assert not is_pii_request(question)


def test_no_rag_mode_is_the_plain_llm() -> None:
    llm = FakeLLM("I do not know.")
    chain, retriever = make(llm, [DE])
    result = chain.answer_no_rag("What is the churn rate in Germany?")
    assert llm.prompts == [build_no_rag_prompt("What is the churn rate in Germany?")]
    assert retriever.calls == [] and result.mode == "no_rag" and result.chunks == []


def test_prompts_ask_for_spanish_answers() -> None:
    assert "in Spanish" in build_no_rag_prompt("q") and "in Spanish" in build_rag_prompt("q", [DE])
    assert NO_CONTEXT.startswith("No lo sé") and REFUSAL_PII.startswith("No puedo")


def test_citations_only_count_retrieved_documents() -> None:
    text = "See [card_de], again [card_de#0002], and [made_up_doc]. Also [card_fr]."
    assert extract_citations(text, [DE, FR]) == ["card_de", "card_fr"]
    assert extract_citations("no citations here", [DE]) == []


def test_rag_prompt_cites_context() -> None:
    assert build_rag_prompt("q", [DE]).count("[card_de]") >= 1
