"""The two answering modes compared in the evaluation: plain LLM and RAG."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from core.logging_setup import get_logger
from core.models import RagAnswer, RetrievedChunk
from rag.llm import LLMClient
from rag.prompts import (
    NO_CONTEXT,
    REFUSAL_PII,
    build_no_rag_prompt,
    build_rag_prompt,
)
from rag.retrievers import Retriever

log = get_logger(__name__)

_NAME_WORDS = r"surname|last\s*name|family\s*name|apellido|nom\s+de\s+famille|nachname"
_PII_PATTERNS = (
    # asks for a person's surname / last name
    re.compile(rf"\b({_NAME_WORDS})\b", re.IGNORECASE),
    # asks about a specific customer id / row number
    re.compile(
        r"\b(customer\s*id|customerid|row\s*number|rownumber)\b\D{0,12}\d{3,}", re.IGNORECASE
    ),
    re.compile(r"\bcustomers?\s+(no\.?\s*)?\d{6,}\b", re.IGNORECASE),
)
_CITATION = re.compile(r"\[([A-Za-z0-9_.\-]+)(?:#\d+)?\]")


def is_pii_request(question: str) -> bool:
    """True when the question asks for a person's surname or a specific customer record."""
    return any(p.search(question) for p in _PII_PATTERNS)


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Document ids cited as ``[doc_id]`` in the answer that were really retrieved (in order)."""
    retrieved = {c.doc_id for c in chunks}
    seen: list[str] = []
    for cited in _CITATION.findall(answer):
        if cited in retrieved and cited not in seen:
            seen.append(cited)
    return seen


class RagChain:
    """Answers a question with or without retrieval, with the same LLM in both modes."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        top_k: int,
        model_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args: llm/retriever: collaborators; top_k: chunks per question; model_id: for logs."""
        self._llm = llm
        self._retriever = retriever
        self._top_k = top_k
        self._model_id = model_id
        self._clock = clock

    def answer_no_rag(self, question: str) -> RagAnswer:
        """Baseline: the plain LLM, no retrieved context."""
        start = self._clock()
        text = self._llm.invoke(build_no_rag_prompt(question))
        return RagAnswer(
            question=question,
            mode="no_rag",
            answer=text.strip(),
            latency_s=self._clock() - start,
            model=self._model_id,
        )

    def answer_rag(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> RagAnswer:
        """Retrieve, then answer only from the retrieved context, citing ``[doc_id]``.

        Personal-data requests are refused and empty retrievals answered "I do not know"
        without calling the LLM.
        """
        start = self._clock()
        if is_pii_request(question):
            return self._fixed(question, REFUSAL_PII, [], start, refused=True)
        chunks = self._retriever.retrieve(question, k or self._top_k, where)
        if not chunks:
            return self._fixed(question, NO_CONTEXT, [], start)
        text = self._llm.invoke(build_rag_prompt(question, chunks)).strip()
        return RagAnswer(
            question=question,
            mode="rag",
            answer=text,
            chunks=chunks,
            cited_ids=extract_citations(text, chunks),
            latency_s=self._clock() - start,
            model=self._model_id,
        )

    def _fixed(
        self,
        question: str,
        text: str,
        chunks: list[RetrievedChunk],
        start: float,
        refused: bool = False,
    ) -> RagAnswer:
        return RagAnswer(
            question=question,
            mode="rag",
            answer=text,
            chunks=chunks,
            latency_s=self._clock() - start,
            model=self._model_id,
            refused=refused,
        )
