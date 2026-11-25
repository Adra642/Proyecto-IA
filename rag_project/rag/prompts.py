"""Prompt templates for the No-RAG baseline and the RAG chain."""

from __future__ import annotations

from core.models import RetrievedChunk

NO_RAG_TEMPLATE = (
    "Answer the question concisely. Always answer in Spanish, whatever the language of the "
    "question. If you do not know the answer or cannot verify it, say so instead of guessing."
    "\n\nQuestion: {question}"
)

RAG_TEMPLATE = """You answer questions about customer churn and retail banking in France, Germany \
and Spain using ONLY the context below.

Rules:
- Use only information that is in the context. If the context does not contain the answer, \
say clearly that you do not know because the context is insufficient. Do not guess and do not \
use outside knowledge.
- Cite the sources that support each statement as [doc_id], using the ids shown in the context.
- Always answer in Spanish, whatever the language of the question or of the context.
- Never reveal personal identifiers such as customer ids or surnames; they are not part of the \
knowledge base.
- Be concise and give exact figures as they appear in the context.

Context:
{context}

Question: {question}

Answer:"""

REFUSAL_PII = (
    "No puedo compartir identificadores personales como apellidos o identificadores de cliente: "
    "se eliminaron de la base de conocimiento antes de indexarla y este sistema solo responde "
    "a partir de documentos agregados y no personales."
)
NO_CONTEXT = "No lo sé: no se encontró información relevante en la base de conocimiento."


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered blocks headed by their ``[doc_id]``."""
    blocks = []
    for chunk in chunks:
        meta = chunk.metadata
        header_path = f" > {meta['header_path']}" if meta.get("header_path") else ""
        head = (
            f"[{chunk.doc_id}] {meta['title']}{header_path} ({meta['source']}, {meta['geography']})"
        )
        blocks.append(f"{head}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Fill the RAG template with the question and the formatted context."""
    return RAG_TEMPLATE.format(context=format_context(chunks), question=question)


def build_no_rag_prompt(question: str) -> str:
    """Fill the baseline template (same model, no retrieved context)."""
    return NO_RAG_TEMPLATE.format(question=question)
