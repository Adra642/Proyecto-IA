"""Chunk-size experiment: retrieval quality of the same corpus chunked at several sizes.

For each size the whole corpus is chunked and embedded into a temporary Chroma index; the
evaluation questions are then run as retrieval-only queries and scored against
``gold_source_ids`` (hit@1/3/5 and MRR at chunk level). Atomic documents (segment cards,
macro notes) are never split, so the size only affects the other documents.

Usage::

    uv run python -m eval.chunking_experiment [--sizes 300 600 1000] [--with-fixed]
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.corpus import load_documents
from core.logging_setup import get_logger, setup_logging
from core.models import Chunk, Document
from core.settings import ChunkingSettings, Settings, load_settings
from eval.questions import Question, load_questions, retrieval_questions
from eval.retrieval_metrics import first_gold_rank, hit_at_k, mean, reciprocal_rank
from factory import Container
from rag.chunking import build_chunker
from rag.embeddings import Embedder
from rag.vectorstore import VectorStoreRepository, open_persistent_client

log = get_logger(__name__)

KS = (1, 3, 5)
OUT_DIR = Path(__file__).parent


class CachedEmbedder:
    """Remembers embeddings by text: identical chunks (atomic documents) are encoded once."""

    def __init__(self, inner: Embedder) -> None:
        """Args: inner: the real embedder."""
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages, reusing earlier results."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._inner.embed_passages(missing), strict=True))
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query (not cached)."""
        return self._inner.embed_query(text)


@dataclass
class Variant:
    """One chunking configuration under test."""

    label: str
    cfg: ChunkingSettings


@dataclass
class QuestionResult:
    """Retrieval outcome of one question under one variant."""

    variant: str
    question_id: str
    rank: int | None
    hits: dict[int, bool]
    rr: float
    context_chars: int


def build_variants(sizes: Sequence[int], ratio: float, with_fixed: bool) -> list[Variant]:
    """Markdown+recursive at each size (and fixed-size at the middle size if requested)."""
    variants = [
        Variant(
            f"markdown_recursive_{s}",
            ChunkingSettings(chunk_size=s, chunk_overlap=round(s * ratio)),
        )
        for s in sizes
    ]
    if with_fixed:
        mid = sorted(sizes)[len(sizes) // 2]
        cfg = ChunkingSettings(strategy="fixed", chunk_size=mid, chunk_overlap=round(mid * ratio))
        variants.append(Variant(f"fixed_{mid}", cfg))
    return variants


def evaluate_variant(
    variant: Variant,
    docs: Sequence[Document],
    questions: Sequence[Question],
    embedder: Embedder,
    workdir: Path,
) -> tuple[dict[str, Any], list[QuestionResult]]:
    """Index the corpus with ``variant`` and score the retrieval questions."""
    chunker = build_chunker(variant.cfg)
    per_doc: list[list[Chunk]] = [chunker.chunk(d) for d in docs]
    chunks = [c for group in per_doc for c in group]
    repo = VectorStoreRepository(
        open_persistent_client(workdir / variant.label), "experiment", embedder
    )
    start = time.monotonic()
    for group in per_doc:
        repo.add(group)
    embed_seconds = time.monotonic() - start
    results: list[QuestionResult] = []
    for q in questions:
        hits = repo.query(q.question, k=max(KS))
        ranked = [h.doc_id for h in hits]
        gold = set(q.gold_source_ids)
        results.append(
            QuestionResult(
                variant=variant.label,
                question_id=q.id,
                rank=first_gold_rank(ranked, gold),
                hits={k: hit_at_k(ranked, gold, k) for k in KS},
                rr=reciprocal_rank(ranked, gold),
                context_chars=sum(len(h.text) for h in hits),
            )
        )
    summary: dict[str, Any] = {
        "variant": variant.label,
        "strategy": variant.cfg.strategy,
        "chunk_size": variant.cfg.chunk_size,
        "chunk_overlap": variant.cfg.chunk_overlap,
        "chunks_total": len(chunks),
        "avg_chunk_chars": round(mean([len(c.text) for c in chunks]), 1),
        **{f"hit@{k}": round(mean([float(r.hits[k]) for r in results]), 3) for k in KS},
        "mrr": round(mean([r.rr for r in results]), 3),
        "avg_context_chars_top5": round(mean([float(r.context_chars) for r in results]), 1),
        "questions": len(results),
        "embed_seconds": round(embed_seconds, 1),
    }
    return summary, results


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write dictionaries as CSV (UTF-8, LF line endings)."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def detail_rows(results: Sequence[QuestionResult]) -> list[dict[str, Any]]:
    """Per-question rows for the CSV with details."""
    return [
        {
            "variant": r.variant,
            "question_id": r.question_id,
            "first_gold_rank": r.rank if r.rank is not None else "",
            **{f"hit@{k}": int(r.hits[k]) for k in KS},
            "reciprocal_rank": round(r.rr, 3),
            "context_chars_top5": r.context_chars,
        }
        for r in results
    ]


def plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Line chart of hit@k and MRR against chunk size (markdown+recursive variants)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = sorted(
        (r for r in rows if r["strategy"] == "markdown_recursive"), key=lambda r: r["chunk_size"]
    )
    sizes = [str(r["chunk_size"]) for r in data]
    series = [  # blue, orange, aqua from the validated palette; MRR in neutral ink, dashed
        ("hit@1", "#2a78d6", "-"),
        ("hit@3", "#eb6834", "-"),
        ("hit@5", "#1baf7a", "-"),
        ("mrr", "#52514e", "--"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    # Series that coincide (e.g. hit@3 = hit@5 = 1.0) are drawn nested (wider line underneath)
    # and share one end label, so none is hidden or unreadable.
    widths = {"hit@1": 2, "hit@3": 5, "hit@5": 2, "mrr": 2}
    end_labels: dict[float, list[str]] = {}
    for key, color, style in series:
        values = [r[key] for r in data]
        ax.plot(sizes, values, style, color=color, linewidth=widths[key], marker="o",
                markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2)  # fmt: skip
        end_labels.setdefault(values[-1], []).append("MRR" if key == "mrr" else key)
    for y, names in end_labels.items():
        ax.annotate(" = ".join(names), (sizes[-1], y), xytext=(10, 0), textcoords="offset points",
                    va="center", color="#0b0b0b", fontsize=10)  # fmt: skip
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Chunk size (characters)", color="#52514e")
    ax.set_ylabel("Share of questions (MRR: mean reciprocal rank)", color="#52514e")
    ax.set_title("Retrieval quality by chunk size", loc="left", color="#0b0b0b", fontsize=13)
    ax.grid(axis="y", color="#e6e5e1", linewidth=1)
    ax.tick_params(colors="#52514e")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(
    settings: Settings, sizes: Sequence[int], ratio: float, with_fixed: bool, out_dir: Path
) -> list[dict[str, Any]]:
    """Run every variant and write the CSVs and the plot into ``out_dir``."""
    docs = load_documents(settings.resolve(settings.paths.processed_dir))
    questions = retrieval_questions(load_questions())
    embedder = CachedEmbedder(Container(settings).embedder())
    summaries: list[dict[str, Any]] = []
    details: list[QuestionResult] = []
    with tempfile.TemporaryDirectory(prefix="chunk_exp_") as tmp:
        for variant in build_variants(sizes, ratio, with_fixed):
            log.info("Variant %s", variant.label)
            summary, results = evaluate_variant(variant, docs, questions, embedder, Path(tmp))
            log.info(
                "  chunks=%d hit@1=%.2f hit@3=%.2f hit@5=%.2f mrr=%.3f",
                summary["chunks_total"],
                summary["hit@1"],
                summary["hit@3"],
                summary["hit@5"],
                summary["mrr"],
            )
            summaries.append(summary)
            details.extend(results)
            # Save after every variant so an interrupted run keeps what it already measured.
            write_csv(out_dir / "chunking_results.csv", summaries)
            write_csv(out_dir / "chunking_detail.csv", detail_rows(details))
    plot(summaries, out_dir / "chunking_results.png")
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Chunk-size retrieval experiment.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 600, 1000])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--with-fixed", action="store_true", help="also test size-only chunking")
    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.resolve(settings.paths.logs_dir))
    run(settings, args.sizes, args.overlap_ratio, args.with_fixed, OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
