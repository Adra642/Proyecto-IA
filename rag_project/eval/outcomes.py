"""Classify every evaluation answer as correct, partial, hallucination or unknown, and plot it.

Rules (deterministic, no LLM):

* Questions with gold figures (or, for L1, L2 and R7, gold key terms): the share ``m`` of them
  found in the answer decides. ``m == 1`` is *correct*, unless the answer also admits it does not
  know (then *partial*). ``0 < m < 1`` is *partial*. ``m == 0`` is *unknown* if the answer admits
  it does not know, otherwise a *hallucination*.
* Behavioural questions (U1, U2): *correct* if the model abstained or refused, else *hallucination*.

Usage::

    uv run python -m eval.outcomes     # eval/outcomes.csv and eval/outcomes.png
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.questions import Question, load_questions  # noqa: E402
from eval.score import auto_score, load_results, says_unknown  # noqa: E402

OUT_DIR = Path(__file__).parent
MODEL_RESULTS = {
    "gemini": OUT_DIR / "results.json",
    "groq": OUT_DIR / "models" / "groq" / "results.json",
    "deepseek": OUT_DIR / "models" / "deepseek" / "results.json",
}
MODEL_LABEL = {
    "gemini": "Gemini (gemini-3.5-flash-lite)",
    "groq": "Groq (gpt-oss-120b)",
    "deepseek": "DeepSeek (deepseek-flash)",
}
MODES = ("no_rag", "rag", "rag_advanced")
MODE_LABEL = {"no_rag": "Sin RAG", "rag": "RAG", "rag_advanced": "RAG avanzado"}
LABELS = ("correct", "partial", "hallucination", "unknown")
LABEL_ES = {
    "correct": "Correcta",
    "partial": "Parcialmente correcta",
    "hallucination": "Alucinación",
    "unknown": "No sabe",
}
COLORS = {
    "correct": "#2a9d5c",
    "partial": "#e9b44c",
    "hallucination": "#d1495b",
    "unknown": "#8d99ae",
}
# Questions without gold figures: key terms (regex, all case-insensitive) the answer must contain.
KEY_TERMS: dict[str, tuple[str, ...]] = {
    "L1": (r"\bANN\b|red neuronal|neural network", r"cinco|five|\b5\b"),
    "L2": (r"FT-?Transformer", r"stacking|apilamiento"),
    "R7": (r"fraud|fraude", r"endeudamiento|indebtedness", r"de-?risking"),
}


def share_found(answer: str, q: Question, auto_match: Any) -> float | None:
    """Share of gold figures or key terms present in ``answer`` (``None`` if neither exists)."""
    if auto_match != "":
        return float(auto_match)
    terms = KEY_TERMS.get(q.id)
    if not terms:
        return None
    return sum(bool(re.search(t, answer, re.IGNORECASE)) for t in terms) / len(terms)


def classify(answer: str, q: Question, auto: dict[str, Any]) -> str:
    """Return one of ``LABELS`` for one answer."""
    if q.behavior != "answer":
        return "correct" if auto["behavior_ok"] == 1 else "hallucination"
    m = share_found(answer, q, auto["numeric_match"])
    if m is None:
        raise ValueError(f"no gold figures or key terms for {q.id}")
    if m >= 1.0 and not says_unknown(answer):  # "I don't know, but..." is not a correct answer
        return "correct"
    if m > 0:
        return "partial"
    return "unknown" if says_unknown(answer) else "hallucination"


def main() -> int:
    """Write ``outcomes.csv`` and the chart ``outcomes.png`` (one panel per available model)."""
    questions = {q.id: q for q in load_questions()}
    rows: list[dict[str, str]] = []
    for model, path in MODEL_RESULTS.items():
        if not path.exists():
            continue
        for r in load_results(path):
            if r["error"]:  # a failed call is not an answer: leave it out of the counts
                continue
            q = questions[r["id"]]
            auto = auto_score(r, q, None)
            outcome = classify(r["answer"], q, auto)
            rows.append({"model": model, "id": q.id, "mode": r["mode"], "outcome": outcome})
    fields = ["model", "id", "mode", "outcome"]
    with (OUT_DIR / "outcomes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    models = [m for m in MODEL_RESULTS if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.3 * len(models) + 0.8), sharex=True)
    for ax, model in zip(axes if len(models) > 1 else [axes], models, strict=True):
        counts = {
            m: Counter(r["outcome"] for r in rows if r["model"] == model and r["mode"] == m)
            for m in MODES
        }
        left = [0] * len(MODES)
        for label in LABELS:
            vals = [counts[m][label] for m in MODES]
            bars = ax.barh([MODE_LABEL[m] for m in MODES], vals, left=left, color=COLORS[label],
                           label=LABEL_ES[label])  # fmt: skip
            for bar, v, lft in zip(bars, vals, left, strict=True):
                if v:
                    ax.text(lft + v / 2, bar.get_y() + bar.get_height() / 2, str(v),
                            ha="center", va="center", color="white", fontweight="bold")  # fmt: skip
            left = [lft + v for lft, v in zip(left, vals, strict=True)]
        ax.invert_yaxis()
        ax.set_title(MODEL_LABEL[model], loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        print(model, {m: {k: counts[m][k] for k in LABELS} for m in MODES})
    axes_last = axes[-1] if len(models) > 1 else axes
    axes_last.set_xlabel("Número de preguntas (15 por modo)")
    handles, labels = axes_last.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_DIR / "outcomes.png", dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
