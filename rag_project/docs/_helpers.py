"""Helpers for the Quarto pages: live data from the repo and exact code excerpts.

Excerpts are cut from the source files with ``ast`` (nothing is imported), so the site always shows
the code that is really in the repository and does not need the heavy dependencies to render.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from IPython.display import Markdown, display

ROOT = Path(__file__).resolve().parents[1]


def read_csv(rel: str) -> list[dict[str, str]]:
    """Rows of a CSV under ``rag_project/`` (empty list if the file does not exist)."""
    path = ROOT / rel
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(rel: str):  # type: ignore[no-untyped-def]
    """Parsed JSON under ``rag_project/``."""
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def md_table(rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    """Display a list of dicts as a Markdown table."""
    if not rows:
        display(Markdown("_Sin datos._"))
        return
    cols = columns or list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in cols) + " |"
        )
    display(Markdown("\n".join(lines)))


def show_source(rel: str, name: str, max_lines: int = 40) -> None:
    """Display the source of a function or ``Class.method`` as a Python code block."""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    text = (ROOT / rel).read_text(encoding="utf-8")
    node: ast.AST | None = tree
    for part in name.split("."):
        node = next(
            (
                n
                for n in ast.walk(node)  # type: ignore[arg-type]
                if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == part
            ),
            None,
        )
        if node is None:
            raise KeyError(f"{name} not found in {rel}")
    lines = (ast.get_source_segment(text, node, padded=True) or "").splitlines()  # type: ignore[arg-type]
    lines = _dedent(lines)[:max_lines]
    display(Markdown("```python\n" + "\n".join(lines) + "\n```"))


def show_file(rel: str, lang: str = "dockerfile", max_lines: int = 60) -> None:
    """Display a file (first ``max_lines`` lines) as a code block."""
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()[:max_lines]
    display(Markdown(f"```{lang}\n" + "\n".join(lines) + "\n```"))


def _dedent(lines: list[str]) -> list[str]:
    indent = min((len(x) - len(x.lstrip()) for x in lines if x.strip()), default=0)
    return [x[indent:] for x in lines]


def pct(value: str | float, digits: int = 1) -> str:
    """Format a 0-1 ratio (or a string of it) as a percentage."""
    return f"{float(value) * 100:.{digits}f} %"
