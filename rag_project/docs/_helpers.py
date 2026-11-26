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


def num_es(value: str | float, decimals: int = 2) -> str:
    """Format a number the Spanish way (decimal comma, dot for thousands); ``—`` if empty."""
    if value in ("", None):
        return "—"
    return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def config_es(name: str) -> str:
    """Spanish label for a retrieval configuration (``+hybrid`` becomes ``+ híbrida``)."""
    for en, es in (
        ("+filters+rerank blend", "+ filtros + reranking mixto"),
        ("+rerank blend", "+ reranking mixto"),
        ("+filters+rerank", "+ filtros + reranking"),
        ("+rerank", "+ reranking"),
        ("+filters", "+ filtros"),
        ("+hybrid", "+ híbrida"),
        ("rerank ", "reranking "),
        ("(hybrid)", "(híbrida)"),
    ):
        name = name.replace(en, es)
    return "Vectorial" if name == "vector" else name


COUNTRY_ES = {"France": "Francia", "Germany": "Alemania", "Spain": "España"}

_TYPE_ES = {"integer": "entero", "decimal": "decimal", "category": "categoría"}

_MEANING_ES = {
    "CreditScore": "Puntuación crediticia del cliente (más alta es mejor).",
    "Geography": "País del cliente: Francia, Alemania o España.",
    "Gender": "Género del cliente: mujer u hombre.",
    "Age": "Edad en años.",
    "Tenure": "Años que el cliente lleva en el banco.",
    "Balance": "Saldo de la cuenta (unidades monetarias del dataset).",
    "NumOfProducts": "Número de productos bancarios contratados (de 1 a 4).",
    "HasCrCard": "1 si el cliente tiene tarjeta de crédito, 0 si no.",
    "IsActiveMember": "1 si el cliente es un miembro activo, 0 si no.",
    "EstimatedSalary": "Salario anual estimado.",
    "Exited": "Objetivo: 1 si el cliente dejó el banco (churn), 0 si no.",
    "Complain": (
        "1 si el cliente presentó una queja, 0 si no. Atención: refleja casi a la perfección "
        "*Exited* (fuga de información del objetivo), por lo que no se crean fichas de segmento."
    ),
    "Satisfaction Score": "Puntuación de satisfacción del cliente, de 1 (baja) a 5 (alta).",
    "Card Type": "Categoría de la tarjeta: DIAMOND, GOLD, PLATINUM o SILVER.",
    "Point Earned": "Puntos de fidelidad acumulados por el cliente.",
}


def dictionary_rows_es(rel: str) -> list[dict[str, object]]:
    """Rows of the generated data dictionary with Spanish type and meaning columns."""
    rows = []
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Column":
            continue
        rng = cells[2].replace(" to ", " a ").replace("Female, Male", "mujer, hombre")
        rng = rng.replace("France, Germany, Spain", "Francia, Alemania, España")
        rows.append(
            {
                "Columna": cells[0],
                "Tipo": _TYPE_ES.get(cells[1], cells[1]),
                "Rango o valores": rng,
                "Significado": _MEANING_ES.get(cells[0], cells[3]),
            }
        )
    return rows


_SOURCE_ES = {
    "A1/A2": (
        "Kaggle Bank Customer Churn (CSV) y documentos analíticos derivados",
        "Ver la página del dataset en Kaggle (licencia sin verificar)",
    ),
    "A3": (
        "Artículos de arXiv sobre predicción de churn (resumen y secciones clave)",
        "arXiv: derechos de autor de los autores; metadatos, resumen y extractos breves con enlace",
    ),
    "A4": ("Páginas conceptuales de Wikipedia", "CC BY-SA 4.0"),
    "A5": (
        "Portal de datos del BCE (tipos oficiales, inflación, tipos bancarios)",
        "Datos estadísticos del BCE: reutilización permitida citando la fuente",
    ),
    "A6": (
        "Textos legales de la UE (EUR-Lex / Cellar)",
        "Derecho de la UE: reutilización autorizada con atribución (Decisión 2011/833/UE)",
    ),
    "B1": (
        "Banco de España, Portal del Cliente Bancario",
        "Contenido web de una institución pública; extractos para investigación con cita (condiciones de reutilización sin verificar)",  # noqa: E501
    ),
    "B2": (
        "API de Eurostat (indicadores de FR, DE y ES)",
        "Eurostat: CC BY 4.0 (reutilización citando la fuente)",
    ),
    "B3": (
        "API del Banco Mundial (indicadores de FR, DE y ES)",
        "Datos abiertos del Banco Mundial: CC BY 4.0",
    ),
    "B4": (
        "Notas de prensa del BCE (canal RSS)",
        "Publicaciones del BCE: reutilización permitida citando la fuente",
    ),
    "B5": (
        "Encuestas de hogares del BCE (riqueza HFCS, pagos SPACE)",
        "Publicaciones del BCE: reutilización permitida citando la fuente",
    ),
    "B6": (
        "Páginas de protección del consumidor de la EBA, directrices e Informe de tendencias del consumidor",  # noqa: E501
        "Publicaciones de la EBA: reutilización autorizada citando la fuente",
    ),
    "B7": (
        "Sistemas de información crediticia (CIRBE, FICP)",
        "Contenido web de una institución pública; extractos para investigación con cita (condiciones de reutilización sin verificar)",  # noqa: E501
    ),
    "B8": (
        "Páginas y informes para el consumidor de Banque de France, BaFin, Bundesbank y Verbraucherzentrale",  # noqa: E501
        "Comisión Europea: reutilización autorizada citando la fuente (Decisión 2011/833/UE); contenido web de instituciones públicas, extractos con cita (condiciones sin verificar)",  # noqa: E501
    ),
}
_LANG_ES = {"en": "inglés", "es": "español", "fr": "francés", "de": "alemán"}


def sources_rows_es(rel: str) -> list[dict[str, str]]:
    """Rows of ``data/SOURCES.md`` translated to Spanish (name, formats, languages, license)."""
    rows = []
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("|---") or "Plan ID" in line:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        name, lic = _SOURCE_ES.get(c[0], (c[1], c[6]))
        langs = ", ".join(_LANG_ES.get(x.strip(), x.strip()) for x in c[4].split(","))
        formats = c[3].replace("generated", "generado")
        rows.append(
            {
                "ID": c[0],
                "Fuente": name,
                "Documentos": c[2],
                "Formatos": formats,
                "Idiomas": langs,
                "Licencia": lic,
            }
        )
    return rows
