"""Gradio Blocks interface (labels in Spanish). Built from an injected ``AppService``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from app.handlers import (
    MODE_ADVANCED,
    MODE_NO_RAG,
    MODE_RAG,
    MODES,
    AppService,
    format_sources,
    format_status,
)
from rag.router import where_from_filters

ABOUT_MD = """
### ¿Qué es esto?
Un sistema RAG sobre **retención de clientes y churn en la banca minorista europea
(Francia, Alemania, España)**. Compara el mismo modelo **sin recuperación** y **con
recuperación** de documentos.

**Datos:** estadísticas calculadas con pandas sobre el dataset *Bank Customer Churn* de Kaggle
(10 000 clientes), artículos de arXiv, Wikipedia, series del BCE, reglamentos de la UE,
Banco de España, Eurostat, Banco Mundial y organismos reguladores.
El detalle y las licencias están en el repositorio (`data/SOURCES.md`).

**Importante:** las respuestas con RAG se basan **solo en los documentos recuperados** y citan
su identificador entre corchetes. Si el contexto no basta, el sistema lo dice. Por privacidad,
no revela apellidos ni identificadores de clientes.
"""


SOURCE_TYPES = ["dataset_stats", "paper", "encyclopedia", "regulation", "macro_data",
                "regulator_guide", "press"]  # fmt: skip
GEOGRAPHIES = ["FR", "DE", "ES", "EU", "global", "US"]
LANGUAGES = ["en", "es", "fr", "de"]


def load_example_questions(path: Path) -> list[str]:
    """Return the example-button questions: the Spanish wording when the file has one."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(q.get("question_es") or q["question"]) for q in data]


def build_demo(service: AppService, examples: list[str]) -> gr.Blocks:
    """Build the Blocks app with a chat tab, a side-by-side comparison tab and an About tab."""
    demo = gr.Blocks(title="RAG: churn en banca europea")
    with demo:
        gr.Markdown("# RAG: churn y retención en banca europea")
        with gr.Tabs():
            _chat_tab(service, examples)
            _compare_tab(service, examples)
            with gr.Tab("Acerca de"):
                gr.Markdown(ABOUT_MD)
    return demo


def _chat_tab(service: AppService, examples: list[str]) -> None:
    with gr.Tab("Chat"):
        mode = gr.Radio(MODES, value=MODE_RAG, label="Modo")
        with gr.Accordion("Filtros (solo en RAG avanzado)", open=False):
            gr.Markdown(
                "Si no eliges ningún filtro, el modo avanzado decide solo (por ejemplo, "
                "preguntas sobre el dataset buscan en `dataset_stats`)."
            )
            f_types = gr.Dropdown(SOURCE_TYPES, multiselect=True, label="Tipo de fuente")
            f_geo = gr.Dropdown(GEOGRAPHIES, multiselect=True, label="Geografía")
            f_lang = gr.Dropdown(LANGUAGES, multiselect=True, label="Idioma")
        chatbot = gr.Chatbot(label="Conversación", height=380)
        status = gr.Markdown()
        with gr.Accordion("Fuentes", open=False):
            sources = gr.Markdown("_Aún no hay fuentes._")
        box = gr.Textbox(label="Pregunta", placeholder="Escribe tu pregunta…", lines=2)
        with gr.Row():
            send = gr.Button("Enviar", variant="primary")
            clear = gr.Button("Limpiar")
        gr.Examples(examples=examples, inputs=box, label="Preguntas de ejemplo")

        def respond(
            question: str,
            history: list[dict[str, str]] | None,
            selected: str,
            types: list[str] | None,
            geos: list[str] | None,
            langs: list[str] | None,
        ) -> tuple[list[dict[str, str]], str, str, str]:
            history = history or []
            where: dict[str, Any] | None = where_from_filters(types or [], geos or [], langs or [])
            text, answer = service.ask(question, selected, where)
            history = [
                *history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": text},
            ]
            src = (
                format_sources(answer.chunks, answer.cited_ids)
                if answer and answer.mode != "no_rag"
                else "_Sin fuentes en este modo._"
            )
            return history, src, format_status(answer), ""

        outputs = [chatbot, sources, status, box]
        inputs = [box, chatbot, mode, f_types, f_geo, f_lang]
        send.click(respond, inputs, outputs)
        box.submit(respond, inputs, outputs)
        clear.click(lambda: ([], "_Aún no hay fuentes._", "", ""), None, outputs)


def _compare_tab(service: AppService, examples: list[str]) -> None:
    with gr.Tab("Comparar"):
        gr.Markdown(
            "La misma pregunta y el mismo modelo, en tres modos: **sin RAG**, **RAG** "
            "(búsqueda vectorial) y **RAG avanzado** (búsqueda híbrida + filtros)."
        )
        box = gr.Textbox(label="Pregunta", lines=2)
        go = gr.Button("Comparar", variant="primary")
        modes = (MODE_NO_RAG, MODE_RAG, MODE_ADVANCED)
        answers: list[gr.Markdown] = []
        statuses: list[gr.Markdown] = []
        with gr.Row():
            for title in modes:
                with gr.Column():
                    gr.Markdown(f"### {title}")
                    answers.append(gr.Markdown())
                    statuses.append(gr.Markdown())
        with gr.Accordion("Fuentes (RAG)", open=False):
            sources_rag = gr.Markdown()
        with gr.Accordion("Fuentes (RAG avanzado)", open=False):
            sources_adv = gr.Markdown()
        gr.Examples(examples=examples, inputs=box, label="Preguntas de ejemplo")

        def compare(question: str) -> list[str]:
            results = [service.ask(question, m) for m in modes]
            out: list[str] = []
            for text, answer in results:
                out += [text, format_status(answer)]
            for _, answer in results[1:]:
                out.append(format_sources(answer.chunks, answer.cited_ids) if answer else "")
            return out

        outputs = [x for pair in zip(answers, statuses, strict=True) for x in pair]
        outputs += [sources_rag, sources_adv]
        go.click(compare, box, outputs)
        box.submit(compare, box, outputs)
