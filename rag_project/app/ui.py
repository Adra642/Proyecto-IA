"""Gradio Blocks interface (labels in Spanish). Built from an injected ``AppService``."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from app.handlers import (
    MODE_NO_RAG,
    MODE_RAG,
    MODES,
    AppService,
    format_sources,
    format_status,
)

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


def load_example_questions(path: Path) -> list[str]:
    """Return the evaluation questions to use as example buttons."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(q["question"]) for q in data]


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
            question: str, history: list[dict[str, str]] | None, selected: str
        ) -> tuple[list[dict[str, str]], str, str, str]:
            history = history or []
            text, answer = service.ask(question, selected)
            history = [
                *history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": text},
            ]
            src = (
                format_sources(answer.chunks, answer.cited_ids)
                if answer and answer.mode == "rag"
                else "_Sin fuentes en este modo._"
            )
            return history, src, format_status(answer), ""

        outputs = [chatbot, sources, status, box]
        send.click(respond, [box, chatbot, mode], outputs)
        box.submit(respond, [box, chatbot, mode], outputs)
        clear.click(lambda: ([], "_Aún no hay fuentes._", "", ""), None, outputs)


def _compare_tab(service: AppService, examples: list[str]) -> None:
    with gr.Tab("Comparar"):
        gr.Markdown(
            "La misma pregunta y el mismo modelo: **sin RAG** (izquierda) y **con RAG** (derecha)."
        )
        box = gr.Textbox(label="Pregunta", lines=2)
        go = gr.Button("Comparar", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Sin RAG")
                left = gr.Markdown()
                left_status = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Con RAG")
                right = gr.Markdown()
                right_status = gr.Markdown()
        with gr.Accordion("Fuentes (RAG)", open=False):
            sources = gr.Markdown()
        gr.Examples(examples=examples, inputs=box, label="Preguntas de ejemplo")

        def compare(question: str) -> tuple[str, str, str, str, str]:
            base_text, base = service.ask(question, MODE_NO_RAG)
            rag_text, rag = service.ask(question, MODE_RAG)
            src = format_sources(rag.chunks, rag.cited_ids) if rag else ""
            return base_text, format_status(base), rag_text, format_status(rag), src

        outputs = [left, left_status, right, right_status, sources]
        go.click(compare, box, outputs)
        box.submit(compare, box, outputs)
