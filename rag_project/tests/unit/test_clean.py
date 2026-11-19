from datetime import date

from core.models import Document, DocumentMetadata
from data_pipeline.clean import (
    clean_documents,
    dedupe_step,
    html_to_markdown,
    looks_like_html,
    min_length_step,
    normalize_text,
)

PAGE = """<html><head><title>x</title><script>var track=1;</script></head><body>
<header><nav><a href="/">Home</a> <a href="/x">Menu item</a></nav></header>
<div class="cookie-banner">We use cookies, accept all</div>
<main>
<h1>Cuentas corrientes</h1>
<p>Una cuenta corriente permite&nbsp;realizar pagos y cobros. Es el producto basico de un banco.
Los clientes pueden cambiar de banco con el servicio de traslado de cuenta
sin perder sus domiciliaciones.</p>
<h2>Comisiones</h2>
<ul><li>Mantenimiento</li><li>Administracion</li></ul>
<table><tr><th>Cuenta</th><th>Coste</th></tr><tr><td>Basica</td><td>0</td></tr></table>
</main>
<footer>Copyright footer text</footer></body></html>"""


def doc(doc_id: str, text: str, title: str = "T") -> Document:
    meta = DocumentMetadata(
        doc_id=doc_id, title=title, source="s", source_type="regulator_guide", url="u",
        language="es", geography="ES", topic="t", retrieved_at=date(2026, 1, 1), license="l",
    )  # fmt: skip
    return Document(text=text, metadata=meta)


def test_looks_like_html() -> None:
    assert looks_like_html(PAGE) and not looks_like_html("# Title\n\nPlain <b>text</b>")


def test_html_to_markdown_strips_boilerplate_and_keeps_headings() -> None:
    md = html_to_markdown(PAGE, "Fallback title")
    assert md.startswith("# Cuentas corrientes")
    assert "## Comisiones" in md and "Mantenimiento" in md and "| Basica | 0 |" in md
    for junk in ("cookies", "Menu item", "Copyright", "track=1", "href"):
        assert junk not in md


def test_title_added_when_no_h1() -> None:
    assert html_to_markdown(
        "<html><body><main><p>" + "word " * 80 + "</p></main></body></html>", "My title"
    ).startswith("# My title")


def test_normalize_is_idempotent_and_cleans() -> None:
    raw = "Café   text​\r\n\r\n\r\n\r\nline   two  \n"
    out = normalize_text(raw)
    assert out == "Café text\n\nline two\n"
    assert normalize_text(out) == out


def test_dedupe_keeps_first_doc_id() -> None:
    a, b, c = doc("b_doc", "Same   TEXT"), doc("a_doc", "same text"), doc("c_doc", "other")
    assert [d.doc_id for d in dedupe_step([a, b, c])] == ["a_doc", "c_doc"]


def test_min_length_drops_tiny() -> None:
    assert [d.doc_id for d in min_length_step([doc("x", "tiny"), doc("y", "w" * 300)])] == ["y"]


def test_pipeline_end_to_end_is_sorted_and_deterministic() -> None:
    docs = [doc("z", PAGE), doc("m", "short"), doc("a", "# Long\n\n" + "plain markdown " * 40)]
    once = clean_documents(docs)
    assert [d.doc_id for d in once] == ["a", "z"]
    assert "cookies" not in once[1].text
    assert [d.text for d in clean_documents(once)] == [d.text for d in once]


def test_body_class_with_sidebar_does_not_remove_the_page() -> None:
    page = (
        '<html><body class="layout-no-sidebars"><nav>menu</nav><main><h1>Droit au compte</h1><p>'
        + "Tout le monde a droit a un compte bancaire en France. " * 8
        + "</p></main></body></html>"
    )
    md = html_to_markdown(page, "T")
    assert md.startswith("# Droit au compte") and "menu" not in md and len(md) > 200
