"""Small shared helpers."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str) -> str:
    """ASCII, lowercase, underscore-separated identifier fragment.

    Args:
        text: Any title or label.

    Returns:
        A slug such as ``general_data_protection_regulation``.
    """
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def pdf_text(data: bytes) -> str:
    """Extract plain text from PDF bytes with pymupdf."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(str(page.get_text()) for page in doc)
