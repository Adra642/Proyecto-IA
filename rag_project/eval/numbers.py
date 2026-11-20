"""Tolerant number extraction and comparison for the automatic numeric check."""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<![\w.])\d[\d.,\u00a0\u202f ]*\d(?![\w])|(?<![\w.])\d(?![\w.])")
_LEADING_ZERO = re.compile(r"^0[.,]\d+$")
_THOUSANDS_COMMA = re.compile(r"^\d{1,3}(,\d{3})+$")
_THOUSANDS_DOT = re.compile(r"^\d{1,3}(\.\d{3})+$")
_THOUSANDS_SPACE = re.compile(r"^\d{1,3}([ \u00a0\u202f]\d{3})+$")


def _parse(token: str) -> float | None:
    token = token.strip(" .,")
    if _THOUSANDS_SPACE.match(token):
        return float(re.sub(r"[ \u00a0\u202f]", "", token))
    token = re.sub(r"[ \u00a0\u202f]", "", token)
    if _LEADING_ZERO.match(token):  # "0.324" is a decimal, not a thousands group
        return float(token.replace(",", "."))
    if "," in token and "." in token:
        # the last separator is the decimal one
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        token = token.replace(thousands, "").replace(decimal, ".")
    elif _THOUSANDS_COMMA.match(token):
        token = token.replace(",", "")
    elif _THOUSANDS_DOT.match(token):
        token = token.replace(".", "")
    else:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """All numbers in ``text`` (handles ``100,000``, ``100 000``, ``32,4`` and ``32.4``)."""
    found = (_parse(m.group(0)) for m in _NUMBER.finditer(text))
    return [n for n in found if n is not None]


def contains_number(text: str, value: float, tolerance: float, percent: bool = False) -> bool:
    """True if ``text`` contains ``value`` within ``tolerance``.

    For percentages the fraction form is accepted too (0.324 for 32.4).
    """
    for n in extract_numbers(text):
        if abs(n - value) <= tolerance:
            return True
        if percent and abs(n - value / 100) <= tolerance / 100:
            return True
    return False
