from data_pipeline.fetchers.eurlex import select_articles

TEXT = (
    "# ACT\n\n(1) Recital.\n\n"
    "### Article 1 - Scope\n\nBody one.\n\n"
    "### Article 2 - Definitions\n\nBody two.\n\n## CHAPTER II - NEXT\n\n"
    "### Article 3 - Other\n\nBody three.\n\n"
    "5. (b) AI systems to evaluate the creditworthiness of persons."
)


def test_keeps_only_wanted_articles_and_drops_trailing_headings() -> None:
    out = select_articles(TEXT, [2])
    assert "Article 2" in out and "Body two." in out
    assert "Article 1" not in out and "Article 3" not in out
    assert "CHAPTER II" not in out and "Recital" not in out


def test_keeps_paragraphs_matching_the_pattern_as_excerpts() -> None:
    out = select_articles(TEXT, [1], keep_matching="creditworthiness")
    assert "### Excerpts" in out and "creditworthiness" in out
    assert out.count("creditworthiness") == 1
