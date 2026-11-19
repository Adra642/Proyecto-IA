import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.build_docs import (
    build_documents,
    compute_stats,
    driver_scores,
    leakage_check,
    overview,
    satisfaction_vs_churn,
    segment_stats,
    with_bands,
)
from data_pipeline.clean import PII_COLUMNS, drop_pii

CSV = Path(__file__).parent.parent / "fixtures" / "tiny_churn.csv"


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.read_csv(CSV)


@pytest.fixture
def df(raw: pd.DataFrame) -> pd.DataFrame:
    return drop_pii(raw)


def test_drop_pii_removes_identifiers(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    assert all(c in raw.columns for c in PII_COLUMNS)
    assert not set(PII_COLUMNS) & set(df.columns)
    assert len(df.columns) == len(raw.columns) - 3


def test_overview(df: pd.DataFrame) -> None:
    ov = overview(df)
    assert (ov["rows"], ov["churned"], ov["retained"]) == (12, 5, 7)
    assert ov["churn_rate"] == pytest.approx(5 / 12)
    assert ov["retained_per_churned"] == pytest.approx(7 / 5)


def test_geography_segments(df: pd.DataFrame) -> None:
    t = segment_stats(df, ["Geography"]).set_index("Geography")
    assert t.loc["Germany", "churn_rate"] == 0.5
    assert t.loc["Germany", "mean_balance"] == 115000
    assert t.loc["Germany", "mean_credit_score"] == 687.5
    assert t.loc["Germany", "share"] == pytest.approx(4 / 12)
    assert t.loc["France", "mean_balance"] == 1500
    assert t.loc["Spain", "churn_rate"] == 0.25
    assert t.loc["Spain", "mean_salary"] == 51250


def test_bands(df: pd.DataFrame) -> None:
    work = with_bands(df)
    age = segment_stats(work, ["AgeBand"]).set_index("AgeBand")
    assert age.loc["<30", "customers"] == 2
    assert age.loc["30-39", "customers"] == 3
    assert age.loc["40-49", "churn_rate"] == 1.0
    assert age.loc["50-59", "churn_rate"] == 1.0
    assert age.loc["60+", "churn_rate"] == 0.0
    ten = segment_stats(work, ["TenureBand"]).set_index("TenureBand")
    assert ten["customers"].sum() == 12
    assert ten.loc["0-2", "customers"] == 4
    assert ten.loc["3-5", "churn_rate"] == pytest.approx(2 / 3)
    assert ten.loc["9-10", "churn_rate"] == 0.5


def test_products_and_card_type(df: pd.DataFrame) -> None:
    p = segment_stats(df, ["NumOfProducts"]).set_index("NumOfProducts")
    assert p.loc[1, "customers"] == 6 and p.loc[1, "churn_rate"] == 0.5
    assert p.loc[2, "churn_rate"] == 0.25
    assert p.loc[4, "churn_rate"] == 1.0
    c = segment_stats(df, ["Card Type"]).set_index("Card Type")
    assert c.loc["GOLD", "mean_points"] == 175
    assert c.loc["SILVER", "churn_rate"] == 0.75
    assert c.loc["PLATINUM", "mean_points"] == 750


def test_leakage_check(df: pd.DataFrame) -> None:
    leak = leakage_check(df)
    assert leak["churn_rate_complained"] == 1.0
    assert leak["churn_rate_not_complained"] == 0.0
    assert leak["share_churners_complained"] == 1.0
    assert leak["correlation"] == pytest.approx(1.0)
    assert leak["complained_and_churned"] == 5


def test_satisfaction(df: pd.DataFrame) -> None:
    sat = satisfaction_vs_churn(df)
    assert sat["max_rate"] == 1.0 and sat["min_rate"] == 0.0 and sat["spread"] == 1.0


def test_drivers_are_deterministic(df: pd.DataFrame) -> None:
    a, b = driver_scores(df), driver_scores(df)
    assert a == b
    assert "Complain" in a["corr"] and "Geography" in a["mi"]


def test_stats_keys(df: pd.DataFrame) -> None:
    stats = compute_stats(df)
    assert stats["overall.rows"] == 12
    assert stats["geography.Germany.churn_rate"] == 0.5
    assert stats["geography_x_num_of_products.Germany|1.churn_rate"] == 1.0
    assert stats["geography_x_num_of_products.France|1.customers"] == 3
    assert stats["age_band.50-59.churn_rate"] == 1.0
    assert stats["card_type.DIAMOND.mean_points"] == 550
    assert stats["leakage.share_churners_complained"] == 1.0
    json.dumps(stats, allow_nan=True)


def test_documents(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    docs, stats = build_documents(df, "https://kaggle.test", date(2026, 11, 19), "lic")
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))
    for needed in (
        "kaggle_data_dictionary",
        "kaggle_overview",
        "kaggle_findings",
        "kaggle_segment_geography_germany",
        "kaggle_summary_geography",
        "kaggle_segment_geography_x_num_of_products_germany_3",
    ):
        assert needed in ids
    germany = next(d for d in docs if d.doc_id == "kaggle_segment_geography_germany")
    assert germany.metadata.geography == "DE"
    assert "Churn rate: 50.0%" in germany.text
    assert "Customers in segment: 4 (33.3% of 12 customers)" in germany.text
    findings = next(d for d in docs if d.doc_id == "kaggle_findings")
    assert "target leakage" in findings.text
    assert "varies noticeably" in findings.text  # satisfaction is not flat in this tiny sample
    assert all(
        d.metadata.source == "kaggle_derived" and "Exited" in d.metadata.columns for d in docs
    )
    corpus = "\n".join(d.text for d in docs)
    for surname in raw["Surname"]:
        assert surname not in corpus
    for cid in raw["CustomerId"]:
        assert str(cid) not in corpus
    assert stats["overall.churn_rate"] == pytest.approx(5 / 12)
