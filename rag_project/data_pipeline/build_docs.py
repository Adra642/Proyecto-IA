"""Turn the churn table into knowledge: statistics (pure functions) and Markdown documents."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from core.models import Document, DocumentMetadata, Geography

DATASET_NAME = 'Kaggle "Bank Customer Churn" dataset'
DATASET_CONTEXT = f"{DATASET_NAME} (customers of a bank in France, Germany and Spain)"
TARGET = "Exited"
GEO_CODES: dict[str, Geography] = {"France": "FR", "Germany": "DE", "Spain": "ES"}

# Thresholds used to phrase findings; kept as named constants (no magic numbers).
LEAKAGE_SHARE = 0.95
FLAT_SPREAD = 0.03
TOP_N = 5

AGE_BINS = [-math.inf, 30, 40, 50, 60, math.inf]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]
TENURE_BINS = [-math.inf, 3, 6, 9, math.inf]
TENURE_LABELS = ["0-2", "3-5", "6-8", "9-10"]

COLUMN_MEANINGS: dict[str, str] = {
    "CreditScore": "Credit score of the customer (higher is better).",
    "Geography": "Country of the customer: France, Germany or Spain.",
    "Gender": "Customer gender: Female or Male.",
    "Age": "Age in years.",
    "Tenure": "Years the customer has been with the bank.",
    "Balance": "Account balance (currency units of the dataset).",
    "NumOfProducts": "Number of bank products held (1 to 4).",
    "HasCrCard": "1 if the customer has a credit card, otherwise 0.",
    "IsActiveMember": "1 if the customer is an active member, otherwise 0.",
    "EstimatedSalary": "Estimated annual salary.",
    "Exited": "Target: 1 if the customer left the bank (churned), otherwise 0.",
    "Complain": "1 if the customer filed a complaint, otherwise 0.",
    "Satisfaction Score": "Satisfaction score given by the customer, 1 (low) to 5 (high).",
    "Card Type": "Credit card tier: DIAMOND, GOLD, PLATINUM or SILVER.",
    "Point Earned": "Loyalty points earned by the customer.",
}


@dataclass(frozen=True)
class Segmentation:
    """A way of splitting customers into segments."""

    name: str  # stats.json key prefix
    title: str
    columns: list[str]  # dataset columns involved
    group_by: list[str]  # columns present in the working frame
    geography_col: str | None = None


SEGMENTATIONS: list[Segmentation] = [
    Segmentation("geography", "Geography", ["Geography"], ["Geography"], "Geography"),
    Segmentation("gender", "Gender", ["Gender"], ["Gender"]),
    Segmentation("age_band", "Age band", ["Age"], ["AgeBand"]),
    Segmentation("tenure_band", "Tenure band", ["Tenure"], ["TenureBand"]),
    Segmentation("num_of_products", "Number of products", ["NumOfProducts"], ["NumOfProducts"]),
    Segmentation("card_type", "Card type", ["Card Type", "Point Earned"], ["Card Type"]),
    Segmentation("has_cr_card", "Has credit card", ["HasCrCard"], ["HasCrCard"]),
    Segmentation("is_active_member", "Active member", ["IsActiveMember"], ["IsActiveMember"]),
    Segmentation(
        "satisfaction_score", "Satisfaction score", ["Satisfaction Score"], ["Satisfaction Score"]
    ),
    Segmentation("complain", "Complaint filed", ["Complain"], ["Complain"]),
    Segmentation(
        "geography_x_num_of_products",
        "Geography x number of products",
        ["Geography", "NumOfProducts"],
        ["Geography", "NumOfProducts"],
        "Geography",
    ),
    Segmentation(
        "geography_x_is_active_member",
        "Geography x active member",
        ["Geography", "IsActiveMember"],
        ["Geography", "IsActiveMember"],
        "Geography",
    ),
]


# ----------------------------------------------------------------------------- statistics
def with_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``AgeBand`` and ``TenureBand`` columns (left-closed bins)."""
    out = df.copy()
    out["AgeBand"] = pd.cut(out["Age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    out["TenureBand"] = pd.cut(
        out["Tenure"], TENURE_BINS, labels=TENURE_LABELS, right=False
    ).astype(str)
    return out


def overview(df: pd.DataFrame) -> dict[str, float]:
    """Row count, churn counts and class balance."""
    churned = int(df[TARGET].sum())
    rows = len(df)
    return {
        "rows": rows,
        "churned": churned,
        "retained": rows - churned,
        "churn_rate": churned / rows,
        "retained_per_churned": (rows - churned) / churned if churned else math.nan,
    }


def segment_stats(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per-segment count, share, churn rate and means (only non-empty segments)."""
    grouped = df.groupby(by, observed=True)
    out = grouped.agg(
        customers=(TARGET, "size"),
        churned=(TARGET, "sum"),
        churn_rate=(TARGET, "mean"),
        mean_balance=("Balance", "mean"),
        mean_salary=("EstimatedSalary", "mean"),
        mean_credit_score=("CreditScore", "mean"),
        mean_age=("Age", "mean"),
        mean_points=("Point Earned", "mean"),
    ).reset_index()
    out["share"] = out["customers"] / len(df)
    return out


def leakage_check(df: pd.DataFrame) -> dict[str, float]:
    """Measure how strongly ``Complain`` mirrors ``Exited``."""
    complained = df["Complain"] == 1
    churned = df[TARGET] == 1
    return {
        "churn_rate_complained": float(df.loc[complained, TARGET].mean()),
        "churn_rate_not_complained": float(df.loc[~complained, TARGET].mean()),
        "share_churners_complained": float(complained[churned].mean()),
        "complained_and_churned": int((complained & churned).sum()),
        "complained_not_churned": int((complained & ~churned).sum()),
        "not_complained_churned": int((~complained & churned).sum()),
        "correlation": float(df["Complain"].corr(df[TARGET])),
    }


def satisfaction_vs_churn(df: pd.DataFrame) -> dict[str, float]:
    """Churn rate per satisfaction score and how much it varies."""
    rates = df.groupby("Satisfaction Score")[TARGET].mean()
    return {
        "spread": float(rates.max() - rates.min()),
        "correlation": float(df["Satisfaction Score"].corr(df[TARGET])),
        "min_rate": float(rates.min()),
        "max_rate": float(rates.max()),
    }


def driver_scores(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Correlation and mutual information of each feature with churn (deterministic)."""
    features = df.drop(columns=[TARGET])
    encoded = pd.DataFrame(index=df.index)
    discrete: list[bool] = []
    for col in features.columns:
        if features[col].dtype == object or str(features[col].dtype) in ("str", "string"):
            encoded[col] = pd.factorize(features[col], sort=True)[0]
            discrete.append(True)
        else:
            encoded[col] = features[col]
            discrete.append(features[col].nunique() <= 20)
    corr = {c: float(encoded[c].corr(df[TARGET])) for c in encoded.columns}
    mi = mutual_info_classif(
        encoded.to_numpy(dtype=float), df[TARGET], discrete_features=discrete, random_state=0
    )
    return {"corr": corr, "mi": {c: float(v) for c, v in zip(encoded.columns, mi, strict=True)}}


def _slug(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("<", "lt").replace("+", "plus")


def segment_key(seg: Segmentation, row: pd.Series) -> str:
    """Stable key for a segment row, e.g. ``Germany`` or ``Germany|3``."""
    return "|".join(str(row[c]) for c in seg.group_by)


def compute_stats(df: pd.DataFrame) -> dict[str, float]:
    """Every number used in the documents, keyed by dotted name (for ``stats.json``)."""
    work = with_bands(df)
    stats: dict[str, float] = {f"overall.{k}": v for k, v in overview(work).items()}
    for seg in SEGMENTATIONS:
        table = segment_stats(work, seg.group_by)
        for _, row in table.iterrows():
            key = segment_key(seg, row)
            for metric in table.columns:
                if metric in seg.group_by:
                    continue
                stats[f"{seg.name}.{key}.{metric}"] = float(row[metric])
    stats.update({f"leakage.{k}": v for k, v in leakage_check(work).items()})
    stats.update({f"satisfaction.{k}": v for k, v in satisfaction_vs_churn(work).items()})
    scores = driver_scores(work.drop(columns=["AgeBand", "TenureBand"]))
    for kind, values in scores.items():
        stats.update({f"drivers.{kind}.{c}": v for c, v in values.items()})
    return {k: _clean_number(v) for k, v in stats.items()}


def _clean_number(value: float) -> float:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return float("nan")
    return round(float(value), 6)


# ----------------------------------------------------------------------------- rendering
def pct(x: float) -> str:
    """Format a fraction as a percentage with one decimal."""
    return f"{x * 100:.1f}%"


def num(x: float) -> str:
    """Format a number with thousands separators and two decimals."""
    return f"{x:,.2f}"


def _comparison(rate: float, overall: float) -> str:
    ratio = rate / overall if overall else math.nan
    if abs(ratio - 1) < 0.05:
        return "about the same as the overall churn rate"
    direction = "higher" if ratio > 1 else "lower"
    return f"{abs(ratio):.2f} times the overall churn rate ({direction})"


def render_segment_card(
    seg: Segmentation, row: pd.Series, overall_rate: float, total_rows: int
) -> str:
    """Markdown card for one segment."""
    label = " and ".join(f"{c} = {row[c]}" for c in seg.group_by)
    rate = float(row["churn_rate"])
    return (
        f"# Churn segment card: {label}\n\n"
        f"Source: {DATASET_CONTEXT}. Segmentation: {seg.title}. Segment: {label}.\n\n"
        f"- Customers in segment: {int(row['customers']):,} ({pct(float(row['share']))} of "
        f"{total_rows:,} customers)\n"
        f"- Churned customers (Exited = 1): {int(row['churned']):,}\n"
        f"- Churn rate: {pct(rate)} (overall churn rate: {pct(overall_rate)})\n"
        f"- Mean balance: {num(float(row['mean_balance']))}\n"
        f"- Mean estimated salary: {num(float(row['mean_salary']))}\n"
        f"- Mean credit score: {float(row['mean_credit_score']):.1f}\n"
        f"- Mean age: {float(row['mean_age']):.1f}\n"
        f"- Mean loyalty points earned: {float(row['mean_points']):.1f}\n\n"
        f"In plain language: customers with {label} left the bank at a rate of {pct(rate)}, "
        f"which is {_comparison(rate, overall_rate)}.\n"
    )


def _name(row: pd.Series, seg: Segmentation) -> str:
    return " / ".join(str(row[c]) for c in seg.group_by)


def render_summary(seg: Segmentation, table: pd.DataFrame, overall_rate: float) -> str:
    """Markdown comparison table of all segments of one segmentation."""
    lines = [
        f"# Churn rate by {seg.title}\n",
        f"Source: {DATASET_CONTEXT}. Overall churn rate: {pct(overall_rate)}.\n",
        "| Segment | Customers | Churned | Churn rate | Mean balance | Mean points |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        label = _name(row, seg)
        lines.append(
            f"| {label} | {int(row['customers']):,} | {int(row['churned']):,} | "
            f"{pct(float(row['churn_rate']))} | {num(float(row['mean_balance']))} | "
            f"{float(row['mean_points']):.1f} |"
        )
    ranked = table.sort_values("churn_rate", kind="stable")
    worst, best = ranked.iloc[0], ranked.iloc[-1]
    lines.append(
        f"\nIn plain language: by {seg.title.lower()}, the highest churn rate is "
        f"{pct(float(best['churn_rate']))} ({_name(best, seg)}) and the lowest is "
        f"{pct(float(worst['churn_rate']))} ({_name(worst, seg)}).\n"
    )
    return "\n".join(lines)


def render_dictionary(df: pd.DataFrame) -> str:
    """Data dictionary: every remaining column with type, range and meaning."""
    lines = [
        f"# Data dictionary: {DATASET_NAME}\n",
        "Personal identifiers (row number, customer id and surname) were removed before "
        "analysis and are not part of the knowledge base.\n",
        "| Column | Type | Range or values | Meaning |",
        "|---|---|---|---|",
    ]
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            kind = "integer" if pd.api.types.is_integer_dtype(series) else "decimal"
            span = f"{series.min():,.2f} to {series.max():,.2f}".replace(".00", "")
        else:
            kind = "category"
            span = ", ".join(sorted(series.unique()))
        lines.append(f"| {col} | {kind} | {span} | {COLUMN_MEANINGS.get(col, '')} |")
    return "\n".join(lines) + "\n"


def render_overview(ov: dict[str, float]) -> str:
    """Dataset overview document."""
    return (
        f"# Overview of the {DATASET_NAME}\n\n"
        f"- Customers (rows): {int(ov['rows']):,}\n"
        f"- Churned customers (Exited = 1): {int(ov['churned']):,}\n"
        f"- Retained customers (Exited = 0): {int(ov['retained']):,}\n"
        f"- Overall churn rate: {pct(ov['churn_rate'])}\n"
        f"- Class balance: about {ov['retained_per_churned']:.1f} retained customers for every "
        "churned customer (imbalanced target).\n\n"
        "Countries covered: France, Germany and Spain.\n"
    )


def render_findings(stats: dict[str, float], df: pd.DataFrame) -> str:
    """Findings document: drivers, leakage check, satisfaction check."""
    leak = {k.split(".", 1)[1]: v for k, v in stats.items() if k.startswith("leakage.")}
    sat = {k.split(".", 1)[1]: v for k, v in stats.items() if k.startswith("satisfaction.")}
    corr = {k.split(".", 2)[2]: v for k, v in stats.items() if k.startswith("drivers.corr.")}
    mi = {k.split(".", 2)[2]: v for k, v in stats.items() if k.startswith("drivers.mi.")}

    def top(values: dict[str, float], key: Callable[[float], float]) -> list[tuple[str, float]]:
        ranked = sorted(
            ((c, v) for c, v in values.items() if c != "Complain"), key=lambda kv: -key(kv[1])
        )
        return ranked[:TOP_N]

    lines = [
        f"# Findings: churn drivers in the {DATASET_NAME}\n",
        f"Overall churn rate: {pct(stats['overall.churn_rate'])}.\n",
        "## Top churn drivers (excluding Complain, see the leakage check)\n",
        "By absolute correlation with Exited:\n",
    ]
    lines += [f"{i}. {c} (correlation {v:+.3f})" for i, (c, v) in enumerate(top(corr, abs), 1)]
    lines += ["", "By mutual information with Exited:\n"]
    lines += [
        f"{i}. {c} (mutual information {v:.4f})" for i, (c, v) in enumerate(top(mi, float), 1)
    ]
    lines += ["", "## Leakage check: Complain versus Exited\n"]
    lines += [
        f"- Churn rate among customers who complained: {pct(leak['churn_rate_complained'])}",
        f"- Churn rate among customers who did not complain: "
        f"{pct(leak['churn_rate_not_complained'])}",
        "- Share of churned customers who had complained: "
        f"{pct(leak['share_churners_complained'])}",
        f"- Correlation between Complain and Exited: {leak['correlation']:.3f}",
    ]
    if leak["share_churners_complained"] >= LEAKAGE_SHARE:
        lines.append(
            "\nConclusion: Complain almost perfectly mirrors Exited, so it is very likely target "
            "leakage (the complaint is probably recorded as part of the departure). A churn "
            "model should not use Complain as a predictor.\n"
        )
    else:
        lines.append("\nConclusion: Complain is related to churn but does not mirror it.\n")
    lines += ["## Satisfaction score versus churn\n"]
    lines.append(
        f"Churn rate ranges from {pct(sat['min_rate'])} to {pct(sat['max_rate'])} across the "
        f"five satisfaction scores (correlation with Exited: {sat['correlation']:.3f})."
    )
    if sat["spread"] <= FLAT_SPREAD:
        lines.append(
            "\nConclusion: satisfaction score alone does not track churn in this dataset; the "
            "churn rate is almost flat across scores.\n"
        )
    else:
        lines.append("\nConclusion: churn varies noticeably with the satisfaction score.\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- documents
def _meta(
    doc_id: str,
    title: str,
    columns: list[str],
    geography: Geography,
    url: str,
    retrieved_at: date,
    license_: str,
) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=doc_id,
        title=title,
        source="kaggle_derived",
        source_type="dataset_stats",
        url=url,
        language="en",
        geography=geography,
        topic="churn",
        columns=sorted({*columns, TARGET}),
        retrieved_at=retrieved_at,
        license=license_,
        format="generated",
    )


def build_documents(
    df: pd.DataFrame, url: str, retrieved_at: date, license_: str
) -> tuple[list[Document], dict[str, float]]:
    """Build every derived document and ``stats.json`` content from a PII-free table.

    Args:
        df: Customer table without personal identifiers.
        url: Dataset URL for the metadata.
        retrieved_at: Date the CSV was downloaded.
        license_: License text for the metadata.

    Returns:
        ``(documents, stats)``.
    """
    work = with_bands(df)
    stats = compute_stats(df)
    overall = stats["overall.churn_rate"]
    total = len(df)
    docs: list[Document] = []

    def add(doc_id: str, title: str, text: str, cols: list[str], geo: Geography = "EU") -> None:
        docs.append(
            Document(
                text=text, metadata=_meta(doc_id, title, cols, geo, url, retrieved_at, license_)
            )
        )

    add(
        "kaggle_data_dictionary",
        "Data dictionary of the Bank Customer Churn dataset",
        render_dictionary(df),
        list(df.columns),
    )
    add(
        "kaggle_overview",
        "Overview of the Bank Customer Churn dataset",
        render_overview(overview(work)),
        [TARGET],
    )
    add(
        "kaggle_findings",
        "Findings: churn drivers, Complain leakage and satisfaction",
        render_findings(stats, work),
        ["Complain", "Satisfaction Score", "NumOfProducts", "Age", "IsActiveMember", "Geography"],
    )
    for seg in SEGMENTATIONS:
        table = segment_stats(work, seg.group_by)
        add(
            f"kaggle_summary_{seg.name}",
            f"Churn rate by {seg.title}",
            render_summary(seg, table, overall),
            seg.columns,
        )
        for _, row in table.iterrows():
            key = "_".join(_slug(row[c]) for c in seg.group_by)
            geo: Geography = (
                GEO_CODES.get(str(row[seg.geography_col]), "EU") if seg.geography_col else "EU"
            )
            label = " and ".join(f"{c} = {row[c]}" for c in seg.group_by)
            add(
                f"kaggle_segment_{seg.name}_{key}",
                f"Churn segment: {label}",
                render_segment_card(seg, row, overall, total),
                seg.columns,
                geo,
            )
    return docs, stats
