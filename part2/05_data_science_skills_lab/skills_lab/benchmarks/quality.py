"""F11 - automated data-quality audit of the real Online Retail II transaction log.

Two design rules shape this module.

*Business events are not defects.* Negative quantities, credit notes and non-product
adjustment codes are legitimate records; statistical outliers are worth a human look but
are not invalid. Both are reported as **review flags**, never as quality violations.

*Scores are rates, not counts.* Each dimension is a normalised 0-100 rate, combined with
documented weights into the composite score. Health states come from score and severity
thresholds, so the verdict means the same thing on 500 rows as on a million. Raw counts
remain visible as diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from skills_lab.config import (
    HEALTH_PASS_MIN,
    HEALTH_WARNING_MIN,
    QUALITY_SAMPLE_ROWS,
    QUALITY_WEIGHTS,
)
from skills_lab.data import retail

PLACEHOLDER_TEXT = {"", "?", "??", "???", "-", "--", "n/a", "na", "none", "null", "unspecified", "unknown"}
_PUNCT_ONLY = re.compile(r"^[\W_]+$")

DIMENSION_DEFINITIONS = {
    "completeness": "Share of cells that are populated (not null).",
    "validity": "Share of populated cells that satisfy that column's typed rule "
                "(e.g. a price must be positive, an identifier must be a positive integer).",
    "uniqueness": "Share of rows that are not exact duplicates of an earlier row.",
    "consistency": "Share of text values already in canonical form - a value written as "
                   "both 'MUG' and 'mug' is inconsistent even though both are populated.",
}

HEALTH_RULE = (
    f"PASS at a score of {HEALTH_PASS_MIN:.0f} or above, WARNING at "
    f"{HEALTH_WARNING_MIN:.0f} or above, CRITICAL below that. A severe breach "
    "(over 20% null or over 20% invalid) is CRITICAL regardless of score; a moderate "
    "breach (over 5% null or over 5% invalid) is at best WARNING. The same rule applies to "
    "each column using that column's rates, and to the dataset using dataset-wide rates - so "
    "a healthy dataset score can coexist with a CRITICAL column, and both are shown."
)


@dataclass
class ColumnReport:
    column: str
    dtype: str
    distinct: int
    nulls: int
    null_pct: float
    invalid: int
    invalid_pct: float
    inconsistent: int
    inconsistent_pct: float
    completeness: float
    validity: float
    consistency: float
    score: float
    health: str
    issues: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "distinct": self.distinct,
            "nulls": self.nulls,
            "null_pct": round(self.null_pct, 2),
            "invalid": self.invalid,
            "inconsistent": self.inconsistent,
            "score": round(self.score, 1),
            "health": self.health,
            "issues": self.issues,
            "review_flags": self.review_flags,
        }


@dataclass
class QualityResult:
    dataset: str
    rows: int
    columns: int
    duplicate_rows: int
    completeness: float
    validity: float
    uniqueness: float
    consistency: float
    score: float
    health: str
    weights: dict[str, float]
    column_reports: list[ColumnReport]
    review_flags: list[str]
    business_events: dict

    def payload(self) -> dict:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "columns": self.columns,
            "duplicate_rows": self.duplicate_rows,
            "dimensions": {
                "completeness": round(self.completeness, 2),
                "validity": round(self.validity, 2),
                "uniqueness": round(self.uniqueness, 2),
                "consistency": round(self.consistency, 2),
            },
            "weights": self.weights,
            "composite_score": round(self.score, 1),
            "health": self.health,
            "health_rule": HEALTH_RULE,
            "columns_detail": [c.as_dict() for c in self.column_reports],
            "review_flags": self.review_flags,
            "business_events": self.business_events,
        }


# --- validity rules -------------------------------------------------------------------


def _text_placeholder_mask(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    lowered = text.str.lower()
    return lowered.isin(PLACEHOLDER_TEXT) | text.map(
        lambda v: bool(_PUNCT_ONLY.match(v)) if isinstance(v, str) else False
    )


def _validity_mask(column: str, values: pd.Series) -> pd.Series:
    """True where a *populated* value breaks that column's rule."""
    present = values.notna()
    if column == "Price":
        invalid = present & (pd.to_numeric(values, errors="coerce") <= 0)
    elif column == "Customer ID":
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = present & ((numeric <= 0) | (numeric != numeric.round()))
    elif column in {"Description", "Country", "StockCode", "Invoice"}:
        invalid = present & _text_placeholder_mask(values).fillna(False)
    elif column == "InvoiceDate":
        parsed = pd.to_datetime(values, errors="coerce")
        invalid = present & parsed.isna()
    else:
        invalid = pd.Series(False, index=values.index)
    return invalid.fillna(False)


def _inconsistency_mask(values: pd.Series) -> pd.Series:
    """True where a text value is a non-canonical spelling of another value present."""
    if values.dtype.kind not in {"O", "U", "S"} and not isinstance(
        values.dtype, pd.StringDtype
    ):
        return pd.Series(False, index=values.index)
    text = values.astype("string")
    canonical = text.str.strip().str.upper()
    # The canonical form is the most frequent surface spelling of each folded key.
    surface_counts = (
        pd.DataFrame({"canonical": canonical, "surface": text})
        .dropna()
        .value_counts()
        .reset_index(name="n")
    )
    preferred = (
        surface_counts.sort_values("n", ascending=False)
        .drop_duplicates("canonical")
        .set_index("canonical")["surface"]
    )
    expected = canonical.map(preferred)
    return (text.notna() & expected.notna() & (text != expected)).fillna(False)


def _outlier_count(values: pd.Series) -> int:
    """IQR outliers, for genuinely numeric columns only (an identifier is not a quantity)."""
    if values.dtype.kind not in {"i", "u", "f"}:
        return 0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)).sum())


def _health(score: float, null_rate: float, invalid_rate: float) -> str:
    if null_rate > 0.20 or invalid_rate > 0.20:
        return "CRITICAL"
    if score < HEALTH_WARNING_MIN:
        return "CRITICAL"
    if null_rate > 0.05 or invalid_rate > 0.05 or score < HEALTH_PASS_MIN:
        return "WARNING"
    return "PASS"


def profile_column(column: str, values: pd.Series) -> ColumnReport:
    rows = len(values)
    nulls = int(values.isna().sum())
    null_rate = nulls / rows if rows else 0.0

    invalid_mask = _validity_mask(column, values)
    invalid = int(invalid_mask.sum())
    populated = rows - nulls
    invalid_rate = invalid / populated if populated else 0.0

    inconsistent_mask = _inconsistency_mask(values)
    inconsistent = int(inconsistent_mask.sum())
    inconsistent_rate = inconsistent / populated if populated else 0.0

    completeness = (1 - null_rate) * 100
    validity = (1 - invalid_rate) * 100
    consistency = (1 - inconsistent_rate) * 100

    # Column score reuses the dataset weights minus uniqueness (a row-level dimension).
    weights = {k: v for k, v in QUALITY_WEIGHTS.items() if k != "uniqueness"}
    total_weight = sum(weights.values())
    score = (
        completeness * weights["completeness"]
        + validity * weights["validity"]
        + consistency * weights["consistency"]
    ) / total_weight

    issues: list[str] = []
    if nulls:
        issues.append(f"{nulls:,} missing values ({null_rate * 100:.1f}%)")
    if invalid:
        issues.append(f"{invalid:,} values break the {column} rule ({invalid_rate * 100:.1f}%)")
    if inconsistent:
        issues.append(
            f"{inconsistent:,} non-canonical spellings ({inconsistent_rate * 100:.1f}%)"
        )

    review_flags: list[str] = []
    outliers = _outlier_count(values)
    if outliers:
        review_flags.append(
            f"{outliers:,} statistical outliers (beyond 3x IQR) - review, not invalid"
        )
    if column == "Quantity":
        negatives = int((pd.to_numeric(values, errors="coerce") < 0).sum())
        if negatives:
            review_flags.append(
                f"{negatives:,} negative quantities - returns/cancellations, not defects"
            )

    return ColumnReport(
        column=column,
        dtype=str(values.dtype),
        distinct=int(values.nunique(dropna=True)),
        nulls=nulls,
        null_pct=null_rate * 100,
        invalid=invalid,
        invalid_pct=invalid_rate * 100,
        inconsistent=inconsistent,
        inconsistent_pct=inconsistent_rate * 100,
        completeness=float(completeness),
        validity=validity,
        consistency=consistency,
        score=score,
        health=_health(score, null_rate, invalid_rate),
        issues=issues,
        review_flags=review_flags,
    )


def run(frame: pd.DataFrame | None = None, rows: int = QUALITY_SAMPLE_ROWS) -> QualityResult:
    data = retail.audit_slice(rows) if frame is None else frame.copy()
    n_rows = len(data)

    reports = [profile_column(column, data[column]) for column in data.columns]

    duplicate_rows = int(data.duplicated(keep="first").sum())
    total_cells = n_rows * len(data.columns)
    completeness = (
        (1 - data.isna().to_numpy().sum() / total_cells) * 100 if total_cells else 100.0
    )
    populated_cells = total_cells - int(data.isna().to_numpy().sum())
    invalid_cells = sum(r.invalid for r in reports)
    inconsistent_cells = sum(r.inconsistent for r in reports)
    validity = (1 - invalid_cells / populated_cells) * 100 if populated_cells else 100.0
    consistency = (
        (1 - inconsistent_cells / populated_cells) * 100 if populated_cells else 100.0
    )
    uniqueness = (1 - duplicate_rows / n_rows) * 100 if n_rows else 100.0

    score = (
        completeness * QUALITY_WEIGHTS["completeness"]
        + validity * QUALITY_WEIGHTS["validity"]
        + uniqueness * QUALITY_WEIGHTS["uniqueness"]
        + consistency * QUALITY_WEIGHTS["consistency"]
    )

    # Dataset health uses dataset-wide rates; a single sparse column is surfaced through
    # its own column health rather than condemning the whole table.
    health = _health(score, 1 - completeness / 100, 1 - validity / 100)

    review_flags = [f"{r.column}: {flag}" for r in reports for flag in r.review_flags]
    critical_columns = [r.column for r in reports if r.health == "CRITICAL"]
    if critical_columns:
        review_flags.insert(
            0, "Columns in CRITICAL state: " + ", ".join(critical_columns)
        )

    return QualityResult(
        dataset=f"Online Retail II (UCI) - first {n_rows:,} transaction lines",
        rows=n_rows,
        columns=len(data.columns),
        duplicate_rows=duplicate_rows,
        completeness=float(completeness),
        validity=validity,
        uniqueness=uniqueness,
        consistency=consistency,
        score=score,
        health=health,
        weights=dict(QUALITY_WEIGHTS),
        column_reports=reports,
        review_flags=review_flags,
        business_events=retail.business_event_summary(retail.audit_slice(rows)),
    )
