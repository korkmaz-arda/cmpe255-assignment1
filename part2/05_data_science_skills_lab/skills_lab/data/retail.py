"""Online Retail II (UCI) - the source for the analytics and data-quality workspaces.

This module draws a line that matters for both workspaces: a *business event* is not a
*data defect*. Credit notes (invoice numbers starting with ``C``), negative quantities and
non-product adjustment codes (postage, bank charges, samples) are genuine commercial
records. They are excluded from the retention population and counted correctly in revenue,
but they are never reported as data-quality violations.
"""

from __future__ import annotations

import pandas as pd

from skills_lab.config import REVENUE_SERIES_DAYS
from skills_lab.data.acquire import load_cached

# StockCodes that are not products. Kept explicit so the rule is auditable.
ADJUSTMENT_CODES = {
    "POST", "D", "DOT", "M", "S", "AMAZONFEE", "BANK CHARGES", "B",
    "CRUK", "PADS", "C2", "C3", "GIFT", "TEST001", "TEST002", "ADJUST", "ADJUST2",
}


def load() -> pd.DataFrame:
    frame = load_cached("retail")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    return frame


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with its business meaning. Adds columns, drops nothing."""
    out = frame.copy()
    invoice = out["Invoice"].astype("string").fillna("")
    stock = out["StockCode"].astype("string").fillna("").str.upper().str.strip()
    out["is_credit_note"] = invoice.str.upper().str.startswith("C").fillna(False)
    out["is_adjustment"] = stock.isin(ADJUSTMENT_CODES)
    out["is_return"] = out["Quantity"] < 0
    out["is_product_line"] = ~out["is_adjustment"]
    out["line_revenue"] = out["Quantity"] * out["Price"]
    return out


def business_event_summary(frame: pd.DataFrame | None = None) -> dict[str, object]:
    """Counts of legitimate business events, reported separately from quality defects."""
    annotated = annotate(load() if frame is None else frame)
    total = len(annotated)
    return {
        "rows": int(total),
        "credit_notes": int(annotated["is_credit_note"].sum()),
        "return_lines": int(annotated["is_return"].sum()),
        "adjustment_lines": int(annotated["is_adjustment"].sum()),
        "returned_value": float(
            annotated.loc[annotated["is_return"], "line_revenue"].sum()
        ),
        "note": (
            "Credit notes, negative quantities and non-product adjustment codes are "
            "legitimate commercial records, not data-quality defects."
        ),
    }


# --- retention -----------------------------------------------------------------------

RETENTION_POPULATION_RULE = (
    "Customers with a non-null Customer ID whose invoice is a product line (not a postage "
    "or bank-charge adjustment), is not a credit note, and has positive quantity and "
    "positive price. A customer's cohort is the calendar month of "
    "their first qualifying purchase; a cohort is 'retained' in period k if that customer "
    "makes any qualifying purchase in the month k months later."
)


def retention_population(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    annotated = annotate(load() if frame is None else frame)
    mask = (
        annotated["Customer ID"].notna()
        & annotated["is_product_line"]
        & ~annotated["is_credit_note"]
        & (annotated["Quantity"] > 0)
        & (annotated["Price"] > 0)
    )
    population = annotated.loc[mask, ["Customer ID", "InvoiceDate", "line_revenue"]].copy()
    population["customer_id"] = population["Customer ID"].astype("int64")
    population["order_month"] = population["InvoiceDate"].dt.to_period("M")
    return population


# --- revenue -------------------------------------------------------------------------

REVENUE_DEFINITIONS = {
    "net": (
        "Net revenue (net of returns): sum of Quantity x Price over all product lines, "
        "including negative return lines, so refunds reduce the day they were booked."
    ),
    "gross": (
        "Gross sales: sum of Quantity x Price over positive product lines only, i.e. "
        "before returns are deducted."
    ),
}


def daily_revenue(
    frame: pd.DataFrame | None = None, days: int = REVENUE_SERIES_DAYS
) -> pd.DataFrame:
    """Trailing daily gross and net revenue for the last ``days`` days on record."""
    annotated = annotate(load() if frame is None else frame)
    products = annotated[annotated["is_product_line"]].copy()
    products["date"] = products["InvoiceDate"].dt.normalize()

    net = products.groupby("date")["line_revenue"].sum()
    gross = (
        products[products["Quantity"] > 0].groupby("date")["line_revenue"].sum()
    )
    series = pd.DataFrame({"net_revenue": net, "gross_sales": gross}).fillna(0.0)
    series = series.sort_index()
    if days:
        series = series.tail(days)
    return series.reset_index().rename(columns={"date": "date"})


# --- quality-audit slice -------------------------------------------------------------


def audit_slice(rows: int, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """A deterministic head slice of the raw table for the data-quality audit.

    A head slice (not a random sample) keeps the audit reproducible and keeps whole
    invoices together, which matters for the duplicate-row check.
    """
    raw = load() if frame is None else frame
    return raw.head(rows).copy()
