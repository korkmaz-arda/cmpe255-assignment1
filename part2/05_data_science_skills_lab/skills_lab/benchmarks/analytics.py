"""F08/F09/F10 - business analytics on real Online Retail II transactions.

Cohort retention and the revenue series are computed from the transaction log. The funnel
is the single illustrative element in the project and is labelled as such everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from skills_lab.config import REVENUE_SERIES_DAYS, SEASONAL_PERIOD
from skills_lab.data import illustrative, retail


@dataclass
class CohortRetention:
    matrix: pd.DataFrame          # rows = cohort month, cols = period index, values = %
    sizes: pd.Series              # customers acquired per cohort
    population_rule: str

    def to_records(self) -> list[dict]:
        records = []
        for cohort, row in self.matrix.iterrows():
            values = {
                f"m{int(period)}": (None if pd.isna(v) else round(float(v), 1))
                for period, v in row.items()
            }
            records.append(
                {"cohort": str(cohort), "customers": int(self.sizes[cohort]), **values}
            )
        return records


@dataclass
class RevenueSeries:
    frame: pd.DataFrame           # date, net_revenue, gross_sales, trend, seasonal
    definitions: dict[str, str]
    days: int
    seasonal_period: int


@dataclass
class Funnel:
    stages: list[dict]
    label: str


@dataclass
class AnalyticsResult:
    retention: CohortRetention
    revenue: RevenueSeries
    funnel: Funnel
    business_events: dict

    def payload(self) -> dict:
        return {
            "dataset": "Online Retail II (UCI), real transaction log",
            "cohort_retention": {
                "population_rule": self.retention.population_rule,
                "cohorts": len(self.retention.sizes),
                "rows": self.retention.to_records(),
            },
            "revenue_series": {
                "days": self.revenue.days,
                "definitions": self.revenue.definitions,
                "net_revenue_total": round(
                    float(self.revenue.frame["net_revenue"].sum()), 2
                ),
                "gross_sales_total": round(
                    float(self.revenue.frame["gross_sales"].sum()), 2
                ),
                "seasonal_period_days": self.revenue.seasonal_period,
            },
            "checkout_funnel": {
                "label": self.funnel.label,
                "stages": self.funnel.stages,
            },
            "business_events": self.business_events,
        }


def cohort_retention(population: pd.DataFrame | None = None) -> CohortRetention:
    """Monthly acquisition cohorts and the share still purchasing in each later month."""
    data = retail.retention_population() if population is None else population

    first_month = data.groupby("customer_id")["order_month"].min().rename("cohort")
    joined = data.join(first_month, on="customer_id")
    joined["period"] = (
        joined["order_month"].astype("int64") - joined["cohort"].astype("int64")
    )

    active = (
        joined.groupby(["cohort", "period"])["customer_id"].nunique().unstack(fill_value=0)
    )
    sizes = joined.groupby("cohort")["customer_id"].nunique()

    # A cohort is only observable for as many periods as the data covers.
    last_period = joined["order_month"].max().ordinal
    observable = {
        cohort: last_period - cohort.ordinal for cohort in sizes.index
    }
    matrix = active.div(sizes, axis=0) * 100
    for cohort, limit in observable.items():
        matrix.loc[cohort, [c for c in matrix.columns if c > limit]] = np.nan

    matrix.index = [str(period) for period in matrix.index]
    sizes.index = [str(period) for period in sizes.index]
    return CohortRetention(
        matrix=matrix.round(1),
        sizes=sizes,
        population_rule=retail.RETENTION_POPULATION_RULE,
    )


def revenue_series(days: int = REVENUE_SERIES_DAYS) -> RevenueSeries:
    """Trailing daily revenue with a genuine weekly seasonal decomposition."""
    frame = retail.daily_revenue(days=days).copy()
    series = frame.set_index("date")["net_revenue"].asfreq("D").fillna(0.0)

    decomposition = seasonal_decompose(
        series, model="additive", period=SEASONAL_PERIOD, extrapolate_trend="period"
    )
    frame = frame.set_index("date")
    frame["trend"] = decomposition.trend
    frame["seasonal"] = decomposition.seasonal
    frame["residual"] = decomposition.resid
    return RevenueSeries(
        frame=frame.reset_index(),
        definitions=retail.REVENUE_DEFINITIONS,
        days=days,
        seasonal_period=SEASONAL_PERIOD,
    )


def checkout_funnel() -> Funnel:
    stages = []
    top = illustrative.FUNNEL_STAGES[0]["users"]
    previous = None
    for stage in illustrative.FUNNEL_STAGES:
        users = stage["users"]
        stages.append(
            {
                "stage": stage["stage"],
                "users": users,
                "share_of_top_pct": users / top * 100,
                "step_conversion_pct": (
                    100.0 if previous is None else users / previous * 100
                ),
                "step_dropoff_pct": (
                    0.0 if previous is None else (previous - users) / previous * 100
                ),
            }
        )
        previous = users
    return Funnel(stages=stages, label=illustrative.FUNNEL_LABEL)


def run() -> AnalyticsResult:
    return AnalyticsResult(
        retention=cohort_retention(),
        revenue=revenue_series(),
        funnel=checkout_funnel(),
        business_events=retail.business_event_summary(),
    )
