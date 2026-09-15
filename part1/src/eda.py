"""Reusable helpers for exploring categorical association with the target.

Every feature in this dataset (and the target) is categorical, so a single
statistic - Cramer's V, built on the chi-square test of independence - is
used throughout instead of correlation.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Sequential single-hue ramp (blue) for a magnitude ranking, per the
# project's data-viz convention: darkest = strongest association.
BAR_COLOR = "#2a78d6"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V for two categorical series: 0 (independent) to 1 (perfect)."""
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency, correction=False)[0]
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim)))


def association_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cramer's V and chi-square test of each feature in X against y.

    Returns one row per feature, sorted by Cramer's V (strongest first),
    with columns: cramers_v, chi2, dof, p_value.
    """
    rows = []
    for col in X.columns:
        contingency = pd.crosstab(X[col], y)
        chi2, p_value, dof, _ = chi2_contingency(contingency, correction=False)
        rows.append({
            "feature": col,
            "cramers_v": cramers_v(X[col], y),
            "chi2": chi2,
            "dof": dof,
            "p_value": p_value,
        })
    return (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("cramers_v", ascending=False)
    )


def plot_association_ranking(table: pd.DataFrame, ax=None):
    """Horizontal bar chart of features ranked by Cramer's V, strongest on top."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ordered = table.sort_values("cramers_v")
    bars = ax.barh(ordered.index, ordered["cramers_v"], color=BAR_COLOR, height=0.7)

    for bar, value in zip(bars, ordered["cramers_v"]):
        ax.text(
            value + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", va="center", fontsize=9, color=MUTED_INK,
        )

    ax.set_xlabel("Cramer's V (association with class)")
    ax.set_xlim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return ax
