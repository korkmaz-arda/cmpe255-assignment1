"""Elbow and silhouette-versus-k analysis (F08).

K-Means is re-fit for every k in the sweep on the shared evaluation matrix,
recording inertia (WCSS) and silhouette. The source hard-highlighted k=5 as both
the elbow point and the silhouette peak regardless of what the data said; here
both are computed, and the pipeline reports them honestly even when they
disagree with the served k.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config


def knee_of(ks: list[int], inertias: list[float]) -> int:
    """Kneedle-style elbow: the k furthest from the chord joining the endpoints.

    Both axes are min-max normalised first so the choice does not depend on the
    units of inertia.
    """
    if len(ks) < 3:
        return ks[0]
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x = (x - x.min()) / (np.ptp(x) or 1.0)
    y = (y - y.min()) / (np.ptp(y) or 1.0)
    # Perpendicular distance to the line from the first to the last point.
    offsets = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(np.argmax(offsets))])


def sweep(X: np.ndarray, ks=config.K_SWEEP, seed: int = config.DEFAULT_SEED,
          served_k: int = config.DEFAULT_K) -> dict:
    """Run the k sweep and derive the empirical elbow and silhouette peak."""
    ks = list(ks)
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(X)
        rows.append(
            {
                "k": int(k),
                "wcss": float(model.inertia_),
                "silhouette": float(silhouette_score(X, model.labels_)),
            }
        )

    silhouette_peak = max(rows, key=lambda r: r["silhouette"])["k"]
    elbow = knee_of([r["k"] for r in rows], [r["wcss"] for r in rows])

    return {
        "rows": rows,
        "elbow_k": int(elbow),
        "silhouette_peak_k": int(silhouette_peak),
        "served_k": int(served_k),
        "agrees_with_served_k": bool(silhouette_peak == served_k),
    }


def verdict(sweep: dict) -> dict:
    """Summarise, honestly, how much empirical support the served k actually has.

    Two standard criteria are computed and they frequently disagree, so neither
    "the data proves k" nor "the data rejects k" is usually true. This returns
    the support level plus ready-made sentences, so every surface that discusses
    k tells the same story from the same numbers.

    A note that belongs with any such comparison: silhouette has a well-known
    bias toward small k, because merging groups raises mean inter-cluster
    distance. A silhouette peak at k=2 is therefore weak evidence for two
    segments; it is often just the coarsest split scoring the metric's favourite
    shape.
    """
    served = sweep["served_k"]
    peak = sweep["silhouette_peak_k"]
    knee = sweep["elbow_k"]

    by_silhouette = peak == served
    by_elbow = knee == served

    if by_silhouette and by_elbow:
        support = "both"
        headline = f"Both criteria select k={served}."
        detail = (
            f"The inertia knee and the silhouette peak both fall at k={served}, which is also the "
            f"served cluster count."
        )
    elif by_elbow:
        support = "elbow"
        headline = f"The inertia knee selects k={served}; silhouette prefers k={peak}."
        detail = (
            f"The two standard ways of choosing a segment count disagree. The elbow method — which "
            f"looks for the point where adding another segment stops buying you a much tighter "
            f"fit — picks k={served}, the served count. Silhouette, which measures how cleanly "
            f"separated the segments are, prefers k={peak} instead; but silhouette systematically "
            f"favours fewer, larger segments, so its vote for the coarsest split is weak evidence "
            f"on its own. k={served} is backed by one of the two criteria and by the business need "
            f"for distinct, actionable personas — it is not asserted against the data."
        )
    elif by_silhouette:
        support = "silhouette"
        headline = f"Silhouette selects k={served}; the inertia knee falls at k={knee}."
        detail = (
            f"The two standard criteria disagree. Silhouette peaks at k={served}, the served cluster "
            f"count, while the inertia knee falls at k={knee}."
        )
    else:
        support = "neither"
        headline = f"Neither criterion selects k={served}."
        detail = (
            f"Silhouette peaks at k={peak} and the inertia knee falls at k={knee}, while the served "
            f"model uses k={served}. Both figures are computed from this sweep rather than asserted. "
            f"k={served} is a product decision about how many personas the marketing team can act "
            f"on, and it is labelled as one rather than presented as an empirical finding."
        )

    return {
        "support": support,
        "headline": headline,
        "detail": detail,
        "served_k": served,
        "silhouette_peak_k": peak,
        "elbow_k": knee,
        "selected_by_silhouette": by_silhouette,
        "selected_by_elbow": by_elbow,
    }
