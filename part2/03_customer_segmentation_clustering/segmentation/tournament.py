"""Multi-algorithm clustering tournament and internal-validity scoring (F03).

Five clustering families are fit on the same standardized matrix and scored on
the same internal-validity metrics, producing a ranked leaderboard. Every
entrant keeps a fixed, documented configuration: no entrant's hyperparameters are
searched against silhouette, because tuning one entrant on the ranking metric
would give it an advantage the others never receive and would make the
leaderboard meaningless.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from . import config

NOISE_LABEL = -1


@dataclass
class Entry:
    """One leaderboard row."""

    algorithm: str
    family: str
    formulation: str
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    n_clusters: int
    noise_ratio: float
    fit_seconds: float
    degenerate: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute internal-validity metrics, excluding noise-labelled points.

    A partition with fewer than two surviving clusters is *degenerate*: the three
    metrics are mathematically undefined there. The source encoded that case as
    silhouette 0 / Davies-Bouldin 99, which silently ranks a failed run above or
    below real ones. Here the metrics are reported as ``None`` and the row is
    flagged, so a degenerate result reads as a failure rather than a score.
    """
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    noise_ratio = float((~mask).mean())
    kept = labels[mask]
    unique = np.unique(kept)

    if len(unique) < 2 or mask.sum() < 3:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "n_clusters": int(len(unique)),
            "noise_ratio": noise_ratio,
            "degenerate": True,
        }

    Xk = X[mask]
    return {
        "silhouette": float(silhouette_score(Xk, kept)),
        "davies_bouldin": float(davies_bouldin_score(Xk, kept)),
        "calinski_harabasz": float(calinski_harabasz_score(Xk, kept)),
        "n_clusters": int(len(unique)),
        "noise_ratio": noise_ratio,
        "degenerate": False,
    }


def silhouette_of(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with degenerate partitions scored as -1 (worst possible).

    Used inside the AutoResearch hill climb, where a candidate that collapses the
    partition must be rejected rather than crash the search. -1 is the true lower
    bound of the silhouette coefficient, so this is a valid ordering, not a
    sentinel.
    """
    result = score_labels(X, labels)
    return -1.0 if result["silhouette"] is None else result["silhouette"]


# --------------------------------------------------------------------------- #
# DBSCAN parameter rule
# --------------------------------------------------------------------------- #

def dbscan_min_samples(n_features: int) -> int:
    """The conventional ``2 * dimensionality`` rule of thumb."""
    return max(4, 2 * n_features)


def dbscan_epsilon(X: np.ndarray, min_samples: int) -> float:
    """Choose eps by the standard k-distance knee heuristic.

    Every point's distance to its ``min_samples``-th nearest neighbour is sorted
    ascending; the knee of that curve is the point of maximum distance from the
    straight line joining its endpoints. This is a purely geometric property of
    the data — it never consults silhouette or any other clustering metric, so
    DBSCAN gets no optimisation advantage over the other tournament entrants.
    """
    n_neighbors = min(min_samples, len(X) - 1)
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(X).kneighbors(X)
    curve = np.sort(distances[:, -1])

    x = np.arange(len(curve), dtype=float)
    x0, y0, x1, y1 = x[0], curve[0], x[-1], curve[-1]
    denominator = np.hypot(y1 - y0, x1 - x0)
    if denominator == 0:  # pragma: no cover - degenerate distance curve
        return float(np.median(curve))
    # Perpendicular distance from each point of the curve to the endpoint chord.
    offsets = np.abs((y1 - y0) * x - (x1 - x0) * curve + x1 * y0 - y1 * x0) / denominator
    return float(curve[int(np.argmax(offsets))])


def dbscan_diagnostics(X: np.ndarray) -> dict:
    """The k-distance curve and chosen eps, for the auditable admin panel."""
    min_samples = dbscan_min_samples(X.shape[1])
    n_neighbors = min(min_samples, len(X) - 1)
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(X).kneighbors(X)
    curve = np.sort(distances[:, -1])
    eps = dbscan_epsilon(X, min_samples)
    # Downsample the curve for plotting; the knee index is preserved by value.
    step = max(1, len(curve) // 300)
    return {
        "min_samples": int(min_samples),
        "eps": eps,
        "rule": (
            f"eps = knee of the sorted {min_samples}-nearest-neighbour distance curve "
            f"(maximum perpendicular distance to the endpoint chord); "
            f"min_samples = 2 x n_features = {min_samples}."
        ),
        "k_distance_curve": [float(v) for v in curve[::step]],
        "k_distance_index": [int(i) for i in range(0, len(curve), step)],
    }


# --------------------------------------------------------------------------- #
# Entrants
# --------------------------------------------------------------------------- #

def build_entrants(X: np.ndarray, k: int, seed: int) -> list[tuple[str, str, str, callable]]:
    """The five clustering families with their fixed configurations."""
    min_samples = dbscan_min_samples(X.shape[1])
    eps = dbscan_epsilon(X, min_samples)

    def fit_kmeans():
        return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit_predict(X)

    def fit_gmm():
        return GaussianMixture(
            n_components=k, covariance_type="full", random_state=seed
        ).fit_predict(X)

    def fit_agglomerative():
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

    def fit_dbscan():
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    def fit_spectral():
        return SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", random_state=seed, assign_labels="kmeans"
        ).fit_predict(X)

    return [
        ("K-Means", "Centroid partitioning", f"k={k}, k-means++, 10 restarts", fit_kmeans),
        ("Gaussian Mixture", "Probabilistic mixture", f"{k} components, full covariance", fit_gmm),
        ("Agglomerative", "Hierarchical", f"{k} clusters, Ward linkage", fit_agglomerative),
        ("DBSCAN", "Density-based", f"eps={eps:.3f} (k-distance knee), min_samples={min_samples}", fit_dbscan),
        ("Spectral", "Graph / spectral", f"{k} clusters, nearest-neighbour affinity", fit_spectral),
    ]


def run(X: np.ndarray, k: int = config.DEFAULT_K, seed: int = config.DEFAULT_SEED) -> list[dict]:
    """Fit and score every entrant on the shared evaluation matrix.

    Returns rows sorted by silhouette descending; degenerate entrants sort last.
    """
    rows: list[Entry] = []
    for algorithm, family, formulation, fit in build_entrants(X, k, seed):
        started = time.perf_counter()
        try:
            labels = fit()
            elapsed = time.perf_counter() - started
            metrics = score_labels(X, labels)
            note = "" if not metrics["degenerate"] else "Fewer than two clusters — metrics undefined."
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = time.perf_counter() - started
            metrics = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None,
                       "n_clusters": 0, "noise_ratio": 0.0, "degenerate": True}
            note = f"Fit failed: {exc}"

        rows.append(
            Entry(
                algorithm=algorithm,
                family=family,
                formulation=formulation,
                silhouette=metrics["silhouette"],
                davies_bouldin=metrics["davies_bouldin"],
                calinski_harabasz=metrics["calinski_harabasz"],
                n_clusters=metrics["n_clusters"],
                noise_ratio=metrics["noise_ratio"],
                fit_seconds=float(elapsed),
                degenerate=metrics["degenerate"],
                note=note,
            )
        )

    rows.sort(key=lambda r: (r.silhouette is not None, r.silhouette or -np.inf), reverse=True)
    return [r.to_dict() for r in rows]


def best_entry(rows: list[dict]) -> dict | None:
    """The highest-silhouette non-degenerate entrant, if any."""
    scored = [r for r in rows if r.get("silhouette") is not None]
    return max(scored, key=lambda r: r["silhouette"]) if scored else None
