"""Production segmentation model: fit, persist, load (F05).

The fitted model, the fitted scaler and the ordered feature-column list are
persisted *together* as one bundle. That coupling is the project's leakage-safety
property: inference reproduces the training transform exactly, in the same column
order, using a scaler that is applied and never re-fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config, tournament

PRODUCTION_N_INIT = 15


@dataclass
class ModelBundle:
    """Everything inference needs, kept in one file so it cannot drift apart."""

    model: KMeans
    scaler: StandardScaler
    columns: list[str]
    k: int
    seed: int

    @property
    def centroids(self) -> np.ndarray:
        return self.model.cluster_centers_


def fit(X: np.ndarray, k: int = config.DEFAULT_K, seed: int = config.DEFAULT_SEED) -> KMeans:
    """Fit the production K-Means on the full standardized training matrix."""
    return KMeans(n_clusters=k, init="k-means++", n_init=PRODUCTION_N_INIT, random_state=seed).fit(X)


def evaluate(model: KMeans, X: np.ndarray) -> dict:
    """Score the production model on the shared evaluation matrix.

    The same matrix is used to fit the model, to run the tournament and to run
    the elbow sweep, which is what makes their silhouettes comparable. The source
    compared a full-data fit against subset-fitted entrants; that is not
    like-for-like and is not reproduced.
    """
    labels = model.predict(X)
    metrics = tournament.score_labels(X, labels)
    metrics["inertia"] = float(model.inertia_)
    metrics["n_customers"] = int(len(X))
    return metrics


def save(bundle: ModelBundle, path=None) -> None:
    path = path or config.MODEL_BUNDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "scaler": bundle.scaler,
            "columns": list(bundle.columns),
            "k": bundle.k,
            "seed": bundle.seed,
        },
        path,
    )


def load(path=None) -> ModelBundle:
    path = path or config.MODEL_BUNDLE
    if not path.exists():
        raise FileNotFoundError(
            f"No model bundle at {path}. Run `python -m segmentation.pipeline` first."
        )
    payload = joblib.load(path)
    return ModelBundle(
        model=payload["model"],
        scaler=payload["scaler"],
        columns=list(payload["columns"]),
        k=int(payload["k"]),
        seed=int(payload["seed"]),
    )
