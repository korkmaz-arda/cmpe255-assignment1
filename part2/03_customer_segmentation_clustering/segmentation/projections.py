"""2D projections of the standardized feature space (F06).

PCA is fitted on the full matrix and *persisted*, because it is reused at
inference to place a newly classified customer on the same scatter canvas.
t-SNE has no out-of-sample transform, so it is fitted on a bounded sample and is
deliberately not persisted — which is exactly why the live-prediction marker can
only be drawn in PCA mode.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import config


def fit_pca(X: np.ndarray, seed: int = config.DEFAULT_SEED) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    return pca, coords


def explained_variance(pca: PCA) -> dict:
    ratios = [float(v) for v in pca.explained_variance_ratio_]
    return {"components": ratios, "total": float(sum(ratios))}


def fit_tsne(X: np.ndarray, seed: int = config.DEFAULT_SEED) -> np.ndarray:
    """t-SNE on a bounded sample; perplexity is clamped for small inputs."""
    perplexity = min(config.TSNE_PERPLEXITY, max(5.0, (len(X) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        max_iter=1000,
    ).fit_transform(X)


def save_pca(pca: PCA, path=None) -> None:
    path = path or config.PCA_BUNDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, path)


def load_pca(path=None) -> PCA:
    path = path or config.PCA_BUNDLE
    if not path.exists():
        raise FileNotFoundError(f"No fitted PCA at {path}. Run the pipeline first.")
    return joblib.load(path)


def build_scatter(frame, X, labels, pca_coords, seed: int = config.DEFAULT_SEED,
                  n_sample: int = config.PROJECTION_SAMPLE) -> dict:
    """Assemble the sampled scatter dataset carrying both coordinate pairs.

    Exactly the sampled customers get t-SNE coordinates, because t-SNE cannot be
    applied out of sample. The sample is drawn deterministically from the seed.
    """
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(frame))
    idx = np.sort(rng.choice(len(frame), size=n_sample, replace=False))

    tsne_coords = fit_tsne(X[idx], seed=seed)

    points = []
    for position, row_index in enumerate(idx):
        row = frame.iloc[row_index]
        points.append(
            {
                "customer_id": int(row["customer_id"]),
                "cluster": int(labels[row_index]),
                "pca_x": float(pca_coords[row_index, 0]),
                "pca_y": float(pca_coords[row_index, 1]),
                "tsne_x": float(tsne_coords[position, 0]),
                "tsne_y": float(tsne_coords[position, 1]),
                **{col: float(row[col]) for col in config.BASE_FEATURES},
            }
        )
    return {"n_sample": n_sample, "seed": seed, "points": points}
