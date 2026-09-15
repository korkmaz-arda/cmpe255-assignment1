"""Real-time single-customer segment classification (F09).

The input passes through the identical feature-engineering step and the
*persisted* scaler, then distances to every centroid are computed and the nearest
wins. Confidence is a normalized inverse-distance share, not a model
probability — it is a heuristic proximity score and is labelled as such
everywhere it is shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features


class ValidationError(ValueError):
    """Raised when a submitted attribute falls outside its permitted range."""


def validate(payload: dict) -> dict:
    """Clean and range-check a submitted customer record.

    Missing fields fall back to mid-range defaults; out-of-range values are
    rejected rather than silently clipped, so a caller always knows what was
    actually scored.
    """
    record: dict[str, float] = {}
    errors: list[str] = []

    for field, (low, high) in config.INPUT_RANGES.items():
        raw = payload.get(field, config.INPUT_DEFAULTS[field])
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{config.ATTRIBUTE_LABELS[field]}: '{raw}' is not a number.")
            continue
        if not np.isfinite(value):
            errors.append(f"{config.ATTRIBUTE_LABELS[field]}: value must be finite.")
        elif value < low or value > high:
            errors.append(
                f"{config.ATTRIBUTE_LABELS[field]}: {value:g} is outside the permitted "
                f"range {low:g}–{high:g}."
            )
        else:
            record[field] = value

    if errors:
        raise ValidationError(" ".join(errors))
    return record


def confidence_shares(distances: np.ndarray) -> np.ndarray:
    """Normalized inverse-distance shares.

    Not a probability: it cannot fall below 1/k, and it reflects only relative
    proximity to the centroids. Reported as a proximity score, never as a
    likelihood.
    """
    inverse = 1.0 / (distances + config.INFERENCE_EPSILON)
    return inverse / inverse.sum()


def classify(payload: dict, bundle, pca, persona_by_cluster: dict[int, dict]) -> dict:
    """Classify one customer and return everything the result card needs."""
    record = validate(payload)

    frame = pd.DataFrame([record])
    enriched = features.engineer(frame)
    # The persisted scaler is applied, never re-fitted, in the persisted column order.
    scaled = features.apply_scaler(bundle.scaler, enriched, bundle.columns)

    distances = np.linalg.norm(bundle.centroids - scaled[0], axis=1)
    cluster = int(np.argmin(distances))
    shares = confidence_shares(distances)
    coords = pca.transform(scaled)[0]

    persona = persona_by_cluster.get(cluster, {})
    return {
        "cluster": cluster,
        "confidence": float(shares[cluster]),
        "confidence_basis": "normalized inverse distance to the cluster centroids (heuristic proximity, not a probability)",
        "distance_to_centroid": float(distances[cluster]),
        "all_distances": [float(d) for d in distances],
        "pca_x": float(coords[0]),
        "pca_y": float(coords[1]),
        "input": record,
        "engineered": {c: float(enriched.iloc[0][c]) for c in config.ENGINEERED_FEATURES},
        "persona": {
            "name": persona.get("name", f"Cluster {cluster}"),
            "tagline": persona.get("tagline", ""),
            "badge": persona.get("badge", "•"),
            "color": persona.get("color", config.FALLBACK_COLOR),
            "description": persona.get("description", ""),
            "strategy": persona.get("strategy", ""),
            "match_quality": persona.get("match_quality"),
            "weak_match": persona.get("weak_match", False),
        },
    }
