"""ULB credit-card fraud data (OpenML 1597) for the imbalanced-classification benchmark.

The full 284,807-row table is used: logistic regression on 29 features is fast, and
subsampling would either distort the base rate or leave too few positives to evaluate.
V1..V28 are anonymised PCA components published with the original dataset.
"""

from __future__ import annotations

import pandas as pd

from skills_lab.data.acquire import load_cached

TARGET = "Class"
COMPONENT_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = COMPONENT_FEATURES + ["Amount"]


def load() -> pd.DataFrame:
    raw = load_cached("creditcard")
    frame = raw[FEATURES + [TARGET]].copy()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce").astype("int64")
    return frame.dropna(subset=FEATURES + [TARGET])


def profile() -> dict[str, object]:
    frame = load()
    positives = int(frame[TARGET].sum())
    return {
        "rows": int(len(frame)),
        "features": len(FEATURES),
        "positives": positives,
        "positive_rate": positives / len(frame),
    }
