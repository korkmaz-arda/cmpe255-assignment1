"""Customer Segmentation & Clustering Intelligence Platform — data-science core.

This package is deliberately free of any UI-framework imports so that the whole
scientific lifecycle (data preparation, clustering, evaluation, inference,
retraining) can be imported, scripted and tested without Streamlit installed.
"""

__all__ = [
    "config",
    "data",
    "features",
    "tournament",
    "model",
    "projections",
    "personas",
    "elbow",
    "inference",
    "autoresearch",
    "pipeline",
    "engine",
]
