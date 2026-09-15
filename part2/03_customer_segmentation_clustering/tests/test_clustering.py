"""Tournament scoring, the DBSCAN parameter rule, and the k sweep."""

from __future__ import annotations

import numpy as np
import pytest

from segmentation import config, elbow, features, tournament


@pytest.fixture(scope="module")
def matrix(request):
    from segmentation import data
    import pandas as pd
    from pathlib import Path

    raw = pd.read_csv(Path(__file__).parent / "fixtures" / "marketing_campaign_sample.csv", sep="\t")
    _, _, X = features.build_training_matrix(data.prepare(raw))
    return X


def test_degenerate_partition_reports_undefined_metrics():
    X = np.random.default_rng(0).normal(size=(30, 3))
    result = tournament.score_labels(X, np.zeros(30, dtype=int))
    assert result["degenerate"] is True
    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None


def test_noise_points_are_excluded_from_metrics():
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)), np.full((5, 2), 50.0)])
    labels = np.array([0] * 10 + [1] * 10 + [tournament.NOISE_LABEL] * 5)
    result = tournament.score_labels(X, labels)
    assert result["n_clusters"] == 2
    assert result["noise_ratio"] == pytest.approx(5 / 25)
    assert result["silhouette"] > 0.9  # the far-away noise points did not distort it


def test_silhouette_of_degenerate_is_the_true_lower_bound():
    X = np.random.default_rng(0).normal(size=(20, 2))
    assert tournament.silhouette_of(X, np.zeros(20, dtype=int)) == -1.0


def test_dbscan_epsilon_is_a_pure_function_of_the_matrix(matrix):
    """The rule must be deterministic and must never consult a clustering metric."""
    min_samples = tournament.dbscan_min_samples(matrix.shape[1])
    first = tournament.dbscan_epsilon(matrix, min_samples)
    second = tournament.dbscan_epsilon(matrix, min_samples)
    assert first == second
    assert first > 0


def test_dbscan_min_samples_follows_the_dimensionality_rule():
    assert tournament.dbscan_min_samples(12) == 24
    assert tournament.dbscan_min_samples(1) == 4  # floor


def test_tournament_runs_every_family_and_sorts_by_silhouette(matrix):
    rows = tournament.run(matrix, k=3, seed=config.DEFAULT_SEED)
    assert len(rows) == 5
    scored = [r["silhouette"] for r in rows if r["silhouette"] is not None]
    assert scored == sorted(scored, reverse=True)
    # Degenerate entrants must sort last, never take a rank on a placeholder score.
    degenerate_positions = [i for i, r in enumerate(rows) if r["degenerate"]]
    assert all(pos >= len(scored) for pos in degenerate_positions)


def test_tournament_is_deterministic(matrix):
    a = tournament.run(matrix, k=3, seed=7)
    b = tournament.run(matrix, k=3, seed=7)
    assert [r["silhouette"] for r in a] == [r["silhouette"] for r in b]


def test_elbow_sweep_reports_the_empirical_peak_not_a_fixed_k(matrix):
    sweep = elbow.sweep(matrix, ks=(2, 3, 4), seed=1, served_k=3)
    assert [r["k"] for r in sweep["rows"]] == [2, 3, 4]
    # WCSS must fall monotonically as k rises.
    wcss = [r["wcss"] for r in sweep["rows"]]
    assert wcss == sorted(wcss, reverse=True)
    peak = max(sweep["rows"], key=lambda r: r["silhouette"])["k"]
    assert sweep["silhouette_peak_k"] == peak
    assert sweep["agrees_with_served_k"] == (peak == 3)


def test_knee_detection_finds_the_bend():
    ks = [2, 3, 4, 5, 6, 7]
    inertias = [100.0, 60.0, 40.0, 38.0, 36.5, 35.5]  # sharp bend at k=4
    assert elbow.knee_of(ks, inertias) == 4


def test_verdict_credits_the_elbow_when_it_selects_the_served_k():
    """Regression guard: the knee agreeing with k must read as support, not as
    evidence against. An earlier wording listed it among the counter-evidence."""
    from segmentation import elbow as elbow_module

    verdict = elbow_module.verdict(
        {"served_k": 5, "silhouette_peak_k": 2, "elbow_k": 5, "rows": []}
    )
    assert verdict["support"] == "elbow"
    assert verdict["selected_by_elbow"] is True
    assert "picks k=5" in verdict["detail"]
    assert "not asserted against" in verdict["detail"]
    # It must not claim the data rejects the served k.
    assert "does not pick" not in verdict["headline"].lower()


def test_verdict_reports_full_agreement():
    from segmentation import elbow as elbow_module

    verdict = elbow_module.verdict(
        {"served_k": 4, "silhouette_peak_k": 4, "elbow_k": 4, "rows": []}
    )
    assert verdict["support"] == "both"
    assert verdict["selected_by_silhouette"] and verdict["selected_by_elbow"]


def test_verdict_labels_an_unsupported_k_as_a_product_decision():
    from segmentation import elbow as elbow_module

    verdict = elbow_module.verdict(
        {"served_k": 5, "silhouette_peak_k": 2, "elbow_k": 3, "rows": []}
    )
    assert verdict["support"] == "neither"
    assert "product decision" in verdict["detail"]
