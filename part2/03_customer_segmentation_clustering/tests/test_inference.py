"""Single-customer classification: validation, consistency and confidence."""

from __future__ import annotations

import numpy as np
import pytest

from segmentation import config, features, inference, model, projections


@pytest.fixture(scope="module")
def served(request):
    import pandas as pd
    from pathlib import Path
    from segmentation import data, personas

    raw = pd.read_csv(Path(__file__).parent / "fixtures" / "marketing_campaign_sample.csv", sep="\t")
    frame = data.prepare(raw)
    enriched, scaler, X = features.build_training_matrix(frame)
    kmeans = model.fit(X, k=4, seed=1)
    bundle = model.ModelBundle(model=kmeans, scaler=scaler,
                              columns=list(config.FEATURE_COLUMNS), k=4, seed=1)
    pca, _ = projections.fit_pca(X, seed=1)
    records = personas.profile(enriched, kmeans.labels_)
    return bundle, pca, personas.by_cluster(records), enriched, X


def _valid_payload(**overrides) -> dict:
    payload = dict(config.INPUT_DEFAULTS)
    payload.update(overrides)
    return payload


def test_missing_fields_fall_back_to_defaults():
    record = inference.validate({"age": 30})
    assert record["age"] == 30
    assert record["income_k"] == config.INPUT_DEFAULTS["income_k"]
    assert set(record) == set(config.BASE_FEATURES)


@pytest.mark.parametrize(
    "field,value",
    [("age", 5), ("age", 200), ("income_k", 1e9), ("spending_score", 0),
     ("recency_days", -1), ("discount_sensitivity", 1.5), ("household_size", 99)],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(inference.ValidationError):
        inference.validate(_valid_payload(**{field: value}))


def test_non_numeric_input_is_rejected():
    with pytest.raises(inference.ValidationError, match="not a number"):
        inference.validate(_valid_payload(age="old"))


def test_nan_input_is_rejected():
    with pytest.raises(inference.ValidationError, match="finite"):
        inference.validate(_valid_payload(age=float("nan")))


def test_all_errors_are_reported_together():
    with pytest.raises(inference.ValidationError) as excinfo:
        inference.validate(_valid_payload(age=200, income_k=999))
    message = str(excinfo.value)
    assert "Age" in message and "income" in message.lower()


def test_confidence_shares_sum_to_one_and_favour_the_nearest():
    shares = inference.confidence_shares(np.array([0.5, 2.0, 4.0]))
    assert shares.sum() == pytest.approx(1.0)
    assert np.argmax(shares) == 0


def test_confidence_never_falls_below_one_over_k():
    """It is a normalized inverse-distance share, so 1/k is its floor."""
    shares = inference.confidence_shares(np.array([3.0, 3.0, 3.0, 3.0, 3.0]))
    assert shares.max() == pytest.approx(0.2)


def test_zero_distance_is_guarded():
    shares = inference.confidence_shares(np.array([0.0, 1.0, 2.0]))
    assert np.isfinite(shares).all()
    assert shares[0] > 0.9


def test_classification_agrees_with_the_training_assignment(served):
    """A training customer, resubmitted through the inference path, must land in
    the cluster the model assigned it. This is the end-to-end guarantee that the
    persisted scaler and column order reproduce the training transform."""
    bundle, pca, by_cluster, enriched, X = served
    expected = bundle.model.labels_

    for row_index in range(0, len(enriched), 37):
        payload = {field: float(enriched.iloc[row_index][field]) for field in config.BASE_FEATURES}
        payload = {
            field: min(max(value, config.INPUT_RANGES[field][0]), config.INPUT_RANGES[field][1])
            for field, value in payload.items()
        }
        result = inference.classify(payload, bundle, pca, by_cluster)
        assert result["cluster"] == expected[row_index]


def test_classification_is_deterministic(served):
    bundle, pca, by_cluster, _, _ = served
    payload = _valid_payload()
    first = inference.classify(payload, bundle, pca, by_cluster)
    second = inference.classify(payload, bundle, pca, by_cluster)
    assert first["cluster"] == second["cluster"]
    assert first["confidence"] == pytest.approx(second["confidence"])
    assert first["pca_x"] == pytest.approx(second["pca_x"])


def test_result_carries_persona_narrative_and_pca_placement(served):
    bundle, pca, by_cluster, _, _ = served
    result = inference.classify(_valid_payload(), bundle, pca, by_cluster)
    assert result["persona"]["name"]
    assert result["persona"]["strategy"]
    assert np.isfinite([result["pca_x"], result["pca_y"]]).all()
    assert result["distance_to_centroid"] == pytest.approx(min(result["all_distances"]))
    assert len(result["all_distances"]) == bundle.k


def test_unreachable_inputs_actually_change_the_outcome(served):
    """Discount sensitivity and household size had no UI control in the source.

    They feed real features, so moving them must move the engineered values.
    """
    bundle, pca, by_cluster, _, _ = served
    low = inference.classify(_valid_payload(discount_sensitivity=0.02), bundle, pca, by_cluster)
    high = inference.classify(_valid_payload(discount_sensitivity=0.95), bundle, pca, by_cluster)
    assert low["engineered"]["deal_affinity"] != high["engineered"]["deal_affinity"]
