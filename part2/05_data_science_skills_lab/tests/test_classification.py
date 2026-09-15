"""F03/F04 - metrics and the live predictor's wiring."""

from __future__ import annotations

import pandas as pd

from skills_lab.benchmarks import classification
from skills_lab.data import titanic


def test_metrics_are_in_range_and_confusion_matches_test_rows(titanic_fixture):
    result = classification.run(titanic_fixture)
    for value in (result.accuracy, result.precision, result.recall, result.f1, result.roc_auc):
        assert 0.0 <= value <= 1.0
    assert sum(result.confusion.values()) == result.test_rows
    assert result.train_rows + result.test_rows == result.rows
    assert result.importances[0].share_pct >= result.importances[-1].share_pct
    assert abs(sum(i.share_pct for i in result.importances) - 100) < 100  # shares, not counts


def test_derived_features_are_recomputed_the_same_way_everywhere():
    frame = pd.DataFrame([{"sibsp": 2, "parch": 3}])
    derived = titanic.add_derived_features(frame)
    assert derived.loc[0, "family_size"] == 6
    assert derived.loc[0, "is_alone"] == "with family"
    alone = titanic.add_derived_features(pd.DataFrame([{"sibsp": 0, "parch": 0}]))
    assert alone.loc[0, "family_size"] == 1
    assert alone.loc[0, "is_alone"] == "alone"


def test_every_exposed_control_reaches_the_model_frame(titanic_fixture, monkeypatch):
    """Each UI control must arrive in the model's input frame with the right value.

    This deliberately does not assert that changing a field changes the prediction: a
    gradient-boosted ensemble can be locally flat or non-monotonic in any feature.
    """
    result = classification.run(titanic_fixture)
    captured = {}

    original = result.pipeline.predict_proba

    def spy(frame):
        captured["frame"] = frame.copy()
        return original(frame)

    monkeypatch.setattr(result.pipeline, "predict_proba", spy)

    classification.predict_profile(
        result.pipeline,
        pclass=2, sex="male", age=44.0, fare=77.5, sibsp=1, parch=2, embarked="Q",
    )
    frame = captured["frame"]
    assert list(frame.columns) == titanic.FEATURES
    row = frame.iloc[0]
    assert row["pclass"] == "2"
    assert row["sex"] == "male"
    assert row["age"] == 44.0
    assert row["fare"] == 77.5
    assert row["sibsp"] == 1
    assert row["parch"] == 2
    assert row["embarked"] == "Q"
    assert row["family_size"] == 4          # recomputed, not passed in
    assert row["is_alone"] == "with family"


def test_predictor_returns_a_probability_and_profiles_can_differ(titanic_fixture):
    result = classification.run(titanic_fixture)
    base = dict(pclass=3, sex="male", age=40, fare=8, sibsp=0, parch=0, embarked="S")
    favourable = dict(pclass=1, sex="female", age=20, fare=200, sibsp=0, parch=0, embarked="C")
    low = classification.predict_profile(result.pipeline, **base)
    high = classification.predict_profile(result.pipeline, **favourable)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert low != high  # representative profiles are distinguishable
