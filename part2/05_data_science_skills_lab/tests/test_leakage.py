"""The project's central claim: transformers never see the evaluation partition."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from skills_lab.benchmarks import classification, regression
from skills_lab.config import CLASSIFICATION_TEST_SIZE, SEED
from skills_lab.data import titanic


def test_imputer_and_scaler_statistics_come_from_training_rows_only(titanic_fixture):
    X = titanic_fixture[titanic.FEATURES]
    y = titanic_fixture[titanic.TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=CLASSIFICATION_TEST_SIZE, random_state=SEED, stratify=y
    )

    pipeline = classification.build_pipeline()
    pipeline.fit(X_train, y_train)

    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    fitted_medians = numeric.named_steps["impute"].statistics_
    expected = X_train[titanic.NUMERIC_FEATURES].median().to_numpy(dtype=float)
    assert np.allclose(fitted_medians, expected)

    fitted_means = numeric.named_steps["scale"].mean_
    imputed_train = numeric.named_steps["impute"].transform(
        X_train[titanic.NUMERIC_FEATURES]
    )
    assert np.allclose(fitted_means, imputed_train.mean(axis=0))

    # And the full-data medians differ, so the test above is not vacuous.
    assert not np.allclose(
        fitted_medians, X[titanic.NUMERIC_FEATURES].median().to_numpy(dtype=float)
    )


def test_unseen_category_does_not_raise(titanic_fixture):
    X = titanic_fixture[titanic.FEATURES].copy()
    y = titanic_fixture[titanic.TARGET]
    pipeline = classification.build_pipeline().fit(X, y)
    novel = X.head(1).copy()
    novel["embarked"] = "ZZ"
    assert 0.0 <= float(pipeline.predict_proba(novel)[0, 1]) <= 1.0


def test_regression_metrics_are_computed_in_dollars(ames_fixture):
    result = regression.run(ames_fixture)
    assert result.target_transform == "log1p on fit, expm1 before scoring"
    # Dollar-scale errors are orders of magnitude larger than log-scale errors would be.
    assert result.rmse > 1_000
    assert result.mae > 1_000
    assert -1.0 <= result.r2 <= 1.0
    residuals = [p["actual"] - p["predicted"] - p["residual"] for p in result.predictions]
    assert max(abs(r) for r in residuals) < 1e-6
