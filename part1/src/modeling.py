"""Model pipelines for the mushroom classification task.

Each pipeline bundles its own preprocessing with its estimator, so calling
`.fit(X_train, y_train)` handles encoding internally and nothing is ever fit
on the test set.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def _one_hot_pipeline(model) -> Pipeline:
    """One-hot encode all feature columns, then fit `model`."""
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), lambda X: list(X.columns))]
    )
    return Pipeline([("encode", encoder), ("model", model)])


def build_dummy_pipeline() -> Pipeline:
    """Always predicts the majority class - the floor any real model must beat."""
    return _one_hot_pipeline(DummyClassifier(strategy="most_frequent"))


def build_odor_only_pipeline() -> Pipeline:
    """Logistic regression on `odor` alone.

    EDA showed odor has by far the strongest association with the target
    (Cramer's V ~0.97), so this checks how far a single, easy-to-observe
    feature gets before adding model complexity.
    """
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), ["odor"])]
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("encode", encoder), ("model", model)])


def build_categorical_nb_pipeline() -> Pipeline:
    """Naive Bayes suited to categorical features directly (via ordinal codes).

    CategoricalNB models each feature's per-category probabilities natively,
    which fits this dataset's structure more directly than one-hot encoding.
    Unknown categories at prediction time are mapped to -1 rather than
    raising an error.
    """
    encoder = ColumnTransformer(
        [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          lambda X: list(X.columns))]
    )
    model = CategoricalNB()
    return Pipeline([("encode", encoder), ("model", model)])


def build_logistic_regression_pipeline() -> Pipeline:
    """A linear model whose coefficients can be read directly for interpretation."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return _one_hot_pipeline(model)


def build_decision_tree_pipeline(max_depth: int = 5) -> Pipeline:
    """A shallow tree whose learned rules can be read and sanity-checked."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_SEED)
    return _one_hot_pipeline(model)


def build_random_forest_pipeline() -> Pipeline:
    """An ensemble model to check whether extra capacity improves on a single tree."""
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    return _one_hot_pipeline(model)


def build_random_forest_on_columns(columns) -> Pipeline:
    """A Random Forest pipeline restricted to only `columns`.

    Used for ablations: dropping a column here means the model never sees it,
    unlike simply ignoring its importance after the fact.
    """
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), list(columns))]
    )
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    return Pipeline([("encode", encoder), ("model", model)])


def build_all_pipelines() -> dict[str, Pipeline]:
    """All planned model pipelines, unfitted, keyed by a short display name."""
    return {
        "Dummy (majority class)": build_dummy_pipeline(),
        "Odor-only": build_odor_only_pipeline(),
        "Categorical Naive Bayes": build_categorical_nb_pipeline(),
        "Logistic Regression": build_logistic_regression_pipeline(),
        "Decision Tree": build_decision_tree_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
    }
