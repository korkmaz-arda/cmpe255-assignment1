"""Reading fitted models: coefficients, tree rules, and feature importances.

Each one-hot pipeline's encoder names its output columns like
"onehot__cap-shape_x" (original column + one category). To compare features
on equal footing across models, the helpers below group each encoded
column's weight back to its original feature name.
"""

import pandas as pd
from sklearn.tree import export_text


def _encoded_feature_names(pipeline) -> list[str]:
    """Encoded column names with the ColumnTransformer prefix stripped."""
    raw = pipeline.named_steps["encode"].get_feature_names_out()
    return [name.split("__", 1)[1] for name in raw]


def _original_column(encoded_name: str) -> str:
    """Map an encoded column like 'cap-shape_x' back to 'cap-shape'."""
    return encoded_name.rsplit("_", 1)[0]


def top_logistic_regression_coefficients(pipeline, top_n: int = 15) -> pd.DataFrame:
    """The `top_n` one-hot categories with the largest |coefficient|.

    A positive coefficient pushes the prediction toward poisonous (the
    positive class); negative pushes toward edible.
    """
    encoded_names = _encoded_feature_names(pipeline)
    coefs = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"encoded_feature": encoded_names, "coefficient": coefs})
    table["feature"] = table["encoded_feature"].apply(_original_column)
    return (
        table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)
        .head(top_n)
        .set_index("encoded_feature")[["feature", "coefficient"]]
    )


def feature_group_importance(pipeline, importances) -> pd.Series:
    """Sum a per-encoded-column importance array up to the original feature."""
    encoded_names = _encoded_feature_names(pipeline)
    grouped = pd.Series(importances, index=encoded_names).groupby(_original_column).sum()
    return grouped.sort_values(ascending=False)


def logistic_regression_feature_importance(pipeline) -> pd.Series:
    """Sum of |coefficient| per original feature - overall influence, any direction."""
    coefs = abs(pipeline.named_steps["model"].coef_[0])
    return feature_group_importance(pipeline, coefs)


def random_forest_feature_importance(pipeline) -> pd.Series:
    """Gini importance per original feature, summed across its one-hot columns."""
    importances = pipeline.named_steps["model"].feature_importances_
    return feature_group_importance(pipeline, importances)


def decision_tree_rules(pipeline, max_depth: int | None = None) -> str:
    """The tree's learned rules as readable if/then text."""
    encoded_names = _encoded_feature_names(pipeline)
    return export_text(
        pipeline.named_steps["model"],
        feature_names=encoded_names,
        max_depth=max_depth,
    )


def decision_tree_feature_importance(pipeline) -> pd.Series:
    """Gini importance per original feature, summed across its one-hot columns."""
    importances = pipeline.named_steps["model"].feature_importances_
    return feature_group_importance(pipeline, importances)


def compare_top_features(rankings: dict, top_n: int = 5) -> pd.DataFrame:
    """How often each feature lands in the top `top_n` of several rankings.

    `rankings` maps a model name to a `pd.Series` of feature -> importance
    (already sorted descending), e.g. the outputs of the functions above.
    Returns one row per feature that appears in any top list, with its rank
    (1 = most important) in each model and a count of how many models rank
    it in their top `top_n`.
    """
    top_lists = {name: ranking.head(top_n) for name, ranking in rankings.items()}
    all_features = sorted(set().union(*[set(t.index) for t in top_lists.values()]))

    table = pd.DataFrame(index=all_features)
    for name, ranking in rankings.items():
        rank = pd.Series(range(1, len(ranking) + 1), index=ranking.index)
        in_top_n = rank.index.isin(top_lists[name].index)
        table[name] = rank.where(in_top_n).reindex(all_features)

    table["in_top_n_count"] = table.notna().sum(axis=1)
    return table.sort_values("in_top_n_count", ascending=False)
