"""S04 - CRISP-DM methodology report, written from the live computed results.

Every figure in this report is read from the results currently in memory. Nothing is
transcribed prose, so the report cannot drift away from what the app just computed.
"""

from __future__ import annotations

from skills_lab.catalog import View, skill_count
from skills_lab.config import (
    CLASSIFICATION_TEST_SIZE,
    FRAUD_TEST_SIZE,
    FRAUD_VALIDATION_SIZE,
    GB_PARAMS,
    QUALITY_WEIGHTS,
    REGRESSION_TEST_SIZE,
    RF_PARAMS,
    SEED,
)
from skills_lab.data import retail


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_report(lab) -> list[tuple[str, str]]:
    """Return the six CRISP-DM phases as (heading, markdown) pairs."""
    cls = lab.result(View.CLASSIFICATION)
    reg = lab.result(View.REGRESSION)
    fra = lab.result(View.FRAUD)
    ana = lab.result(View.ANALYTICS)
    qual = lab.result(View.QUALITY)

    missing = [
        name
        for name, value in (
            ("classification", cls), ("regression", reg), ("fraud", fra),
            ("analytics", ana), ("quality", qual),
        )
        if value is None
    ]
    preamble = (
        ""
        if not missing
        else f"\n\n> Not all workspaces are available ({', '.join(missing)}); "
        "their figures are omitted below rather than substituted.\n"
    )

    business = f"""
This is a teaching workbench, not a production modelling system. Its goal is to show
*correct methodology* on real public data: split before you fit, choose a metric that
matches the decision, set decision thresholds deliberately, and separate statistical
significance from practical significance.

The catalog currently holds **{skill_count()} skills** across four categories, and every one
of them routes to a workspace that actually computes a result.{preamble}
"""

    data_understanding_rows = []
    if cls:
        data_understanding_rows.append(
            f"- **Titanic** (OpenML 40945): {cls.rows:,} passengers, "
            f"{cls.positive_rate * 100:.1f}% survived, "
            f"{cls.age_missing_rate * 100:.1f}% of ages missing in the source data."
        )
    if reg:
        data_understanding_rows.append(
            f"- **Ames house sales** (OpenML 42165): {reg.rows:,} sales, median price "
            f"${reg.price_median:,.0f}. Living area and basement area correlate at "
            f"{reg.area_basement_correlation:.2f}."
        )
    if fra:
        data_understanding_rows.append(
            f"- **Credit-card fraud** (OpenML 1597): {fra.rows:,} transactions, "
            f"{fra.positives} frauds - a {fra.positive_rate * 100:.3f}% base rate."
        )
    if ana:
        data_understanding_rows.append(
            f"- **Online Retail II** (UCI): {ana.business_events['rows']:,} transaction "
            f"lines, of which {ana.business_events['credit_notes']:,} are credit notes and "
            f"{ana.business_events['return_lines']:,} are return lines - business events, "
            f"not data defects."
        )
    if qual:
        data_understanding_rows.append(
            f"- **Quality-audit slice**: the first {qual.rows:,} Online Retail II lines, "
            f"profiled across {qual.columns} columns."
        )

    data_prep = f"""
Every supervised workspace splits first and fits transformers afterwards, inside a
scikit-learn `Pipeline`, so no imputation or scaling statistic can be estimated from data the
model is later scored on.

- Numeric block: median imputation, then standardisation.
- Categorical block: most-frequent imputation, then one-hot encoding with
  `handle_unknown="ignore"` so an unseen level cannot raise at serving time.
- Engineered features (Titanic): `family_size = sibsp + parch + 1` and an
  `is_alone` indicator, recomputed identically for the live predictor.
- Post-outcome columns (`boat`, `body`) are excluded from the Titanic features; they record
  what happened after the sinking and would leak the target.
- Retail retention population: {retail.RETENTION_POPULATION_RULE}
- Revenue is reported as net of returns, with gross sales shown alongside so the definition
  is never ambiguous.
"""

    modeling_rows = [
        f"- **Classification**: gradient-boosted trees "
        f"({GB_PARAMS['n_estimators']} stages, depth {GB_PARAMS['max_depth']}, "
        f"learning rate {GB_PARAMS['learning_rate']}), stratified "
        f"{int((1 - CLASSIFICATION_TEST_SIZE) * 100)}/{int(CLASSIFICATION_TEST_SIZE * 100)} "
        f"hold-out, seed {SEED}.",
        f"- **Regression**: random forest ({RF_PARAMS['n_estimators']} trees, max depth "
        f"{RF_PARAMS['max_depth']}) fitted on `log1p(SalePrice)`; predictions are "
        f"back-transformed with `expm1` before any metric is computed, so errors are in "
        f"dollars. Hold-out {int(REGRESSION_TEST_SIZE * 100)}%.",
        f"- **Imbalanced**: two logistic regressions identical except for `class_weight`, on a "
        f"stratified {int((1 - FRAUD_TEST_SIZE - FRAUD_VALIDATION_SIZE) * 100)}/"
        f"{int(FRAUD_VALIDATION_SIZE * 100)}/{int(FRAUD_TEST_SIZE * 100)} train/validation/test "
        f"split. No hyperparameter search is performed anywhere in this project, and none is "
        f"claimed.",
    ]

    evaluation_parts = []
    if cls:
        evaluation_parts.append(
            f"**Survival classification** - ROC-AUC {_fmt(cls.roc_auc)}, F1 {_fmt(cls.f1)}, "
            f"accuracy {cls.accuracy * 100:.1f}% on {cls.test_rows} held-out passengers."
        )
    if reg:
        evaluation_parts.append(
            f"**House prices** - R-squared {_fmt(reg.r2)}, RMSE ${reg.rmse:,.0f}, "
            f"MAE ${reg.mae:,.0f}, in dollars after back-transformation."
        )
    if fra:
        evaluation_parts.append(
            f"**Fraud** - average precision is the primary metric: "
            f"{_fmt(fra.weighted_test.average_precision)} for the class-weighted model versus "
            f"{_fmt(fra.baseline_test.average_precision)} unweighted (ROC-AUC "
            f"{_fmt(fra.weighted_test.roc_auc)} vs {_fmt(fra.baseline_test.roc_auc)} as "
            f"secondary context). At the default cutoff the unweighted model catches "
            f"{fra.baseline_test.recall * 100:.0f}% of fraud; the weighted model catches "
            f"{fra.weighted_test.recall * 100:.0f}% but raises "
            f"{fra.weighted_test.confusion['false_positive']:,} false alarms. "
            f"The F1-optimal cutoff chosen on **validation** (log-odds "
            f"{fra.selected_threshold:.2f}, validation F1 {_fmt(fra.selected_validation_f1)}) "
            f"scores F1 {_fmt(fra.weighted_selected_test.f1)} on the test partition, which was "
            f"never used to choose the cutoff. The gap between those two numbers indicates how "
            f"much of the validation score came from fitting the cutoff to validation data; "
            f"tuning on the test set would have hidden it."
        )
    if qual:
        evaluation_parts.append(
            f"**Data quality** - composite score {qual.score:.1f}/100 "
            f"({qual.health}): completeness {qual.completeness:.1f}, validity "
            f"{qual.validity:.1f}, uniqueness {qual.uniqueness:.1f}, consistency "
            f"{qual.consistency:.1f}, weighted "
            f"{', '.join(f'{k} {v}' for k, v in QUALITY_WEIGHTS.items())}."
        )
    if ana:
        evaluation_parts.append(
            f"**Analytics** - {len(ana.retention.sizes)} monthly cohorts; the A/B calculator "
            f"runs a pooled two-proportion z-test on counts you enter, reports a 95% "
            f"confidence interval, and bases its recommendation on effect direction, "
            f"magnitude and interval width rather than on p < 0.05 alone."
        )

    deployment = """
The app is a single local Streamlit process. Benchmarks are computed once at startup and
cached in memory for the session, so switching workspaces never recomputes. There are no
persisted model artifacts: reproducibility comes from pinned seeds and a pinned
configuration, and every dataset is re-derived from the local parquet cache created by
`scripts/prepare_data.py`.

Honest limitations: single hold-out splits throughout (no cross-validation), no resampling
methods such as SMOTE, no forecasting, and one illustrative element - the checkout funnel,
which is labelled as such wherever it appears because no comparable free event-stream data
exists.
"""

    return [
        ("1. Business understanding", business),
        ("2. Data understanding", "\n".join(data_understanding_rows) or "_No data available._"),
        ("3. Data preparation", data_prep),
        ("4. Modeling", "\n".join(modeling_rows)),
        ("5. Evaluation", "\n\n".join(evaluation_parts) or "_No results available._"),
        ("6. Deployment", deployment),
    ]
