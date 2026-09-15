"""The skills catalog: what each technique is for, and which workspace demonstrates it.

Every entry routes to a benchmark that really runs. There is no catch-all branch that
returns a success message without computing anything, and the skill count shown anywhere
in the UI is always ``len(SKILLS)``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class View(str, Enum):
    """The six application views. Single source of truth for navigation keys."""

    CATALOG = "catalog"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FRAUD = "fraud"
    ANALYTICS = "analytics"
    QUALITY = "quality"


WORKSPACE_LABELS = {
    View.CLASSIFICATION: "Survival classification - Titanic (OpenML 40945)",
    View.REGRESSION: "House-price regression - Ames, Iowa (OpenML 42165)",
    View.FRAUD: "Imbalanced classification - credit-card fraud (OpenML 1597)",
    View.ANALYTICS: "Business analytics - Online Retail II (UCI)",
    View.QUALITY: "Data-quality audit - Online Retail II (UCI)",
}

CATEGORIES = [
    "Data preparation & feature engineering",
    "Modeling & evaluation",
    "Data quality & validation",
    "Business & statistical analytics",
]


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    category: str
    origin: str
    purpose: str
    intuition: str
    pitfalls: tuple[str, ...]
    view: View

    @property
    def workspace_label(self) -> str:
        return WORKSPACE_LABELS[self.view]

    def as_dict(self) -> dict:
        data = asdict(self)
        data["view"] = self.view.value
        data["pitfalls"] = list(self.pitfalls)
        data["workspace"] = self.workspace_label
        return data


SKILLS: tuple[Skill, ...] = (
    Skill(
        id="exploratory-data-analysis",
        name="Exploratory data analysis",
        category=CATEGORIES[0],
        origin="Tukey, Exploratory Data Analysis (1977)",
        purpose=(
            "Describe a table before modelling it: shape, types, missingness, class "
            "balance and the distribution of every candidate feature."
        ),
        intuition=(
            "EDA estimates the marginal distribution of each variable and its joint "
            "behaviour with the target. Everything you learn here is a claim about the "
            "sample, not the population, until you quantify the uncertainty."
        ),
        pitfalls=(
            "Looking at the test partition during exploration - that is leakage by eye.",
            "Reading a mean without its spread, or a rate without its denominator.",
            "Treating a column with 20% missing values as if the missingness were random.",
        ),
        view=View.CLASSIFICATION,
    ),
    Skill(
        id="leakage-free-imputation",
        name="Leakage-free cleaning and imputation",
        category=CATEGORIES[0],
        origin="Kaufman et al., Leakage in Data Mining (2012)",
        purpose=(
            "Fill missing values without letting information from the evaluation data "
            "influence the fill."
        ),
        intuition=(
            "A median imputer has parameters. Fitting it on all rows estimates those "
            "parameters partly from the rows you are about to be scored on, which "
            "flatters the score by an amount you cannot measure afterwards."
        ),
        pitfalls=(
            "Calling fillna() on the full frame before splitting.",
            "Imputing a categorical with a new level the model has never seen.",
            "Silently dropping rows, which changes the denominator of every later rate.",
        ),
        view=View.QUALITY,
    ),
    Skill(
        id="feature-engineering",
        name="Feature engineering",
        category=CATEGORIES[0],
        origin="Zheng & Casari, Feature Engineering for Machine Learning (2018)",
        purpose=(
            "Derive columns that express domain structure the raw table only implies, "
            "such as household size from sibling and parent counts."
        ),
        intuition=(
            "A derived feature is a hand-built basis function. It helps when it encodes "
            "structure the model would otherwise need many splits to approximate."
        ),
        pitfalls=(
            "Deriving a feature from the target, which is leakage in disguise.",
            "Computing a feature differently at training time and at serving time.",
            "Adding correlated variants and reading the split importances as causal.",
        ),
        view=View.CLASSIFICATION,
    ),
    Skill(
        id="imbalanced-data",
        name="Imbalanced classification",
        category=CATEGORIES[0],
        origin="He & Garcia, Learning from Imbalanced Data (2009)",
        purpose=(
            "Train and evaluate when the positive class is rare, as in card fraud at a "
            "0.17% base rate."
        ),
        intuition=(
            "Accuracy is dominated by the majority class: always predicting 'not fraud' "
            "scores 99.83%. Class weighting rescales the loss so minority errors cost "
            "more, which changes what the model optimises - and distorts its probability "
            "scale, so the decision cutoff must then be chosen deliberately."
        ),
        pitfalls=(
            "Reporting accuracy as the headline number.",
            "Assuming 0.5 is still a sensible cutoff after reweighting.",
            "Choosing the cutoff on the same data you report the score from.",
        ),
        view=View.FRAUD,
    ),
    Skill(
        id="pipelines",
        name="Pipelines as leakage control",
        category=CATEGORIES[1],
        origin="scikit-learn Pipeline / ColumnTransformer",
        purpose=(
            "Bind preprocessing and estimator into one object so that fitting can only "
            "ever see the training partition."
        ),
        intuition=(
            "A pipeline makes the leakage-safe order structural rather than a habit: "
            "fit() fits every step on train, transform() applies the fitted parameters."
        ),
        pitfalls=(
            "Scaling outside the pipeline and then splitting.",
            "Refitting the encoder on the test partition and quietly changing the columns.",
            "Forgetting handle_unknown='ignore', so an unseen category raises at serving.",
        ),
        view=View.CLASSIFICATION,
    ),
    Skill(
        id="model-configuration",
        name="Deterministic model configuration",
        category=CATEGORIES[1],
        origin="Pedregosa et al., scikit-learn (2011)",
        purpose=(
            "Pin the hyperparameters and random seeds that make a result reproducible, "
            "and state them where a reader can see them."
        ),
        intuition=(
            "Every number reported here is conditional on a configuration. Reproducibility "
            "means the configuration is written down, not that the numbers are stable by luck."
        ),
        pitfalls=(
            "Reporting a score without the seed and split that produced it.",
            "Claiming a search was run when the parameters were chosen by hand.",
            "Tuning until the headline looks good and reporting only the winner.",
        ),
        view=View.REGRESSION,
    ),
    Skill(
        id="target-transforms",
        name="Target transforms and back-transformation",
        category=CATEGORIES[1],
        origin="Box & Cox, An Analysis of Transformations (1964)",
        purpose=(
            "Fit on a transformed target when errors scale with magnitude, then invert "
            "the transform before computing any metric."
        ),
        intuition=(
            "Sale prices are right-skewed, so squared error on the raw scale is dominated "
            "by expensive houses. Fitting on log1p makes the error roughly proportional; "
            "expm1 returns predictions to dollars so RMSE keeps its units."
        ),
        pitfalls=(
            "Reporting R-squared on the log scale and calling it accuracy in dollars.",
            "Forgetting that expm1 of the mean log is not the mean price.",
            "Applying log to a target that can be zero or negative without an offset.",
        ),
        view=View.REGRESSION,
    ),
    Skill(
        id="model-evaluation",
        name="Choosing the right metric",
        category=CATEGORIES[1],
        origin="Saito & Rehmsmeier, PLoS ONE (2015)",
        purpose=(
            "Pick metrics that reflect the decision being made: PR-AUC and recall for rare "
            "positives, RMSE in original units for prices."
        ),
        intuition=(
            "ROC-AUC integrates over the false-positive rate, whose denominator is huge "
            "when negatives dominate, so it stays high even for a weak detector. Average "
            "precision uses precision, whose denominator is the alerts you actually raise."
        ),
        pitfalls=(
            "Comparing two models at different cutoffs.",
            "Quoting a threshold-free metric to justify a thresholded decision.",
            "Ignoring the operational cost of the false positives a recall gain buys.",
        ),
        view=View.FRAUD,
    ),
    Skill(
        id="programmatic-profiling",
        name="Programmatic data profiling",
        category=CATEGORIES[2],
        origin="Abedjan et al., Data Profiling (VLDB 2015)",
        purpose=(
            "Compute per-column type, cardinality, missingness and rule violations "
            "automatically so an audit is repeatable rather than manual."
        ),
        intuition=(
            "A profile is a set of rates. Expressing every dimension as a rate keeps the "
            "verdict comparable across tables of wildly different sizes."
        ),
        pitfalls=(
            "Scoring a table by counting issues, which punishes large tables.",
            "Coercing identifiers to numbers and calling the result an outlier.",
            "Profiling a filtered view and reporting it as the source table.",
        ),
        view=View.QUALITY,
    ),
    Skill(
        id="data-quality-audit",
        name="Data-quality scorecard",
        category=CATEGORIES[2],
        origin="Wang & Strong, Data Quality Dimensions (1996)",
        purpose=(
            "Combine completeness, validity, uniqueness and consistency into one score "
            "with explicit weights, and separate defects from legitimate business events."
        ),
        intuition=(
            "Each dimension is a 0-100 rate, so the weighted sum stays on the same scale "
            "and the weights are the only judgement call - and they are written down."
        ),
        pitfalls=(
            "Mixing a percentage with a raw row count in one score.",
            "Flagging returns and credit notes as corruption.",
            "Treating a statistical outlier as invalid rather than as something to review.",
        ),
        view=View.QUALITY,
    ),
    Skill(
        id="cohort-analysis",
        name="Cohort retention analysis",
        category=CATEGORIES[3],
        origin="Standard subscription/e-commerce retention practice",
        purpose=(
            "Group customers by acquisition month and track what share purchases again in "
            "each later month."
        ),
        intuition=(
            "Retention is a conditional rate: the denominator is the cohort, not the "
            "customer base, and a recent cohort is simply observed for fewer periods."
        ),
        pitfalls=(
            "Comparing a young cohort's period-2 value against an old cohort's period-8.",
            "Leaving credit notes in the population so a refund counts as activity.",
            "Reading a retention drop as churn when the observation window just ended.",
        ),
        view=View.ANALYTICS,
    ),
    Skill(
        id="funnel-analysis",
        name="Conversion funnel analysis",
        category=CATEGORIES[3],
        origin="Standard product-analytics practice",
        purpose=(
            "Decompose a conversion path into stages and locate the largest drop-off."
        ),
        intuition=(
            "Overall conversion is the product of the step rates, so the stage with the "
            "lowest step rate bounds what any downstream improvement can achieve."
        ),
        pitfalls=(
            "Mixing users and sessions between stages.",
            "Assuming a strictly ordered path when users skip or re-enter stages.",
            "Presenting illustrative stage counts as measured telemetry.",
        ),
        view=View.ANALYTICS,
    ),
    Skill(
        id="ab-testing",
        name="Two-proportion A/B testing",
        category=CATEGORIES[3],
        origin="Kohavi et al., Trustworthy Online Controlled Experiments (2020)",
        purpose=(
            "Decide whether a conversion-rate difference between two variants is "
            "distinguishable from sampling noise, and whether it is large enough to ship."
        ),
        intuition=(
            "Under the null the pooled difference is approximately normal with a standard "
            "error set by the sample sizes. The p-value answers only 'is it distinguishable "
            "from zero'; the confidence interval answers 'how big could it plausibly be'."
        ),
        pitfalls=(
            "Peeking repeatedly and stopping when p first dips below 0.05.",
            "Reading a small p-value as a large effect.",
            "Using post-hoc power as if it were independent evidence.",
        ),
        view=View.ANALYTICS,
    ),
    Skill(
        id="time-series-decomposition",
        name="Time-series decomposition",
        category=CATEGORIES[3],
        origin="Cleveland et al., STL decomposition (1990)",
        purpose=(
            "Separate a daily revenue series into trend, a repeating weekly component and "
            "the remainder."
        ),
        intuition=(
            "An additive decomposition assumes observation = trend + seasonal + residual. "
            "The trend is a centred moving average; the seasonal term is the average "
            "detrended deviation for each weekday."
        ),
        pitfalls=(
            "Choosing a period that does not match the real cycle.",
            "Reading the residual as noise when it contains promotions and holidays.",
            "Extending a decomposition into a forecast - it is descriptive, not predictive.",
        ),
        view=View.ANALYTICS,
    ),
)

SKILLS_BY_ID = {skill.id: skill for skill in SKILLS}


def skill_count() -> int:
    return len(SKILLS)


def search(query: str = "", category: str | None = None) -> list[Skill]:
    """Free-text search across name, purpose and origin, composed with a category filter."""
    text = query.strip().lower()
    results = []
    for skill in SKILLS:
        if category and skill.category != category:
            continue
        if text and text not in " ".join(
            (skill.name, skill.purpose, skill.origin)
        ).lower():
            continue
        results.append(skill)
    return results
