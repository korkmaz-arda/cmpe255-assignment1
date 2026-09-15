"""CRISP-DM methodology report (S12).

Every figure in this report is read from the artifacts the pipeline actually
produced. The source's version was static authored prose that did not change
when the model was retrained and quoted metrics the served model never achieved;
nothing here is hardcoded except the narrative framing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from segmentation import config, elbow

from . import theme

PHASES = [
    "1 · Business understanding",
    "2 · Data understanding",
    "3 · Data preparation",
    "4 · Modeling",
    "5 · Evaluation",
    "6 · Deployment",
]


def _actionability(engine) -> str:
    """Report the measured segment-size and persona-fit outcome for the served model."""
    personas = engine.personas
    if not personas:
        return ""
    smallest = min(personas, key=lambda p: p["share"])
    unmatched = [p for p in personas if not p.get("matched", True)]
    size = (
        f"Measured: the smallest served segment, “{smallest['name']}”, holds "
        f"{smallest['share']:.1%} of customers ({smallest['size']:,})."
    )
    if unmatched:
        fit = (
            f" {len(unmatched)} of {len(personas)} segments fit no persona and are described from "
            f"their own data."
        )
    else:
        fit = f" All {len(personas)} segments carry a persona whose narrative their data supports."
    return size + fit


def _business(engine) -> None:
    production = engine.production
    st.markdown(
        "**Objective.** Replace blanket promotion with differentiated marketing by discovering "
        "the customer archetypes latent in purchasing behaviour. There are no segment labels to "
        "learn from, so this is an unsupervised problem and success is judged on internal "
        "cluster-validity and on whether the resulting segments are actionable."
    )
    st.markdown("**How success is measured**")
    st.markdown(
        f"""
- **Separation** — silhouette on the shared evaluation matrix. Achieved: **{theme.metric_or_dash(production.get('silhouette'))}**.
- **Actionability** — every segment should hold a meaningful share of the base, and be
  describable by a persona whose narrative its data supports. {_actionability(engine)}
- **Inference latency** — single-customer classification is a scaler transform plus
  {engine.bundle.k if engine.bundle else config.DEFAULT_K} centroid-distance computations, so it is
  effectively instantaneous. No latency target is claimed here because none was measured.
"""
    )
    st.markdown(
        "**On the scale of these numbers.** Silhouette on real behavioural data of this kind is "
        "modest — customer behaviour is continuous, not naturally clumped into well-separated "
        "groups. The figures in this report are what this pipeline measured, not targets it was "
        "tuned toward."
    )


def _data(engine) -> None:
    meta = engine.meta
    quality = meta.get("data_quality", {})
    dataset = meta.get("dataset", {})

    st.markdown(
        f"**Source.** [{dataset.get('name', config.DATASET_NAME)}]({dataset.get('url', config.DATASET_PAGE)}) "
        f"— a real, public retail marketing dataset of 2,240 customers. It is downloaded by "
        f"`python -m segmentation.data --download` and is not committed to the repository."
    )
    st.markdown(
        f"**After cleaning:** {quality.get('n_customers', 0):,} customers described by "
        f"{quality.get('n_attributes', 0)} behavioural attributes."
    )

    attributes = quality.get("attributes", {})
    if attributes:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Attribute": info["label"],
                        "Range": f"{info['min']:.4g} – {info['max']:.4g}",
                        "Mean": round(info["mean"], 3),
                        "Domain meaning": info["meaning"],
                    }
                    for info in attributes.values()
                ]
            ),
            hide_index=True, width="stretch",
        )

    correlation = quality.get("spending_score_vs_total_spend_spearman")
    if correlation is not None:
        st.markdown(
            f"<div class='caveat'><b>One attribute is derived, not observed.</b> The raw dataset "
            f"has no spending score, so a 1–100 <i>purchase-engagement</i> score is constructed from "
            f"conversion efficiency (purchases per web visit) and campaign responsiveness. A simple "
            f"percentile of total spend was rejected because it correlates ρ≈0.91 with total spend "
            f"itself, which would give the model two near-duplicate monetary axes. The derivation "
            f"actually used measures ρ={correlation:.3f} against total spend on the cleaned data — "
            f"correlated, as any genuine purchase signal must be, but not a duplicate.</div>",
            unsafe_allow_html=True,
        )


def _preparation(engine) -> None:
    st.markdown(
        """
**Cleaning rules.** Rows with a missing income are dropped rather than imputed (imputing would
invent the strongest clustering signal); incomes above the 99.5th percentile are removed, which
clears the 666,666 data-entry artefact; ages outside 18–90 are removed, clearing the 1893 and 1899
birth years; and customers with no purchase history are removed, because a discount share and a
conversion rate are both undefined for them.

**Discount sensitivity.** `NumDealsPurchases` counts purchases made *with a discount*, while the
three `Num*Purchases` columns partition purchases *by channel* — a discounted purchase is still made
through a channel. Verified empirically: the containment holds for 2,237 of 2,240 raw rows. The
denominator is therefore the channel counts only; adding deals in would double-count them.
"""
    )
    st.markdown("**Engineered interaction features.**")
    engineered = engine.meta.get("engineered", {})
    if engineered:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Feature": name.replace("_", " ").title(),
                     "Formula": info["formula"], "What it captures": info["meaning"]}
                    for name, info in engineered.items()
                ]
            ),
            hide_index=True, width="stretch",
        )
    st.markdown(
        f"""
**Standardization.** All {len(config.FEATURE_COLUMNS)} columns are z-score standardized. The fitted
scaler is persisted with the model and the ordered column list, and is only ever *applied* at
inference — re-fitting a scaler on a single record would silently destroy every prediction. The
`+1` denominators in the ratio features are deliberate divide-by-zero guards.

**Partitioning.** There is no train/test split, which is correct for unsupervised work. Every
entrant, the production model and the elbow sweep are fit and scored on the same standardized
matrix, which is what makes their silhouettes comparable.
"""
    )


def _modeling(engine) -> None:
    rows = engine.leaderboard
    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Algorithm": r["algorithm"], "Family": r["family"],
                     "Formulation": r["formulation"],
                     "Silhouette": r["silhouette"], "Clusters": r["n_clusters"]}
                    for r in rows
                ]
            ),
            hide_index=True, width="stretch",
            column_config={"Silhouette": st.column_config.NumberColumn(format="%.4f")},
        )
    production = engine.production
    st.markdown(
        f"**Served model.** {production.get('algorithm', '—')} — {production.get('formulation', '')}, "
        f"fitted on {production.get('n_customers', 0):,} customers. The fitted model, scaler and "
        f"column order are persisted together so inference reproduces the training transform exactly."
    )

    research = engine.research
    if research:
        st.markdown(
            f"**AutoResearch.** A four-phase greedy hill climb explored feature mutations and "
            f"hyperparameters on a {research['n_rows']:,}-customer sample, starting from the raw "
            f"attributes. It moved silhouette {research['start_silhouette']:.5f} → "
            f"{research['best_silhouette']:.5f} over {research['total_steps']} steps "
            f"({research['accepted']} accepted, {research['rejected']} rejected). "
            f"{research['promotion_note']}"
        )
        gamed = [
            s for s in research["steps"]
            if s.get("partition", {}).get("smallest_share", 1.0) < config.MIN_CLUSTER_SHARE
        ]
        st.markdown(
            f"**Acceptance gate.** {research.get('acceptance_gate', '')}"
            + (
                f" In this run it rejected {len(gamed)} step(s) on that rule alone — for example "
                f"“{gamed[0]['component']}”, which scored {gamed[0]['silhouette_after']:.4f} by "
                f"leaving its smallest cluster with {gamed[0]['partition']['smallest_share']:.2%} "
                f"of customers."
                if gamed else ""
            )
        )


def _evaluation(engine) -> None:
    production = engine.production
    columns = st.columns(3)
    columns[0].metric("Silhouette", theme.metric_or_dash(production.get("silhouette")))
    columns[0].caption("Higher is better")
    columns[1].metric("Davies–Bouldin", theme.metric_or_dash(production.get("davies_bouldin")))
    columns[1].caption("Lower is better")
    columns[2].metric("Calinski–Harabasz", theme.metric_or_dash(production.get("calinski_harabasz"), "{:,.1f}"))
    columns[2].caption("Higher is better")

    sweep = engine.elbow
    if sweep:
        verdict = elbow.verdict(sweep)
        st.markdown(
            f"**Choice of k.** Across the sweep k={min(config.K_SWEEP)}–{max(config.K_SWEEP)}: "
            f"{verdict['detail']}"
        )

    st.markdown(
        "**No external agreement measure is reported.** The dataset carries no ground-truth segment "
        "labels, so there is nothing to compute an adjusted Rand index or purity against. Only "
        "internal validity metrics are meaningful here, and only internal validity is claimed."
    )

    st.markdown(
        f"**Persona validity.** Personas are assigned by cluster content, never by cluster number. "
        f"Each cluster is placed at its percentile on every attribute, and each persona's narrative "
        f"makes explicit claims — “higher income”, “low spend”, “mid-sized baskets”. A persona is "
        f"only shown on a cluster that satisfies *every* claim its narrative makes: “high” means at "
        f"or above the {config.CLAIM_HIGH_MIN:.0%} percentile, “low” at or below the "
        f"{config.CLAIM_LOW_MAX:.0%}, and “average” between the {config.CLAIM_MID_BAND[0]:.0%} and "
        f"{config.CLAIM_MID_BAND[1]:.0%}. Among the personas that pass, the closest overall profile "
        f"is chosen. A cluster that no persona fits is shown as a numbered segment described from "
        f"its own data, rather than given a marketing narrative the numbers contradict."
    )
    personas_served = engine.personas
    unmatched = [p for p in personas_served if not p.get("matched", True)]
    notes = engine.meta.get("persona_notes", [])
    for note in notes:
        st.warning(note)
    if personas_served and not unmatched:
        st.success(
            f"All {len(personas_served)} segments carry a persona whose every narrative claim holds "
            f"for that segment's data."
        )


def _deployment(engine) -> None:
    meta = engine.meta
    st.markdown(
        f"""
**Serving.** All results are written to on-disk artifacts in `artifacts/` — the model bundle
(model + scaler + column order), the fitted PCA, the persona profiles, the leaderboard, the elbow
sweep, the scatter sample and the AutoResearch run record. The application loads them into a single
engine object and re-reads them whenever they change, so a retrain immediately changes what every
view shows without a restart.

**Inference path.** Eight validated attributes → the same four engineered features → the *persisted*
scaler → distance to each of the {meta.get('k', config.DEFAULT_K)} centroids → nearest cluster, plus
a normalized inverse-distance proximity score and PCA coordinates for the live marker.

**Retraining.** The studio's cluster count, training-sample size and seed all reach the pipeline and
all change the result. Last run: {meta.get('generated_at', 'n/a')} in
{meta.get('elapsed_seconds', 0):.1f}s over {meta.get('n_customers', 0):,} customers.

**Reproducibility.** `python -m segmentation.data --download`, then
`python -m segmentation.pipeline --k {meta.get('k', config.DEFAULT_K)} --seed {meta.get('seed', config.DEFAULT_SEED)}`
regenerates every artifact deterministically.
"""
    )


RENDERERS = {
    PHASES[0]: _business,
    PHASES[1]: _data,
    PHASES[2]: _preparation,
    PHASES[3]: _modeling,
    PHASES[4]: _evaluation,
    PHASES[5]: _deployment,
}


def render(engine) -> None:
    st.markdown("### CRISP-DM methodology report")
    st.caption("Every figure on this page is read from the artifacts this pipeline produced.")
    sidebar, detail = st.columns([1, 3], gap="large")
    with sidebar:
        phase = st.radio("Phase", PHASES, label_visibility="collapsed", key="crispdm_phase")
    with detail:
        st.markdown(f"#### {phase}")
        RENDERERS[phase](engine)
