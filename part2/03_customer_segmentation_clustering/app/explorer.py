"""Customer-facing segment explorer: problem framing, persona cards, manifold
scatter and the live classifier (S01–S04, F06, F07, F09)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from segmentation import config, features, inference

from . import state, theme

AXIS_PAD = 3.5


# --------------------------------------------------------------------------- #
# S04 — problem-framing banner
# --------------------------------------------------------------------------- #

def _banner(engine) -> None:
    production = engine.production
    meta = engine.meta
    research = engine.research

    st.markdown(
        "<div class='panel'><h3>Why segment at all?</h3>"
        "<div class='note'>Blanket promotions convert poorly because they treat a heterogeneous "
        "customer base as one audience. There are no segment labels to learn from, so the task is "
        "unsupervised: discover the archetypes that are already latent in purchasing behaviour, then "
        "attach a marketing action to each one.</div></div>",
        unsafe_allow_html=True,
    )

    # Every figure below is read from the artifacts this pipeline produced.
    columns = st.columns(4)
    silhouette_entry = config.METRIC_GLOSSARY["silhouette"]
    columns[0].metric(
        "Served silhouette",
        theme.metric_or_dash(production.get("silhouette")),
        help=f"{silhouette_entry['plain']}\n\n{silhouette_entry['reading']}",
    )
    columns[1].metric(
        "Customers segmented",
        f"{meta.get('n_customers', 0):,}",
        help=f"Real records from the {config.DATASET_NAME} dataset after cleaning.",
    )
    columns[2].metric(
        "Segments served",
        meta.get("k", "—"),
        help="Cluster count the production model was actually fitted with.",
    )
    if research:
        columns[3].metric(
            "AutoResearch best (separate search)",
            f"{research['best_silhouette']:.4f}",
            f"{research['improvement_pct']:+.1f}% vs its own start",
            help=(
                f"Best silhouette from the exploratory search, measured on its own "
                f"{research['n_rows']:,}-customer sample with {len(research['active_features'])} "
                f"features. Silhouette is not comparable across different samples and feature "
                f"sets, so this is not a like-for-like comparison with the served model."
            ),
        )
    else:
        columns[3].metric("AutoResearch best (separate search)", "—", help="No search has been run yet.")

    engineered = meta.get("engineered", {})
    if engineered:
        chips = "".join(
            f"<span class='chip'><b>{name.replace('_', ' ').title()}</b> &nbsp;<code>{info['formula']}</code></span>"
            for name, info in engineered.items()
        )
        st.markdown(
            f"<div class='panel'><h3>Engineered behavioural signals</h3>{chips}</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# S01 — persona cards with cross-filtering
# --------------------------------------------------------------------------- #

def _persona_cards(engine) -> None:
    st.markdown("#### Discovered personas")
    st.caption("Click a card to filter the manifold to that segment; click it again to clear.")

    personas = engine.personas
    columns = st.columns(len(personas))
    for column, record in zip(columns, personas):
        selected = st.session_state.selected_cluster == record["cluster"]
        means = record["means"]
        background = f"background:{record['color']}1f;" if selected else ""
        border = f"border-left-color:{record['color']};"
        outline = f"box-shadow:0 0 0 2px {record['color']};" if selected else ""

        column.markdown(
            theme.compact(
            f"""<div class='persona-card' style="{border}{background}{outline}">
              <div class='idx'>Cluster {record['cluster']} · {record['share']:.1%} · {record['size']:,} customers</div>
              <div class='nm'>{record['badge']} {record['name']}</div>
              <div class='tl'>{record['tagline']}</div>
              <div class='stat'><span>Avg income</span><b>{theme.money(means['income_k'] * 1000)}</b></div>
              <div class='stat'><span>Spending score</span><b>{means['spending_score']:.0f}/100</b></div>
              <div class='stat'><span>Avg spend (2-yr)</span><b>{theme.money(means['total_spend'])}</b></div>
            </div>"""
            ),
            unsafe_allow_html=True,
        )
        label = "✓ Filtering" if selected else "Filter manifold"
        if column.button(label, key=f"persona-{record['cluster']}", width="stretch"):
            st.session_state.selected_cluster = None if selected else record["cluster"]
            st.rerun()

        if not record.get("matched", True):
            column.caption("◇ No persona fits this segment — described from its own data.")
        elif record["weak_match"]:
            column.caption(f"⚠ Weak persona match ({record['match_quality']:.0%})")


# --------------------------------------------------------------------------- #
# S02 — interactive 2D manifold scatter
# --------------------------------------------------------------------------- #

def _scatter(engine) -> None:
    points = engine.scatter.get("points", [])
    if not points:
        st.info("No projection sample available. Run the pipeline to generate one.")
        return

    frame = pd.DataFrame(points)

    head = st.columns([3, 2])
    with head[0]:
        st.markdown("#### 2D manifold")
    with head[1]:
        st.segmented_control(
            "Projection", state.PROJECTIONS, key="projection_control",
            label_visibility="collapsed",
        )
    projection = state.current_projection()
    prefix = "pca" if projection == "PCA" else "tsne"

    selected = st.session_state.selected_cluster
    figure = go.Figure()

    for record in engine.personas:
        cluster = record["cluster"]
        subset = frame[frame["cluster"] == cluster]
        if subset.empty:
            continue
        dimmed = selected is not None and selected != cluster
        figure.add_trace(
            go.Scattergl(
                x=subset[f"{prefix}_x"],
                y=subset[f"{prefix}_y"],
                mode="markers",
                name=record["name"],
                marker=dict(
                    size=7 if not dimmed else 5,
                    color=record["color"],
                    opacity=0.12 if dimmed else 0.82,
                    line=dict(width=0.5, color="rgba(255,255,255,.35)"),
                ),
                customdata=subset[
                    ["customer_id", "age", "income_k", "spending_score",
                     "total_spend", "recency_days", "discount_sensitivity"]
                ].to_numpy(),
                hovertemplate=(
                    f"<b>{record['name']}</b> · cluster {cluster}<br>"
                    "Customer %{customdata[0]}<br>"
                    "Age %{customdata[1]:.0f} · Income $%{customdata[2]:.1f}k<br>"
                    "Spending score %{customdata[3]:.0f}/100<br>"
                    "Spend $%{customdata[4]:,.0f} · Recency %{customdata[5]:.0f}d<br>"
                    "Discount share %{customdata[6]:.0%}<extra></extra>"
                ),
            )
        )

    # S02 — the live-classified customer, PCA only: t-SNE has no out-of-sample
    # transform, so there is no honest place to put this marker on that canvas.
    prediction = st.session_state.prediction
    if prediction and projection == "PCA":
        figure.add_trace(
            go.Scatter(
                x=[prediction["pca_x"]], y=[prediction["pca_y"]],
                mode="markers", name="Your customer",
                marker=dict(size=20, color=prediction["persona"]["color"], symbol="circle",
                            line=dict(width=4, color="rgba(255,255,255,.95)")),
                hovertemplate=f"<b>Your customer</b><br>{prediction['persona']['name']}<extra></extra>",
            )
        )

    x_all = frame[f"{prefix}_x"]
    y_all = frame[f"{prefix}_y"]
    # Axis bounds always include at least +/-AXIS_PAD so filtered views stay stable.
    bound_x = max(AXIS_PAD, abs(x_all.min()), abs(x_all.max())) * 1.08
    bound_y = max(AXIS_PAD, abs(y_all.min()), abs(y_all.max())) * 1.08

    figure.update_layout(
        height=470, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
        xaxis=dict(range=[-bound_x, bound_x], zeroline=True, zerolinecolor="rgba(148,163,184,.35)",
                   showgrid=True, gridcolor="rgba(148,163,184,.12)", title=f"{projection} 1"),
        yaxis=dict(range=[-bound_y, bound_y], zeroline=True, zerolinecolor="rgba(148,163,184,.35)",
                   showgrid=True, gridcolor="rgba(148,163,184,.12)", title=f"{projection} 2"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    sample_size = engine.scatter.get("n_sample", len(frame))
    if projection == "PCA":
        variance = engine.meta.get("pca_explained_variance", {}).get("total")
        caption = (
            f"Linear PCA projection of the 12-dimensional standardized feature space, "
            f"capturing {variance:.1%} of total variance."
            if variance else "Linear PCA projection of the standardized feature space."
        )
    else:
        caption = (
            "Non-linear t-SNE embedding (perplexity "
            f"{config.TSNE_PERPLEXITY}). t-SNE has no out-of-sample transform, so a newly "
            "classified customer cannot be placed on this canvas — switch to PCA to see the marker."
        )
    st.caption(f"{caption} Showing {sample_size:,} deterministically sampled customers; hover any point for the record behind it.")


# --------------------------------------------------------------------------- #
# S03 — live classifier
# --------------------------------------------------------------------------- #

def _predict(engine, payload: dict) -> None:
    try:
        st.session_state.prediction = engine.classify(payload)
        st.session_state.prediction_error = None
    except inference.ValidationError as exc:
        st.session_state.prediction = None
        st.session_state.prediction_error = str(exc)


def _classifier(engine) -> None:
    st.markdown("#### Classify a customer")
    st.caption(
        "All eight attributes that drive the model are adjustable — including discount sensitivity "
        "and household size, which feed two of the engineered features."
    )

    with st.form("classifier"):
        payload = {}
        columns = st.columns(2)
        for index, field in enumerate(config.BASE_FEATURES):
            low, high = config.INPUT_RANGES[field]
            default = config.INPUT_DEFAULTS[field]
            step = 0.01 if field == "discount_sensitivity" else 1.0
            payload[field] = columns[index % 2].slider(
                config.ATTRIBUTE_LABELS[field],
                min_value=float(low), max_value=float(high), value=float(default),
                step=step, key=f"input-{field}", help=config.ATTRIBUTE_MEANINGS[field],
            )
        submitted = st.form_submit_button("Classify customer", width="stretch", type="primary")

    if submitted:
        with st.spinner("Scoring against the served centroids…"):
            _predict(engine, payload)

    # A prediction is issued automatically on first load using the default values.
    if st.session_state.prediction is None and st.session_state.prediction_error is None:
        _predict(engine, {field: config.INPUT_DEFAULTS[field] for field in config.BASE_FEATURES})

    if st.session_state.prediction_error:
        st.error(st.session_state.prediction_error)
        return

    result = st.session_state.prediction
    if not result:
        return

    persona = result["persona"]
    st.markdown(
        theme.compact(
        f"""<div class='result-card' style="background:linear-gradient(135deg,{persona['color']}ee,{persona['color']}99);">
          <div class='cap'>Cluster {result['cluster']} · {result['confidence']:.1%} proximity score</div>
          <div class='nm'>{persona['badge']} {persona['name']}</div>
          <div class='desc'>{persona['description']}</div>
          <div class='callout'><b>Recommended action</b><br>{persona['strategy']}</div>
        </div>"""
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Distance to centroid {result['distance_to_centroid']:.3f} in standardized space. "
        f"The percentage is a normalized inverse-distance **proximity score**, not a probability — "
        f"with {engine.bundle.k} centroids it can never fall below {1 / engine.bundle.k:.0%}."
    )
    with st.expander("Engineered features for this record"):
        engineered = result["engineered"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": [name.replace("_", " ").title() for name in engineered],
                    "Formula": [features.FEATURE_FORMULAS[name] for name in engineered],
                    "Value": [f"{value:.3f}" for value in engineered.values()],
                }
            ),
            hide_index=True, width="stretch",
        )


# --------------------------------------------------------------------------- #

def render(engine) -> None:
    if not engine.trained:
        st.warning("No trained model found. Open the retraining studio, or run "
                   "`python -m segmentation.pipeline` from the project root.")
        return

    _banner(engine)
    _persona_cards(engine)
    st.divider()

    left, right = st.columns([3, 2], gap="large")
    # Fill the classifier column first. It computes this run's prediction, and the
    # scatter draws the live marker from that prediction; drawing the scatter first
    # would place the marker one submission behind the result card. Column order on
    # screen is set by st.columns, not by the order the columns are filled.
    with right:
        _classifier(engine)
    with left:
        _scatter(engine)
