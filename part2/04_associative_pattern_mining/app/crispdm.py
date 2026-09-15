"""CRISP-DM report, written against what this project actually does.

Every figure is read from the artifacts of the current run, so the report cannot
drift away from the pipeline.
"""

from __future__ import annotations

import streamlit as st

from basket import config


@st.dialog("CRISP-DM methodology report", width="large")
def show(engine) -> None:
    headline = engine.headline
    summary = engine.benchmarks
    production = summary.get("production_params", config.PRODUCTION)
    benchmark_params = summary.get("params", {})
    search = engine.search or {}
    reference = engine.reference_row

    st.caption(
        f"Generated from the current artifacts "
        f"({engine.run_meta.get('generated_at', 'unknown date')}). "
        f"Every number below is read from this run."
    )

    split = engine.run_meta.get("corpus", {})
    products_clause = (
        f" over {split['split_distinct_products']:,} distinct products"
        if "split_distinct_products" in split else ""
    )
    shape_clause = (
        f" The single most common product ({split['split_most_common_product']}) appears "
        f"in {split['split_most_common_share']:.1%} of orders; the median product appears "
        f"in {split['split_median_product_share']:.3%}."
        if "split_most_common_share" in split else ""
    )

    phases = {
        "1 · Business understanding": f"""
**Question.** A grocery retailer wants to know, for any basket a shopper has
assembled, which product to suggest next — and wants the evidence for the
suggestion, not just the suggestion.

**Why association mining.** There is no label to predict here. The task is to
find combinations that occur together more often than chance explains, which is
unsupervised frequent-itemset mining followed by rule scoring. There is no
training set, no test set and no fitted parameters: a rule is a counted fact
about the corpus.

**What success means.** Rules that are frequent enough to be trustworthy, strong
enough to be worth acting on, and numerous enough to cover real baskets. This run
covers **{headline.get('basket_coverage', 0):.1%}** of baskets — that share fires at
least one rule. The rest legitimately have no recommendation.
""",
        "2 · Data understanding": f"""
**Source.** {config.DATASET_NAME}, the public Instacart order data
({config.DATASET_PAGE}). The `order_products__train` split holds
**{split.get('split_orders', config.TRAIN_SPLIT_ORDERS):,} real orders**{products_clause},
with each product's real aisle and department.

**Shape of the problem.** Real grocery baskets are extremely diverse.{shape_clause}
Pair frequencies are therefore small, which is why the production support
threshold is **{production.get('min_support', 0):.2%}** of orders — about
{production.get('min_support', 0) * headline.get('n_transactions', 0):,.0f} orders behind
every itemset — rather than the several percent a small synthetic catalog would
allow. That is a property of real data, not a weakness of the method.

**What the data does not have.** No prices. The unit prices in this app are
generated from a fixed seed so the basket panel has something to add up; they are
labelled illustrative everywhere they appear and never touch the mining, the rule
metrics, the affinity graph or the ranking.
""",
        "3 · Data preparation": f"""
**Catalog.** A product joins the catalog when it appears in at least
{config.CATALOG_MIN_FREQUENCY:.1%} of orders, measured over the **full** train
split before any sampling — so the catalog is a property of the dataset, not of
the sample. That admits **{headline.get('catalog_size', 0)} products**. The floor is
five times the production support threshold: an itemset can never be more common
than its rarest member, so a product below that line has no headroom to take part
in a frequent pair. It is a frequency rule, applied blind to what the products are.

**Baskets.** Each order becomes a set of `product_id` values — the canonical
identity throughout. Names, aisles, departments and colours are display metadata
joined at the last moment. There are no quantities and no timestamps; market
basket analysis works on set membership.

**The support denominator.** Orders are projected onto the catalog, and an order
left with fewer than two catalog products is **kept**. It cannot contribute a
combination, but it is a real order that did not contain the pairs being scored,
so it is negative evidence. Of {headline.get('n_transactions', 0):,} orders mined,
{headline.get('n_transactions_with_two_or_more', 0):,} can contribute a candidate — and
all {headline.get('n_transactions', 0):,} sit in the denominator of every support figure.
Dropping the sparse ones would shrink that denominator and inflate every metric
derived from it.

**No scaling, encoding or imputation.** None of it is meaningful for set-based
mining, and none is performed.
""",
        "4 · Modeling": f"""
**Three algorithms, three implementations.** Apriori (level-wise candidate
generation), FP-Growth (a frequency-ordered prefix tree mined through conditional
pattern bases) and ECLAT (vertical tidsets intersected depth-first). Each one
genuinely computes its own itemsets and reports its own measured wall-clock time.
Because all three are exact, they must agree — and they do, which the benchmark
view asserts on every run.

**Benchmark pass.** support ≥ {benchmark_params.get('min_support')}, itemsets up to
{benchmark_params.get('max_len')} products, confidence ≥ {benchmark_params.get('min_confidence')},
lift ≥ {benchmark_params.get('min_lift')}. Same corpus, same thresholds for every
entrant: that discipline is the only reason the leaderboard can be read.

**Production pass.** support ≥ {production.get('min_support')}, confidence ≥
{production.get('min_confidence')}, lift ≥ {production.get('min_lift')}, mined by
**{headline.get('production_algorithm')}** — the fastest of the three on measured
time. It yields **{headline.get('n_frequent_itemsets', 0):,} frequent itemsets** and
**{headline.get('n_active_rules', 0):,} rules**.

**Rules.** Every frequent itemset of two or more products is split into all
antecedent/consequent partitions and scored on support, confidence, lift, leverage
and conviction. A partition whose either side is not itself frequent is skipped
rather than approximated.
""",
        "5 · Evaluation": f"""
**There is no held-out evaluation, and none would be meaningful.** A rule is a
counted property of the corpus, not a prediction fitted to it. Quality is judged
by the interestingness metrics themselves.

**This run.** top lift **{headline.get('top_lift', 0):.2f}x**, mean lift
**{headline.get('mean_lift', 0):.2f}x**, mean confidence
**{headline.get('mean_confidence', 0):.1%}**, basket coverage
**{headline.get('basket_coverage', 0):.1%}**.

**External check.** {'The reference library (' + reference['algorithm'] + ') was run on the same corpus at the same thresholds and found the same frequent itemsets.' if reference else 'No external reference row in the current artifacts.'}
It is a correctness check, not a competitor and not a state-of-the-art claim.

**Parameter search.** {'A four-phase hill climb over the thresholds, optimising mean lift subject to a minimum rule count. Last run: ' + f"{search.get('initial_mean_lift', 0):.3f} → {search.get('best_mean_lift', 0):.3f} mean lift ({search.get('accepted', 0)} trials accepted, {search.get('rejected', 0)} rejected)." if search else 'Not run yet.'}
The rule-count floor is deliberate: mean lift on its own is trivially inflated by
tightening thresholds until a handful of freak rules remain, which would be
useless to the recommender. That is a rule-set quality heuristic, not a
generalization estimate.
""",
        "6 · Deployment": f"""
**Artifact-backed serving.** The pipeline writes the rule set, the affinity graph,
the benchmark summary and the catalog to `{config.ARTIFACT_DIR.name}/`. The app
loads them once and answers from memory — it never mines at request time, which is
why a suggestion takes well under a millisecond.

**Recommendation.** A rule fires when its whole antecedent is already in the
basket. Candidates are ranked by lift x confidence, and the lift and confidence
shown on a card are those of the rule that produced the ranking, so the badge and
the ordering can never disagree.

**Re-mining.** The admin view re-runs the pipeline with new thresholds and a new
corpus size, rewrites the artifacts and hot-reloads them. All four controls change
the result.

**The affinity network** draws only rules with a single product on each side,
because those are the only rules that assert an affinity between exactly two
products. This run has {headline.get('n_pairwise_rules', 0)} of them out of
{headline.get('n_active_rules', 0):,} rules; the rest keep their full meaning in the
recommender and the rules table.
""",
    }

    for title, body in phases.items():
        with st.expander(title, expanded=title.startswith("1")):
            st.markdown(body)
