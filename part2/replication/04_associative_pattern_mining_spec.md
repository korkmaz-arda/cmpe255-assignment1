# Project Specification — Market Basket Intelligence & Associative Pattern Mining

Reimplementation target derived from static inspection of the source project `04_associative_pattern_mining`.

---

## 1. Project Overview

A full-stack market-basket analysis application for grocery/e-commerce transaction data. It mines frequent itemsets and association rules from a corpus of shopping baskets, exposes the mined rules through a live cross-sell recommender, and presents the mining process and its results in an analyst-facing dashboard.

The user experience has two halves:

* A **shopper/merchandiser workspace** where a basket is assembled interactively and the system instantly returns ranked cross-sell add-ons with the association metrics that justify each one, alongside a 2D product-affinity network diagram.
* A **data-science admin workspace** presenting algorithm benchmarks, an autonomous hyperparameter-search trajectory, and a searchable/sortable rule table, plus a CRISP-DM methodology report and a live re-mining control.

Analytical identity: unsupervised frequent-itemset mining (Apriori, FP-Growth, ECLAT) over a transaction database, with rules scored by support, confidence, lift, leverage and conviction. There is no supervised model, no train/test split, and no learned parameters — all "inference" is rule lookup over a persisted rule set.

---

## 2. Required Features

### Core Requirements

**`F01` — Transaction corpus generation**

Behavior: The system produces the transaction database it mines from, deterministically, rather than reading an external dataset.

Important details:

* A fixed product catalog of 33 named grocery items across 6 departments (Produce; Dairy & Eggs; Bakery & Deli; Pantry; Beverages; Snacks). Each product carries a display name, department, unit price, and an independent baseline purchase probability (~0.10–0.38).
* Five "basket archetypes" (guacamole/Mexican, Italian pasta dinner, coffee & breakfast, PB&J snack, green salad & protein) each define a small set of *core* items and a set of *optional* items, and a selection weight. Weights are normalized into a categorical distribution over archetypes.
* Each transaction: pick one archetype by weight; include each core item with probability 0.88; include each optional item with probability 0.52; then add every remaining catalog item independently with probability equal to 0.18 × its baseline probability (impulse noise). If the result has fewer than 2 items, two random catalog items are added.
* Items are stored as a sorted, deduplicated set of product names per basket; there is no quantity or timestamp dimension.
* Default corpus size 10,000 baskets, fixed random seed (42). The seed and parameters must make the corpus and therefore all downstream metrics reproducible.
* The archetype design is the entire reason co-occurrence structure exists; it must be preserved for the mined rules to be meaningful.

**`F02` — Frequent itemset mining via three algorithm families**

Behavior: The same transaction corpus is mined by three distinct frequent-itemset algorithms under identical thresholds, so their outputs and runtimes can be compared.

Important details:

* **Level-wise candidate generation (Apriori):** frequent 1-itemsets by support count, then iterative join of (k−1)-itemsets into k-candidates, counted by scanning the transaction database, pruned at minimum support; capped at a maximum itemset length (default 4).
* **Prefix-tree mining (FP-Growth):** items filtered by minimum support and ordered by descending frequency; transactions inserted into a shared-prefix count tree.
* **Vertical tidset mining (ECLAT):** each item mapped to the set of transaction indices containing it; itemsets extended depth-first by tidset intersection, with support read directly from intersection cardinality; extension stops at the maximum length.
* All three report: the set of frequent itemsets with their support, and a wall-clock execution time.
* Under identical thresholds, the three must agree on the frequent itemsets and their supports (this is the invariant the benchmark comparison relies on); only runtime and memory characteristics should differ. See caveats — the source satisfies this by construction rather than by independent computation.

**`F03` — Association rule generation with multi-metric scoring**

Behavior: Every frequent itemset of size ≥ 2 is split into all non-empty antecedent/consequent partitions, and each candidate rule is scored and filtered.

Important details:

* Metrics per rule: rule support = support of the union; confidence = union support ÷ antecedent support; lift = union support ÷ (antecedent support × consequent support); leverage = union support − (antecedent support × consequent support); conviction = (1 − consequent support) ÷ (1 − confidence), with the denominator floored at a small epsilon to avoid division by zero.
* Antecedent and consequent supports are looked up from the frequent-itemset table; a partition whose antecedent or consequent is not itself frequent is skipped.
* Rules are retained only when confidence ≥ a minimum confidence and lift ≥ a minimum lift.
* Rules are sorted by lift descending, then confidence descending.
* Each rule is published with both machine-readable item lists and a human-readable string form (`A + B ➔ C`), plus the antecedent and consequent marginal supports.
* Multi-item consequents are allowed and common.

**`F04` — Production rule set and persisted artifacts**

Behavior: A pipeline run mines a canonical "production" rule set and writes it, plus supporting artifacts, to durable storage that the serving layer reads.

Important details:

* Benchmark pass: all three algorithms at min support 0.035, max length 4, rule filters confidence ≥ 0.30 and lift ≥ 1.20.
* Production pass: min support 0.03, max length 4, confidence ≥ 0.35, lift ≥ 1.25.
* Artifacts written: the production rule list; the affinity graph (`F05`); a benchmark summary containing the champion algorithm name, production headline metrics (transaction count, catalog size, frequent-itemset count, active rule count, top rule lift, mean rule confidence) and the algorithm leaderboard; and the product catalog.
* On the reference corpus this yields on the order of 1,900 production rules from ~380 frequent itemsets, with a top lift above 6 and mean confidence around 0.60.

**`F05` — Product affinity network graph**

Behavior: The strongest rules are projected into a 2D node-link diagram of product affinities that the UI renders directly.

Important details:

* Built from the top 50 production rules by lift.
* Nodes are the distinct products appearing in those rules, carrying department, price, a department-derived color, and the product's marginal transaction frequency.
* Each rule contributes one edge, carrying the rule's lift, confidence, support and readable rule string.
* Node positions are precomputed server-side on a circle (fixed center/radius) so the client does no layout work; the layout is static, not force-simulated, despite the "force-directed" label in documentation.
* Department→color mapping is part of the visual contract (one distinct color per department).

**`F06` — Interactive basket cross-sell recommender**

Behavior: The user builds a basket; for every change the system returns ranked add-on recommendations derived from the rule set.

Important details:

* A rule fires when its full antecedent is a subset of the current basket. Each consequent item not already in the basket becomes a candidate.
* Candidate ranking score = lift × confidence, taking the maximum over firing rules. Separately, the maximum lift and maximum confidence seen for that candidate are reported, along with the supporting rule text.
* Top 6 candidates returned, each with product name, department, price, score, lift, confidence percentage, a headline supporting rule, and up to three supporting rules.
* The response also reports basket item count, current basket monetary total (sum of catalog prices), and a "potential GMV uplift" equal to the summed price of the top 3 recommendations.
* Recommendation is pure in-memory lookup — no model is evaluated — and is expected to be sub-millisecond.
* Empty/no-match state: when no rule fires, the UI must say so and suggest trying complementary items rather than showing an empty panel.

**`F07` — Algorithm benchmark comparison**

Behavior: The admin view presents a leaderboard comparing the three mining algorithms on the same corpus and thresholds, against a labeled external reference baseline.

Important details:

* Per entry: algorithm name, mining paradigm description, frequent-itemset count, rule count, top lift, mean confidence, execution time, and a qualitative memory-efficiency note.
* Ordering places the external reference row first, then implemented algorithms by ascending runtime; the fastest implemented algorithm is badged as production champion.
* The external "Kaggle Grandmaster SOTA" row (284 itemsets, 142 rules, 4.85 lift, 68.4% confidence, 3.82 s) is a static constant, not a computed or reproduced result, and is flagged as a baseline row in the data itself.
* Accompanied by headline metric tiles (transactions, frequent itemsets at 3% support, active rules at 35% confidence, peak lift) and three explanatory cards describing the complexity trade-offs of the three paradigms.

**`F08` — Autonomous mining-parameter search ("AutoResearch") with inspectable trajectory**

Behavior: A four-phase automated search over mining configurations runs against a smaller corpus, records every trial as an accept/reject step with rationale, and the dashboard replays that trajectory.

Important details:

* Objective metric: **mean lift across the surviving rule set**.
* Phase 1 — backbone tournament: all three algorithms at min support 0.035 / confidence 0.30 / lift 1.20; results ranked by runtime; a baseline step (step 0) records the starting mean lift.
* Phase 2 — five threshold-mutation trials (tightening the lift gate to 1.60; raising confidence to 0.45; lowering support to 0.020 to reach niche pairings; restricting itemsets to pairs; a high-leverage configuration at support 0.025 / confidence 0.40 / lift 1.75).
* Phase 3 — four-point hyperparameter grid over (min support, min confidence, min lift).
* Phase 4 — a bundle-optimization step for 3–4 item antecedents.
* Acceptance rule: a trial is accepted only if it improves the incumbent mean lift by more than 0.02 **and** retains at least a minimum rule count (10 in phase 2, 8 in phase 3). Accepted trials update the incumbent; rejected ones do not.
* Each step records: step id, iteration, phase, category, a natural-language hypothesis, a short code-snippet illustrating the change, the hyperparameter set, mean lift before/after, delta, decision, a written reflection, and a timestamp.
* The exported run summary reports initial mean lift, best mean lift, percentage improvement, total iterations, and accepted/rejected counts, plus the backbone leaderboard and the full trajectory.
* Corpus for the search defaults to a few thousand baskets (not the full 10,000), so its absolute numbers differ from the production artifacts by design.
* The search must be re-runnable on demand from the UI, refreshing the displayed trajectory on completion.

**`F09` — Association rules explorer**

Behavior: A table of the active rules that the analyst can search and sort.

Important details:

* Columns: rank, antecedent items, consequent items, support (as %), confidence (as %), lift (×), leverage, conviction.
* Free-text search filters on antecedent or consequent text.
* Support, confidence, lift, leverage and conviction columns are click-sortable with direction toggling; default sort is lift descending.
* The table is fed by a top-N rules query (60 by default) and displays at most 50 rows.

**`F10` — Live re-mining control**

Behavior: The user can trigger a full re-run of the mining pipeline from the UI and see the refreshed results without restarting anything.

Important details:

* Exposed controls: minimum support, minimum confidence, minimum lift, and corpus size, with validated ranges (support 0.01–0.20, confidence 0.10–0.90, lift 1.0–5.0, transactions 1,000–30,000).
* On completion the serving layer reloads the regenerated artifacts, the UI refetches every dataset, and the new active-rule count is reported back to the user.
* See caveats: in the source only the corpus size actually influences the re-mined output.

---

### Secondary Requirements

**`S01` — Two-level navigation.** A top-level switch between the basket/graph workspace and the admin workspace, and within admin a sub-switch between benchmarks, AutoResearch, and rules explorer. Tab state resets to the basket workspace on load.

**`S02` — Service status indicator.** A health readout in the header showing the service is reachable and how many rules are currently loaded, plus the champion algorithm and production metrics available to the app.

**`S03` — Quick basket presets.** One-click preset baskets (guacamole fiesta, Italian pasta dinner, morning espresso bar, PB&J snack) that replace the current basket, matching the archetypes the corpus was generated from. The app opens with a non-empty default basket (avocados + limes) so results are visible immediately.

**`S04` — Basket editing affordances.** Catalog dropdown (showing name, department, price; already-basketed items disabled), add button, per-item remove, live item count, and an "add to cart" action on each recommendation card that promotes it into the basket and re-triggers recommendation.

**`S05` — Basket economics display.** Current basket value and projected uplift shown as currency, side by side.

**`S06` — Graph interaction.** Hovering a node enlarges it, highlights only the rules touching that product, and shows a tooltip with the product's department, price and marginal transaction frequency. With no hover, a bounded subset of edges (top 35) is drawn. Edge thickness encodes lift.

**`S07` — Metric formula panel.** A persistent explanatory strip on the basket workspace stating the support, confidence, lift and conviction definitions, plus a problem-statement banner framing the cross-sell task and three headline figures (external SOTA lift, evolved mean lift, claimed inference latency). These three figures are static display constants, not computed at render time.

**`S08` — AutoResearch trajectory chart.** A line/point chart of mean lift versus iteration, with accepted steps in one color and rejected in another; points are clickable to open the step inspector.

**`S09` — AutoResearch step inspector.** A modal showing one trial's hypothesis, decision and delta, before→after metric, the agent's written reflection, the illustrative code change, and the hyperparameter set.

**`S10` — Trajectory filtering.** Filter the mutation table by phase and by accept/reject decision.

**`S11` — Backbone tournament cards.** Per-algorithm summary cards from the search run (family, itemsets, mean lift, runtime) with the fastest badged as tournament champion.

**`S12` — CRISP-DM report.** A six-phase modal report (business understanding, data understanding, data preparation, modeling, evaluation & benchmark, deployment) with a per-phase executive summary and a longer body including the metric definitions and a benchmark comparison table. Content is static authored prose rendered as preformatted text, not generated from the current run.

**`S13` — Run feedback.** Long-running actions (re-mining, AutoResearch) show an in-progress label, disable their trigger, and celebrate completion with a brief visual effect and a success message.

---

## 3. User Workflow

1. The app loads, fetching health, catalog, graph, benchmarks, top rules and search history in parallel. `[S01, S02]`
2. The basket workspace opens with a seeded basket; recommendations and the affinity graph are already populated. `[S03, F06, F05]`
3. The user edits the basket — presets, dropdown add, remove, or accepting a recommendation — and recommendations plus basket economics update on every change. `[S04, F06, S05]`
4. The user explores the affinity graph by hovering products to isolate their rules. `[S06]`
5. The user opens the CRISP-DM report for methodology context. `[S12]`
6. The user switches to admin and compares algorithm backbones against the reference baseline. `[F07]`
7. The user reviews the automated search trajectory, filters it, and inspects individual accepted/rejected trials. `[F08, S08, S09, S10, S11]`
8. The user searches and sorts the active rule table. `[F09]`
9. Optionally the user re-runs the search `[F08]` or re-mines with new parameters `[F10]`; on success all views refresh. `[S13]`

---

## 4. Data, State, and Data-Science Behavior

**Data origin.** Entirely self-generated (`F01`). No external dataset is downloaded or bundled, despite consistent "Kaggle Instacart" branding throughout the documentation and UI. The catalog is 33 hand-authored products, not the 500+ SKUs claimed in the papers.

**Preprocessing.** Minimal by design: baskets are sets of product-name strings. ECLAT additionally inverts the corpus into per-item transaction-index sets. There is no encoding, scaling, imputation, or train/test partitioning, and none is meaningful for this method.

**Thresholds that matter.**

| Context | min support | max itemset length | min confidence | min lift |
|---|---|---|---|---|
| Benchmark comparison | 0.035 | 4 | 0.30 | 1.20 |
| Production rule set | 0.030 | 4 | 0.35 | 1.25 |
| Search baseline | 0.035 | 4 | 0.30 | 1.20 |
| Search trials | 0.020–0.040 | 2 or 4 | 0.30–0.60 | 1.20–2.20 |
| Re-mining UI ranges | 0.01–0.20 | — | 0.10–0.90 | 1.0–5.0 |

**Evaluation.** There is no held-out evaluation; rule quality is judged by the interestingness metrics themselves. The automated search optimizes mean lift subject to a rule-count floor, which is a rule-set quality heuristic, not a generalization estimate. Claims of "leakage-free cross-validation" in the project's documentation have no counterpart in the implementation and are not applicable to this method.

**State and persistence.** All state is file-backed JSON artifacts loaded once into memory at service start: the rule list, the graph, the benchmark summary, the catalog, and the search history. Recommendation, rule listing, graph retrieval and benchmark retrieval all read that in-memory state. Re-mining rewrites the artifacts on disk and reloads them in place, so results persist across restarts; the search run likewise overwrites its history artifact. Client state (active basket, tabs, filters, sort) is ephemeral and resets on reload.

**Inference behavior.** Scoring is subset matching plus arithmetic over precomputed rule metrics. Nothing is fitted at request time. Latency figures displayed in the UI are static text, not measurements.

---

## 5. Outputs and UI Behavior

* **Recommendation cards** — per suggested product: name, department, price, lift badge, the readable rule that justifies it, confidence percentage, and an add action. Communicates *what to suggest next and why*, with the statistical evidence visible rather than hidden.
* **Basket economics** — current basket value and projected added value from the top three suggestions.
* **Affinity network** — colored product nodes on a fixed circular layout, edges whose thickness grows with lift, hover isolation of a product's rules, and a tooltip giving department, price and how often the product appears in transactions. Communicates which products sit at the center of the affinity structure.
* **Benchmark leaderboard and metric tiles** — runtime/quality comparison of the three algorithms against a labeled external baseline, plus corpus- and rule-level headline numbers.
* **Search trajectory chart and mutation table** — the accept/reject shape of the automated search over iterations, with per-step drill-down into hypothesis, reasoning, parameters and code change. Communicates *how* the final configuration was reached, including the dead ends.
* **Rules explorer table** — the full interestingness profile of each rule (support, confidence, lift, leverage, conviction) with search and sort.
* **CRISP-DM report** — six-phase methodology narrative with formulas and a comparison table.
* **Important states** — loading text while recommendations are being fetched; an explicit "no rules triggered" message for unmatched baskets; an "empty basket" prompt; in-progress labels and disabled buttons during re-mining and search; a header badge that only appears once the service responds.

---

## 6. Important Semantic Mechanisms

* **Artifact-backed serving.** The serving layer never mines at request time; it answers from rule artifacts produced by a separate pipeline run. Replacing this with on-request mining would change the latency story and the meaning of the "active rules" counters.
* **Recompute-and-reload.** The re-mining path regenerates artifacts and swaps them into the running service, which is what makes the parameter studio feel live; the client must refetch every dataset afterwards for the views to agree.
* **Subset-match rule firing with max-aggregation.** A recommendation's rank comes from the best lift×confidence among all firing rules, while its displayed lift and confidence are independent maxima. This aggregation choice, not any single rule, determines the ordering users see.
* **Archetype-driven corpus structure.** The specific mixture of archetypes and impulse noise is what produces the lift values the whole application is about; changing the generator changes every downstream number.
* **Identical-threshold benchmark premise.** The leaderboard is only interpretable because all backbones are run on the same corpus at the same thresholds, so any reimplementation must preserve that discipline.
* **Fixed seeding.** Determinism of the corpus is what makes the committed artifacts, tests, and documented numbers comparable at all.

---

## 7. Source Caveats

### Apparent implementation defects or quirks

* **The prefix-tree miner does not mine.** It builds a frequency-ordered prefix tree from the transactions and then discards it, returning the level-wise algorithm's itemsets instead; no conditional pattern bases or recursive projections are extracted from the tree. The invariant violated is that a named algorithm's reported output should be produced by that algorithm. Its correctness is therefore untested and its independent behavior unknown.
* **The prefix-tree runtime is synthesized.** The reported execution time is the measured tree-build time scaled by a constant factor (~0.45) with a floor, explicitly to make it appear faster than the level-wise algorithm. It is then used as the ranking key for the "production champion" badge and the tournament ordering. A benchmark timing should reflect the work actually performed for the reported result.
* **Committed artifacts make this visible.** All three backbones report byte-identical itemset counts, rule counts, top lift and mean confidence (e.g. 364 / 1944 / 6.069 / 0.589), differing only in runtime — consistent with two of the three sharing one computation.
* **Graph edges misrepresent multi-item rules.** Each rule contributes a single edge between its *first* antecedent item and its *first* consequent item, while the edge tooltip shows the full rule string. Edges therefore claim affinities between item pairs that the rule does not assert, and distinct rules collapse onto the same visual edge. The committed graph has 50 edges over only 10 nodes for this reason. The intended invariant: an edge should represent an affinity actually asserted between the two products it connects.
* **Re-mining ignores three of its four controls.** The support, confidence and lift sliders are validated and transmitted but the pipeline is invoked with its hardcoded defaults; only corpus size takes effect. Users are shown parameter controls whose settings do not affect the result they receive.
* **Reported lift/confidence may come from different rules than the ranking score.** Per-candidate maxima are tracked independently, and the headline supporting rule is simply the first firing rule in lift order, which need not be the rule that produced either maximum.
* **Trajectory chart uses a hardcoded y-range.** The chart maps mean lift onto a fixed window (roughly 3.4–3.9); trials outside it (the committed history contains one at 2.87) render off-chart.
* **Search run is synchronous.** The endpoint documented as running the search "in background" executes it inline and blocks until completion.
* **Vertical-tidset extension records supports without re-checking the minimum.** Extensions are filtered before recursion, so in practice the emitted itemsets are frequent, but the emission step itself applies no threshold check.
* **Fallback constants in the UI diverge from the artifacts.** Default metric tiles (364 itemsets, 4.48 peak lift) and default search KPIs (3 accepted / 7 rejected) differ from the committed artifacts (382 itemsets, 6.069 lift, 4 accepted / 6 rejected). These only surface when data is missing, but they will mislead.

### Conflicting evidence

* **Dataset provenance.** README, abstract, paper, article, CRISP-DM report and UI all describe the Kaggle Instacart dataset with "500+ distinct SKUs". The implementation generates the corpus synthetically from 33 authored products. The synthetic generator is the real data source.
* **Headline numbers.** Documentation cites top lift 4.48–4.85 and mean confidence 52.8%/64.2%; committed artifacts show top lift 6.069 and mean confidence 58.9% (benchmark) / 60.2% (production). Documentation runtimes (0.382 / 0.026 / 0.889 s) also differ from the artifacts (0.402 / 0.031 / 0.889 s).
* **Rule counts.** The CRISP-DM table and benchmark artifact disagree with the production artifact (1944 vs 1921 rules, 364 vs 382 itemsets) because they come from different threshold passes; the tables do not label which pass they describe.
* **External baseline.** The "Kaggle Grandmaster SOTA" leaderboard row and the SOTA figure on the basket workspace banner are static constants embedded in the pipeline and the UI respectively. They are not reproduced, measured, or derived from any competition result available in the repository.
* **Audit report.** The committed audit asserts a 99.3% compliance grade, zero leakage, deterministic seeding "across NumPy, PyTorch and Scikit-Learn", and verification of loss functions and cross-validation folds. The project contains no PyTorch, no scikit-learn, no loss function and no cross-validation. The report is generic boilerplate and should not be treated as evidence of anything.
* **Quick-start instructions are wrong.** The README instructs running from `backend/` and `frontend/` directories; the actual directories are `server/` and `client/`.
* **Architecture prose is inaccurate.** The paper and article describe TypeScript, Server-Sent Events, Express, rolling walk-forward cross-validation and force-directed layout. The implementation is plain JavaScript/React with REST polling on user action, a Python service, and a precomputed circular layout.

### Documented but unimplemented intent

* A second packaged skill for conversion-funnel analysis is present and references an analyzer script, a design guide and a report template — none of which exist in the repository. The application has no funnel functionality.
* Claimed distributed multi-node scaling and edge quantization are stated as future work only.
* The visual-audit automation script is present but hardcodes an absolute Windows artifact path and browser profile directory from the original author's machine; it captures six named screenshots, only three of which are committed.

### Unresolved uncertainty

* Because the prefix-tree miner delegates, static inspection cannot establish what an independent prefix-tree implementation would produce on this corpus, nor whether the committed runtime ordering would survive an honest implementation.
* The exact provenance of the committed artifacts is unclear: the production artifact's metrics are self-consistent with the current production thresholds, but the search-history artifact was produced with a corpus size that neither the on-demand endpoint (4,000) nor the script default (5,000) unambiguously identifies.

---

## 8. Acceptance Checklist

* [ ] `F01` — Deterministic archetype-driven transaction corpus over the 33-product, 6-department catalog.
* [ ] `F02` — Three distinct frequent-itemset algorithms, each genuinely producing its own itemsets and its own honest timing, agreeing on results under identical thresholds.
* [ ] `F03` — Rules generated over all partitions of frequent itemsets, scored on support, confidence, lift, leverage and conviction, filtered and ranked as specified.
* [ ] `F04` — Pipeline produces and persists the production rule set, benchmark summary, graph and catalog artifacts.
* [ ] `F05` — Affinity graph whose every edge represents an affinity actually asserted between the two products it connects, with department coloring and marginal frequencies.
* [ ] `F06` — Basket-driven cross-sell recommendations with consistent, explainable ranking and supporting-rule attribution, plus basket and uplift totals.
* [ ] `F07` — Algorithm leaderboard with headline metrics and an explicitly labeled, non-computed external reference row.
* [ ] `F08` — Four-phase parameter search with the stated acceptance rule, full per-step trajectory record, and on-demand re-run.
* [ ] `F09` — Searchable, multi-column-sortable rules table.
* [ ] `F10` — Re-mining control in which every exposed parameter affects the regenerated rule set, with refreshed views afterwards.
* [ ] `S01`–`S13` — Navigation, health indicator, presets and default basket, basket editing, economics display, graph hover isolation, formula/problem panel, trajectory chart, step inspector, trajectory filters, tournament cards, CRISP-DM report, run feedback.
