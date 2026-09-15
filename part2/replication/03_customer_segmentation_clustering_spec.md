# Project Specification — Customer Segmentation & Clustering Intelligence Platform

*Replication target derived from static inspection of source project `03_customer_segmentation_clustering`.*

---

## 1. Project Overview

A full-stack unsupervised-learning application for **retail customer segmentation**. It generates a synthetic customer behavioural dataset, engineers domain features, runs a tournament of five clustering algorithm families, trains a production K-Means model, profiles the resulting clusters into named marketing personas, projects customers into 2D via PCA and t-SNE, and serves all of it through a REST API consumed by a two-view web dashboard.

The user experience has two halves:

* a **customer-facing segment explorer** — persona summary cards, an interactive 2D manifold scatter plot, and a live "classify a new customer" form that returns a persona plus a marketing recommendation;
* a **data-science admin console** — algorithm leaderboard with clustering-validity metrics, elbow/silhouette-vs-k analysis, an autonomous "AutoResearch" hill-climbing experiment log with step-level inspection, and normalized per-persona radar profiles.

Two global modals sit above both views: a six-phase CRISP-DM methodology report and a retraining studio that re-runs the whole pipeline on demand.

Its data-science identity is **unsupervised clustering with internal validity metrics** (silhouette, Davies–Bouldin, Calinski–Harabasz, WCSS) plus **centroid-distance inference** for new records.

---

## 2. Required Features

### Core Requirements

---

**F01 — Generate the customer behavioural dataset**

**Behavior:** The dataset is *synthesized in code*, not loaded from a file, despite being presented throughout the UI and documentation as "the Kaggle Customer Personality dataset." Default size is 10,000 records; the size and random seed are parameters.

**Important details:**

* Records are drawn from five hand-specified ground-truth generator groups of equal size (N/5 each), one per intended persona. A ground-truth group label is carried on each record but is **never used for fitting or evaluation** — clustering is fully unsupervised.
* Eight per-customer attributes: age, annual income (thousands), spending score (1–100), recency in days since last order, total annual spend, monthly web visits, discount sensitivity (0–1), household size.
* Each group uses distinct distributions per attribute (normal, exponential, Poisson, beta, and a categorical choice for household size), each clipped to a plausible range. Total annual spend is a noisy linear function of income and spending score, then clipped. The distinguishing design intent per group:

  | Group | Intended archetype | Income | Spending score | Notable |
  |---|---|---|---|---|
  | A | affluent high spenders | ~110k | ~84 | low recency, high total spend (6k–15k), low discount sensitivity |
  | B | affluent low spenders | ~105k | ~24 | older, high recency, few web visits |
  | C | young high spenders | ~38k | ~82 | youngest (~26), highest web visits (~14/mo), small households |
  | D | low-income low spenders | ~32k | ~20 | highest discount sensitivity (0.65–0.98), lowest total spend |
  | E | mid-range steady | ~68k | ~50 | mid on every axis |

* Rows are shuffled with the same seed and re-indexed, so generation is fully deterministic for a given seed.
* Records carry a stable customer identifier used for scatter-point tooltips.

---

**F02 — Engineer behavioural interaction features**

**Behavior:** Four ratio/interaction features are derived from the eight base attributes and appended, giving a 12-column feature matrix used for all clustering and inference.

**Important details:**

* Discretionary ratio: income ÷ (spending score + 1)
* Monetary velocity: total annual spend ÷ (recency days + 1)
* Digital engagement: web visits × (spending score ÷ 100)
* Deal affinity: discount sensitivity × (1 − spending score ÷ 100)
* The `+1` denominators are deliberate divide-by-zero guards and should be retained.
* All 12 columns are z-score standardized before any distance computation. The fitted scaler is persisted and re-applied (never re-fit) at inference time — this is the project's stated leakage-safety property.

---

**F03 — Run a multi-algorithm clustering tournament**

**Behavior:** Five clustering families are fit on the same standardized matrix and scored on the same internal-validity metrics, producing a ranked leaderboard.

**Important details:**

* Families and their configured settings: K-Means with k-means++ seeding (k=5, 10 restarts); Gaussian mixture (5 components, full covariance); agglomerative hierarchical (5 clusters, Ward linkage); DBSCAN (ε=0.85, min samples 20); spectral clustering (5 clusters, nearest-neighbour affinity).
* To keep exact metric computation affordable, the tournament is evaluated on a **fixed random 3,500-row subset** of the full matrix, drawn with the pipeline seed. The same subset is reused later to score the production model, so leaderboard and production numbers are comparable.
* Metrics per entrant: silhouette, Davies–Bouldin, Calinski–Harabasz, resulting cluster count, noise ratio (density-based methods can emit a noise label), and fit wall-time.
* Noise-labelled points are excluded from metric computation; degenerate results (fewer than two clusters) are recorded as silhouette 0 / Davies–Bouldin 99.
* The leaderboard is sorted by silhouette descending.

---

**F04 — Insert an external "Kaggle top-1%" reference row into the leaderboard**

**Behavior:** A sixth leaderboard row representing an external state-of-the-art baseline is displayed alongside the real entrants and is badged distinctly in the table.

**Important details:**

* Its metrics are **statically written constants**, not computed from any experiment: silhouette 0.3850, Davies–Bouldin 0.9820, Calinski–Harabasz 2150.0, 5 clusters, fit time 4.12s.
* It participates in the same silhouette sort, and in the committed artifact it ranks **first**, above every actually-trained model. The UI's "champion / production model" highlight is applied to whichever row sorts first, so the badge logic and the baseline row interact (see caveats).

---

**F05 — Train and persist the production segmentation model**

**Behavior:** After the tournament, a K-Means model (k=5, k-means++, 15 restarts, seeded) is fit on the **full** standardized dataset and becomes the model served for live inference. Its silhouette, Davies–Bouldin and Calinski–Harabasz are reported on the 3,500-row evaluation subset.

**Important details:**

* The fitted model, the fitted scaler, and the ordered feature-column list are persisted together so that inference reproduces the training transform exactly.
* Reference values from the committed artifact: silhouette 0.3497, Davies–Bouldin 1.0201, Calinski–Harabasz 1966.8, over 10,000 customers.

---

**F06 — Compute 2D PCA and t-SNE projections**

**Behavior:** Two 2D embeddings of the standardized feature space are produced for the scatter visualization.

**Important details:**

* PCA is fit on the full matrix with 2 components; the per-component and total explained-variance ratios are retained and surfaced in the UI (reference value: 69.6% total).
* t-SNE (perplexity 30, seeded, bounded iterations) is fit on a **1,200-row random sample** — t-SNE has no out-of-sample transform, so only these sampled customers have t-SNE coordinates.
* The PCA transform is persisted because it is reused at inference to place a newly classified customer on the same scatter canvas. t-SNE is not persisted and cannot be applied to new inputs.
* The exported scatter dataset is exactly the 1,200 sampled customers, each carrying both coordinate pairs, its assigned cluster, and the raw attributes shown on hover.

---

**F07 — Profile clusters into named marketing personas**

**Behavior:** For each of the five discovered clusters the application computes member count, share of the customer base, and the mean of all eight base attributes, then attaches a persona name, tagline, badge, colour, prose description, and a marketing strategy recommendation.

**Important details:**

* The five persona identities are: *VIP Champions*, *Prudent Affluents*, *Young Trendsetters*, *Bargain Hunters*, *Mainstream Loyalists* — each with a fixed colour used consistently across cards, scatter points, radar polygons and prediction results.
* **The mechanism by which personas are attached is positional**: persona *n* in the fixed metadata table is assigned to cluster index *n* emitted by K-Means. K-Means label indices are arbitrary, so this mapping is not guaranteed to match cluster content, and in the committed artifact it does not (see caveats). The replication requirement is semantic: each persona's identity — name, tagline, badge, description and marketing strategy — must correspond to the behavioral characteristics of the cluster it is attached to.
* Persona descriptions and marketing strategies are authored content, not derived from the cluster statistics.

---

**F08 — Elbow and silhouette-versus-k analysis**

**Behavior:** K-Means is re-fit for each k from 2 through 9 on the evaluation subset, recording within-cluster sum of squares (inertia) and silhouette per k, for display as an elbow curve and a per-k score list.

**Important details:**

* The UI hard-highlights k=5 as the "elbow point" / "peak" regardless of the data. In the committed artifact the maximum silhouette actually occurs at k=6 (0.3615) and k=5 scores 0.3500 (see caveats).

---

**F09 — Real-time single-customer segment classification**

**Behavior:** A user submits the eight customer attributes and receives a cluster assignment with persona identity, confidence, distance to centroid, PCA coordinates, the full persona narrative, and a tailored marketing action.

**Important details:**

* The input passes through the identical feature-engineering step and the *persisted* scaler, then distances to all five centroids are computed; the nearest wins.
* Confidence is a **normalized inverse-distance share**, not a probability from a fitted model: each centroid distance is inverted (with a small epsilon guard), the inverses are normalized to sum to one, and the winner's share is reported as a percentage. It is therefore always > 20% for k=5 and is a heuristic score.
* The record's PCA coordinates are returned so the UI can drop a live marker onto the scatter canvas.
* Every input field has a permitted range enforced server-side (age 18–90, income 10–250k, spending score 1–100, recency 1–365 days, annual spend 50–50,000, web visits 0–50, discount sensitivity 0–1, household 1–10); out-of-range submissions are rejected as validation errors. Missing fields fall back to mid-range defaults.

---

**F10 — Autonomous "AutoResearch" hill-climbing optimizer**

**Behavior:** A separate optimization routine searches for a better clustering configuration in four sequential phases, recording every trial as an inspectable step with a hypothesis, the transformation applied, before/after silhouette, delta, an accept/reject decision, and a written reflection. Results are persisted as a single run record and displayed in the admin console.

**Important details:**

* It operates on its own generated dataset (5,000 rows by default; 4,000 when triggered from the API) and starts from the **eight base attributes only**, not the engineered set — the engineered features are what it rediscovers.
* **Phase 1 — backbone tournament.** The same five families are evaluated; the best by silhouette becomes the incumbent and its score the baseline, logged as step 0.
* **Phase 2 — feature mutations.** Six candidate features are proposed one at a time and greedily kept if they improve silhouette: monetary velocity, discretionary ratio, digital engagement, deal affinity, log-transformed total spend, and a power-transformed (Gaussianized) income. Accepted features persist into the working matrix for all later steps.
* **Phase 3 — hyperparameter trials.** Five configurations are tried against the evolved matrix: K-Means with more restarts, k=6, k=4, agglomerative with average linkage, and a tied-covariance mixture.
* **Phase 4 — consensus ensembling.** K-Means and a Gaussian mixture are both fit on the evolved matrix, but the co-association consensus is **not actually computed or scored**; the step is logged as accepted with a zero delta and a fixed narrative claiming ">91.4% co-assignment" (see caveats).
* **Acceptance gate:** a candidate is accepted only if silhouette improves by more than +0.0005; otherwise the working state is left unchanged and the step is marked rejected.
* The run record summarizes starting and best silhouette, percentage improvement over the *starting* score, total steps, accepted and rejected counts, the phase-1 leaderboard, and the final active feature list.
* Reference committed run: baseline 0.34549 → best 0.41803, "+21.0%", 12 steps, 7 accepted / 5 rejected. All five feature mutations except monetary velocity were accepted; the best score came from the **k=4** hyperparameter trial.

---

**F11 — On-demand retraining**

**Behavior:** From the retraining studio the user picks a target cluster count and a dataset size, executes the pipeline, and the running service reloads all artifacts so every view reflects the new model without a restart. Success is confirmed with a message and a celebratory confetti animation; failures surface as an error message.

**Important details:**

* Accepted ranges: cluster count 2–10, dataset size 1,000–30,000 server-side (the slider offers 2,000–20,000), plus a seed.
* **The cluster-count choice is accepted and validated but has no effect** — the pipeline is hard-wired to five clusters (see caveats). Dataset size and seed do take effect.
* Retraining is synchronous and re-runs everything: generation, tournament, production fit, PCA, t-SNE, profiling, elbow sweep, and artifact export.

---

**F12 — On-demand AutoResearch execution**

**Behavior:** The admin console can launch a fresh AutoResearch run, which replaces the stored run record; the view reloads and confetti fires on success.

**Important details:** the run is synchronous and blocks until all four phases finish; failures surface inline in the panel.

---

### Secondary Requirements

---

**S01 — Persona summary cards with cross-filtering**

Five cards, one per persona, showing cluster index, share and member count, persona name and tagline, and average income / spending score / annual spend. Clicking a card filters the scatter plot to that cluster; clicking the selected card clears the filter. Selection is reflected in the card's border and background.

---

**S02 — Interactive 2D manifold scatter plot**

An SVG canvas rendering the sampled customers coloured by cluster, with a PCA/t-SNE toggle, centre crosshair guides, and axis bounds that always include at least ±3.5 so sparse filtered views stay stable. Hovering a point enlarges it and opens a detail panel with the customer identifier, persona name, age, income, spending score, annual spend, recency and discount sensitivity. The caption states the projection in use and, for PCA, its explained variance. The live-classified customer is overlaid as a pulsing ringed marker in the persona colour — **only in PCA mode**, since no t-SNE transform exists for new points.

---

**S03 — Live classifier form**

Six range sliders (age, income, spending score, recency, annual spend, web visits) with live value read-outs, a submit button with a pending state, and a result card in the persona colour showing cluster index, confidence, persona name, description, and the marketing action in a highlighted callout. A prediction is issued automatically on first load using the default slider values. Discount sensitivity and household size are **sent with fixed default values but have no UI control** (see caveats).

---

**S04 — Problem-framing banner**

A header panel stating the business problem (generic blanket promotions convert poorly; discover archetypes without supervisory labels), a three-value strip (external baseline silhouette 0.3850; "AutoResearch peak" 0.4180 with "+21.0% gain"; optimal clusters k=5), and chips naming the three headline engineered signals with their formulas. All five of these numbers are **statically written into the view**, not read from the API.

---

**S05 — Benchmark KPI tiles and leaderboard table**

Four KPI tiles (production algorithm, silhouette, Davies–Bouldin, Calinski–Harabasz — each with a short interpretive caption such as "lower is better") and a nine-column leaderboard table: rank, algorithm, family/formulation, cluster count, the three validity metrics, fit time, and a status badge distinguishing the external baseline row, the top-ranked "production" row, and plain benchmarked rows. Tiles fall back to hardcoded literals when the API payload is missing.

---

**S06 — Elbow and silhouette-vs-k panels**

A WCSS-versus-k polyline with a point per k, k-labelled ticks, and an enlarged marker at k=5; alongside it a per-k list showing WCSS and silhouette with the k=5 row emphasized.

---

**S07 — AutoResearch trajectory and experiment stream**

A backbone-tournament card grid (family, name, the three metrics, fit time, champion badge on the leader); four KPI tiles (starting silhouette, best silhouette with the percent gain, total steps, accepted/rejected counts); a trajectory chart plotting silhouette after each step, green for accepted and red for rejected, with step numbers; and a table of every step with round, phase, hypothesis, component, resulting silhouette, signed delta, and decision badge.

---

**S08 — Step inspection modal**

Clicking any trajectory point, table row or inspect button opens a modal with a before/after/gate banner, the transformation snippet for that step, the agent's written reflection, and the active hyperparameters as formatted JSON.

---

**S09 — AutoResearch filtering**

Phase filter (all, backbone battle, feature evolution, hyperparameter tuning, consensus ensembling) and decision filter (all, accepted, rejected), combined conjunctively; the table header shows the filtered step count.

---

**S10 — Run-record export**

A download control writes the complete AutoResearch run record to a timestamped JSON file on the user's machine.

---

**S11 — Persona radar profiles**

A radar chart over six normalized axes — income, spending score, annual spend, monthly web visits, age, recency — each divided by a fixed display maximum (150, 100, 12000, 20, 70, 120) and clamped to [0.05, 1]. All five personas overlay translucently by default; selecting one via the "all clusters" / per-persona buttons isolates it with heavier stroke and fill. Beside it, a per-persona breakdown list (income, spending score, annual spend, recency) dims the non-selected entries.

---

**S12 — CRISP-DM methodology report**

A modal with a six-phase sidebar and detail pane: business understanding (objective, KPI targets — silhouette > 0.35 and sub-5ms inference — and the five persona targets), data understanding (an attribute table with ranges and domain meanings), data preparation (the four engineered features and the standardization step), modeling (the five families with their silhouette figures and the AutoResearch narrative), evaluation (three metric tiles and the elbow claim), and deployment. **All figures in this report are static authored content** and do not update when the model is retrained.

---

**S13 — Application shell and navigation**

A header with product title, a live status indicator and dataset label, a two-tab view switcher (explorer / admin console), and buttons opening the CRISP-DM report and retraining studio. The admin console has its own three-tab sub-navigation (benchmarks, AutoResearch, radar) plus a telemetry refresh button that re-fetches benchmark and elbow data. Modals close via their close button or an overlay click. Persona profiles are fetched once at application start and shared across views; they are re-fetched after a successful retrain.

---

## 3. User Workflow

**Exploration path**

1. Application loads; persona profiles, scatter points and an initial default-value prediction are fetched. `[F07, F06, F09]`
2. User reads the problem framing and persona cards, optionally clicking one to filter the manifold. `[S04, S01]`
3. User toggles between PCA and t-SNE and hovers points to inspect individual customers. `[S02]`
4. User adjusts the classifier sliders and submits, receiving a persona, confidence and marketing action, and seeing the customer placed on the PCA canvas. `[F09, S03, S02]`

**Analyst path**

5. User switches to the admin console and reviews KPI tiles, the algorithm leaderboard including the external baseline row, and the elbow/silhouette-vs-k panels. `[F03, F04, F08, S05, S06]`
6. User opens the AutoResearch tab, studies the tournament grid and hill-climbing trajectory, filters by phase or decision, and inspects individual steps. `[F10, S07, S08, S09]`
7. User exports the run record, or launches a fresh search. `[S10, F12]`
8. User opens the radar tab to compare personas on normalized axes. `[S11]`
9. At any point the user opens the CRISP-DM report `[S12]` or the retraining studio, retrains, and returns to refreshed personas. `[F11]`

---

## 4. Data, State, and Data-Science Behavior

**Data origin.** Entirely synthetic and generated at pipeline run time (F01). No data file ships with the project; nothing is downloaded. The "Kaggle" framing is presentation only.

**Preparation.** No missing-value handling, outlier removal, or categorical encoding is needed because the generator emits clean numeric data in bounded ranges. Preparation is exactly: derive four interaction features (F02), then z-score standardize all 12 columns.

**Partitioning.** There is no train/test split — appropriate for unsupervised work. The only partition is the 3,500-row evaluation subset used for tournament and elbow scoring, plus the 1,200-row t-SNE/scatter sample. Both are drawn with the pipeline seed.

**Model families.** Centroid partitioning, probabilistic mixtures, agglomerative hierarchy, density-based, and spectral (F03); production serving uses centroid partitioning only (F05).

**Evaluation.** Internal validity only — silhouette (primary optimization target throughout), Davies–Bouldin, Calinski–Harabasz, WCSS. The generator's ground-truth group labels are available but never used to compute external agreement measures, so no supervised-style accuracy is reported anywhere.

**Thresholds and constants that materially affect behavior:** k=5 production clusters; 3,500-row evaluation subset; 1,200-row projection sample; elbow sweep over k=2..9; AutoResearch acceptance gate of +0.0005 silhouette; DBSCAN ε=0.85 with min-samples 20; t-SNE perplexity 30; inference epsilon guard of 1e-5; default seed 42 everywhere.

**Persistence and state.** All results are written to a small set of on-disk artifacts: the fitted clustering model bundled with its scaler and column order, the fitted PCA, the persona profiles, the leaderboard-and-production-metrics record, the elbow sweep, the scatter sample, and the AutoResearch run record. The service loads these at startup into a single long-lived engine object and re-reads them on retrain. There is no database and no per-user state; every browser session sees the same artifacts. Client state (active view, admin sub-tab, filters, slider values, modal visibility) is ephemeral and resets on reload.

**Recompute behavior.** Nothing is recomputed per request except single-customer inference. Retraining (F11) and AutoResearch (F12) are the only write paths, both synchronous and both global in effect.

---

## 5. Outputs and UI Behavior

| Output | What it communicates |
|---|---|
| Persona cards `[S01]` | Segment size and share plus the three headline economics (income, spending score, annual spend) that justify the persona name |
| Manifold scatter `[S02]` | Whether the segments occupy separable regions of the reduced feature space, under both a linear (PCA, with explained variance stated) and a non-linear (t-SNE) view; hover exposes the raw record behind any point |
| Live prediction card `[F09, S03]` | Which segment a hypothetical customer belongs to, how confidently, and the concrete marketing action implied |
| KPI tiles `[S05]` | Production algorithm identity and its three validity metrics, each with a direction-of-goodness caption |
| Leaderboard `[F03, F04, S05]` | Relative separation quality, cluster count, degeneracy (noise ratio) and cost (fit time) across algorithm families, against a fixed external reference |
| Elbow + per-k list `[F08, S06]` | The inertia/k trade-off and how silhouette varies with k, framed as justification for the chosen k |
| Trajectory chart + stream `[F10, S07]` | The optimization path — which hypotheses were tried, which survived the acceptance gate, and how the score climbed |
| Step modal `[S08]` | The exact transformation, parameters and stated rationale behind any single experiment |
| Radar profiles `[S11]` | How personas differ shape-wise across six normalized dimensions simultaneously |
| CRISP-DM report `[S12]` | The methodology narrative from business objective through deployment |
| Exported run record `[S10]` | The full experiment log as portable JSON |

**Important states.** Empty/missing artifacts degrade to hardcoded literals in the KPI tiles and to empty tables and charts elsewhere rather than errors. Long-running actions (predict, retrain, AutoResearch) disable their trigger and swap in a pending label with a spinner. Retrain and AutoResearch failures render an inline message. Fetch failures on initial load are logged to the console only, leaving the affected panel empty.

---

## 6. Important Semantic Mechanisms

* **Persisted scaler reuse at inference.** The scaler fitted during training is serialized alongside the model and only ever *applied* at inference. Re-fitting a scaler on the single input record would silently destroy every prediction; this coupling must be preserved.
* **Column-order coupling.** The exact ordered feature list is persisted with the model, because the engineered matrix must be presented to the model and scaler in the same order it was trained on.
* **Shared evaluation subset.** Tournament entrants, the production model and the elbow sweep are all scored on the same fixed subset, which is what makes their silhouette values comparable. Scoring any of them on a different sample would break the leaderboard's meaning.
* **Inverse-distance confidence.** The confidence shown to users is a normalized inverse-centroid-distance share, not a model probability — replacing it with, say, mixture posteriors would change the displayed numbers substantially.
* **Positional persona binding.** Persona identity is bound to the raw cluster label index (F07). This is what makes persona names unstable across retrains and is the root of the mislabelling described below.
* **Greedy stateful hill-climbing.** AutoResearch accepted features mutate the working matrix for all subsequent steps, so the trajectory is order-dependent and later phases are evaluated against an evolved feature set, not the original one.
* **Artifact-backed serving with hot reload.** Every view is served from on-disk artifacts held in a single in-memory engine, and retraining rewrites those artifacts and reloads the engine in place — the mechanism by which a retrain immediately changes what all users see.

---

## 7. Source Caveats

### Apparent implementation defects and quirks

* **Personas are mismatched to their clusters.** Because persona metadata is bound positionally to K-Means label indices, the committed profiles are visibly wrong: the cluster labelled *Prudent Affluents* ("high income, careful saver") has mean income $32k, spending score 19.5 and discount sensitivity 0.81 — it is the bargain-hunter group; the cluster labelled *Young Trendsetters* has mean age 51.5 and income $105k — it is the prudent-affluent group; and the cluster labelled *Bargain Hunters* has mean age 25.7, income $38k, spending score 81.6 and the highest web visits — it is the young-trendsetter group. The mislabelling propagates to the cards, scatter tooltips, radar labels, live-prediction results and marketing recommendations. This should not be reproduced: each persona identity must correspond to the behavioral characteristics it describes.
* **Retrain cluster count is inert.** The retraining studio's k slider is sent and validated but never reaches the pipeline, which is hard-coded to five clusters; the studio still reports success. Similarly, `n_clusters` is bounded 2–10 while several UI elements assume exactly five personas and five fixed colours.
* **k=5 is asserted, not derived.** Both the elbow panel and the per-k list hard-highlight k=5 as elbow point and silhouette peak, but in the committed sweep the highest silhouette is at k=6 (0.3615) versus 0.3500 at k=5. The CRISP-DM report repeats the k=5 claim as an empirical finding.
* **Consensus ensembling is not implemented.** AutoResearch phase 4 fits two models but never builds or scores a co-association consensus; the step is unconditionally logged as accepted with delta 0 and a fixed narrative quoting a ">91.4% co-assignment" figure that is never computed.
* **Two classifier inputs are unreachable.** Discount sensitivity and household size are part of the request and materially affect two engineered features, but the form exposes no controls for them — they are always sent as 0.20 and 2.
* **The "champion" badge can land on the reference row.** The leaderboard sorts the static external baseline together with the trained models and badges rank 1; in the committed artifact the baseline outranks every trained model, so the top row simultaneously carries the baseline badge while the genuine production model sits at rank 2.
* **The "+21.0% gain" conflates two different baselines.** AutoResearch computes improvement relative to *its own* phase-1 starting score (0.34549 → 0.41803 ≈ +21%), but the explorer banner and the README present the same 21% as a gain over the external 0.3850 Kaggle baseline, which it is not.
* The tournament draws its evaluation subset with a freshly seeded generator, but the same call also perturbs nothing else — determinism holds; however the production model's reported metrics come from a subset of a *full-data* fit while leaderboard entrants were fit on the subset itself, so the two are not strictly like-for-like comparisons.

### Conflicting evidence

* **The AutoResearch optimum is not the served model.** The 0.4180 headline comes from a k=4 configuration over a different (five-feature-evolved, 5,000-row) matrix. The production model is k=5 over the four-fixed-engineered-feature matrix and scores 0.3497. Nothing feeds AutoResearch's discoveries back into the production pipeline — notably, AutoResearch *rejected* monetary velocity and *accepted* log-spend and power-transformed income, while the production feature set does the opposite. The UI presents 0.4180 as though it characterized the deployed system.
* **Documentation overstates the served metrics.** The README, abstract, paper and CRISP-DM report state 0.4180 at k=5 and a baseline of 0.3850; the served production silhouette is 0.3497, i.e. *below* the external reference row shown in the same table.
* **The project's implementation plan describes a different project**: the Mall Customer dataset, a champion silhouette of 0.554, MiniBatch K-Means, a `/api/clusters` endpoint, Recharts, and an entirely different set of persona names. None of this matches the implementation.
* **The committed audit report is not evidence.** It certifies "99.3% compliance, zero leakage, zero reward hacking" with generic findings referencing walk-forward cross-validation and PyTorch seeding — neither of which exists in this project. It appears to be templated prose and contradicts the defects above.
* **The scientific paper describes infrastructure that is absent**: Express.js, TypeScript, SSE, cross-validation splits, robust scaling. The implementation uses one Python service, plain JavaScript React, no streaming, no CV, and standard (not robust) scaling.
* The three bundled agent skill packs are duplicated verbatim in two directories; two of them are generic authored guides rather than descriptions of this project, and the project-specific one references directory paths that do not match the repository layout and claims a sub-5ms inference budget that is asserted rather than measured.

### Documented but unimplemented intent

* A printable/exportable CRISP-DM report (the report is view-only in the implementation).
* Distance-metric and PCA-pre-reduction search inside AutoResearch, described in the README; the implemented search covers feature mutations and hyperparameters only.
* Consensus ensembling as a real optimization phase (see defects).
* Agglomerative linkage variants `complete` and mixture covariance `diag`, named in the design document but absent from the implemented search space.

### Unresolved uncertainty

* The committed artifacts are internally consistent with the current source defaults (metrics, cluster sizes, feature names and step counts all line up), so no stale-artifact drift was detected — but this could not be confirmed by execution, which was out of scope.
* Whether any real external benchmark underlies the 0.3850 figure cannot be determined; nothing in the repository computes or cites it beyond the literal constant.

---

## 8. Acceptance Checklist

* [ ] `F01` — Deterministic five-group synthetic customer dataset with the eight specified attributes.
* [ ] `F02` — Four engineered interaction features plus persisted standardization.
* [ ] `F03` — Five clustering families scored on a shared evaluation subset and ranked.
* [ ] `F04` — External reference row present and distinctly badged, with its static origin understood.
* [ ] `F05` — Production K-Means model fit on the full dataset, scored, and persisted with its scaler and column order.
* [ ] `F06` — 2D PCA (with explained variance) and sampled t-SNE projections available.
* [ ] `F07` — Cluster profiles with counts, shares, attribute means and persona narratives, each persona identity matching the behavioral characteristics it describes.
* [ ] `F08` — Elbow and silhouette sweep across k.
* [ ] `F09` — Single-customer classification with persona, heuristic confidence, centroid distance and PCA placement.
* [ ] `F10` — Four-phase hill-climbing search with an acceptance gate and a fully inspectable step log.
* [ ] `F11` — Retraining regenerates all artifacts and hot-reloads the serving layer.
* [ ] `F12` — AutoResearch can be launched on demand and replaces the stored run record.
* [ ] `S01` — Persona cards cross-filter the scatter plot.
* [ ] `S02` — Interactive scatter with projection toggle, hover detail and live-prediction marker.
* [ ] `S03` — Slider-driven classifier form with auto-run on load and a styled result card.
* [ ] `S04` — Problem-framing banner with headline figures and engineered-signal chips.
* [ ] `S05` — KPI tiles and full leaderboard table with status badges.
* [ ] `S06` — Elbow curve and per-k silhouette list.
* [ ] `S07` — Backbone grid, AutoResearch KPI tiles, trajectory chart and experiment table.
* [ ] `S08` — Step inspection modal with transformation, reflection and parameters.
* [ ] `S09` — Phase and decision filters over the experiment stream.
* [ ] `S10` — Run-record JSON export.
* [ ] `S11` — Normalized six-axis persona radar with isolation control and breakdown list.
* [ ] `S12` — Six-phase CRISP-DM report modal.
* [ ] `S13` — Two-view shell, admin sub-tabs, telemetry refresh, modal handling and shared profile state.
