# Project Specification — Data Science Skills Mastery Lab

*Implementation-neutral replication specification derived from static inspection of the source project `05_data_science_skills_lab`.*

---

## 1. Project Overview

The project is an interactive, browser-based teaching workbench for data-science practice. It presents a browsable encyclopedia of named analytical "skills" (each with purpose, mathematical intuition, input/output expectations, and common beginner pitfalls) and pairs that catalog with five self-contained analytical benchmark workspaces that actually execute: a binary-classification benchmark (Titanic-style survival), a regression benchmark (house sale prices), an imbalanced-classification benchmark (card-fraud detection), a business-analytics benchmark (cohort retention, checkout funnel, A/B experiment, revenue time series), and an automated data-quality audit of a deliberately dirty table.

Its analytical identity is *pedagogical demonstration of correct methodology* rather than production modeling: the recurring theme across every workspace is leakage-safe preprocessing, choosing metrics appropriate to the problem (ROC-AUC/F1 rather than raw accuracy on skewed targets), decision-threshold calibration, and statistical-significance discipline.

All five datasets are **synthesized programmatically at runtime from parameterized generative processes**; no Kaggle files or any other external data are bundled or downloaded, despite consistent "Kaggle benchmark" labeling throughout the UI and documentation.

The user experience is a single-page application with a persistent top navigation bar switching between six views (catalog + five benchmarks), plus a modal methodology report. All benchmark computations run server-side on demand and are recomputed fresh for every request; nothing is persisted.

---

## 2. Required Features

### Core Requirements

---

**F01 — Generate the five synthetic benchmark datasets**

**Behavior:** The application generates every dataset it analyzes from seeded generative processes, so results are reproducible without external data.

**Important details:**

* *Survival/classification set* (891 rows, seed 42): ticket class drawn from a fixed 3-category distribution; sex drawn at fixed proportions; age drawn from class-conditional normals (higher class → older, clipped to plausible ranges); sibling/spouse and parent/child counts from fixed discrete distributions; derived family-size and travelling-alone indicators; fare drawn from class-conditional exponential distributions with class-specific floors and caps; port of embarkation categorical. The binary target is sampled from a logistic model of sex (large positive weight for female), ticket class (penalties for 2nd and 3rd), a small negative age effect, a small positive fare effect, and a penalty for large families. Missingness is then injected: roughly 20% of ages set to missing. The published coefficients of this generative logit matter, because the UI's interactive simulator re-uses them (see F04).
* *Regression set* (1,460 rows, seed 42): living area, overall quality (skewed 1–10 distribution), year built, basement area (derived as a random fraction of living area), garage capacity, full baths, and a 7-level neighborhood factor with fixed per-neighborhood price multipliers. The target is generated on a log scale as a linear combination of those drivers plus the log neighborhood multiplier plus Gaussian noise, then exponentiated and rounded to the nearest hundred. Basement area being a deterministic fraction of living area makes the two predictors strongly collinear by construction.
* *Imbalanced fraud set* (5,000 rows, seed 42): eight anonymized "component" features plus a transaction amount. Legitimate rows are standard-normal components with exponentially distributed small amounts; fraudulent rows are drawn from a shifted, wider component distribution with larger amounts. Positive-class share is **1.7%** by parameter, then rows are shuffled. The two classes are well separated by construction, which is why downstream recall is very high.
* *Business-analytics bundle* (not a table but a structured result): an acquisition-cohort retention matrix of 8 monthly cohorts with given cohort sizes and a fixed period-decay profile (100%, ~48%, ~36%, ~29%, ~24%, ~21%, ~18%, ~16%) perturbed by small uniform jitter and floored/capped, each cohort observable for fewer periods as it gets more recent (triangular matrix); a fixed 5-stage checkout funnel with hardcoded user counts and per-stage drop rates; a fixed stored A/B experiment record; and a 60-day daily revenue series built as a linear upward trend plus a 7-day sinusoidal seasonal component plus Gaussian noise, returned alongside its own trend component.
* *Dirty-data set* (500 base rows + 15 exact duplicate rows appended, seed 42): customer id, name with ~5% nulls, an age column deliberately containing impossible values (a negative sentinel, an implausibly large value) and nulls, an email column where ~8% of values are a literal malformed placeholder, a revenue column with a rare extreme outlier and nulls, and an account-status column with inconsistent casing variants (`ACTIVE`/`active`/`Active`) plus nulls.

---

**F02 — Browse, search, and filter the skills catalog**

**Behavior:** The default view is a searchable card grid of analytical skills. Each card shows the skill's category, its upstream origin attribution, name, purpose, a short mathematical/statistical intuition note, and a bulleted list of common pitfalls to avoid.

**Important details:**

* Catalog data is served from a structured catalog resource; the shipped catalog contains **13 skills** across four categories: data preparation & feature engineering (EDA, cleaning/leakage-free imputation, feature engineering, imbalanced data), modeling & evaluation (pipelines, hyperparameter tuning, model evaluation), data quality & validation (programmatic EDA, data-quality audit), and business & statistical analytics (cohort, funnel, A/B test, time-series decomposition).
* Free-text search matches across skill name, purpose, and origin attribution; a category pill row filters to one category or shows all. Filter and search compose.
* The count displayed in the header badge, navigation label, and "all skills" pill is a **hardcoded 46**, unrelated to the catalog's actual length (see caveats).

---

**F03 — Execute and report the survival classification benchmark**

**Behavior:** The classification workspace trains a model on the generated survival data and reports held-out performance.

**Important details:**

* Feature set: ticket class, sex, age, sibling/spouse count, parent/child count, fare, embarkation port, family size, travelling-alone flag.
* Single stratified hold-out split, 25% test, fixed seed.
* Preprocessing is encapsulated so that it is fitted only on the training partition: median imputation then standardization for the numeric block, one-hot encoding (tolerating unseen categories) for the categorical block.
* Estimator: gradient-boosted trees, 100 stages, depth 3, learning rate 0.08, fixed seed.
* Reported: accuracy, precision, recall, F1, ROC-AUC; a 2×2 confusion-matrix breakdown; a downsampled ROC curve point series; and the top 8 features by model importance with percentage shares. A small sample of raw rows is also returned with missing ages rendered as an explicit "missing" marker so the payload stays serializable.

---

**F04 — Interactive survival-probability simulator**

**Behavior:** The user adjusts a hypothetical passenger profile — sex selector, ticket-class selector, age slider (1–80), fare slider (5–300) — and immediately sees a survival probability and a survived/perished verdict at a 0.50 cutoff, updating live with no round trip.

**Important detail — mechanism:** The probability is **not produced by the trained model**. The client re-evaluates the same logistic expression used to *generate* the synthetic labels (same coefficients for sex, class, age, fare, large-family penalty), clamped to [0.01, 0.99]. The family-size term is present in the expression but the UI exposes no control that changes it, so it never contributes.

---

**F05 — Execute and report the house-price regression benchmark**

**Behavior:** The regression workspace fits a model that predicts sale price and reports error and fit metrics plus a calibration scatter.

**Important details:**

* Features: living area, overall quality, year built, basement area, garage capacity, full baths, neighborhood.
* Unstratified hold-out split, 25% test, fixed seed. Same leakage-safe preprocessing pattern as F03 (median impute + scale numerics, one-hot the neighborhood factor).
* Estimator: random forest regressor, 100 trees, max depth 10, fixed seed, **fitted against the log1p-transformed target**; predictions are back-transformed with expm1 before scoring, so RMSE, MAE and R² are reported in original currency units. The applied target transform is reported to the user as metadata.
* Also returns up to 60 (actual, predicted, residual) triples from the evaluation partition for plotting, and a small sample of raw feature rows.

---

**F06 — Demonstrate imbalanced-classification failure and remedy**

**Behavior:** The fraud workspace trains two classifiers on the same skewed data and contrasts them, showing that the naive model's headline accuracy hides poor minority-class recall.

**Important details:**

* Stratified hold-out split, 30% test, fixed seed. Features are standardized with a scaler fitted on the training partition only and applied to the test partition.
* Model A: logistic regression with default (unweighted) class treatment. Model B: identical logistic regression with inverse-frequency class weighting.
* Both models report accuracy, precision, recall, F1. The weighted model additionally yields a precision–recall curve (downsampled point series).
* A decision-threshold search sweeps 30 cutoffs across 0.10–0.90 on the weighted model's scores and reports the cutoff maximizing F1 together with that peak F1. **This search is conducted against the evaluation partition** (see caveats).
* The comparison is presented as a two-row table framing model A as operationally unsafe (missed fraud) and model B as production-safe.

---

**F07 — Interactive decision-threshold workbench**

**Behavior:** A slider over cutoff values 0.10–0.90 (step 0.02, default 0.42) shows how recall, precision, and their harmonic mean trade off as the cutoff moves, with an explanatory note that lower cutoffs favor recall and higher cutoffs favor precision.

**Important detail — mechanism:** The displayed recall and precision are **synthetic illustrative functions of the slider value** (recall decreases linearly with the cutoff, precision increases linearly, both clamped), not re-scored predictions from the trained model. F1 is computed correctly from those two synthetic inputs. The behavior to preserve is the live, direction-correct precision/recall trade-off demonstration; the source obtains it by formula rather than by re-thresholding real scores.

---

**F08 — Cohort retention matrix**

**Behavior:** A triangular heat-table of monthly acquisition cohorts showing cohort label, cohort size, and retention percentage per elapsed period, with cell background intensity scaling with retention so decay is visually obvious.

---

**F09 — Checkout funnel breakdown**

**Behavior:** A five-stage conversion funnel (homepage → product page → add to cart → initiate checkout → purchase) rendered as proportional bars with absolute user counts and conversion percentages, exposing the largest drop-off stages.

**Important detail:** Stage counts and rates are fixed constants in the data layer, not computed from an event stream.

---

**F10 — Live two-proportion A/B significance calculator**

**Behavior:** The user sets control and treatment visitor counts (5,000–50,000) and conversion counts (100–5,000) with sliders — live-computed observed conversion rates are shown next to each — then triggers recalculation and receives a fresh statistical verdict.

**Important details:**

* Computation: pooled conversion proportion, pooled standard error, two-proportion z statistic, two-tailed p-value via the normal approximation, absolute lift in percentage points, relative lift in percent, significance at α = 0.05, and a plain-language rollout recommendation ("deploy treatment" only when significant *and* the lift is positive; otherwise "keep control").
* Degenerate guards: non-positive sample sizes are rejected as an input error; a zero standard error yields z = 0 and p = 1; relative lift divides by a small floor to avoid division by zero.
* Before the user recalculates, the panel displays the stored experiment record from F01's analytics bundle; after recalculation it displays the live result. The displayed metrics are z-score, p-value, relative lift, and a rollout decision badge whose color and wording reflect significance.
* The catalog advertises a confidence interval and statistical power for this skill; neither is computed (see caveats).

---

**F11 — Automated data-quality audit scorecard**

**Behavior:** The quality workspace profiles the dirty table and produces a scorecard plus per-column integrity assessment.

**Important details:**

* Dataset-level outputs: row count, column count, overall completeness percentage (share of non-null cells across the whole table), exact-duplicate row count, and an overall quality score defined as completeness minus half a point per duplicate row, floored at zero.
* Per-column outputs: column name, inferred storage type, distinct-value count, null count and null percentage, a list of detected issue strings, and a health label derived purely from issue count — more than one issue is `CRITICAL`, exactly one is `WARNING`, none is `PASS`.
* Issue detectors actually implemented: any missing values (reported with percentage), negative values in the age column (reported with a count), and presence of the malformed-email placeholder. The injected inconsistent-casing status values and the implausible large age and the extreme revenue outlier are generated but **not** detected by any rule.

---

**F12 — Execute a catalog skill on demand**

**Behavior:** Every skill card offers a "live execute" action that runs an associated analysis and shows the raw result payload back to the user, so the abstract skill description is immediately tied to a concrete computed output.

**Important details:**

* Skill identifiers are routed to one of the five benchmark computations: EDA / feature-engineering / pipelines route to the classification benchmark; hyperparameter-tuning and dataframe-patterns route to the regression benchmark; imbalanced-data and model-evaluation route to the fraud benchmark; cohort, funnel and time-series route to the analytics bundle; programmatic EDA, quality audit and cleaning route to the quality audit; the A/B skill runs the significance calculation on a fixed illustrative sample.
* Any unrecognized identifier returns a **canned success acknowledgement** with no computation performed. This matters because the header advertises far more skills than the catalog contains; in a larger catalog most executions would be simulated.

---

### Secondary Requirements

**S01 — Six-view tabbed shell.** Persistent header with product title, upstream-source attribution line, six navigation tabs (catalog, classification, regression, fraud, analytics, quality), and a "skills installed" count badge. Exactly one view is mounted at a time; the catalog is the initial view.

**S02 — Jump from a skill to its benchmark.** Each catalog card offers a secondary action that switches the application to the benchmark workspace the skill is associated with, labeled with that benchmark's example dataset. The required semantics are that every catalog skill's jump target resolves to a real workspace (see caveats — two of the four targets do not).

**S03 — Skill execution output modal.** Executing a skill opens a modal showing the skill name, a success banner naming the verification dataset, and the full result payload pretty-printed as JSON for inspection; the modal is dismissible via close button, footer button, or overlay click. A brief celebratory confetti burst fires on successful execution.

**S04 — CRISP-DM methodology report modal.** A header action opens a scrollable modal walking through the six CRISP-DM phases as applied to this project: pedagogical framing, dataset inventory with sizes, the leakage-protection approach, models benchmarked and the log-target transform, evaluation and significance findings, and packaging/distribution. Its content is static prose, and some of its numbers are stated rather than read from live results.

**S05 — Eager multi-benchmark preload.** On startup the application concurrently requests the catalog and all five benchmark results, holds them in memory, and renders each view from that store; tab switching therefore does not re-fetch. Failures are logged and leave the affected view on its fallback values.

**S06 — Fallback display values.** Every metric tile, confusion-matrix cell, and scorecard figure renders a plausible hardcoded default when live data is absent. This means the UI never shows an empty or loading state — it silently shows stand-in numbers instead (see caveats).

**S07 — Supporting visualizations and tables.** Feature-importance bar list with percentage labels (classification); 2×2 confusion-matrix tiles plus a precision/recall summary line (classification); actual-vs-predicted scatter with a perfect-prediction identity reference line and per-point hover detail (regression); raw sample-row tables (regression); per-column integrity table with color-coded health badges (quality).

**S08 — Service health reporting.** The backend exposes a health/status summary naming the service, its advertised skill count, and the list of active benchmarks.

**S09 — Packaged agent skill documents.** The repository ships a set of standalone, self-contained skill guides (EDA, feature engineering, data cleaning, imbalanced data, model evaluation, data-quality audit, plus a lab-overview document) as a deliberate deliverable advertised in the project README. They are duplicated byte-identically into a second agent-tooling directory. Their instructional content — particularly the "fit every cleaning statistic on the training split only" rule — is the methodological thesis the application demonstrates.

---

## 3. User Workflow

1. The application opens on the skills catalog; all benchmark results are fetched in the background. `[F02, S05]`
2. The user searches or filters the catalog to find a technique, and reads its purpose, mathematical intuition, and pitfalls. `[F02]`
3. The user either runs the skill in place and inspects the returned payload `[F12, S03]`, or jumps to the associated benchmark workspace. `[S02]`
4. In the classification workspace the user reviews held-out metrics, the confusion matrix, and feature importances `[F03, S07]`, then experiments with hypothetical passenger profiles. `[F04]`
5. In the regression workspace the user reviews error metrics and the calibration scatter and inspects raw rows. `[F05, S07]`
6. In the fraud workspace the user compares the naive and class-weighted models, notes the recall gap behind similar accuracy, reads the F1-optimal cutoff `[F06]`, and moves the cutoff slider to feel the precision/recall trade-off. `[F07]`
7. In the analytics workspace the user reads the retention decay and funnel drop-offs `[F08, F09]`, then configures and runs a significance test on experiment counts. `[F10]`
8. In the quality workspace the user reads the overall scorecard and the per-column issue flags. `[F11]`
9. At any point the user opens the methodology report for the end-to-end CRISP-DM narrative. `[S04]`

---

## 4. Data, State, and Data-Science Behavior

**Data origin.** Entirely synthetic and generated in-process (F01). Named after well-known public benchmarks but sharing only schema shape and headline row counts with them. The generative processes encode the ground truth the models then recover, so reported metrics measure how well each estimator recovers a known signal, not real-world performance.

**Preprocessing and leakage discipline.** Every supervised workspace splits first and fits transformers afterwards. Imputation and scaling are fitted on the training partition and applied unchanged to the evaluation partition; categorical encoding tolerates unseen levels. This is the project's central methodological claim and must be preserved. Note that the claim holds for *transformers* only: no cross-validation is used anywhere (single hold-out splits throughout), and the fraud cutoff search is tuned on the evaluation partition.

**Feature engineering.** Family size and travelling-alone indicators are materialized in the classification data generator and then consumed as model features. The regression workspace's only transform is the log1p target transform with back-transformation before scoring. No polynomial expansion, no target encoding, and no resampling (no SMOTE) exists anywhere in the implementation despite all three being described in documentation.

**Models.** Gradient-boosted trees (classification), random forest (regression), logistic regression in unweighted and class-weighted variants (imbalanced comparison). Every model uses a fixed random seed. There is no model selection, no hyperparameter search, and no cross-validated tuning, despite "hyperparameter tuning" and "Bayesian parameter sweeps" being named in catalog and report copy.

**Evaluation.** Classification: accuracy/precision/recall/F1/ROC-AUC plus confusion matrix and ROC points. Regression: RMSE, MAE, R² in original units. Imbalanced: the same classification metrics for both variants plus a precision–recall curve and an F1-maximizing cutoff. Analytics: two-proportion z-test with normal-approximation two-tailed p-value at α = 0.05. Quality: completeness-driven composite score with a duplicate-row penalty.

**Statistical/inference behavior at serving time.** There is **no model persistence of any kind**. Every benchmark request regenerates its dataset, re-splits, re-fits, re-scores, and returns fresh results. There are no saved model artifacts, no result caches, no database, and no user-session state on the server. Because the generators are seeded, repeated requests are nevertheless expected to produce identical numbers — with one exception noted in the caveats.

**Client state.** Benchmark payloads live in application memory for the page's lifetime; the active tab, catalog search text and category filter, simulator inputs, threshold slider value, and A/B slider values are transient component state. Nothing survives a page reload.

**Thresholds and parameters worth preserving:** 25%/25%/30% test fractions for the three supervised workspaces; stratification on the two classification splits; seed 42 throughout; the gradient-boosting and forest hyperparameters listed in F03/F05; the 1.7% positive-class rate; the 0.10–0.90 / 30-point cutoff sweep; α = 0.05; the 0.5-point-per-duplicate quality penalty; the issue-count-to-health-label mapping.

---

## 5. Outputs and UI Behavior

* **Catalog view** — card grid; each card carries a category chip, origin attribution, name, purpose paragraph, a highlighted "statistical & mathematical intuition" block in monospace, and an amber-flagged pitfalls list. Search box and category pill row above. Two actions per card (execute, jump to benchmark).
* **Classification view** — banner with the benchmark framing and three headline tiles (ROC-AUC, F1, accuracy as a percentage); an interactive simulator panel whose result box turns green/red with a survived/perished verdict and a large probability readout; a confusion-matrix panel with four labeled tiles (true/false positives and negatives, color-coded) and a precision/recall summary line; a horizontal feature-importance bar list with percentage labels. The returned ROC curve series and raw sample rows are received but not rendered.
* **Regression view** — banner with three headline tiles (R², RMSE and MAE formatted as currency); a scatter plot of actual versus predicted sale price with a dashed identity reference line and per-point tooltips showing both values; a raw feature sample table with formatted area, quality, year, garage, neighborhood and price columns.
* **Fraud view** — banner with three headline tiles (balanced-model recall, optimal cutoff, peak F1); a two-row comparison table with per-model accuracy, precision, recall (annotated with how much fraud is missed or caught), F1, and an operational status badge contrasting "critical fraud leakage" against "production safe"; the threshold slider workbench with three live tiles (effective recall, effective precision, F1).
* **Analytics view** — retention matrix table with green intensity shading and a legend note; funnel bar list with per-stage counts and conversion percentages; A/B panel with two input cards (control/treatment) showing live observed rates, a recalculate action, and a four-tile result strip (z-score, p-value, relative lift, rollout decision) whose background switches between significant and non-significant styling.
* **Quality view** — banner with three headline tiles (quality score out of 100, completeness percentage, duplicate row count); a per-column table with type, distinct count, null count and percentage, a PASS/WARNING/CRITICAL badge, and a comma-joined list of detected anomalies or a "no integrity issues found" placeholder.
* **Modals** — skill execution output (raw JSON payload) and the CRISP-DM report; both dismissible by overlay click, close icon, or footer button.
* **States** — there is no explicit loading, empty, or error state anywhere; missing data falls through to the hardcoded fallback figures described in S06. There are no export, download, or reset controls, and no keyboard shortcuts.

---

## 6. Important Semantic Mechanisms

* **Generator-mirrored client-side inference.** The classification simulator's live probability comes from re-implementing the label-generating logistic function in the browser. This is what makes the control feel instantaneous and always self-consistent with the synthetic data, and it means the simulator's answers cannot disagree with the data even if the trained model would.
* **Recompute-on-every-request serving.** No artifacts, no caches. Determinism comes entirely from seeding the generators and estimators, so seed handling is load-bearing for reproducibility rather than incidental.
* **Fallback-value rendering.** Because every display value has an inline default, a fully failed backend still renders a complete, plausible-looking dashboard. Any replication must decide deliberately how to handle unavailable data; silently showing stand-in numbers is a source behavior, not an obviously desirable one.
* **Skill-identifier routing with a catch-all.** Skill execution is a lookup from identifier to one of five real computations, with an unconditional success response for anything unmatched.
* **Contrast-by-construction in the fraud workspace.** The whole pedagogical payload depends on the generated classes being separable enough that class weighting produces a dramatic recall jump while accuracy barely moves. The generator's class-separation parameters are therefore behaviorally meaningful, not arbitrary.

---

## 7. Source Caveats

### Conflicting evidence

* **Skill count is inconsistent across every source.** The catalog resource contains **13** skills. The backend health response, API description, UI header badge, navigation label, filter pill, CRISP-DM report copy, and packaged skill documents all state **46**. The README, abstract, paper, and article all state **54**. The committed catalog screenshot renders "(0 Skills)" and a further screenshot renders "(13 Skills)". The implemented number is 13.
* **Screenshots are stale and largely duplicated.** Five of the six committed screenshots are byte-identical files, all showing the catalog view rather than the benchmarks they are captioned as. The screenshots also depict an earlier UI revision: a different category taxonomy (ML / Data Quality / Analytics / Statistical Analysis / SQL & Data Modeling) than the four categories in the current code, and an empty "Input / Output" block on each card where the current code renders the mathematical-intuition block. Screenshots are therefore weak evidence for current behavior.
* **Class imbalance is misdescribed.** The generator produces a 1.7% positive rate and the backend reports 1.7%, but the UI banner and README describe the benchmark as a 0.17% base-rate problem.
* **Documented metrics vs. current code paths.** The lab skill document reports specific headline figures (ROC-AUC 0.7267, accuracy 83.5%, F1 0.812, R² 0.8105, RMSE $28,450, recall 54%→96%, cutoff 0.42, quality score 87.8). These same values also appear as the UI's hardcoded fallbacks. Some are internally inconsistent (an accuracy/F1 pair that high alongside an AUC of 0.727), so they should be treated as recorded history rather than expected outputs.
* **Audit report claims are not substantiated by the code.** It certifies cross-validated fold-scoped transformers and rolling walk-forward validation; the implementation uses single hold-out splits only and no temporal validation. It certifies deterministic seed pinning across PyTorch; no deep-learning framework is used. It certifies "zero reward-hacking traps" while the fraud cutoff is tuned on the evaluation partition. It reports inspecting 12 TypeScript/JavaScript files; the client is plain JavaScript/JSX.
* **Hardcoded numbers presented as measured.** The fraud comparison table's accuracy column (99.2% and 97.8%) is literal markup, not read from the model results shown in the same row. The funnel stage counts and the pre-recalculation A/B record are fixed constants. The CRISP-DM report's figures are static prose.
* **Startup instructions do not match the layout.** The README's quick start refers to `backend/` and `frontend/` directories; the source ships `server/` and `client/`. No Python dependency manifest or JavaScript lockfile is committed, so the exact environment cannot be established from the repository.

### Apparent implementation defects or quirks

* **Two catalog jump targets navigate nowhere.** The catalog associates skills with benchmark keys including `house-prices` and `data-quality`, but the application's view keys are `house` and `quality`. Selecting "open benchmark" on the regression, cleaning, programmatic-EDA, or quality-audit skills switches to a view key that matches no view, producing a blank content area with no way back except the header tabs. Invariant violated: every advertised navigation affordance should reach the workspace it names.
* **Decision cutoff tuned on the evaluation partition.** The F1-maximizing cutoff (F06) is selected using the same held-out data whose metrics are then reported, so both the cutoff and the peak F1 are optimistically biased. Invariant violated: a reported held-out score should not be conditioned on a choice made using that same held-out data.
* **Retention matrix jitter is unseeded.** The cohort generator draws its per-cell perturbation before the seed is set in that function, so retention percentages depend on whatever consumed the global random stream earlier in the process. This contradicts the project's blanket determinism claims. Elsewhere, the shared global random state is also reseeded inside several generators, making per-request results order-dependent in principle.
* **Retention table renders only six period columns.** Headers cover months 0–5 while the earliest cohorts carry up to eight period values, so the oldest cohorts emit cells with no corresponding header.
* **Quality scoring is narrower than advertised.** The README and catalog describe scoring across completeness, validity, uniqueness, and consistency; the implementation computes completeness with a flat duplicate penalty. The injected inconsistent-casing status values, the implausible high age, and the extreme revenue outlier are generated as teaching material but never flagged. The composite score also mixes a percentage with an unscaled row-count penalty, so its units are not meaningful.
* **A/B confidence interval and statistical power are advertised but absent.** Both the catalog's stated output schema and the CRISP-DM report promise a 95% confidence interval (and power); neither is computed or displayed.
* **Feature-importance bars use an arbitrary visual multiplier.** Bar widths are the importance percentage multiplied by 2.5, so any feature above 40% importance saturates the track and relative magnitudes above that point are not readable.
* **Computed outputs that are never displayed.** The ROC curve series, classification sample rows, and precision–recall curve series are computed and transmitted but rendered nowhere in the client; the README nonetheless advertises PR-AUC optimization and residual analysis views.
* **Collinear regression predictors by construction.** Basement area is generated as a random fraction of living area, which inflates apparent feature redundancy in the regression benchmark.
* **Fraud test asserts nothing about quality.** The API test suite asserts response shape and two threshold numbers (AUC ≥ 0.70, R² ≥ 0.80) that are satisfied by the synthetic generators by construction; passing tests do not evidence methodological correctness.

### Documented but unimplemented intent

* SMOTE/resampling, focal loss, Bayesian hyperparameter sweeps, cross-validation, polynomial feature expansion, out-of-fold target encoding, and time-series decomposition with ACF/forecasting are all described in catalog copy, README, or report prose; none are implemented.
* Server-sent-event streaming, live telemetry, and a TypeScript codebase are described in the paper and article; the implementation is request/response JSON over plain JavaScript.
* Model cards in the Mitchell et al. sense are claimed to be maintained; no model card exists in the repository.
* LTV/CAC ratios and churn-curve projections are advertised for the analytics view; neither is computed or displayed.

### Unresolved uncertainty

* Whether the 46- or 54-skill catalog ever existed cannot be determined from this repository; only the 13-entry catalog is present.
* Without a dependency manifest, the library versions that produced the documented metrics cannot be established, so exact numeric reproduction of the recorded figures is not verifiable statically.

---

## 8. Acceptance Checklist

* [ ] `F01` — All five datasets are generated from seeded, parameterized processes with the stated shapes, distributions, injected missingness, and injected data-quality defects.
* [ ] `F02` — Skills catalog is browsable with working free-text search across name/purpose/origin and category filtering, and the displayed skill count agrees with the catalog actually served.
* [ ] `F03` — Classification benchmark trains on a stratified hold-out split with train-only preprocessing and reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and feature importances.
* [ ] `F04` — Passenger simulator updates a survival probability and verdict live from the user's inputs.
* [ ] `F05` — Regression benchmark fits on a log-transformed target, back-transforms before scoring, and reports RMSE, MAE, R², plus actual-vs-predicted pairs.
* [ ] `F06` — Unweighted and class-weighted classifiers are compared on the same imbalanced split, with a precision–recall curve and an F1-optimal decision cutoff whose reported performance is not conditioned on the data used to report it.
* [ ] `F07` — Threshold control demonstrates the precision/recall trade-off live and in the correct direction.
* [ ] `F08` — Cohort retention matrix renders every cohort's full observable period series with intensity-coded cells, reproducibly across runs.
* [ ] `F09` — Five-stage checkout funnel shows per-stage counts and conversion rates.
* [ ] `F10` — A/B calculator computes pooled two-proportion z, two-tailed p-value, absolute and relative lift, an α = 0.05 verdict, and a rollout recommendation from user-supplied counts, with degenerate inputs handled.
* [ ] `F11` — Quality audit reports row/column counts, completeness, duplicate rows, a composite score, and per-column type, distinct count, nulls, health label, and detected issues.
* [ ] `F12` — Executing a catalog skill produces and displays a real computed result for that skill's associated analysis.
* [ ] `S01` — Six-view tabbed shell with attribution header and skill-count badge, defaulting to the catalog.
* [ ] `S02` — Every catalog skill's benchmark jump reaches the workspace it names.
* [ ] `S03` — Skill execution opens a dismissible modal showing the full result payload.
* [ ] `S04` — CRISP-DM methodology report is available as a dismissible modal.
* [ ] `S05` — Catalog and all benchmark results load on startup and back tab switching without refetching.
* [ ] `S06` — Unavailable data is handled with a deliberate, non-misleading display state.
* [ ] `S07` — Feature-importance list, confusion-matrix panel, actual-vs-predicted scatter with identity reference, raw sample tables, and the per-column integrity table are all present and legible across their full value ranges.
* [ ] `S08` — Service health/status summary is available.
* [ ] `S09` — Standalone methodology skill guides are shipped as part of the deliverable.
