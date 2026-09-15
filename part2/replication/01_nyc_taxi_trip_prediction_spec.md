# Project Specification — NYC Taxi Trip Duration & Fare Prediction Platform

Build-facing specification derived from static inspection of the source project. It describes required behavior for an independent reimplementation; it is not a transcription of the original code.

---

## 1. Project Overview

A full-stack, end-to-end supervised-regression platform framed around the NYC taxi trip-duration problem. It has two faces:

* **A public ride estimator.** A rider-style form where a person picks a pickup and a dropoff from a fixed set of NYC landmarks, sets a pickup timestamp and passenger count, and immediately sees a predicted trip duration, an itemized fare estimate, spatial route telemetry, and an animated map showing a taxi travelling the route.
* **A data-science admin console.** A transparency surface over the modelling lifecycle: a benchmark matrix of competing model families, feature importances, learning curves, dataset distribution histograms, per-distance-tier error diagnostics, an autonomous hill-climbing "AutoResearch" experiment log, and a control that retrains and hot-swaps the production model from the browser.

The analytical identity is **tabular gradient-boosted regression on engineered spatial-temporal features, with a log-transformed duration target**. Distances (great-circle, city-block), compass bearing, hour/day/month calendar signals, congestion flags, and proximity to six NYC transit hubs are the feature vocabulary. The scored metric throughout is RMSLE (RMSE in log-duration space), with R², RMSE and MAE in seconds reported alongside.

Critically: **the dataset is synthesized in-process, not loaded from the Kaggle competition**, despite extensive documentation to the contrary (see §7).

---

## 2. Required Features

### Core Requirements

---

**`F01` — Generate the trip dataset from a parameterized simulator**

**Behavior:** The training corpus is produced programmatically from a seeded stochastic model of NYC taxi activity. No external dataset file is read.

**Important details:**

* Pickup and dropoff points are drawn from a weighted set of ~7 named geographic zones (Midtown, Downtown, Uptown, Brooklyn, Queens, JFK, LaGuardia), each a lat/lon rectangle with a sampling weight (Midtown heaviest, airports lightest) and a nominal cruising speed. Coordinates are uniform within the chosen rectangle. Pickup and dropoff zones are sampled independently.
* Pickup timestamps are uniform over a fixed six-month window.
* **Ground-truth duration is generated from physics, not sampled:** a baseline city speed is multiplied by time-of-day and trip-type modifiers — slower during weekday rush hours (07–09, 16–19), faster late at night (00–05), faster for long/airport-corridor trips (great-circle displacement beyond a threshold). The city-block distance is inflated by a random road-detour factor, divided by the effective speed to yield a base duration, and an exponentially distributed boarding/stoplight delay is added. The result is rounded and clipped to a plausible range.
* Auxiliary columns are sampled independently: passenger count (1–6, heavily skewed to 1), vendor id, a store-and-forward flag, and a fare amount computed from a published-rate formula.
* Generation is deterministic under a fixed seed.

---

**`F02` — Clean the generated data with domain filters**

**Behavior:** Rows outside plausible operating bounds are dropped before modelling: durations outside a minimum/maximum window (roughly one minute to three hours), and pickup or dropoff coordinates outside a NYC bounding box. Row count after filtering is reported as the training corpus size.

---

**`F03` — Engineer the spatial-temporal feature set**

**Behavior:** A fixed, ordered feature matrix of 20 columns is derived from raw coordinates, timestamp, and passenger count. The same transformation is used at training time and at single-request inference time, so the serving path cannot drift from the training path.

**Important details:**

* Raw pickup and dropoff latitude/longitude are retained as features.
* **Great-circle distance** (haversine, in km) between pickup and dropoff.
* **City-block ("Manhattan") distance**: the sum of the two orthogonal great-circle legs (pure longitude displacement at the pickup latitude, plus pure latitude displacement), in km.
* **Compass bearing**, normalized to 0–360°.
* Calendar features: hour, day-of-week, month.
* Binary flags: weekend, rush hour (hours 07–09 and 16–19), late night (hours 00–05).
* Passenger count.
* **Six transit-hub proximity features** — JFK, LaGuardia, Newark, Times Square, Wall Street, Grand Central. Each is the *minimum* of (pickup→hub) and (dropoff→hub) great-circle distance, i.e. "how close did this trip come to that hub", not a pickup-only offset.
* When a request supplies no timestamp or no passenger count, neutral defaults are substituted so a prediction is still produced.

---

**`F04` — Train and benchmark a three-model comparison on a log-transformed target**

**Behavior:** One training run fits three model families on the same train split and evaluates all three on the same held-out split, producing a comparison record for each.

**Important details:**

* Target transform: natural log of (1 + duration in seconds). Predictions are inverted back to seconds for the seconds-denominated metrics.
* Split: a single random 80/20 partition with a fixed seed. There is no temporal or grouped split and no cross-validation.
* Models: (1) an L2-regularized linear baseline, (2) a bagged decision-forest ensemble (tens of trees, depth-capped), (3) the production gradient-boosted tree regressor, tuned with a moderate number of boosting rounds, mid-range depth, a low learning rate, and row/column subsampling, fitted with early stopping against the validation split.
* Metrics recorded per model: RMSLE (RMSE computed in log space), RMSE in seconds, MAE in seconds, R² in seconds space, and wall-clock training time. The boosted model is flagged as the active production model.
* Ordering expectation from the generated data: the linear baseline is far weaker (RMSLE ≈ 0.39, R² ≈ 0.70); the forest and boosted models are close together and strong (RMSLE ≈ 0.15, R² ≈ 0.97).

---

**`F05` — Persist model and telemetry artifacts consumed by the application**

**Behavior:** Training writes a serialized model plus three telemetry documents that the running application reads; the application never recomputes these at request time.

**Important details:**

* **Model metadata**: model name, version, training timestamp, corpus size, feature count, active model name, the metric block, the hyperparameters actually used, the ordered feature list, and a descending feature-importance ranking (importance and percentage per feature).
* **Experiment records**: the per-model benchmark rows from `F04`.
* **Diagnostics document**, containing: per-iteration training and validation loss curves (downsampled to roughly 30 points for charting); a random sample of up to 400 validation rows as (actual minutes, predicted minutes, error minutes); a 15-bin histogram of residual minutes over a fixed ±15-minute range; a 12-bin histogram of trip durations over a fixed 1–60 minute range; and a 24-bucket hourly pickup count.

---

**`F06` — Serve single-trip predictions with fare decomposition and route telemetry**

**Behavior:** Given pickup coordinates, dropoff coordinates, an optional timestamp and an optional passenger count, the service returns a predicted duration, a confidence band, an itemized fare, and derived route metrics.

**Important details:**

* The request is turned into a one-row feature matrix via the same engineering path as `F03`, scored by the persisted boosted model, inverted from log space, and floored at a minimum plausible duration.
* Duration is returned in seconds, in minutes, and as a human-readable "Nm Ss" string.
* **Confidence band:** reported as a "95% confidence interval" but computed as fixed multiplicative offsets around the point prediction (roughly −10% / +12%). It is a presentation heuristic, not an estimated interval.
* **Fare breakdown** is rule-based, not modelled: a flat base fare, a per-mile distance charge applied to the city-block distance, a per-minute time charge applied to the *predicted* duration, a rush-hour surcharge (weekday peak hours only), a late-night surcharge, a flat congestion fee always applied, and an airport access fee applied when either endpoint falls within a small radius of JFK or LaGuardia. The total is the sum. Note the fare therefore depends on the model output — a duration error propagates into the quoted price.
* **Route metrics:** great-circle and city-block distance in both km and miles, bearing in degrees, and an implied average speed (city-block distance divided by predicted duration) in km/h and mph.
* Timestamp handling: absent timestamps default to "now", so an otherwise identical request can score differently over time.

---

**`F07` — Serve bulk predictions**

**Behavior:** A list of trip requests returns the same per-trip prediction payload for each, plus a count. Semantics are identical to `F06` applied per row.

---

**`F08` — Present the interactive ride estimator**

**Behavior:** The primary screen lets a user compose a trip and see the prediction update automatically.

**Important details:**

* Pickup and dropoff are each chosen from a shared list of ~10 named NYC landmarks (Times Square, JFK, LaGuardia, Central Park, Wall Street, Brooklyn Bridge, Grand Central, Empire State, Williamsburg, Astoria), via a dropdown plus a row of quick-select chips for the first few. Each landmark carries a borough/zone label shown next to the field.
* Pickup datetime is a datetime picker defaulting to the current moment; passenger count is a 1–6 segmented selector.
* **There is no submit button.** Any change to pickup, dropoff, time, or passenger count re-requests a prediction immediately.
* Results panel shows the formatted duration and total fare as the hero pair, the confidence band beneath, the itemized fare rows (surcharge rows appear only when non-zero), and a four-tile telemetry grid: city-block distance, great-circle distance, compass heading, and predicted average speed.
* A banner above the form frames the problem as the Kaggle challenge and shows a three-cell comparison: a competition reference RMSLE, the production model's RMSLE/R², and a mean time error. **These banner numbers are statically written into the UI, not read from the live model metadata** (see §7).

---

**`F09` — Render an animated spatial route simulator**

**Behavior:** Beside the form, a stylized vector map of New York shows the selected trip.

**Important details:**

* Geographic coordinates are projected into map space by linear normalization against a fixed NYC lat/lon bounding box, with clamping to keep markers inside the frame.
* The map draws simplified borough landmasses (Manhattan, Brooklyn, Queens, Bronx, Staten Island) with labels, and highlighted circles marking the JFK and LaGuardia areas.
* Every preset landmark appears as a labelled clickable node; the current pickup renders green and enlarged, the current dropoff red and enlarged.
* The route is a curved spline between pickup and dropoff, drawn with a glow underlay and a green→amber→red gradient stroke.
* Pickup and dropoff markers pulse continuously; a taxi glyph animates along the route path on a repeating loop, rotating to follow the heading.
* **Clicking a landmark node sets it as the dropoff** (clicks on the current pickup are ignored), which in turn re-triggers the prediction.

---

**`F10` — Present the model benchmark matrix**

**Behavior:** The admin console shows one row per benchmarked model with its architecture name, the hyperparameters used, algorithm family, RMSLE, R², MAE in seconds (plus a rounded minutes equivalent), and training time.

**Important details:**

* RMSLE is color-coded by quality bands.
* Status column distinguishes three states: a competition-reference baseline row, the model currently active in production, and plain benchmarked rows.

---

**`F11` — Present feature importance, learning curves, and dataset distributions**

**Behavior:** Three visualizations render from the persisted diagnostics.

**Important details:**

* **Feature importance:** a horizontal ranked bar chart of the top 10 features, bar length proportional to importance relative to the strongest feature, with the percentage shown.
* **Learning curves:** a dual-line plot of training loss and validation loss against boosting iteration, auto-scaled to the observed loss range, with a legend distinguishing the two series. This is the evidence the user has for over/under-fitting.
* **Dataset explorer:** a trip-duration histogram and a 24-hour pickup-density histogram; rush-hour bars in the hourly chart are visually distinguished from off-peak bars. Both bars expose exact counts on hover.

---

**`F12` — Present an autonomous hill-climbing research log ("AutoResearch")**

**Behavior:** A dedicated admin view exposes the full trajectory of an automated model-search session as an inspectable experiment stream.

**Important details — the search procedure:**

1. **Backbone tournament.** Six model families are fitted on the base feature set and ranked by RMSLE: a boosted-tree regressor, a histogram-based gradient booster, an extremely-randomized-trees ensemble, a random forest, a small feed-forward neural network, and a regularized linear model. Each records RMSLE, RMSE, MAE, R², training time, and per-sample inference latency. The best becomes the incumbent score.
2. **Feature mutation.** A fixed catalogue of eight candidate derived features is tried one at a time, each with a stated hypothesis and a code fragment: log of city-block distance; tortuosity (city-block ÷ great-circle); an airport-expressway flag combining hub proximity with long displacement; sine and cosine cyclical encodings of pickup hour; a rush-hour × distance interaction; displacement efficiency (great-circle ÷ city-block); and a late-night-long-trip flag. A candidate is **accepted only if validation RMSLE improves by more than a small epsilon**, in which case it is kept in the working feature set and becomes the new incumbent; otherwise it is reverted.
3. **Hyperparameter annealing.** Five named boosted-tree configurations (deeper trees, higher subsampling, added L2 regularization, more rounds at a lower learning rate, shallow-and-fast) are evaluated against the current feature set under the same accept/reject gate.
4. **Ensemble blending.** The two strongest boosting families are fitted and blended at two fixed weightings; each blend is evaluated under the same gate.

**Important details — the recorded trajectory:** every step records an iteration number, phase name, category, hypothesis text, the feature/component touched, a code fragment representing the change, the active hyperparameters, RMSLE before and after, the delta, an ACCEPTED/REJECTED decision, a written "reflection" explaining the outcome, and a timestamp. The session summary records the initial and best RMSLE, percent improvement, total steps, accepted and rejected counts, the backbone leaderboard, and the final active feature list.

**Important details — the view:** a leaderboard grid of the six backbones with the champion badged; four KPI tiles (initial RMSLE, best evolved RMSLE with percent gain, total steps, accepted-vs-rejected gate counts); a trajectory scatter/line chart of RMSLE per step where accepted steps are larger and green, rejected smaller and red, each point clickable; phase filters (all / backbone / feature / hyperparameter / ensemble) and status filters (all / accepted / rejected); and the step table itself.

---

**`F13` — Retrain and hot-swap the production model from the browser**

**Behavior:** A modal lets the user choose dataset size and boosting hyperparameters, launch a full retraining run, and have the new model become the live serving model without a restart.

**Important details:**

* Controls: dataset sample size (slider, 10k–60k), boosting rounds (slider, 50–400), tree depth (slider, 4–10), learning rate (numeric input, 0.01–0.30). The service additionally validates a subsample fraction and enforces its own broader bounds.
* Launching runs the *entire* `F04`/`F05` pipeline — regenerate data, re-engineer features, re-benchmark all three models, re-export every artifact — then reloads the model and telemetry into the serving process.
* While running, the control shows a training state and is disabled; failures surface as an inline error message in the modal.
* On success the app celebrates (confetti), closes the modal, and refetches the overview so the header KPI badge and admin tiles reflect the new model.
* This is destructive to the previous artifacts: any hand-curated rows in the experiment records (notably the competition-reference row) are overwritten by the freshly computed three-model set.

---

**`F14` — Trigger an AutoResearch session on demand**

**Behavior:** A control in the AutoResearch view starts a fresh hill-climbing session on the server and, on completion, reloads and re-renders the trajectory. Documented to return the best RMSLE reached and the number of recorded steps.

**Apparent implementation defect:** the request path invokes a search routine that does not exist on the search component, so the call fails and the view surfaces its error state instead of a new trajectory. The intended semantics are that this control produces a new persisted trajectory of the kind described in `F12`. See §7.

---

### Secondary Requirements

---

**`S01` — Service health and readiness reporting**
A health signal reports service status, whether a model artifact is loaded, and the name of the active model. The application is expected to remain serviceable (with a warning) when no model artifact exists yet, though prediction requests then fail.

**`S02` — A single consistent landmark preset catalogue**
The named NYC landmarks offered to the user are backed by one authoritative catalogue — each entry carrying a stable identifier, display name, coordinates, and borough/zone label (plus an icon hint used for presentation). Wherever presets appear — the pickup and dropoff selectors, the quick-select chips, and the map's clickable nodes — they must resolve to the same entries with the same coordinates, so a landmark's identity, position, and zone label never disagree between views.

**`S03` — Model overview summary**
An overview payload combines the full model metadata, a count of recorded experiments, and the ranked feature-importance list. It backs the header badge (active model and R²) and the four admin KPI tiles: active algorithm with version, RMSLE, R², and training corpus size.

**`S04` — Data-scientist deep-dive diagnostics**
A third admin tab shows: a **model-accuracy-by-distance-tier** table (short / commute / cross-borough / airport-express tiers with MAE seconds, MAPE, R², and validation sample count); a **residual diagnostics** card (skewness, kurtosis, mean bias error, overall MAPE, each annotated with its ideal value); and a **feature-correlation table** (Pearson r against the log-duration target, with a signal-type label and significance).
**Mechanism:** every number in this view is a fixed constant written into the service response. Nothing is computed from the current model or validation set, and the values do not change after a retrain.

---

**`S05` — CRISP-DM research report modal**
A full-screen report reachable from the header, with a six-phase sidebar (Business Understanding; Data Understanding; Data Preparation & Feature Engineering; Modeling & Benchmarks; Evaluation & Diagnostics; Deployment & AutoResearch) and a content pane per phase covering objectives and success criteria, a column dictionary with valid domain ranges and missingness, the feature-engineering rationale including the log-target transform, a three-card benchmark summary, the residual diagnostic figures, and deployment/SLA claims.
**Mechanism:** the report is entirely static authored prose and static numbers; it does not read live metadata, so its figures diverge from the served model after any retrain.

---

**`S06` — Export the AutoResearch session**
The full trajectory document (summary, leaderboard, active features, all steps) can be downloaded as a JSON file with a timestamped filename.

**`S07` — Step inspection modal**
Clicking any trajectory point or table row opens a detail modal showing the score before, score after, and the accept/reject decision with its delta; the code fragment for the change in a monospace block; the written reflection; and the active hyperparameters as formatted JSON.

**`S08` — Filtering of the experiment stream**
Phase and decision filters narrow the step table; the table header reflects the filtered step count.

**`S09` — Tabbed navigation and shared shell**
A persistent header provides the two top-level tabs (estimator, admin), the live model status badge, and entry points to the report and retraining modals. The admin area has its own three-tab sub-navigation (benchmarks & training, AutoResearch, deep-dive). Modals dismiss on overlay click and on an explicit close control.

**`S10` — Client resilience states**
Prediction, telemetry, and trajectory fetches that fail are caught and logged without crashing the view; sections whose data is absent render nothing or fall back to placeholder defaults rather than erroring. The prediction panel is simply absent until the first successful prediction returns.

---

## 3. User Workflow

**Rider / estimator path**

1. The app opens on the estimator with a default trip already selected and a prediction already requested. `[F08]`
2. The user changes pickup, dropoff, time, or passenger count — via the dropdowns, chips, pickers, or by clicking a landmark on the map. `[F08, F09]`
3. A new prediction is fetched on every change, and the duration, confidence band, fare breakdown, and telemetry tiles update. `[F06, F08]`
4. The map re-renders the route and replays the taxi animation. `[F09]`

**Data-scientist path**

1. The user switches to the admin console and reads the KPI tiles. `[S03]`
2. On the benchmarks tab they compare model families, then inspect feature importance, learning curves, and dataset distributions. `[F10, F11]`
3. On the AutoResearch tab they read the backbone leaderboard and search KPIs, scan the trajectory chart, filter the step stream, and open individual steps to read the hypothesis, code change and reflection. `[F12, S07, S08]` They may export the session `[S06]` or launch a fresh search `[F14]`.
4. On the deep-dive tab they review tiered errors, residual moments, and target correlations. `[S04]`
5. They open the CRISP-DM report for methodology narrative. `[S05]`
6. They open the retraining studio, adjust sample size and boosting hyperparameters, launch a run, and on completion the served model, artifacts, and all admin telemetry reflect the new model. `[F13]`

---

## 4. Data, State, and Data-Science Behavior

**Data origin.** Fully synthetic and regenerated on demand (`F01`). Nothing is read from disk as input; the only persisted inputs to the running application are the model and telemetry artifacts produced by training (`F05`). Because durations are generated from an explicit speed/distance/noise model, the learnable signal is largely a deterministic function of city-block distance, hour and day — which is why tree models reach R² ≈ 0.97 and why distance and the congestion flags dominate feature importance. A reimplementation must preserve this: the reported accuracy is a property of the simulator, not of real NYC traffic.

**Preprocessing and features.** Domain-bound filtering (`F02`) then the fixed 20-column matrix (`F03`). No scaling, imputation, or encoding is applied — all features are numeric by construction and tree models are used. The identical transformation runs at inference, so there is no train/serve skew in the feature path.

**Target and metrics.** Models are fitted on log1p(duration) and scored as RMSE in that log space (called RMSLE), with predictions exponentiated back for RMSE/MAE/R² in seconds. All model comparison, all accept/reject decisions in the search engine, and all headline KPIs use RMSLE.

**Validation.** A single random 80/20 split with a fixed seed, reused across all models in a run and across all search steps so scores are comparable. There is no cross-validation, no temporal split, and no separate test set held out from the search — the same validation split drives both model selection and the reported metrics.

**Model families.** Regularized linear, bagged forest, and gradient-boosted trees in the main pipeline; the search engine adds histogram-based boosting, extremely-randomized trees, and a small MLP. The boosted-tree model is the only one persisted and served.

**Hyperparameters that matter.** Boosting rounds, tree depth, learning rate, and row/column subsample fractions are the tuned surface, all user-adjustable through retraining (`F13`). Early stopping against the validation split governs the effective number of rounds and therefore the length of the learning curves.

**Thresholds worth preserving.**

* Rush-hour hours 07–09 and 16–19; late-night hours 00–05 — used by the generator, the features, and the fare rules.
* The search engine's acceptance gate: a candidate must improve RMSLE by more than ~1e-4 to be kept.
* The airport fee radius around JFK/LaGuardia; the confidence-band multipliers; the minimum served duration floor.
* Data-cleaning bounds on duration and the coordinate bounding box.

**State and persistence.** The serving process holds the model and all three telemetry documents in memory, loaded once at startup. Requests never recompute telemetry. Retraining is the only mutation path: it rewrites all artifacts on disk and reloads them in place. The AutoResearch trajectory is a separate persisted document, read from disk per request rather than cached, and rewritten by a search run. There is no database, no user accounts, and no per-user state.

---

## 5. Outputs and UI Behavior

**Prediction outputs** (`F06`, `F08`) communicate three distinct things: how long the ride will take (point estimate plus a band), what it will cost and why (line-by-line, so a user can see the surge and airport fees appear when applicable), and the geometry of the trip (two different distance notions side by side, plus heading and implied speed — the two distances differ materially and the contrast is the point).

**Benchmark matrix** (`F10`) communicates that model choice matters: a linear baseline is visibly inadequate on this target while tree ensembles are close to each other, with the active model identified.

**Feature importance** (`F11`) communicates which engineered signals carry the prediction — expected to be dominated by city-block distance with the temporal congestion flags secondary.

**Learning curves** (`F11`) communicate convergence and the train/validation gap across boosting iterations.

**Dataset histograms** (`F11`) communicate the right-skewed duration distribution (the motivation for the log target) and the hourly demand profile with peak hours highlighted.

**AutoResearch views** (`F12`) communicate the search narrative: which model family won the tournament, how the score moved step by step, which mutations were kept versus reverted and on what numeric grounds, and what the agent's stated reasoning was for each.

**Deep-dive views** (`S04`) communicate error behavior conditional on trip length and the shape of the residual distribution — but as fixed illustrative values, not measurements.

**Report modal** (`S05`) communicates methodology as a structured six-phase narrative.

**Exports** (`S06`) hand the user the full research record as JSON.

**Visual language.** Dark "NYC taxi" theme, yellow/cyan/emerald accents, monospace for all numeric readouts, glass-panel cards, custom hand-rolled SVG for every chart and the map (no charting library). Charts auto-scale to their data; bars expose exact values on hover.

---

## 6. Important Semantic Mechanisms

* **Artifact-backed serving.** Every admin number except the deep-dive tab originates from documents written at training time and loaded into memory, not from computation at request time. Replacing this with per-request computation would change both the meaning of the displayed values and the behavior of retraining.
* **Shared feature transformation across train and serve.** The single-row inference path reuses the training-time feature pipeline; this is what makes a prediction meaningful and must not be reimplemented as a parallel code path.
* **Retraining as a full pipeline rerun with hot swap.** Retraining is not a warm update — it regenerates data, re-benchmarks everything, rewrites all artifacts, and reloads the serving model. The user-visible consequence is that the whole admin console changes at once.
* **Debounce-free reactive prediction.** The estimator issues a prediction request on every input change with no submit step and no debouncing; this drives the perceived liveness of the tool.
* **Hill-climbing accept/reject gate.** The search is greedy with an explicit improvement epsilon and full reversion on rejection; the incumbent score is what every candidate is measured against. Removing the gate or the reversion would turn the trajectory into a plain sweep.
* **Fare derived from the model output.** The time component of the quoted fare is computed from the predicted duration, coupling price accuracy to model accuracy.
* **Missing-timestamp default to current time.** Predictions are not reproducible across calls when a timestamp is omitted.
* **Client-side geographic projection.** Map placement is a fixed linear lat/lon→canvas mapping with clamping; landmark positions, the JFK/LGA highlight circles and the drawn landmasses only agree visually because they were tuned to that specific bounding box.

---

## 7. Source Caveats

### Conflicting evidence

* **Dataset provenance.** The README, design document, walkthrough, paper, abstract, article, and the report modal all state the system is trained on the Kaggle NYC Taxi Trip Duration dataset (the paper and abstract cite N = 1,458,644 real trips). The implementation trains exclusively on the in-process simulator (`F01`). All accuracy claims should be read as accuracy against synthetic data.
* **Committed artifacts do not match current training defaults.** The persisted metadata records ~10k training rows with 50 boosting rounds, depth 5 and learning rate 0.1, whereas the pipeline's defaults are ~40k rows with 250 rounds, depth 7 and learning rate 0.08. The artifacts appear to originate from a small retraining run (matching the test suite's retrain payload), not from a default training run.
* **Headline metrics disagree across sources.** Documentation reports RMSLE 0.1479 / R² 0.9679; the artifacts and the live UI show RMSLE 0.1531 / R² 0.9697; the estimator banner hardcodes 0.1531 / 96.97% / ±2.14 min / 128 s MAE. The abstract and paper instead claim the system *achieves* RMSLE 0.3680 — which is the number the experiment records attribute to the external competition reference, not to this model.
* **A competition-reference row is injected into the experiment records by hand.** The persisted experiment list contains a "Kaggle 1st Place Grandmaster Baseline" entry (ensemble weights, OSRM features, 10-fold CV) that the training pipeline never produces, and the benchmark table has a dedicated badge for it. Any retraining silently deletes it. Separately, this row's RMSLE (0.368, from a real-data competition) is presented as directly comparable to this project's synthetic-data RMSLE (0.153), which it is not.
* **Deep-dive diagnostics are fabricated constants.** Tiered errors, residual moments, MAPE, normality p-value, and feature correlations (`S04`) are fixed values in the service response, presented in the UI as measured model diagnostics. The report modal repeats the same constants.
* **Audit report claims contradicted by the source.** The committed audit asserts "walk-forward rolling window splits", "temporal ordering strictly preserved", and "explicit cross-validation partition", and awards a 99.3% / Grade A+ certification. The implementation uses a single random split with no temporal awareness and no cross-validation. The audit appears to be generic boilerplate rather than a finding about this project.
* **Walkthrough test count.** The walkthrough reports an 8-test suite; the committed suite contains 10 tests (adding deep-dive and AutoResearch-history checks).
* **Data era.** The report modal documents the timestamp domain as 2016 (matching the real competition); the generator produces timestamps in a 2026 six-month window.
* **Background retraining.** Documentation describes a background retraining worker; retraining is performed synchronously inside the request.

### Apparent implementation defects or quirks

* **The AutoResearch launch control cannot succeed.** The request handler calls a search method that is not defined on the search class (the class exposes a full multi-phase run routine under a different name), so the endpoint raises and the UI shows its error state. The intended behavior is that the control runs a session and refreshes the trajectory (`F14`). The handler also fixes the sample size and step count rather than exposing them.
* **The retraining modal returns before declaring its state.** The component exits early when closed, ahead of its state declarations, so the number of hooks differs between the closed and open renders. Opening the modal is expected to fail at render under React's hook-ordering rules. The intended behavior is simply that the modal renders its controls when opened.
* **The search engine measures candidates against a different model's incumbent score.** The incumbent RMSLE is taken from the winning backbone (in the committed run, the histogram-based booster at 0.14959), while every feature-mutation and (initially) hyperparameter candidate is evaluated with a fixed boosted-tree model that alone scores worse (0.15157). The comparison is therefore not like-for-like, and in the committed trajectory every one of the 13 feature and hyperparameter candidates is rejected, with only a 50/50 ensemble blend squeaking through. The invariant violated is that a greedy search should compare each candidate against the score of the same configuration it is mutating.
* **`rmsle_before` is recorded as the incumbent rather than the prior step's score**, so the trajectory chart's "before" column is constant across all rejected steps.
* **The rush-hour training feature omits the weekday condition** that the data generator and the fare rules both apply, so the model's congestion flag fires on weekend peak hours where no slowdown was generated. The design document describes this feature as weekday-only.
* **The retraining modal exposes no subsample control** although it maintains that value and sends it; and its slider bounds (10k–60k samples) are narrower than what the service accepts (5k–100k), while its default sample size differs from the service default.
* **The report modal renders LaTeX markup as literal text** — formulas appear as raw `$...$` source because no math renderer is present. It also states the city-block distance as a raw degree difference, which is not what is computed.
* **Two landmark catalogues exist** — one served by the API, one embedded in the client — and the client uses its own. They currently agree in content but can drift. Newark appears as a proximity feature but is not among the landmark presets.
* **Landmark identity on the map is matched by coordinate equality** rather than identity, so two presets sharing coordinates would both highlight.
* **The browser audit script hardcodes a Windows browser path**, so it will not run on other platforms without modification.

### Documented but unimplemented intent

* Cross-validation and leakage-prevention machinery (fold-fitted transformers, walk-forward splits, Mitchell-style model cards) described in the paper and audit are not present.
* Server-sent events / streaming and a TypeScript frontend are described in the paper; the client is plain JavaScript and uses ordinary request/response.
* CatBoost and LightGBM are named in documentation and in the injected reference row; neither is used (a histogram-based gradient booster stands in for LightGBM in the search engine).
* A batch-prediction UI is not present although the batch endpoint exists (`F07`) — it is exercised only by the test suite.

### Unresolved uncertainty

* Whether the committed AutoResearch trajectory and the committed model artifacts came from the same run or configuration cannot be determined statically; their sample sizes and hyperparameters differ.
* No dependency manifest exists for the Python side (no requirements file or lockfile), so exact library versions are unknown; the client does have a lockfile.
* Whether the reported "0 console errors" browser audit was performed against a build in which the retraining modal defect was present cannot be established without execution.

---

## 8. Acceptance Checklist

* [ ] `F01` — Trip dataset is generated from a seeded, zone-weighted simulator with physics-derived durations.
* [ ] `F02` — Domain filters remove implausible durations and out-of-bounds coordinates.
* [ ] `F03` — The 20-column spatial-temporal feature set is produced identically for training and inference.
* [ ] `F04` — Three model families are trained and evaluated on a common split against a log-transformed target, with RMSLE/RMSE/MAE/R²/time recorded per model.
* [ ] `F05` — Model, metadata, experiment records, and diagnostics (curves, residual sample, error histogram, duration and hourly distributions) are persisted for the application to consume.
* [ ] `F06` — Single-trip requests return duration, confidence band, itemized rule-based fare, and route telemetry.
* [ ] `F07` — Bulk trip requests return per-trip predictions with a count.
* [ ] `F08` — The estimator lets the user compose a trip from presets, time, and passenger count, and updates the prediction without an explicit submit.
* [ ] `F09` — The map projects coordinates onto a stylized NYC canvas, draws the curved route with animated taxi and pulsing endpoints, and lets a landmark click set the dropoff.
* [ ] `F10` — The benchmark matrix compares models and identifies the active one.
* [ ] `F11` — Feature importance, learning curves, and duration/hourly distributions are visualized from the persisted diagnostics.
* [ ] `F12` — The AutoResearch view presents a backbone leaderboard, search KPIs, an RMSLE trajectory, and a filterable, inspectable step stream reflecting a greedy accept/reject search over features, hyperparameters, and blends.
* [ ] `F13` — Retraining from the browser reruns the full pipeline with user-chosen parameters and hot-swaps the served model and all telemetry.
* [ ] `F14` — Launching an AutoResearch session from the UI produces and displays a new trajectory.
* [ ] `S01` — Health reporting exposes service status and loaded-model identity.
* [ ] `S02` — Landmark presets resolve to one consistent catalogue across every place they are offered.
* [ ] `S03` — Model overview backs the header badge and admin KPI tiles.
* [ ] `S04` — Deep-dive view presents tiered error, residual-moment, and target-correlation diagnostics that correspond to the current model and validation data.
* [ ] `S05` — A six-phase CRISP-DM report is reachable and navigable, with figures consistent with the model actually served.
* [ ] `S06` — The AutoResearch session can be exported as JSON.
* [ ] `S07` — Any search step can be opened to inspect scores, decision, code change, reflection, and hyperparameters.
* [ ] `S08` — Phase and decision filters narrow the step stream and the displayed count.
* [ ] `S09` — Top-level and admin sub-navigation, the status badge, and dismissible modals behave as described.
* [ ] `S10` — Failed fetches and absent data degrade gracefully rather than crashing a view.
