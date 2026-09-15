# NYC Taxi Trip Duration — prediction platform

Predicts how long a New York City yellow-cab ride will take, using the real **Kaggle "NYC Taxi Trip Duration"**
trips from January–June 2016. Models are trained on `log(1 + seconds)` and scored with **RMSLE** (error in log space).
One local Streamlit app provides:

- **Ride estimator.** Choose any pickup and dropoff point on a real street map, or use the landmark shortcuts or
  type exact coordinates. Set a ride date and an exact pickup time and a passenger count to get a predicted
  duration, an empirical 90% prediction band, an itemised *illustrative* fare, and route telemetry.
- **Data-science console.** Benchmark matrix, feature importance, learning curves, dataset distributions,
  deep-dive diagnostics, an AutoResearch hill-climbing log, retraining with hot swap, and a CRISP-DM report built
  from the live artifacts.

## Quick start

```bash
conda activate cmpe255
pip install -r requirements.txt

# 1. Data: place the Kaggle archive here (not committed; no download or credentials needed)
#    data/raw/nyc-taxi-trip-duration.zip
python scripts/prepare_data.py          # extract, clean, split  (~5 s)

# 2. Train, benchmark and promote a model version (validation metrics only)
python scripts/train.py                 # full fit pool, ~45 s on 24 CPU cores

# 3. Optional: AutoResearch hill-climbing session (validation only)
python scripts/autoresearch.py          # 150k training rows, a few minutes

# 4. Once the configuration is settled, run the one-time final holdout evaluation
python scripts/finalize.py

# Tests (offline, synthetic fixtures, ~1 min)
python -m pytest -q

# App
streamlit run app/streamlit_app.py      # http://localhost:8501 (next free port if taken), no onboarding prompts
```

Useful variants: `python scripts/train.py --sample-size 50000 --n-estimators 120 --max-depth 6 --learning-rate 0.1 --subsample 0.8 --seed 7`,
and `python scripts/autoresearch.py --fit-sample-size 30000 --seed 1`.

## Data

| | |
|---|---|
| Source | Kaggle competition *NYC Taxi Trip Duration* (`nyc-taxi-trip-duration.zip`, NYC TLC yellow cab trips 2016) |
| Location | `data/raw/nyc-taxi-trip-duration.zip` (outer zip holds nested `train.zip`/`test.zip`; both CSVs are extracted to `data/raw/`) |
| Used | `train.csv` only. Kaggle `test.csv` has no `trip_duration` label, so it cannot evaluate anything and is not used. |
| Processed | `data/processed/trips_clean.parquet` + `data/processed/data_report.json` |
| Git | Everything under `data/`, all zips, CSVs, parquet files and `artifacts/` is ignored by the project-local `.gitignore`. |

**Cleaning** uses domain rules only, applied in order, with the removed count recorded for each:

| Rule | Removed |
|---|---:|
| Duration between 60 s and 3 h | 10,707 |
| Pickup inside NYC bounding box (lat 40.49–40.92, lon −74.27 to −73.68, includes Newark) | 128 |
| Dropoff inside NYC bounding box | 914 |
| Passenger count 1–6 | 19 |
| Great-circle speed ≤ 120 km/h (GPS/clock faults) | 21 |

That leaves 1,458,644 raw trips → **1,446,855 usable trips**. Trips whose pickup and dropoff are at the same
point are kept, because they are valid records.

**Splits** come from a fixed hash of the trip `id` (md5, per-mille buckets). They do not depend on row order,
sample size or seed:

| Role | Share | Rows | Used for |
|---|---:|---:|---|
| fit | 65% | 941,429 | model fitting; retraining samples are drawn only from here |
| calibration | 5% | 72,669 | prediction-band residual quantiles **only** |
| validation | 15% | 216,206 | early stopping, benchmark comparisons, retrain feedback, AutoResearch, diagnostics; always the full split, so scores stay comparable across retrains and sessions |
| test | 15% | 216,551 | **one-time final holdout evaluation** (`scripts/finalize.py`) |

Retraining and AutoResearch never read the test split. The tests check this with an instrumented data store.
`finalize.py` refuses to re-score a version unless you pass `--force`, and every test evaluation is appended to
`artifacts/holdout_log.json`.

The splits are random over Jan–Jun 2016 (the same setup as Kaggle), not time-ordered. The scores therefore do
not measure performance on later periods.

## Features (20, identical for training and serving)

- Pickup and dropoff latitude/longitude.
- Great-circle (haversine) distance.
- City-block distance: the east–west leg along the pickup latitude plus the north–south leg, in km.
- Compass bearing.
- Hour, day of week and month.
- Weekend flag.
- **Weekday-only** rush-hour flag (07–09, 16–19).
- Late-night flag (00–05).
- Passenger count.
- Six hub-proximity features (JFK, LGA, EWR, Times Square, Wall Street, Grand Central). Each is the *minimum* of
  the pickup→hub and dropoff→hub distances.

**Leakage guard:** `taxi.features.build_features` reads only the request-time columns (`config.REQUEST_COLUMNS`).
`dropoff_datetime` and `trip_duration` are used only for the target and for cleaning, and `id` only for
splitting. Tests confirm that changing those fields leaves features and predictions unchanged.

**Pure core:** the feature and inference functions never read the clock. A missing pickup time or passenger
count raises `ValueError`. Defaults belong to the caller: the estimator form starts at 2026-01-01 12:00 AM with 1
passenger, and batch CSV rows missing a field are rejected with their row numbers.

## Models and evaluation

- **Benchmark** (same training sample, same validation split): median baseline (a locally computed reference),
  Ridge, random forest, and **XGBoost** (the served model, early-stopped on validation).
- **Prediction band:** per distance tier, the 5th–95th percentiles of log residuals on the calibration split.
  Coverage is reported on validation (descriptive) and on test (at finalize). It is an empirical prediction
  band, not a confidence interval.
- **Deep-dive:** error by distance tier, residual skewness/kurtosis/bias/MAPE, and feature–target Pearson
  correlations. Everything is computed on the validation split at training time; no constants.
- **Artifacts** are immutable version directories `artifacts/versions/<version>/` containing `model.ubj`,
  `metadata.json`, `experiments.json`, `diagnostics.json` and, once finalized, `holdout.json`.
  - A new version is fully written and then validated (files present, model reloads, finite predictions on
    probe trips) before `artifacts/current.json` is atomically replaced.
  - If training fails, the pointer stays on the previous version.
  - The app caches by version id, so a promotion hot-swaps every view.

### Measured results

These come from the default run on 2026-09-17 (logs in `logs/`, which is gitignored). The model was trained on
all 941,429 fit-pool trips with seed 42. Your numbers will differ if you retrain with other settings.

Validation split (216,206 trips; used for selection):

| Model | RMSLE | R² (s) | MAE (s) | Fit time |
|---|---:|---:|---:|---:|
| Median baseline (reference) | 0.7279 | −0.072 | 442 | <0.1 s |
| Ridge | 0.5008 | 0.222 | 310 | 0.2 s |
| Random forest | 0.3336 | 0.776 | 186 | 31 s |
| **XGBoost (active)** | **0.3141** | **0.799** | **173** | 4.5 s |

The 90% prediction band (fitted on calibration) covered 90.0% of validation trips.

**Final holdout evaluation.** `finalize.py` ran once, on the test split (216,551 trips): RMSLE **0.3142**,
R² **0.803**, MAE **173 s**, band coverage **90.1%**. A second `finalize.py` was refused, as designed.

**AutoResearch** (150,000 training trips, seed 42, 317 s). All scores are validation RMSLE.

- **Tournament:** HistGradientBoosting 0.31782 (champion), XGBoost 0.31930, random forest 0.33943, MLP 0.34798,
  extra trees 0.35418, Ridge 0.50083.
- **Hill climb:** the XGBoost configuration went from 0.31930 to **0.31276** (−2.05%), with 7 of 15 gated
  candidates accepted.
- **Accepted changes:** the `airport_expressway` feature; the `deeper_trees`, `higher_subsampling`,
  `l2_regularization` and `more_rounds_lower_lr` configurations; and both HistGB blends.

The served model does **not** use the AutoResearch configuration, and the search did not read test.

## AutoResearch

A greedy search run on validation only (`taxi/autoresearch.py`). It has four phases:

1. A tournament of six model families: XGBoost, HistGradientBoosting, ExtraTrees, RandomForest, MLP, Ridge.
2. Eight candidate derived features.
3. Five named XGBoost configurations.
4. Two log-space blends of the evolved XGBoost with HistGradientBoosting.

**Like-for-like gate:** each candidate is compared with the current score of the *same* XGBoost configuration it
modifies. The comparison is not against the tournament champion. A change is kept only if it improves RMSLE by
more than ε = 1e-4; otherwise it is reverted. Every step records:

- candidate score, incumbent before and after, delta, and decision;
- the exact code expression that ran;
- a timestamp;
- a reflection sentence, **generated from a template and the measured numbers**.

Sessions go to `artifacts/autoresearch/<session>.json`. They never touch the test split and do not replace the
served model.

## Adaptations from the source specification

| Spec | Adaptation | Why |
|---|---|---|
| F01 synthetic simulator | Real Kaggle 2016 trips; "regenerate" means a seeded re-draw of the training sample | Real public data for a real problem (user-approved) |
| F04 single 80/20 split | Hash split: fit / calibration / validation / test (65/5/15/15) with explicit roles | No selection on reported data; band calibration kept separate |
| F06 fixed −10%/+12% "95% CI" | Empirical 90% prediction band from calibration residuals per tier | The source band was a presentation heuristic |
| F06 missing timestamp → now | The core requires explicit inputs; the UI supplies defaults | Deterministic, reproducible predictions |
| F03 rush-hour flag | Weekday-only | Source defect: weekend peaks were flagged |
| F08 hard-coded banner | Banner reads live validation metrics (plus test metrics if finalized) | Source numbers were stale |
| F10 injected "Kaggle 1st place" row | Locally computed median reference baseline; external scores are not shown | Not comparable (different test set and protocol) |
| F12 incumbent from a different model | Like-for-like incumbent | Source compared candidates against another model's score |
| F13 no subsample control | Subsample slider added; sample slider spans the actual fit pool | Source defect |
| F14 launch control crashed | Working launch with sample size and seed | Source defect |
| S04 constant diagnostics | Computed on validation per version | Source values were fabricated |
| S05 static report | Generated from live metadata; formulas rendered with LaTeX | Source figures were stale |
| F09 map | Hand-simplified borough outlines; all shapes share one lat/lon projection; landmark matching by id | Consistency (S02) |

The **fare** is an illustrative rule-based estimate (`config.FARE_CARD`). It is not a TLC meter quote and is
not fitted to fare data, and its time component uses the *predicted* duration.

## Estimator map and time controls

- **Street map** (Leaflet 1.9.4 from jsDelivr, standard OpenStreetMap tiles, © OpenStreetMap contributors).
  - An explicit **"Map click sets: Pickup / Dropoff"** selector decides which endpoint a map click moves. The
    current mode is shown on the map and by its coloured frame.
  - Clicking a landmark snaps the selected endpoint to that landmark.
  - The green **P** and red **D** markers can be dragged; dragging always moves that marker, whatever the mode.
  - Dropdowns, quick-pick chips and the latitude/longitude fields stay in sync with the markers. A point that is
    not exactly a landmark is labelled *Custom point*.
- **Python is authoritative.** The map sends only raw lat/lon events. `app/interaction.py` validates them (for
  example, points outside the NYC service box are rejected with a notice), and the prediction path is unchanged:
  coordinates → `build_features()` → model.
- **Needs internet.** The street map needs network access for the library and the tiles. If tiles fail, a notice
  appears and clicks still return exact coordinates. If the library cannot load, the offline stylised map
  appears; there you can click landmarks, and the coordinate fields and shortcuts keep working. Predictions never
  depend on the tiles.
  - The standard OSM tile servers are meant for light interactive use (OSMF tile usage policy). For heavier or
    public deployment, switch `TILE_URL` in `app/map_component.py` to a provider you have access to.
- **When:** separate **Ride date** and a larger **Pickup time** field (exact hour and minute).
  - Shortcuts: ±15 min and ±1 h buttons (these change only the time), quick times (03:00, 08:30, 12:00, 17:30,
    22:00), and a button that jumps to a date inside the training period (Tue 2016-03-15).
  - Selectable dates run from 2009 to 2035. Streamlit's default ±10-year window would have blocked every 2016
    date.
  - A readout shows the date, weekday, hour and month the model receives, plus the weekday-rush / late-night /
    weekend flags. These come from `build_features` itself.

## Project layout

```
taxi/        framework-independent core: config, landmarks, data, features, models, train, predict,
             diagnostics, artifacts, holdout, autoresearch, report
app/         Streamlit UI: streamlit_app.py (entry), estimator, interaction (framework-free rules), map_component,
             admin_*, dialogs, state
scripts/     prepare_data.py, train.py, autoresearch.py, finalize.py
tests/       offline tests on synthetic Kaggle-schema fixtures (no Kaggle data committed)
```

## Limitations

- Predictions reflect Jan–Jun 2016 traffic. Other dates are flagged as temporal extrapolation.
- Random (not temporal) splits. Validation scores are slightly optimistic after interactive tuning; the
  one-time holdout is the unbiased estimate.
- No weather, events or routing-engine features. Landmark coordinates are approximate; the dashed map line is a
  straight connection, not the driven route. There is no address search.
- The street map needs internet access (jsDelivr + tile.openstreetmap.org); offline, the stylised fallback allows
  landmark clicks only.
- Retraining and AutoResearch run synchronously in the app session (a status panel shows progress).
