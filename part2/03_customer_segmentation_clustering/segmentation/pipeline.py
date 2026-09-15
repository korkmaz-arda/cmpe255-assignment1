"""End-to-end pipeline orchestration and retraining (F11).

Running the pipeline regenerates every artifact the dashboard serves:
the model bundle, the fitted PCA, the persona profiles, the leaderboard and
production metrics, the elbow sweep and the scatter sample. The serving layer
reloads them, so a retrain immediately changes what every view shows.

Unlike the source, the requested cluster count actually reaches the pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from . import config, data, elbow, features, model, personas, projections, tournament


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run(k: int = config.DEFAULT_K, seed: int = config.DEFAULT_SEED,
        n_rows: int | None = None, frame: pd.DataFrame | None = None) -> dict:
    """Execute the full pipeline and write all artifacts.

    Parameters
    ----------
    k
        Number of clusters actually fitted and served.
    seed
        Seed for every stochastic step — fitting, sampling and projection.
    n_rows
        Deterministic training-sample size. ``None`` uses the whole cleaned
        dataset. This replaces the source's synthetic dataset-size control: the
        real dataset is a fixed 2,197 customers, so the meaningful analogue is
        how much of it the model is trained on.
    """
    if not (config.MIN_K <= k <= config.MAX_K):
        raise ValueError(f"Cluster count must be between {config.MIN_K} and {config.MAX_K}, got {k}.")

    started = time.perf_counter()
    full = data.load_clean() if frame is None else frame

    if n_rows is not None and n_rows < config.MIN_TRAIN_SAMPLE:
        raise ValueError(f"Training sample must be at least {config.MIN_TRAIN_SAMPLE} customers.")
    training = data.sample(full, n_rows, seed=seed)

    # --- preparation -------------------------------------------------------
    enriched, scaler, X = features.build_training_matrix(training)

    # --- tournament (F03) --------------------------------------------------
    # Every entrant, the production model and the elbow sweep are fit and scored
    # on this same matrix, which is what makes their silhouettes comparable.
    leaderboard = tournament.run(X, k=k, seed=seed)
    dbscan_info = tournament.dbscan_diagnostics(X)

    # --- production model (F05) -------------------------------------------
    production = model.fit(X, k=k, seed=seed)
    production_metrics = model.evaluate(production, X)
    labels = production.labels_

    bundle = model.ModelBundle(
        model=production, scaler=scaler, columns=list(config.FEATURE_COLUMNS), k=k, seed=seed
    )
    model.save(bundle)

    # --- projections (F06) -------------------------------------------------
    pca, pca_coords = projections.fit_pca(X, seed=seed)
    projections.save_pca(pca)
    variance = projections.explained_variance(pca)
    scatter = projections.build_scatter(enriched, X, labels, pca_coords, seed=seed)

    # --- personas (F07) ----------------------------------------------------
    persona_records = personas.profile(enriched, labels)
    notes = personas.sanity_notes(persona_records)

    # --- elbow sweep (F08) -------------------------------------------------
    sweep = elbow.sweep(X, seed=seed, served_k=k)

    elapsed = time.perf_counter() - started

    # --- persist -----------------------------------------------------------
    production_row = {
        "algorithm": "K-Means",
        "family": "Centroid partitioning",
        "formulation": f"k={k}, k-means++, {model.PRODUCTION_N_INIT} restarts",
        **{key: production_metrics[key] for key in
           ("silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters", "inertia", "n_customers")},
    }

    benchmarks = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k,
        "seed": seed,
        "n_customers": int(len(training)),
        "leaderboard": leaderboard,
        "production": production_row,
        "dbscan": dbscan_info,
        "evaluation_note": (
            f"All entrants, the production model and the elbow sweep are fit and scored on the same "
            f"{len(training)}-customer standardized matrix, so their metrics are directly comparable."
        ),
    }
    _write_json(config.BENCHMARKS_JSON, benchmarks)
    _write_json(config.PERSONAS_JSON, persona_records)
    _write_json(config.ELBOW_JSON, sweep)
    _write_json(config.SCATTER_JSON, scatter)

    quality = data.quality_report(training)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k,
        "seed": seed,
        "n_customers": int(len(training)),
        "n_available": int(len(full)),
        "trained_on_full_dataset": bool(n_rows is None or n_rows >= len(full)),
        "elapsed_seconds": float(elapsed),
        "dataset": {
            "name": config.DATASET_NAME,
            "url": config.DATASET_PAGE,
            "synthetic": False,
        },
        "feature_columns": list(config.FEATURE_COLUMNS),
        "engineered": {
            name: {"formula": features.FEATURE_FORMULAS[name], "meaning": features.FEATURE_MEANINGS[name]}
            for name in config.ENGINEERED_FEATURES
        },
        "pca_explained_variance": variance,
        "data_quality": quality,
        "persona_notes": notes,
    }
    _write_json(config.RUN_META_JSON, meta)

    return {"benchmarks": benchmarks, "meta": meta, "personas": persona_records, "elbow": sweep}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full segmentation pipeline.")
    parser.add_argument("--k", type=int, default=config.DEFAULT_K, help="number of clusters to serve")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=None, help="training sample size (default: all)")
    args = parser.parse_args()

    result = run(k=args.k, seed=args.seed, n_rows=args.rows)
    meta, benchmarks = result["meta"], result["benchmarks"]
    production = benchmarks["production"]
    sweep = result["elbow"]

    print(f"Customers:   {meta['n_customers']} of {meta['n_available']}   k={meta['k']}   seed={meta['seed']}")
    print(f"Production:  silhouette {production['silhouette']:.4f} | "
          f"Davies-Bouldin {production['davies_bouldin']:.4f} | "
          f"Calinski-Harabasz {production['calinski_harabasz']:.1f}")
    print(f"PCA:         {meta['pca_explained_variance']['total']:.1%} explained variance (2 components)")
    print(f"Elbow:       inertia knee at k={sweep['elbow_k']}, silhouette peak at k={sweep['silhouette_peak_k']}")
    print("Personas:")
    for record in result["personas"]:
        print(f"  cluster {record['cluster']}  {record['name']:22s} "
              f"{record['size']:5d} ({record['share']:.1%})  match {record['match_quality']:.0%}")
    for note in meta["persona_notes"]:
        print(f"  ! {note}")
    print(f"Elapsed:     {meta['elapsed_seconds']:.1f}s -> {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
