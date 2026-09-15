"""Train, benchmark and promote a new model version (validation metrics only; test is untouched).

Usage:
  python scripts/train.py                                  # full fit pool, default XGBoost settings
  python scripts/train.py --sample-size 50000 --n-estimators 120 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxi import config  # noqa: E402
from taxi.artifacts import ArtifactStore  # noqa: E402
from taxi.train import TrainParams, run_training  # noqa: E402


def main() -> None:
    d = config.DEFAULT_XGB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-size", type=int, default=None, help="rows drawn from the fit pool (default: all)")
    ap.add_argument("--n-estimators", type=int, default=d["n_estimators"])
    ap.add_argument("--max-depth", type=int, default=d["max_depth"])
    ap.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    ap.add_argument("--subsample", type=float, default=d["subsample"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    params = TrainParams(sample_size=args.sample_size, n_estimators=args.n_estimators, max_depth=args.max_depth,
                         learning_rate=args.learning_rate, subsample=args.subsample, seed=args.seed)
    t0 = time.perf_counter()
    version = run_training(params, progress=lambda m: print(f"[{time.perf_counter() - t0:7.1f}s] {m}", flush=True))
    bundle = ArtifactStore().load_current()
    print("\nValidation metrics (fixed validation split):")
    for e in bundle.experiments:
        v = e["validation"]
        print(f"  {e['name']:<18} RMSLE {v['rmsle']:.4f}  R2 {v['r2']:.4f}  MAE {v['mae_s']:.1f}s  "
              f"fit {e['fit_seconds']:.1f}s  [{e['status']}]")
    print("Band validation coverage:", json.dumps(round(bundle.metadata["band_validation_coverage"]["coverage"], 4)))
    print(f"Active version: {version}  (not finalized; run scripts/finalize.py once when settled)")


if __name__ == "__main__":
    main()
