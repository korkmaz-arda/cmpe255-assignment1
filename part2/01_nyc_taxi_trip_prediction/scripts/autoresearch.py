"""Run a validation-only AutoResearch hill-climbing session and persist its trajectory.

Usage:
  python scripts/autoresearch.py                          # 150k fit-sample rows, seed 42
  python scripts/autoresearch.py --fit-sample-size 30000 --seed 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxi.autoresearch import run_session  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-sample-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.perf_counter()
    s = run_session(fit_sample_size=args.fit_sample_size, seed=args.seed,
                    progress=lambda m: print(f"[{time.perf_counter() - t0:7.1f}s] {m}", flush=True))
    print("\nLeaderboard (validation RMSLE):")
    for r in s["leaderboard"]:
        print(f"  {r['rank']}. {r['name']:<22} {r['rmsle']:.5f}  fit {r['fit_seconds']:.1f}s  "
              f"{r['latency_us_per_row']:.2f} us/row{'  <- champion' if r['champion'] else ''}")
    print("\nGated steps:")
    for st in s["steps"]:
        if st["decision"] != "BENCHMARK":
            print(f"  {st['iteration']:>2} {st['phase']:<14} {st['component']:<26} "
                  f"{st['rmsle_candidate']:.5f} vs {st['incumbent_before']:.5f}  {st['decision']}")
    sm = s["summary"]
    print(f"\nInitial {sm['initial_rmsle']:.5f} -> best {sm['best_rmsle']:.5f} "
          f"({sm['improvement_pct']:.2f}%), accepted {sm['accepted']}/{sm['gated_steps']}, "
          f"final model {sm['final_model']}, runtime {s['runtime_seconds']:.0f}s -> {s['session_id']}")


if __name__ == "__main__":
    main()
