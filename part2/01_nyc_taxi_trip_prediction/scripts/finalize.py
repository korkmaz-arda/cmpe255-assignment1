"""One-time final holdout evaluation of the active (or given) model version on the test split.

Usage:
  python scripts/finalize.py                 # evaluate the active version once
  python scripts/finalize.py --version v...  # a specific version
  python scripts/finalize.py --force         # re-evaluate (refused otherwise; logged in holdout_log.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxi.holdout import HoldoutAlreadyEvaluated, finalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    try:
        result = finalize(version=args.version, force=args.force)
    except HoldoutAlreadyEvaluated as exc:
        print(f"Refused: {exc}")
        sys.exit(2)
    print(json.dumps({k: result[k] for k in ("version", "evaluated_at", "metrics", "forced_rerun")}, indent=2))
    print(f"Band coverage on test: {result['band_coverage']['coverage']:.4f} "
          f"(nominal {result['band_coverage']['nominal']:.2f})")


if __name__ == "__main__":
    main()
