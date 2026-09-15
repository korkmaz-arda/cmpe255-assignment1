"""Score the promoted version (artifacts/current.json) on the final held-out test sets.

Run this once, after the production checkpoint has been trained and promoted.
Writes final_eval.json into that version's directory. Refuses to run again on an
already finalized version unless --force is given; forced runs are recorded.

    python scripts/finalize.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanollama.finalize import FinalizeRefused, finalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate an already finalized version (recorded)")
    ap.add_argument("--device")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    try:
        finalize(Path(args.artifacts), Path(args.data), force=args.force, device=args.device)
    except FinalizeRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
