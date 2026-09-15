"""Download the four public datasets into the local parquet cache.

    python scripts/prepare_data.py           # download anything missing
    python scripts/prepare_data.py --force   # re-download everything

Nothing else in the project touches the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills_lab.config import CACHE_FILES  # noqa: E402
from skills_lab.data import acquire  # noqa: E402

SOURCES = {
    "titanic": "Titanic passenger list - OpenML data id 40945",
    "ames": "Ames, Iowa house sales - OpenML data id 42165",
    "creditcard": "ULB credit-card fraud - OpenML data id 1597 (~150 MB download)",
    "retail": "Online Retail II - UCI archive (~45 MB zip, ~1.07M rows)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download everything")
    parser.add_argument(
        "--only", choices=sorted(CACHE_FILES), help="prepare a single dataset"
    )
    args = parser.parse_args()

    keys = [args.only] if args.only else list(CACHE_FILES)
    for key in keys:
        print(f"\n== {key}: {SOURCES[key]}")
        if acquire.is_prepared(key) and not args.force:
            print(f"   already cached at {CACHE_FILES[key]}")
            frame = acquire.load_cached(key)
        else:
            print("   downloading ...")
            frame = acquire.prepare(key, force=args.force)
        print(f"   {len(frame):,} rows x {frame.shape[1]} columns -> {CACHE_FILES[key]}")

    print("\nAll requested datasets are ready. Start the app with:")
    print("   streamlit run app/main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
