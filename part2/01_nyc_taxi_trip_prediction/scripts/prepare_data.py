"""Extract the local Kaggle archive, clean train.csv, assign fixed splits, write the data report.

Usage: python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxi import data  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    report = data.prepare()
    print(json.dumps(report, indent=2))
    print(f"Done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
