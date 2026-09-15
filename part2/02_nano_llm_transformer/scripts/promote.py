"""Verify, probe and promote an existing version (atomically updates artifacts/current.json).

    python scripts/promote.py <version-id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from nanollama import artifacts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    root = Path(args.artifacts)
    vdir = artifacts.versions_dir(root) / args.version
    if not vdir.exists():
        sys.exit(f"no such version: {vdir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        record = artifacts.promote(root, vdir, device)
    except artifacts.PromotionError as exc:
        sys.exit(f"promotion refused: {exc} (current.json unchanged)")
    print(f"promoted {vdir.name}: {record}")


if __name__ == "__main__":
    main()
