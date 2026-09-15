"""Run the fixed validation-safe generation probes against one version (default: the promoted one)
and write artifacts/logs/probes-<version>-<time>.json. Uses the canonical decoding defaults.

    python scripts/probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from nanollama import artifacts  # noqa: E402
from nanollama.compare import compare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    arts = Path(args.artifacts)
    vdir = artifacts.versions_dir(arts) / args.version if args.version else artifacts.current_version(arts)
    model, tok, _ = artifacts.load_model(vdir, "cuda" if torch.cuda.is_available() else "cpu")
    result = compare({vdir.name: model}, tok, Path(args.data))
    out = arts / "logs" / f"probes-{vdir.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(out)


if __name__ == "__main__":
    main()
