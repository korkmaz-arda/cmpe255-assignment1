"""Train NanoLlama into a new artifacts/versions/<run-id>/ directory and promote it.

Uses only the train and validation splits. It never runs the final test evaluation;
that is a separate, explicit step: python scripts/finalize.py

    python scripts/train.py --preset smoke     # short validation run
    python scripts/train.py                    # full run with default settings
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanollama.train import TrainConfig, run_training  # noqa: E402

PRESETS = {
    "full": {},
    "smoke": {"epochs": 1, "max_story_windows": 4000, "tag": "smoke"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="full")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-story-windows", type=int)
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile (it is used by default on CUDA)")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    cfg = TrainConfig(**PRESETS[args.preset])
    for name in ("epochs", "batch_size", "lr", "seed", "max_story_windows"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    cfg.compile = not args.no_compile

    log_dir = Path(args.artifacts) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train-{time.strftime('%Y%m%d-%H%M%S')}-{cfg.tag}.log"
    fh = open(log_path, "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    log(f"log file: {log_path}")
    log(f"config: {cfg.to_dict()}")
    run_training(Path(args.data), Path(args.artifacts), cfg, device=args.device, promote=not args.no_promote, log=log)


if __name__ == "__main__":
    main()
