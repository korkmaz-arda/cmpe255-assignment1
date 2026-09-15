"""Stage-2 conversation adaptation: continue training an existing version with a
conversation-heavy mixture and a lower learning rate, then compare it with its
parent on validation losses and fixed generation probes.

The adapted model is written as a NEW version directory and is NOT promoted here.
Promote it explicitly with scripts/promote.py if it is accepted. Test splits are never read.

    python scripts/adapt.py                       # adapts the version in artifacts/current.json
    python scripts/adapt.py --init 20260916-181935-run
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
from nanollama.train import TrainConfig, run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", help="version id to start from (default: the promoted version)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--everyday-repeat", type=int, default=2)
    ap.add_argument("--kb-repeat", type=int, default=4)
    ap.add_argument("--story-windows-per-epoch", type=int, default=3000)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    arts = Path(args.artifacts)
    init_dir = artifacts.versions_dir(arts) / args.init if args.init else artifacts.current_version(arts)
    if init_dir is None or not init_dir.exists():
        sys.exit("no version to adapt")
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, everyday_repeat=args.everyday_repeat,
                      kb_repeat=args.kb_repeat, story_windows_per_epoch=args.story_windows_per_epoch,
                      init_from=str(init_dir), compile=True, tag="adapt")

    log_path = arts / "logs" / f"adapt-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    log(f"config: {cfg.to_dict()}")
    vdir = run_training(Path(args.data), arts, cfg, promote=False, log=log)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parent, tok, _ = artifacts.load_model(init_dir, device)
    adapted, _, _ = artifacts.load_model(vdir, device)
    t0 = time.time()
    result = compare({"before": parent, "after": adapted}, tok, Path(args.data))
    tel = json.loads((vdir / "telemetry.json").read_text())
    init = tel["initialized_from"]
    best = tel["best_epoch"] - 1
    result["validation"] = {
        "before": {**{s: v["loss"] for s, v in init["validation"].items()}, "selection": init["selection_loss"]},
        "after": {**{s: tel["curves"]["val_loss"][s][best] for s in ("tinystories", "everyday", "kb")},
                  "selection": tel["best_val_selection_loss"]},
        "note": "kb is a diagnostic only; selection = (tinystories + everyday) / 2",
    }
    result["before_version"] = init_dir.name
    result["after_version"] = vdir.name
    artifacts.write_json(vdir / "adaptation_comparison.json", result)
    log(f"comparison written ({time.time() - t0:.0f}s): {vdir / 'adaptation_comparison.json'}")
    log(json.dumps({"validation": result["validation"], "summary": result["summary"]}, indent=1))
    log(f"NOT promoted. To accept: python scripts/promote.py {vdir.name}")


if __name__ == "__main__":
    main()
