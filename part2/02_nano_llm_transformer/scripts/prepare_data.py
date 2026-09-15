"""Download the open datasets (if missing) and build the windowed train/val/test splits.

    python scripts/prepare_data.py                      # default sample size
    python scripts/prepare_data.py --stories 20000      # smaller TinyStories training sample
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanollama.data import everyday, prepare  # noqa: E402

TS_BASE = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/"
TS_FILES = ["TinyStories-Instruct-valid.txt", "TinyStories-Instruct-train.txt"]


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name in TS_FILES:
        dest = raw / name
        if dest.exists():
            continue
        print(f"downloading {name} (the train file is ~2.7 GB) ...", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(TS_BASE + name, tmp)
        tmp.rename(dest)
    if not all((raw / "everyday" / f).exists() for f in everyday.FILES.values()):
        print("downloading everyday-conversations parquet files ...", flush=True)
        everyday.download(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--stories", type=int, default=60000, help="TinyStoriesInstruct training sample size")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    download(Path(args.raw))
    manifest = prepare.run(Path(args.raw), Path(args.out), args.stories, args.context, args.seed)
    rep = manifest["report"]
    print(json.dumps({k: rep[k] for k in ("tinystories", "everyday")}, indent=1))
    for src, splits in rep["windows"].items():
        for split, s in splits.items():
            print(f"{src:12s} {split:5s} examples={s['examples']:6d} fit={s['fit_directly']:6d} "
                  f"turn_windowed={s['turn_windowed']:4d} chunked={s['chunked']:5d} dropped={s['dropped']:3d} "
                  f"windows={s['windows']:6d} target_tokens={s['target_tokens']:9d} "
                  f"replaced={s['char_replacement_rate']:.2e}")
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
