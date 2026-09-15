"""Shared offline fixtures: a tiny raw corpus in the real file formats, prepared data, a tiny trained version."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanollama.model import ModelConfig  # noqa: E402

TINY_CONTEXT = 384


def tiny_model_config(context_length: int = TINY_CONTEXT) -> ModelConfig:
    return ModelConfig(vocab_size=104, n_layers=2, d_model=32, n_heads=4, d_ff=64, context_length=context_length)


def _story(i: int, long: bool) -> str:
    body = f"Tom had a red ball number {i}. He liked to play with it in the park every day. "
    if long:
        body = body * 8  # well past the tiny context, forcing chunked windows
    return body.strip()


def write_raw_fixture(raw: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw.mkdir(parents=True, exist_ok=True)
    fields_first = "Features: Dialogue\nWords: ball, park, red\nSummary: Tom plays with a ball.\nStory: \n"
    recs_train = [fields_first + _story(i, long=i % 3 == 0) for i in range(30)]
    recs_train.append("Features: Dialogue\nSummary: a record with no story field")
    recs_valid = [
        f"Random sentence: The sun was warm.\nStory: \n{_story(100 + i, long=i % 2 == 0)}\nSummary: A day number {i}."
        for i in range(12)
    ]
    sep = "\n<|endoftext|>\n"
    (raw / "TinyStories-Instruct-train.txt").write_text(sep.join(recs_train) + sep)
    (raw / "TinyStories-Instruct-valid.txt").write_text(sep.join(recs_valid) + sep)

    def convo(i: int, trailing_user: bool = False):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": f"What is a good snack number {i}?"},
            {"role": "assistant", "content": f"An apple is a good snack, and number {i} is a fine choice for a quick bite."},
        ]
        if trailing_user:
            msgs.append({"role": "user", "content": "Thanks!"})
        return {"full_topic": "Food/Snacks", "messages": msgs}

    d = raw / "everyday" / "data"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([convo(i, i % 4 == 0) for i in range(40)]), d / "train_sft-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist([convo(1000 + i) for i in range(6)]), d / "test_sft-00000-of-00001.parquet")


@pytest.fixture(scope="session")
def prepared(tmp_path_factory) -> Path:
    from nanollama.data import prepare

    base = tmp_path_factory.mktemp("data")
    raw, out = base / "raw", base / "processed"
    write_raw_fixture(raw)
    prepare.run(raw, out, n_train_stories=24, context_length=TINY_CONTEXT, seed=7,
                n_val_stories=4, n_test_stories=4, everyday_val_fraction=0.1)
    return out


@pytest.fixture(scope="session")
def trained(prepared, tmp_path_factory):
    """A tiny CPU-trained, promoted version. Returns (artifacts_root, version_dir)."""
    from nanollama.train import TrainConfig, run_training

    arts = tmp_path_factory.mktemp("artifacts")
    cfg = TrainConfig(epochs=2, batch_size=8, lr=3e-3, model=tiny_model_config(), tag="test", kb_repeat=2)
    vdir = run_training(prepared, arts, cfg, device="cpu", log=lambda *_: None)
    return arts, vdir
