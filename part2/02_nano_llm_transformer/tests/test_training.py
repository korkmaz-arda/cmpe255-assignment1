import json
import shutil

import numpy as np
import pytest
import torch

from nanollama import artifacts
from nanollama.data import store
from nanollama.finalize import FinalizeRefused, finalize
from nanollama.metrics import inputs_and_labels, masked_loss_sum
from nanollama.model import NanoLlama
from nanollama.train import SELECTION_TOLERANCE, TrainConfig, improves, lr_at, run_training
from tests.conftest import tiny_model_config


def test_masked_positions_do_not_affect_loss():
    torch.manual_seed(0)
    m = NanoLlama(tiny_model_config()).eval()
    tokens = torch.randint(6, 104, (2, 41))
    flags = torch.zeros(2, 41, dtype=torch.bool)
    flags[:, 30:] = True  # only the last 11 positions are assistant targets
    x, y = inputs_and_labels(tokens, flags)
    other = tokens.clone()
    other[:, 1:30] = torch.randint(6, 104, (2, 29))  # rewrite every masked (prompt) label
    _, y_other = inputs_and_labels(other, flags)
    assert torch.equal(y, y_other)  # masked labels are ignored entirely
    with torch.no_grad():
        logits, _ = m(x)
    loss, count = masked_loss_sum(logits, y)
    assert count == 2 * 11
    full = torch.nn.functional.cross_entropy(logits[:, 29:].reshape(-1, 104), tokens[:, 30:].reshape(-1), reduction="sum")
    assert torch.allclose(loss, full, atol=1e-4)


def test_selection_prefers_earlier_epoch_on_effective_ties():
    assert improves(1.0, float("inf"))
    assert not improves(1.0 - SELECTION_TOLERANCE / 2, 1.0)
    assert improves(1.0 - 2 * SELECTION_TOLERANCE, 1.0)


def test_lr_schedule_warmup_and_cosine_floor():
    cfg = TrainConfig(lr=1e-3, warmup_frac=0.1, min_lr_frac=0.05)
    lrs = [lr_at(s, 100, cfg) for s in range(100)]
    assert lrs[0] < lrs[9] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(5e-5, rel=0.05)
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))


def test_training_writes_version_with_validation_only(trained):
    arts, vdir = trained
    for f in artifacts.REQUIRED_FILES + ("promotion.json",):
        assert (vdir / f).exists(), f
    assert not (vdir / artifacts.FINAL_EVAL).exists()
    tel = json.loads((vdir / "telemetry.json").read_text())
    assert set(tel["curves"]["val_loss"]) == {"tinystories", "everyday", "kb"}
    assert "(tinystories_val_loss + everyday_val_loss) / 2" in tel["selection_definition"]
    sel = tel["curves"]["val_selection"]
    for e in range(len(sel)):
        assert sel[e] == pytest.approx((tel["curves"]["val_loss"]["tinystories"][e] + tel["curves"]["val_loss"]["everyday"][e]) / 2)
    assert tel["best_val_selection_loss"] == pytest.approx(sel[tel["best_epoch"] - 1])
    assert tel["diagnostic_only_sources"] == ["kb"]
    assert tel["curves"]["train_loss"][-1] < tel["curves"]["train_loss"][0]
    val = json.loads((vdir / "validation.json").read_text())
    assert "validation" in val["split"] and val["probes"]
    assert artifacts.current_version(arts) == vdir


def test_checkpoint_reload_reproduces_logits(trained):
    _, vdir = trained
    m1, tok, cfg = artifacts.load_model(vdir)
    m2, _, _ = artifacts.load_model(vdir)
    x = torch.randint(6, 104, (1, 50))
    with torch.no_grad():
        assert torch.equal(m1(x)[0], m2(x)[0])
    assert m1.lm_head.weight.data_ptr() == m1.embed.weight.data_ptr()
    assert cfg.n_layers == 2 and tok.vocab_size == 104


def test_training_never_opens_test_split(prepared, tmp_path, monkeypatch):
    opened = []
    real = store._read

    def spy(root, split, source):
        opened.append(split)
        return real(root, split, source)

    monkeypatch.setattr(store, "_read", spy)
    cfg = TrainConfig(epochs=1, batch_size=8, model=tiny_model_config(), tag="spy")
    run_training(prepared, tmp_path, cfg, device="cpu", promote=False, log=lambda *_: None)
    assert opened and set(opened) <= {"train", "val"}


def test_failed_probe_leaves_pointer_untouched(trained, tmp_path, monkeypatch):
    arts, vdir = trained
    root = tmp_path / "arts"
    shutil.copytree(arts, root)
    before = (root / artifacts.POINTER).read_text()
    bad = artifacts.new_version_dir(root, "bad")
    for f in artifacts.REQUIRED_FILES:
        shutil.copy(vdir / f, bad / f)

    def broken_probe(model, tok):
        raise artifacts.PromotionError("probe produced non-finite logits")

    monkeypatch.setattr(artifacts, "probe", broken_probe)
    with pytest.raises(artifacts.PromotionError):
        artifacts.promote(root, bad)
    assert (root / artifacts.POINTER).read_text() == before
    assert json.loads((bad / "promotion.json").read_text())["status"] == "failed"

    missing = artifacts.new_version_dir(root, "missing")
    with pytest.raises(artifacts.PromotionError):
        artifacts.promote(root, missing)
    assert (root / artifacts.POINTER).read_text() == before


def test_finalize_guard_and_version_scoping(trained, prepared, tmp_path):
    arts, vdir = trained
    root = tmp_path / "arts"
    shutil.copytree(arts, root)
    active = artifacts.current_version(root)
    out = finalize(root, prepared, device="cpu", log=lambda *_: None)
    report = json.loads(out.read_text())
    assert out.parent == active and report["finalize_count"] == 1
    assert set(report["sources"]) == {"tinystories", "everyday", "kb", "kb_rephrase"}
    assert "answer seen during training" in report["sources"]["kb_rephrase"]["label"]

    with pytest.raises(FinalizeRefused):
        finalize(root, prepared, device="cpu", log=lambda *_: None)
    forced = json.loads(finalize(root, prepared, force=True, device="cpu", log=lambda *_: None).read_text())
    assert forced["finalize_count"] == 2 and forced["finalize_history"][-1]["forced"] is True

    # A newly promoted version starts not finalized.
    new = artifacts.new_version_dir(root, "retrain")
    for f in artifacts.REQUIRED_FILES:
        shutil.copy(active / f, new / f)
    artifacts.promote(root, new)
    assert artifacts.current_version(root) == new
    assert not (new / artifacts.FINAL_EVAL).exists()


def test_finalize_refuses_when_prepared_data_changed(trained, prepared, tmp_path):
    arts, _ = trained
    root = tmp_path / "arts"
    shutil.copytree(arts, root)
    data = tmp_path / "data"
    shutil.copytree(prepared, data)
    m = json.loads((data / "manifest.json").read_text())
    m["seed"] = 999
    (data / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(FinalizeRefused):
        finalize(root, data, device="cpu", log=lambda *_: None)


def test_stage2_adaptation_starts_from_checkpoint(trained, prepared, tmp_path):
    _, parent = trained
    cfg = TrainConfig(epochs=1, batch_size=8, lr=3e-4, model=tiny_model_config(), tag="adapt", everyday_repeat=2,
                      kb_repeat=1, story_windows_per_epoch=5, init_from=str(parent))
    vdir = run_training(prepared, tmp_path, cfg, device="cpu", promote=False, log=lambda *_: None)
    tel = json.loads((vdir / "telemetry.json").read_text())
    assert tel["stage"] == 2 and tel["initialized_from"]["version"] == parent.name
    assert "stage-2 conversation adaptation" in tel["training_from"]
    assert tel["mixture"]["story_windows_per_epoch"] == 5
    # the recorded initial validation is the parent's validation at its saved epoch
    parent_tel = json.loads((parent / "telemetry.json").read_text())
    before = parent_tel["curves"]["val_loss"]["everyday"][parent_tel["best_epoch"] - 1]
    assert tel["initialized_from"]["validation"]["everyday"]["loss"] == pytest.approx(before, rel=1e-4)
    assert artifacts.current_version(tmp_path) is None  # not promoted
    assert (parent / "checkpoint.pt").exists()  # parent untouched

    wrong = TrainConfig(epochs=1, batch_size=8, model=tiny_model_config(context_length=256), init_from=str(parent))
    with pytest.raises(Exception):
        run_training(prepared, tmp_path, wrong, device="cpu", promote=False, log=lambda *_: None)


def test_comparison_probes_are_validation_safe(trained, prepared, monkeypatch):
    from nanollama.compare import FIXED_PROBES, assert_validation_safe, compare

    assert_validation_safe()
    _, vdir = trained
    m, tok, _ = artifacts.load_model(vdir)
    import nanollama.compare as cmp
    monkeypatch.setattr(cmp, "GREEDY", cmp.DecodeSettings(**{**cmp.GREEDY.to_dict(), "max_new_tokens": 8}))
    monkeypatch.setattr(cmp, "SAMPLED", cmp.DecodeSettings(**{**cmp.SAMPLED.to_dict(), "max_new_tokens": 8}))
    res = compare({"before": m, "after": m}, tok, prepared)
    assert len(res["probes"]) >= len(FIXED_PROBES)
    assert all(not r["source"].startswith("test") for r in res["probes"])
    assert res["probes"][0]["before"]["greedy"] == res["probes"][0]["after"]["greedy"]
    assert res["decoding"]["greedy"]["loop_guard"] is False


def test_repromotion_does_not_modify_version_directory(trained, tmp_path):
    arts, vdir = trained
    root = tmp_path / "arts"
    shutil.copytree(arts, root)
    old = artifacts.current_version(root)
    first_record = (old / "promotion.json").read_bytes()
    new = artifacts.new_version_dir(root, "retrain")
    for f in artifacts.REQUIRED_FILES:
        shutil.copy(old / f, new / f)
    artifacts.promote(root, new)
    assert artifacts.current_version(root) == new
    artifacts.promote(root, old)  # restore the earlier version
    assert artifacts.current_version(root) == old
    assert (old / "promotion.json").read_bytes() == first_record  # immutable
    history = [json.loads(line) for line in (root / artifacts.HISTORY).read_text().splitlines()]
    assert [h["version"] for h in history[-2:]] == [new.name, old.name] and history[-1]["repromotion"] is True
