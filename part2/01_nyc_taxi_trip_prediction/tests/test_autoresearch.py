import numpy as np
import pytest

from taxi import autoresearch as ar
from taxi.artifacts import ArtifactStore
from taxi.data import TripStore
from taxi.models import FittedModel


def test_gate_requires_improvement_above_epsilon():
    assert ar.gate(0.3000, 0.3002, epsilon=1e-4)
    assert not ar.gate(0.30005, 0.3001, epsilon=1e-4)
    assert not ar.gate(0.31, 0.30, epsilon=1e-4)


class _Pred:
    def __init__(self, y, sigma, seed):
        self.y, self.sigma, self.seed = y, sigma, seed

    def predict(self, X):
        noise = np.random.default_rng(self.seed).standard_normal(len(self.y))
        return self.y + self.sigma * noise


def make_stub(y_val_holder):
    """Error scale depends deterministically on the configuration, so outcomes are known."""
    def fit_fn(key, params, X_tr, y_tr, X_val, y_val):
        sigma = {"xgboost": 0.30, "hist_gb": 0.29, "ridge": 0.50}.get(key, 0.40)
        if key == "xgboost":
            if "good_feature" in X_val.columns:
                sigma -= 0.02
            if "bad_feature" in X_val.columns:
                sigma += 0.02
            if params.get("max_depth") == 99:
                sigma -= 0.01
        return FittedModel(key=key, model=_Pred(y_val, sigma, seed=7), fit_seconds=0.0, hyperparameters=params)
    return fit_fn


def test_session_is_like_for_like_reverts_rejections_and_skips_test(processed_path, tmp_path):
    trips = TripStore(processed_path)
    feats = [
        {"name": "bad_feature", "expr": "df['hour'] * 0.0", "hypothesis": "worse"},
        {"name": "good_feature", "expr": "df['hour'] * 1.0", "hypothesis": "better"},
    ]
    hps = [{"name": "magic_depth", "override": {"max_depth": 99}, "hypothesis": "better"},
           {"name": "noop", "override": {"reg_lambda": 1.0}, "hypothesis": "same"}]
    s = ar.run_session(fit_sample_size=None, seed=1, trips=trips, store=ArtifactStore(tmp_path),
                       backbone_keys=["xgboost", "hist_gb", "ridge"], feature_candidates=feats,
                       hyperparameter_candidates=hps, blend_weights=[0.5], fit_fn=make_stub(None))
    assert "test" not in trips.accessed
    assert s["config"]["test_split_used"] is False
    steps = {st["component"]: st for st in s["steps"]}
    xgb_backbone = next(r for r in s["leaderboard"] if r["key"] == "xgboost")
    # The hill climb starts from the XGBoost configuration's own score, not the champion's.
    assert s["summary"]["initial_rmsle"] == pytest.approx(xgb_backbone["rmsle"])
    assert s["leaderboard"][0]["key"] == "hist_gb"
    assert steps["bad_feature"]["decision"] == "REJECTED"
    assert steps["bad_feature"]["incumbent_before"] == pytest.approx(xgb_backbone["rmsle"])
    assert steps["good_feature"]["decision"] == "ACCEPTED"
    assert "bad_feature" not in s["active_features"] and "good_feature" in s["active_features"]
    assert steps["magic_depth"]["decision"] == "ACCEPTED"
    assert steps["noop"]["decision"] == "REJECTED"
    # Each gated step is measured against the incumbent left by the previous gated step.
    gated = [st for st in s["steps"] if st["decision"] in ("ACCEPTED", "REJECTED")]
    for prev, cur in zip(gated, gated[1:]):
        assert cur["incumbent_before"] == pytest.approx(prev["incumbent_after"])
    assert s["summary"]["accepted"] + s["summary"]["rejected"] == len(gated)
    assert ArtifactStore(tmp_path).load_latest_session()["session_id"] == s["session_id"]


def test_feature_code_fragment_is_the_code_that_runs(processed_path):
    import pandas as pd
    from taxi.features import build_features
    df = build_features(TripStore(processed_path).role("validation").head(50))
    for cand in ar.FEATURE_CANDIDATES:
        out = ar.add_feature(df, cand)
        expected = eval(cand["expr"], {"np": np}, {"df": df})  # noqa: S307
        pd.testing.assert_series_equal(out[cand["name"]], pd.Series(expected, name=cand["name"]),
                                       check_dtype=False)
        assert out[cand["name"]].notna().all()


def test_real_minimal_session_runs(processed_path, tmp_path):
    s = ar.run_session(fit_sample_size=None, seed=0, trips=TripStore(processed_path), store=ArtifactStore(tmp_path),
                       backbone_keys=["xgboost", "hist_gb"], feature_candidates=ar.FEATURE_CANDIDATES[:1],
                       hyperparameter_candidates=ar.HYPERPARAMETER_CANDIDATES[-1:], blend_weights=[0.5])
    assert len(s["steps"]) == 5
    assert all(np.isfinite(st["rmsle_candidate"]) for st in s["steps"])
