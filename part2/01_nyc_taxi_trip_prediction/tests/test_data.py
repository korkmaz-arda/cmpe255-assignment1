import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from taxi import config, data

from conftest import make_synthetic_trips


def test_cleaning_rules_count_removals():
    df = make_synthetic_trips(50, seed=1)
    df.loc[0, "trip_duration"] = 30                  # too short
    df.loc[1, "trip_duration"] = 4 * 3600            # too long
    df.loc[2, "pickup_latitude"] = 0.0               # outside bbox
    df.loc[3, "dropoff_longitude"] = -80.0           # outside bbox
    df.loc[4, "passenger_count"] = 0                 # invalid passengers
    df.loc[5, ["dropoff_latitude", "dropoff_longitude"]] = [40.91, -73.69]  # far away ...
    df.loc[5, "trip_duration"] = 61                  # ... in one minute -> implausible speed
    cleaned, counts = data.clean(df)
    removed = {c["rule"]: c["removed"] for c in counts}
    assert removed == {"duration_bounds": 2, "pickup_in_nyc_bbox": 1, "dropoff_in_nyc_bbox": 1,
                       "passenger_count_range": 1, "implied_speed": 1}
    assert len(cleaned) == 44
    assert counts[0]["rows_before"] == 50


def test_zero_displacement_trips_are_kept():
    df = make_synthetic_trips(5, seed=2)
    df["dropoff_latitude"], df["dropoff_longitude"] = df["pickup_latitude"], df["pickup_longitude"]
    cleaned, _ = data.clean(df)
    assert len(cleaned) == 5


def test_extract_nested_archive(tmp_path):
    csv = make_synthetic_trips(10).to_csv(index=False).encode()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("train.csv", csv)
    inner_test = io.BytesIO()
    with zipfile.ZipFile(inner_test, "w") as z:
        z.writestr("test.csv", b"id,vendor_id\nx,1\n")
    archive = tmp_path / "nyc-taxi-trip-duration.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train.zip", inner.getvalue())
        z.writestr("test.zip", inner_test.getvalue())
    paths = data.extract_archive(archive, tmp_path / "raw")
    loaded = data.load_raw_train(paths["train"])
    assert len(loaded) == 10 and paths["test"].exists()
    assert pd.api.types.is_datetime64_any_dtype(loaded["pickup_datetime"])


def test_split_is_deterministic_order_independent_and_disjoint():
    ids = [f"id{i}" for i in range(20000)]
    a = data.assign_split(ids)
    shuffled = list(np.random.default_rng(0).permutation(ids))
    b = dict(zip(shuffled, data.assign_split(shuffled)))
    assert all(b[i] == r for i, r in zip(ids, a))
    fractions = {r: np.mean(a == r) for r in data.ROLES}
    for name, lo, hi in config.SPLIT_BOUNDS:
        assert fractions[name] == pytest.approx((hi - lo) / 1000, abs=0.015)
    assert set(a) == set(data.ROLES)


def test_role_reads_are_disjoint_and_validation_is_fixed(processed_path):
    store = data.TripStore(processed_path)
    frames = {r: store.role(r) for r in data.ROLES}
    id_sets = [set(f["id"]) for f in frames.values()]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            assert not id_sets[i] & id_sets[j]
    # Changing the training sample size/seed never changes the validation or calibration rows.
    s1 = data.TripStore(processed_path)
    s1.fit_sample(500, seed=1)
    v1, c1 = s1.role("validation"), s1.role("calibration")
    s2 = data.TripStore(processed_path)
    s2.fit_sample(1500, seed=99)
    pd.testing.assert_frame_equal(v1, s2.role("validation"))
    pd.testing.assert_frame_equal(c1, s2.role("calibration"))


def test_seeded_sampling_is_repeatable_and_drawn_from_fit_only(processed_path):
    store = data.TripStore(processed_path)
    a = store.fit_sample(700, seed=5)
    b = store.fit_sample(700, seed=5)
    c = store.fit_sample(700, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["id"]) != set(c["id"])
    assert (a["split"] == "fit").all()
