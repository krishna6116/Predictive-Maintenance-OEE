"""Phase 3 feature tests: pure functions, no database needed."""
import numpy as np
import pandas as pd
import pytest

from features.build import (RUL_CAP, add_rolling, add_rul, apply_scaler, build_features, feature_columns,
                            fit_scaler, split_engines)
from pipeline.config import SENSORS


def readings(n_units=4, n_cycles=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1 + u):  # engines of different lengths
            rows.append({"unit": u, "cycle": c, **{s: 100 + 0.1 * c + rng.normal(0, 1) for s in SENSORS}})
    return pd.DataFrame(rows)


@pytest.fixture
def scaler():
    df = readings()
    return fit_scaler(df, df.unit.unique())


def test_rul_counts_down_to_zero_and_is_capped():
    df = add_rul(pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)}))
    assert df.RUL.iloc[-1] == 0
    assert df.RUL.max() == RUL_CAP
    assert (df.RUL.iloc[:75] == RUL_CAP).all()  # first 75 cycles have >=125 cycles left


def test_split_is_by_engine_and_repeatable():
    s1, s2 = split_engines(range(1, 101)), split_engines(range(1, 101))
    assert s1 == s2
    assert set(s1["train"]).isdisjoint(s1["val"])
    assert sorted(s1["train"] + s1["val"]) == list(range(1, 101))
    assert len(s1["val"]) == 20


def test_scaler_ignores_validation_engines():
    df = readings()
    df.loc[df.unit == 4, SENSORS] = 1e6  # an extreme validation engine must not move the scaler
    sc = fit_scaler(df, [1, 2, 3])
    assert abs(sc["s2"]["mean"]) < 1000
    scaled = apply_scaler(df[df.unit.isin([1, 2, 3])], sc)
    assert abs(scaled["s2"].mean()) < 1e-9 and abs(scaled["s2"].std() - 1) < 1e-9


def test_slope_matches_a_known_line():
    df = pd.DataFrame({"unit": [1] * 30, "cycle": range(1, 31), **{s: [2.0 * c + 5 for c in range(1, 31)] for s in SENSORS}})
    out = add_rolling(df, cols=["s2"], windows=(10,))
    assert np.allclose(out["s2_slope10"].iloc[1:], 2.0)
    assert out["s2_slope10"].iloc[0] == 0.0  # one point has no slope


def test_windows_never_mix_engines():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2, 2], "cycle": [1, 2, 3, 1, 2, 3],
                       **{s: [0, 0, 0, 100, 100, 100] for s in SENSORS}})
    out = add_rolling(df, cols=["s2"], windows=(5,))
    assert (out.loc[out.unit == 2, "s2_mean5"] == 100).all()


def test_features_are_causal(scaler):
    """Changing a future reading must not change any earlier feature (no look-ahead leakage)."""
    df = readings()
    base = build_features(df, scaler)
    changed = df.copy()
    changed.loc[(changed.unit == 2) & (changed.cycle == 40), SENSORS] += 50
    after = build_features(changed, scaler)
    past = (base.unit == 2) & (base.cycle < 40)
    pd.testing.assert_frame_equal(base[past], after[past])


def test_incremental_build_matches_full_build(scaler):
    """Features for a new batch, computed from the history up to it, equal a full rebuild."""
    df = readings()
    full = build_features(df, scaler)
    upto = df[df.cycle <= 45]
    batch = build_features(upto, scaler)
    batch = batch[batch.cycle > 40].reset_index(drop=True)
    expected = full[(full.cycle > 40) & (full.cycle <= 45)].reset_index(drop=True)
    pd.testing.assert_frame_equal(batch, expected)


def test_feature_table_shape(scaler):
    out = build_features(readings(), scaler)
    assert list(out.columns) == ["unit", "cycle"] + feature_columns()
    assert len(feature_columns()) == 14 * 12
    assert not out[feature_columns()].isna().any().any()
