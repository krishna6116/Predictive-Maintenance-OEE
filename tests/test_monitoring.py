"""Phase 7 monitoring: label timing, model health, KS drift with Holm, and the retraining rule."""
import math

import numpy as np
import pandas as pd
import pytest

from pipeline import monitoring as M
from pipeline.config import SENSORS


def last_preds(rows):
    return pd.DataFrame(rows, columns=["engine_id", "cycle", "rul_point", "rul_low", "rul_high"])


def test_label_becomes_available_at_the_failure_cycle():
    fail = M.failures({1: 100, 2: 50}, {1: 20, 2: 5})
    assert fail == {1: 120, 2: 55}
    assert not M.label_available(120, 119) and M.label_available(120, 120)


def test_health_scores_only_engines_whose_failure_has_happened():
    lp = last_preds([(1, 100, 25, 15, 35), (2, 50, 3, 0, 8), (3, 80, 60, 40, 80)])
    fail = {1: 120, 2: 55, 3: 200}
    h = M.health(lp, fail, shift=150)
    assert h["n_labelled"] == 2 and h["cum_n_scored"] == 2
    # engine 1: truth 20, error +5; engine 2: truth 5, error -2
    assert math.isclose(h["rmse"], math.sqrt((25 + 4) / 2)) and math.isclose(h["mae"], 3.5)
    assert h["coverage_90"] == 1.0 and h["lower_bound_miss_rate"] == 0.0
    assert M.health(lp, fail, shift=54)["n_labelled"] == 0 and M.health(lp, fail, shift=54)["rmse"] is None


def test_no_future_label_leaks_into_an_earlier_shift():
    rng = np.random.default_rng(0)
    lp = last_preds([(e, 50, rng.uniform(0, 100), 0, 125) for e in range(1, 31)])
    fail = {e: 60 + 3 * e for e in range(1, 31)}
    for shift in (70, 100, 130):
        before = M.health(lp, fail, shift)
        later = {e: (L if L <= shift else L + int(rng.integers(1, 200))) for e, L in fail.items()}  # change only unknown labels
        assert M.health(lp, later, shift) == before


def test_health_window_uses_most_recent_labels_and_cap():
    lp = last_preds([(e, 10, 125, 100, 150) for e in range(1, 26)])
    fail = {e: 10 + 200 + e for e in range(1, 26)}           # true RUL 201..225 -> capped truth 125
    h = M.health(lp, fail, shift=1000, window=20)
    assert h["n_scored"] == 20 and h["cum_n_scored"] == 25 and h["rmse"] == 0.0
    assert h["lower_bound_miss_rate"] == 0.0


def test_miss_rate_counts_true_life_below_lower_bound():
    lp = last_preds([(1, 100, 30, 20, 40), (2, 100, 30, 20, 40)])
    h = M.health(lp, {1: 110, 2: 125}, shift=200)            # true RUL 10 (< 20: miss) and 25
    assert h["lower_bound_miss_rate"] == 0.5 and h["near_failure_mae"] == pytest.approx((20 + 5) / 2)


def test_holm_matches_hand_values():
    adj = M.holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])         # 4x0.01, max(3x0.03, .)=0.09, max(2x0.04, 0.09), 0.2


def readings(n_eng, cycles, shift=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"unit": u, "cycle": c, **{s: 100 + rng.normal() for s in SENSORS}} for u in range(n_eng) for c in cycles]
    df = pd.DataFrame(rows)
    if shift:
        df[shift[0]] = df[shift[0]] * (1 + shift[1])
    return df


def test_drift_flags_a_shifted_sensor_and_not_the_others():
    ref, rec = readings(80, [1, 2, 3], seed=1), readings(100, [1, 2, 3], shift=("s4", 0.05), seed=2)
    d = M.drift(rec, ref, alpha=0.01)
    assert d.set_index("sensor").drifted.to_dict() == {s: s == "s4" for s in SENSORS}
    assert (d.n_recent == 100).all() and (d.n_reference == 80).all()      # engines, not readings


def test_drift_is_not_tested_on_too_few_engines():
    d = M.drift(readings(M.MIN_ENGINES - 1, [1, 2, 3]), readings(80, [1, 2, 3]), alpha=0.01)
    assert not d.tested.any() and not d.drifted.any()


def test_reference_is_age_matched():
    train = readings(5, range(1, 11))
    ref = M.reference_for(train, shift=6)
    assert sorted(ref.cycle.unique()) == [4, 5, 6] and M.window_cycles(1) == (1, 1)


RULE = {"drift": {"min_drifted_sensors": 1}, "health": {"min_labelled_engines": 20, "rmse_multiplier": 1.45,
                                                        "miss_rate_limit": 0.2}}


def test_retraining_rule_parts():
    clean = pd.DataFrame({"sensor": SENSORS, "drifted": False})
    ok = {"n_scored": 20, "rmse": 12.0, "lower_bound_miss_rate": 0.05}
    assert M.retraining(RULE, clean, ok, ref_rmse=11.0) == (False, [])
    flag, why = M.retraining(RULE, clean.assign(drifted=clean.sensor == "s4"), ok, 11.0)
    assert flag and why == ["drift in s4"]
    assert M.retraining(RULE, clean, {**ok, "rmse": 16.0}, 11.0)[0]                  # 16 > 1.45 x 11 = 15.95
    assert not M.retraining(RULE, clean, {**ok, "rmse": 15.9}, 11.0)[0]
    assert M.retraining(RULE, clean, {**ok, "lower_bound_miss_rate": 0.25}, 11.0)[0]
    assert not M.retraining(RULE, clean, {**ok, "lower_bound_miss_rate": 0.20}, 11.0)[0]   # limit is "above 20%"
    assert not M.retraining(RULE, clean, {**ok, "n_scored": 19, "rmse": 99.0}, 11.0)[0]    # too few labels yet


def test_committed_rule_was_chosen_within_budgets():
    r = M.load_rule()
    assert r["drift"]["alpha"] in (0.05, 0.01, 0.005, 0.001, 0.0005)
    assert max(r["drift"]["false_alarm_chosen_by_dataset"].values()) <= r["budgets"]["drift"]
    assert max(r["health"]["false_alarm_rmse_chosen"].values()) <= r["budgets"]["rmse"]
    assert max(r["health"]["false_alarm_miss_chosen"].values()) <= r["budgets"]["miss"]
    # +5% (the guide's test): always caught when it persists; caught in >= 99% of runs even when only the newest
    # shift is shifted (the 3 misses are late FD003 windows with 15-19 validation engines, Holm p 0.011-0.014)
    assert r["drift"]["power"]["persistent: +5%"]["min"] == 1.0 and r["drift"]["power"]["newest shift: +5%"]["min"] >= 0.99
    assert r["combined_false_alarm_upper_estimate"] <= 0.05


def test_engine_means_make_ties_exact_whatever_the_addition_order():
    # 0.1 + 0.2 + 0.3 and 0.3 + 0.2 + 0.1 differ in the last binary digit; rounded means must not
    a = pd.DataFrame({"unit": [1, 1, 1, 2, 2, 2], "s4": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]})
    m = M.engine_means(a, ["s4"])
    assert m.loc[1, "s4"] == m.loc[2, "s4"]
    rng = np.random.default_rng(3)
    vals = np.round(rng.normal(100, 1, size=(40, 3)), 2)      # 2-decimal readings, like NASA's
    fwd = pd.DataFrame({"unit": np.repeat(range(40), 3), "s4": vals.ravel()})
    rev = pd.DataFrame({"unit": np.repeat(range(40), 3), "s4": vals[:, ::-1].ravel()})
    ref = pd.DataFrame({"unit": np.repeat(range(100, 140), 3), "s4": np.round(rng.normal(100, 1, size=120), 2)})
    d1, d2 = M.drift(fwd, ref, 0.01, ["s4"]), M.drift(rev, ref, 0.01, ["s4"])
    assert d1.ks_stat.iloc[0] == d2.ks_stat.iloc[0]


def test_rounding_fix_does_not_change_the_pre_registered_rule():
    a, b = M.load_rule(), M.load_rule(M.RULE_FILE.with_name("monitoring_rule_recheck.json"))
    for part, key in (("drift", "alpha"), ("drift", "min_drifted_sensors"), ("health", "rmse_multiplier"),
                      ("health", "miss_rate_limit"), ("health", "min_labelled_engines")):
        assert a[part][key] == b[part][key]
