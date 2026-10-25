"""Phase 7b monitoring: wear-adjusted residuals, KS against an engine-weighted reference, persistence, health."""
import numpy as np
import pandas as pd
import pytest

from pipeline import monitoring2 as M2
from pipeline.calibrate_monitoring2 import decide
from pipeline.config import SENSORS


def fleet(n_eng, life=200, wear=True, shift=0.0, seed=0):
    """Synthetic engines: every sensor rises as remaining life falls (wear), plus noise; rhat = true RUL + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_eng):
        L = int(life + rng.integers(-40, 40))
        for c in range(1, L + 1):
            r = L - c
            level = (5.0 * np.exp(-r / 30.0) if wear else 0.0)
            rows.append({"unit": u, "cycle": c, "rhat": max(0.0, min(125.0, r + rng.normal(0, 5))),
                         **{s: 100 + level + rng.normal(0, 1) + shift for s in SENSORS}})
    return pd.DataFrame(rows)


def test_residuals_remove_wear():
    tr = fleet(40, seed=1)
    lev = M2.ExpectedLevel().fit(tr)
    z = lev.z(tr)
    near, far = z[tr.rhat < 20], z[tr.rhat > 100]
    assert abs(near.s4.mean()) < 0.3 and abs(far.s4.mean()) < 0.3      # raw s4 differs by ~5 units between them
    assert 0.6 < z.s4.std() < 1.4


def test_healthy_survivors_do_not_look_drifted_but_a_real_shift_does():
    tr = fleet(80, seed=2)
    lev = M2.ExpectedLevel().fit(tr)
    ref, n_ref = M2.reference_windows(lev.z(tr))
    test = fleet(100, seed=3)
    # survivors: engines still far from failure (the selection that broke Phase 7)
    healthy = test[(test.groupby("unit").cycle.transform("max") - test.cycle) > 60]
    win = healthy.groupby("unit").tail(M2.WINDOW_SHIFTS)
    d = M2.drift(lev.z(win), ref, n_ref)
    assert (d.p_holm > 0.01).all()
    shifted = win.assign(s4=win.s4 + 1.0)
    d2 = M2.drift(lev.z(shifted), ref, n_ref).set_index("sensor")
    assert d2.loc["s4", "p_holm"] < 1e-6 and (d2.drop("s4").p_holm > 0.01).all()


def test_engines_with_a_short_window_are_left_out():
    z = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2], **{s: [0.0] * 5 for s in SENSORS}})
    assert list(M2.window_means(z).index) == [1]


def test_reference_weights_each_engine_once():
    z = pd.DataFrame({"unit": [1] * 10 + [2] * 4, "cycle": list(range(1, 11)) + list(range(1, 5)),
                      **{s: [0.0] * 10 + [1.0] * 4 for s in SENSORS}})
    ref, n = M2.reference_windows(z)
    v, w = ref["s4"]
    assert n == 2 and np.isclose(w[v == 0.0].sum(), 0.5) and np.isclose(w[v == 1.0].sum(), 0.5)


def test_ks_against_itself_is_not_significant_and_far_is():
    rng = np.random.default_rng(0)
    r = np.sort(rng.normal(size=500))
    w = np.full(500, 1 / 500)
    assert M2.ks_vs_reference(rng.normal(size=60), r, w, 80)[1] > 0.01
    d, p = M2.ks_vs_reference(rng.normal(3, 1, size=60), r, w, 80)
    assert d > 0.8 and p < 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_vectorised_decision_equals_flow_decision(seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 0.03, size=(40, len(SENSORS))) ** 2 * 50
    A[rng.random(40) < 0.2] = np.nan                         # untested runs
    rule = {"alpha": 0.01, "k": 2, "m": 3, "alpha_strong": 1e-4}
    t, f = decide(A, rule["alpha"], rule["k"], rule["m"], rule["alpha_strong"])
    hist = []
    for i in range(len(A)):
        hist.append({} if np.isnan(A[i]).any() else dict(zip(SENSORS, A[i])))
        per, sud = M2.drift_decision(hist, rule)
        assert set(per) | set(sud) == ({s for j, s in enumerate(SENSORS) if f[i, j]} if t[i] else set())


def test_health_uses_only_failed_engines_and_life_matched_error():
    lp = pd.DataFrame({"engine_id": [1, 2, 3], "cycle": [50, 50, 50], "rul_point": [10.0, 30.0, 40.0],
                       "rul_low": [5.0, 25.0, 60.0], "rul_high": [20.0, 40.0, 90.0]})
    fail = {1: {"at": 60, "life": 60}, 2: {"at": 90, "life": 90}, 3: {"at": 500, "life": 500}}
    err = M2.ExpectedError().fit(np.array([60, 90, 150, 200, 300] * 5), np.array([100.0] * 25))
    h = M2.health(lp, fail, shift=100, expected_error=err, p0=0.05)
    assert h["n_labelled"] == 2 and h["n_scored"] == 2                 # engine 3 has not failed yet
    assert np.isclose(h["rmse"], np.sqrt((0 + 100) / 2))              # truths 10 and 40
    assert np.isclose(h["error_ratio"], 100 / 200, atol=0.05)
    assert h["miss_rate"] == 0.0 and np.isclose(h["miss_p"], 1.0)


def test_retraining_reasons_distinguish_sudden_and_persistent():
    rule = {"health": {"min_labelled_engines": 20, "error_ratio_limit": 1.7, "miss_alpha": 0.01, "p0": 0.05}}
    flag, why = M2.retraining(rule, ["s4"], ["s11"], {"n_scored": 5})
    assert flag and why[0] == "persistent drift in s4" and "sensor fault suspected" in why[1]
    assert M2.retraining(rule, [], [], {"n_scored": 20, "error_ratio": 1.0, "miss_p": 0.5, "miss_rate": 0.05}) == (False, [])


def test_log_spread_stays_positive_where_a_linear_fit_goes_negative():
    # noise shrinks steeply with age and few readings are old: a linear spread fit dips below zero there
    rng = np.random.default_rng(4)
    rows = []
    for u in range(60):
        L = 120 if u < 55 else 400                                   # five long-lived engines: a sparse corner
        for c in range(1, L + 1):
            r = L - c
            rows.append({"unit": u, "cycle": c, "rhat": float(min(125, r)),
                         **{s: 100 + rng.normal(0, 3.0 * np.exp(-c / 40) + 0.01) for s in SENSORS}})
    tr = pd.DataFrame(rows)
    lin, log = M2.ExpectedLevel().fit(tr), M2.ExpectedLevel(spread="log").fit(tr)
    X = M2._inputs(tr.rhat, tr.cycle)
    assert lin.kind == "linear" and M2.SPREAD["v2"] == "linear"                       # pre-registered default unchanged
    assert (log.mad("s4", X) > 0).all()
    assert np.isclose(log.mad("s4", X).mean(), np.abs(tr.s4 - lin.mean["s4"].predict(X)).mean(), rtol=0.05)
    assert np.abs(log.z(tr)[SENSORS]).max().max() < 50


def test_rule_variants_resolve_to_their_files():
    assert M2.RULE_FILES["v2"].name == "monitoring_rule_v2.json" and M2.RULE_FILES["v2.1"].name == "monitoring_rule_v2_1.json"
    assert M2.load_rule("v2") == M2.load_rule(M2.RULE_FILE)
