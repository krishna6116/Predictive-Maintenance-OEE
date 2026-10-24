"""Phase 7b monitoring rules: wear-adjusted drift with persistence, and life-matched model health.

Why: Phase 7's drift test compared raw sensor readings with training engines of the same AGE. Late in a replay the
test engines still reporting are healthier than training engines of that age, so normal wear looked like drift
(11.6% false alarms on clean test runs). Here every reading is first compared with what that sensor should read
for an engine of that age and PREDICTED remaining life:

  z = (reading - expected level) / expected spread          (standardised residual, per sensor)

ExpectedLevel learns both curves on the TRAINING engines (spline regression on predicted remaining life and age),
using out-of-fold predictions (models/oof.py), so the inputs look like the flow's predictions on unseen engines.

Drift: each engine's mean z over its readings in the last WINDOW_SHIFTS shifts (engines with all 3) is compared,
sensor by sensor, with the same summary over all windows of the training engines, by a KS statistic. Each training
engine counts once (its windows share its weight), and the p-value uses the number of training engines as the
reference size, because engines, not windows, are the independent units. Holm over the 14 sensors.

Decision per sensor (parameters chosen on validation, pipeline/monitoring_rule_v2.json):
  persistent drift  - adjusted p < alpha in at least K of the last M runs
  sudden fault      - adjusted p < alpha_strong in one run ("sensor fault suspected: check the sensor first")
Health: lower-bound miss rate of the last 20 labelled engines tested against the validation miss rate (binomial,
one-sided), and their mean squared error divided by the error expected for engines of the same life.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, kstwo
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from features.build import RUL_CAP
from pipeline.config import SENSORS

RULE_FILE = Path("pipeline/monitoring_rule_v2.json")
WINDOW_SHIFTS = 3
MIN_ENGINES = 15
HEALTH_WINDOW = 20
NEAR = 50
DECIMALS = 9          # exact ties (see pipeline/monitoring.engine_means)
MIN_SPREAD = 1e-6


def _design():
    return make_pipeline(ColumnTransformer([
        ("rul", SplineTransformer(n_knots=8, degree=3, extrapolation="constant"), [0]),
        ("age", SplineTransformer(n_knots=6, degree=3, extrapolation="constant"), [1])]), Ridge(alpha=1.0))


def _inputs(rhat, cycle) -> np.ndarray:
    return np.column_stack([np.clip(np.asarray(rhat, float), 0, RUL_CAP), np.asarray(cycle, float)])


class ExpectedLevel:
    """Expected level and spread of each sensor given predicted remaining life and age (fitted on training engines)."""

    def __init__(self, sensors=SENSORS):
        self.sensors, self.mean, self.spread = list(sensors), {}, {}

    def fit(self, readings: pd.DataFrame, rhat: str = "rhat") -> "ExpectedLevel":
        X = _inputs(readings[rhat], readings.cycle)
        for s in self.sensors:
            y = readings[s].to_numpy(float)
            self.mean[s] = _design().fit(X, y)
            resid = np.abs(y - self.mean[s].predict(X))
            self.spread[s] = _design().fit(X, resid)
        return self

    def z(self, readings: pd.DataFrame, rhat: str = "rhat") -> pd.DataFrame:
        """Standardised residuals (rounded to DECIMALS so ties are exact); keeps unit and cycle."""
        X = _inputs(readings[rhat], readings.cycle)
        out = readings[["unit", "cycle"]].copy()
        for s in self.sensors:
            sd = np.maximum(self.spread[s].predict(X) * np.sqrt(np.pi / 2), MIN_SPREAD)   # mean |dev| -> sd (normal)
            out[s] = np.round((readings[s].to_numpy(float) - self.mean[s].predict(X)) / sd, DECIMALS)
        return out


def window_means(z: pd.DataFrame, sensors=SENSORS) -> pd.DataFrame:
    """Per engine, the mean z over its readings in the window; only engines with all WINDOW_SHIFTS readings."""
    g = z.groupby("unit")
    full = g.cycle.transform("size") == WINDOW_SHIFTS
    return z[full].groupby("unit")[list(sensors)].mean().round(DECIMALS)


def reference_windows(z: pd.DataFrame, sensors=SENSORS) -> tuple[dict, int]:
    """Per-engine means over every run of WINDOW_SHIFTS consecutive cycles of the training engines.
    Returns ({sensor: (sorted values, weights)}, number of engines); each engine's windows share weight 1."""
    parts = []
    for u, g in z.sort_values(["unit", "cycle"]).groupby("unit"):
        r = g[list(sensors)].rolling(WINDOW_SHIFTS).mean().dropna()
        parts.append(r.assign(_w=1.0 / max(len(r), 1)))
    allw = pd.concat(parts)
    ref = {}
    for s in sensors:
        v = np.round(allw[s].to_numpy(float), DECIMALS)
        o = np.argsort(v, kind="stable")
        ref[s] = (v[o], allw._w.to_numpy(float)[o] / allw._w.sum())
    return ref, len(parts)


def ks_vs_reference(sample, ref_sorted, ref_w, n_ref) -> tuple[float, float]:
    """KS distance between the sample's ECDF and the (engine-weighted) reference ECDF, over all jump points;
    p-value from the Kolmogorov distribution with effective size n*m/(n+m), m = number of reference engines."""
    a = np.sort(np.asarray(sample, float))
    x = np.concatenate([a, ref_sorted])
    fa = np.searchsorted(a, x, side="right") / len(a)
    cw = np.concatenate([[0.0], np.cumsum(ref_w)])
    fb = cw[np.searchsorted(ref_sorted, x, side="right")]
    d = float(np.max(np.abs(fa - fb)))
    en = len(a) * n_ref / (len(a) + n_ref)
    return d, float(np.clip(kstwo.sf(d, np.round(en)), 0.0, 1.0))


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def drift(zwin: pd.DataFrame, ref: dict, n_ref: int, sensors=SENSORS) -> pd.DataFrame:
    """One run's drift test: zwin = z of the readings received in the window (unit, cycle, sensors)."""
    wm = window_means(zwin, sensors)
    rows = []
    for s in sensors:
        a = wm[s].dropna().to_numpy(float)
        if len(a) < MIN_ENGINES:
            rows.append({"sensor": s, "n_recent": len(a), "n_reference": n_ref, "ks_stat": np.nan, "p_value": np.nan})
            continue
        d, p = ks_vs_reference(a, *ref[s], n_ref)
        rows.append({"sensor": s, "n_recent": len(a), "n_reference": n_ref, "ks_stat": d, "p_value": p})
    out = pd.DataFrame(rows)
    out["tested"] = out.p_value.notna()
    out["p_holm"] = np.nan
    if out.tested.any():
        out.loc[out.tested, "p_holm"] = holm(out.loc[out.tested, "p_value"])
    return out


def drift_decision(history: list[dict], rule: dict) -> tuple[list[str], list[str]]:
    """history: per-run {sensor: adjusted p or None}, oldest first, ending with the current run.
    Returns (persistent sensors, sudden-fault sensors) under the rule's K-of-M and alpha_strong."""
    a, K, Mw, strong = rule["alpha"], rule["k"], rule["m"], rule["alpha_strong"]
    now, recent = history[-1], history[-Mw:]
    if not now:                              # this run could not be tested: no new evidence, no decision
        return [], []
    persistent = [s for s in SENSORS if sum(1 for h in recent if h.get(s) is not None and h[s] < a) >= K]
    sudden = [s for s in SENSORS if now.get(s) is not None and now[s] < strong and s not in persistent]
    return persistent, sudden


class ExpectedError:
    """Expected squared error of an engine's last prediction, as a function of its life (from out-of-fold
    predictions of training engines at test-like readings: cycle >= 31, true RUL <= 145)."""

    def fit(self, life, sq_err) -> "ExpectedError":
        self.model = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="constant"), Ridge(alpha=1.0))
        self.model.fit(np.asarray(life, float).reshape(-1, 1), np.asarray(sq_err, float))
        self.floor = float(np.quantile(sq_err, 0.1))
        return self

    def __call__(self, life) -> np.ndarray:
        return np.maximum(self.model.predict(np.asarray(life, float).reshape(-1, 1)), self.floor)


def health(last_preds: pd.DataFrame, fail: dict, shift: int, expected_error: ExpectedError, p0: float) -> dict:
    """Health on the last HEALTH_WINDOW labelled engines (labels only once the failure has happened).
    last_preds: engine_id, cycle, rul_point, rul_low, rul_high. fail: engine -> failure shift (on the plant clock),
    life: engine -> life in cycles."""
    known = [e for e in last_preds.engine_id.astype(int) if fail[e]["at"] <= shift]
    d = last_preds[last_preds.engine_id.isin(known)].copy()
    d["at"] = d.engine_id.map(lambda e: fail[e]["at"])
    d["life"] = d.engine_id.map(lambda e: fail[e]["life"])
    d = d.sort_values(["at", "engine_id"]).tail(HEALTH_WINDOW)
    out = {"n_labelled": len(known), "n_scored": len(d)}
    if not len(d):
        return {**out, "rmse": None, "error_ratio": None, "miss_rate": None, "miss_p": None, "coverage_90": None}
    true = d.life - d.cycle
    err2 = (d.rul_point - true.clip(upper=RUL_CAP)) ** 2
    misses = int((true < d.rul_low).sum())
    return {**out, "rmse": float(np.sqrt(err2.mean())), "error_ratio": float(err2.sum() / expected_error(d.life).sum()),
            "miss_rate": misses / len(d), "miss_p": float(binom.sf(misses - 1, len(d), p0)),
            "coverage_90": float(((d.rul_low <= true.clip(upper=RUL_CAP)) & (true.clip(upper=RUL_CAP) <= d.rul_high)).mean())}


def retraining(rule: dict, persistent: list, sudden: list, h: dict) -> tuple[bool, list[str]]:
    reasons = [f"persistent drift in {s}" for s in persistent]
    reasons += [f"sudden shift in {s}: sensor fault suspected, check the sensor before retraining" for s in sudden]
    hr = rule["health"]
    if h.get("n_scored", 0) >= hr["min_labelled_engines"]:
        if h["error_ratio"] > hr["error_ratio_limit"]:
            reasons.append(f"prediction error {h['error_ratio']:.2f} x the expected error for these engines' lives")
        if h["miss_p"] < hr["miss_alpha"]:
            reasons.append(f"lower-bound miss rate {h['miss_rate']:.0%} (binomial p {h['miss_p']:.3g} vs expected {hr['p0']:.1%})")
    return bool(reasons), reasons


def load_rule(path: Path = RULE_FILE) -> dict:
    return json.loads(Path(path).read_text())
