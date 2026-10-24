"""Phase 7 monitoring: when a label exists, model health, data drift, and the retraining rule.

Label timing (decision D4). An engine's true failure point is L = its last NASA test reading + NASA's true RUL.
In the replay, shift k means every engine has run k cycles. The failure of an engine is known (its failure is
in the plant's log) from shift L onwards, and only then can its predictions be scored: label_available(L, k)
is L <= k. Between its last reading and L the engine is silent but still running.

Model health (D5). Each labelled engine contributes ONE prediction: its last one (the official benchmark view),
scored against truth = L - cycle (capped at 125 for RMSE/MAE/coverage, as in Phases 4-6). The rule uses the
most recent HEALTH_WINDOW labelled engines; totals over all labelled engines are stored too.

Drift (D1, D6, D7). Every reading RECEIVED in the last WINDOW_SHIFTS shifts (rows the validator quarantined for
range are included: drift is about what the sensors send) is summarised per engine (the mean of that engine's
readings in the window), then compared sensor by sensor with the same summary of the TRAINING engines at the same
cycle numbers, using a two-sample KS test. Engines, not readings, are the independent units: consecutive readings
of one engine are near-copies, and treating them as independent made clean validation streams look drifted
(pre-registration evidence in pipeline/monitoring_rule.json). p-values are Holm-adjusted across the 14 sensors; a
sensor has drifted when its adjusted p-value is below ALPHA from the pre-registered rule file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from features.build import RUL_CAP
from pipeline.config import SENSORS

RULE_FILE = Path("pipeline/monitoring_rule.json")
WINDOW_SHIFTS = 3      # drift window: readings received in the last 3 shifts (~300 test readings per dataset)
MIN_ENGINES = 15       # no KS test with fewer engines than this on either side
HEALTH_WINDOW = 20     # model-health rule: the 20 most recently labelled engines
NEAR = 50              # near failure: true RUL <= 50 cycles (as in Phases 4-6)


def label_available(failure_cycle: float, shift: int) -> bool:
    return failure_cycle <= shift


def failures(last_cycle: dict, true_rul: dict) -> dict:
    """{engine: true failure cycle L} from the last observed cycle and NASA's (uncapped) true RUL."""
    return {int(e): int(last_cycle[e] + true_rul[e]) for e in last_cycle}


def health(last_preds: pd.DataFrame, fail: dict, shift: int, window: int = HEALTH_WINDOW) -> dict:
    """Score the last prediction of every engine whose label is available at `shift`.
    last_preds: engine_id, cycle, rul_point, rul_low, rul_high (one row per engine, its newest prediction)."""
    known = [e for e in last_preds.engine_id.astype(int) if label_available(fail[e], shift)]
    d = last_preds[last_preds.engine_id.isin(known)].copy()
    d["L"] = d.engine_id.map(fail)
    d = d.sort_values(["L", "engine_id"])                     # order in which the labels arrived
    d["true_rul"] = d.L - d.cycle
    d["truth"] = d.true_rul.clip(upper=RUL_CAP)
    out = {"n_labelled": len(d)}
    for tag, part in (("", d.tail(window)), ("cum_", d)):
        out[f"{tag}n_scored"] = len(part)
        if len(part) == 0:
            for k in ("rmse", "mae", "near_failure_mae", "coverage_90", "lower_bound_miss_rate"):
                out[f"{tag}{k}"] = None
            continue
        err = part.rul_point - part.truth
        near = part[part.true_rul <= NEAR]
        out[f"{tag}rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{tag}mae"] = float(np.mean(np.abs(err)))
        out[f"{tag}near_failure_mae"] = float(np.mean(np.abs(near.rul_point - near.truth))) if len(near) else None
        out[f"{tag}coverage_90"] = float(np.mean((part.rul_low <= part.truth) & (part.truth <= part.rul_high)))
        out[f"{tag}lower_bound_miss_rate"] = float(np.mean(part.true_rul < part.rul_low))
    return out


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (running maximum, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


MEAN_DECIMALS = 9      # NASA readings have <= 4 decimals; see engine_means


def engine_means(readings: pd.DataFrame, sensors=SENSORS) -> pd.DataFrame:
    """One row per engine: the mean of its readings (missing values skipped), rounded to MEAN_DECIMALS.

    Rounding matters for the KS test: many engines have exactly equal means (readings have 2-4 decimals), and
    without it the last binary digit of each mean depends on the order of the additions, which decides how tied
    values are ordered and moved the KS statistic by a whole step (found by the Phase 7 verifier). Nine decimals
    keep every digit of the data and make ties exact."""
    return readings.groupby("unit")[list(sensors)].mean().round(MEAN_DECIMALS)


def drift(recent: pd.DataFrame, reference: pd.DataFrame, alpha: float, sensors=SENSORS) -> pd.DataFrame:
    """KS test per sensor on per-engine means, Holm-adjusted. recent / reference are readings with a `unit`
    column. Returns one row per sensor; tested=False when either side has fewer than MIN_ENGINES engines."""
    ra, rb = engine_means(recent, sensors), engine_means(reference, sensors)
    rows = []
    for s in sensors:
        a, b = ra[s].dropna().to_numpy(float), rb[s].dropna().to_numpy(float)
        if min(len(a), len(b)) < MIN_ENGINES:
            rows.append({"sensor": s, "n_recent": len(a), "n_reference": len(b), "ks_stat": None, "p_value": None})
            continue
        r = ks_2samp(a, b, method="asymp")
        rows.append({"sensor": s, "n_recent": len(a), "n_reference": len(b), "ks_stat": float(r.statistic),
                     "p_value": float(r.pvalue)})
    out = pd.DataFrame(rows)
    out["tested"] = out.p_value.notna()
    out["p_holm"] = np.nan
    if out.tested.any():
        out.loc[out.tested, "p_holm"] = holm(out.loc[out.tested, "p_value"])
    out["drifted"] = out.tested & (out.p_holm < alpha)
    return out


def window_cycles(shift: int) -> tuple[int, int]:
    return max(1, shift - WINDOW_SHIFTS + 1), shift


def reference_for(train: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Training-engine readings at the same cycle numbers as the drift window (age-matched reference)."""
    lo, hi = window_cycles(shift)
    return train[(train.cycle >= lo) & (train.cycle <= hi)]


def load_rule(path: Path = RULE_FILE) -> dict:
    return json.loads(Path(path).read_text())


def retraining(rule: dict, drift_rows: pd.DataFrame, h: dict, ref_rmse: float) -> tuple[bool, list[str]]:
    """The pre-registered rule (pipeline/monitoring_rule.json). Returns (flag, reasons)."""
    reasons = []
    drifted = drift_rows[drift_rows.drifted].sensor.tolist()
    if len(drifted) >= rule["drift"]["min_drifted_sensors"]:
        reasons.append(f"drift in {', '.join(drifted)}")
    if h.get("n_scored", 0) >= rule["health"]["min_labelled_engines"] and h.get("rmse") is not None:
        if h["rmse"] > rule["health"]["rmse_multiplier"] * ref_rmse:
            reasons.append(f"rolling RMSE {h['rmse']:.2f} > {rule['health']['rmse_multiplier']:g} x validation {ref_rmse:.2f}")
        if h["lower_bound_miss_rate"] > rule["health"]["miss_rate_limit"]:
            reasons.append(f"lower-bound miss rate {h['lower_bound_miss_rate']:.0%} > {rule['health']['miss_rate_limit']:.0%}")
    return bool(reasons), reasons
