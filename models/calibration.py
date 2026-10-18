"""Phase 5b: the interval-calibration rule, fixed on FD001 validation BEFORE any FD003 modeling.

Run:  python -m models.calibration      (writes models/calibration_rule.json; reads only FD001 validation files)

Phase 5 scaled the MC-dropout interval (mean +/- k x sd) with one k chosen on ALL validation readings. On the
readings that matter, that k is too small: leave-engines-out it covers ~86% of test-like readings and ~83% of
near-failure readings at a nominal 90%. This script compares candidate rules on FD001 validation only and
records the chosen one. The committed JSON (and its git timestamp) is the pre-registration: FD003 is then
scored once with this rule, re-learning k on FD003's own validation engines with the same procedure.

A rule may only switch on what is known at prediction time: the PREDICTED remaining life, never the true one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("models/calibration_rule.json")
NOMINAL = 0.9
NEAR = 50            # cycles: the near-failure zone used throughout Phases 4-5
TEST_LIKE = (31, 145)  # FD001's test-like definition (cycle >= 31, true RUL <= 145), kept fixed for FD003
CHOSEN = "E"


def test_like(cycle, rul_true) -> np.ndarray:
    return (np.asarray(cycle) >= TEST_LIKE[0]) & (np.asarray(rul_true) <= TEST_LIKE[1])


def zone_scales(y, mean, std, cycle, rul_true, nominal=NOMINAL) -> tuple[float, float]:
    """Rule E: two factors learned on test-like validation readings, one for readings whose PREDICTED RUL is at
    most NEAR cycles and one for the rest. Returns (k_near, k_far)."""
    y, mean, std = (np.asarray(a, float) for a in (y, mean, std))
    z = np.abs(y - mean) / np.maximum(std, 1e-6)
    tl, pn = test_like(cycle, rul_true), mean <= NEAR
    return float(np.quantile(z[tl & pn], nominal)), float(np.quantile(z[tl & ~pn], nominal))


def zone_interval(mean, std, k_near, k_far) -> tuple[np.ndarray, np.ndarray]:
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    k = np.where(mean <= NEAR, k_near, k_far)
    return np.maximum(mean - k * std, 0.0), mean + k * std


def _candidates(z, tl, nf, pn):
    """Each candidate maps a training mask to a per-reading k (same length as z)."""
    q = lambda m: np.quantile(z[m], NOMINAL)
    one = lambda f: (lambda m: np.full(len(z), f(m)))
    return {
        "A": ("one k from all validation readings (Phase 5)", one(lambda m: q(m))),
        "B": ("one k from test-like readings", one(lambda m: q(m & tl))),
        "C": ("one k from readings with TRUE RUL <= 50 (not usable in deployment as a zone)", one(lambda m: q(m & nf))),
        "E": ("two k from test-like readings, switched on PREDICTED RUL <= 50",
              lambda m: np.where(pn, q(m & tl & pn), q(m & tl & ~pn))),
    }


def evaluate(v: pd.DataFrame) -> dict:
    """Leave-engines-out coverage of every candidate: each engine is scored with k learned on the others."""
    y, mean, std = v.RUL.to_numpy(float), v["MC mean"].to_numpy(float), v["MC std"].to_numpy(float)
    z = np.abs(y - mean) / np.maximum(std, 1e-6)
    tl, nf, pn = test_like(v.cycle, v.RUL_true), y <= NEAR, mean <= NEAR
    units = v.unit.to_numpy()
    out = {}
    for name, (desc, rule) in _candidates(z, tl, nf, pn).items():
        cov, width = np.zeros(len(z), bool), np.zeros(len(z))
        for u in np.unique(units):
            held = units == u
            k = rule(~held)
            cov[held] = (z <= k)[held]
            width[held] = (np.minimum(2 * k * std, mean + k * std))[held]  # lower edge clipped at 0
        k_all = rule(np.ones(len(z), bool))
        k_in = {"near": float(k_all[pn][0]), "far": float(k_all[~pn][0])} if name == "E" else float(k_all[0])
        out[name] = {"rule": desc, "k_in_sample": k_in, "loeo_coverage_all": float(cov.mean()), "loeo_coverage_testlike": float(cov[tl].mean()),
                     "loeo_coverage_near_failure": float(cov[nf].mean()), "mean_width_testlike": float(width[tl].mean()),
                     "mean_width_near_failure": float(width[nf].mean())}
    return out


def main() -> int:
    v = pd.read_csv("models/val_predictions_phase5.csv", float_precision="round_trip")
    ev = evaluate(v)
    for n, e in ev.items():
        print(f"{n}: test-like {e['loeo_coverage_testlike']:.1%}, near failure {e['loeo_coverage_near_failure']:.1%}, "
              f"width {e['mean_width_testlike']:.1f} / {e['mean_width_near_failure']:.1f}  ({e['rule']})")
    rule = {"chosen": CHOSEN, "nominal": NOMINAL, "near_zone_cycles": NEAR, "test_like": {"min_cycle": TEST_LIKE[0], "max_true_rul": TEST_LIKE[1]},
            "procedure": "k_near, k_far = 90th percentile of |RUL - MC mean| / MC sd over test-like validation readings with "
                         "predicted MC mean <= 50 and > 50; interval = MC mean +/- k x MC sd (lower edge clipped at 0), k "
                         "picked by the reading's predicted RUL. Re-learned on each dataset's own validation engines.",
            "criterion": "coverage closest to 90% on BOTH test-like and near-failure validation readings (leave-engines-out), "
                         "using only information available at prediction time",
            "evidence": "FD001 validation (20 engines), seed-0 LSTM MC passes from Phase 5; no test set used",
            "candidates": ev}
    OUT.write_text(json.dumps(rule, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} (chosen rule {CHOSEN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
