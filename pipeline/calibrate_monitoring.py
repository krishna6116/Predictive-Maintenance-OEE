"""Phase 7 pre-registration: choose the drift and retraining rules on VALIDATION engines only.

Run:  python -m pipeline.calibrate_monitoring     (writes pipeline/monitoring_rule.json; reads no test file)

Committed on its own before the flow touches any test data, the same way Phase 5b pre-registered rule E.

Drift. The 20 validation engines of each dataset are replayed shift by shift (all start at cycle 1, as in the flow).
Each shift, their readings from the last 3 shifts are averaged per engine and KS-tested against the same summary of
the 80 TRAINING engines at the same cycles, Holm-adjusted over 14 sensors. These streams are clean, so every run that flags a sensor is a false
alarm. ALPHA = the largest value in the grid whose per-run false-alarm rate is within the drift budget on each
dataset. Power: one sensor at a time is shifted by +1%, +2%, +5% of its reading (the guide's test) and by
+0.25, +0.5, +1 training sd (shifts small enough to pass the range validator), either over the whole window
(persistent drift) or in the newest shift only (a sudden fault).

Health. On unseen test engines the interval's k is learned from other engines, so each validation engine gets
bounds with k learned from the other 19 (leave-one-engine-out, rule E). A "window" is 20 engines drawn with
replacement, each scored at one random test-like reading (cycle >= 31, true RUL <= 145), the same view as a
test engine's last reading. 5,000 windows per dataset give the false-alarm rate of each candidate RMSE
multiplier (against the dataset's validation RMSE on test-like readings) and lower-bound miss-rate limit.
Each is the least strict value whose rate is within its budget on BOTH datasets (starting from the 30%
multiplier suggested in the brief).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import split_file
from models.calibration import test_like, zone_interval, zone_scales
from pipeline import monitoring as M
from pipeline.check_raw_data import load
from pipeline.config import RAW, SENSORS

CSV = {"float_precision": "round_trip"}
VAL = {"FD001": "models/val_predictions_phase5.csv", "FD003": "models/val_predictions_phase5b.csv"}
BUDGET = {"drift": 0.02, "rmse": 0.015, "miss": 0.015}          # per-run false-alarm budgets; total <= ~5%
ALPHAS = [0.05, 0.01, 0.005, 0.001, 0.0005]
MULTIPLIERS = [round(1.30 + 0.05 * i, 2) for i in range(35)]     # 1.30 .. 3.00
MISS_LIMITS = [round(j / M.HEALTH_WINDOW, 2) for j in range(1, M.HEALTH_WINDOW + 1)]
PCT_SHIFTS, SD_SHIFTS = (0.01, 0.02, 0.05), (0.25, 0.5, 1.0)
N_BOOT, SEED = 5000, 0


def streams(ds: str):
    """Yield (shift, recent window of validation readings, newest-shift mask, age-matched training reference)."""
    split = json.loads(Path(split_file(ds)).read_text())
    raw = load(RAW / f"train_{ds}.txt")
    val, train = raw[raw.unit.isin(split["val"])], raw[raw.unit.isin(split["train"])]
    for k in range(1, int(val.cycle.max()) + 1):
        lo, hi = M.window_cycles(k)
        rec = val[(val.cycle >= lo) & (val.cycle <= hi)]
        ref = M.reference_for(train, k)
        if min(rec.unit.nunique(), ref.unit.nunique()) >= M.MIN_ENGINES:
            yield k, rec, (rec.cycle == k).to_numpy(), ref, train[SENSORS].std()


def calibrate_drift() -> dict:
    runs = []
    for ds in VAL:
        for k, rec, newest, ref, sd in streams(ds):
            clean = M.drift(rec, ref, alpha=1.0)                  # alpha irrelevant here: keep the p-values
            runs.append({"ds": ds, "k": k, "rec": rec, "newest": newest, "ref": ref, "sd": sd,
                         "p": clean.p_value.to_numpy(float)})
    fa_ds = {a: {ds: float(np.mean([(M.holm(r["p"]) < a).any() for r in runs if r["ds"] == ds])) for ds in VAL} for a in ALPHAS}
    fa = {a: float(np.mean([(M.holm(r["p"]) < a).any() for r in runs])) for a in ALPHAS}
    ok = [a for a in ALPHAS if max(fa_ds[a].values()) <= BUDGET["drift"]]   # within budget on EACH dataset
    if not ok:
        raise SystemExit(f"no alpha meets the drift budget: {fa_ds}")
    alpha = max(ok)
    per_ds = fa_ds[alpha]

    from scipy.stats import ks_2samp
    power = {}
    sizes = [("pct", s) for s in PCT_SHIFTS] + [("sd", s) for s in SD_SHIFTS]
    for mode in ("persistent", "newest shift"):
        for kind, size in sizes:
            hits = {s: [] for s in SENSORS}
            for r in runs:
                for j, s in enumerate(SENSORS):
                    x = r["rec"][s].to_numpy(float).copy()
                    mask = np.ones(len(x), bool) if mode == "persistent" else r["newest"]
                    x[mask] += x[mask] * size if kind == "pct" else r["sd"][s] * size
                    p = r["p"].copy()
                    shifted = r["rec"].assign(**{s: x})
                    p[j] = ks_2samp(M.engine_means(shifted)[s].to_numpy(float), M.engine_means(r["ref"])[s].to_numpy(float),
                                    method="asymp").pvalue
                    hits[s].append(M.holm(p)[j] < alpha)
            label = f"+{size:.0%}" if kind == "pct" else f"+{size:g} sd"
            rates = {s: float(np.mean(v)) for s, v in hits.items()}
            power[f"{mode}: {label}"] = {"mean": float(np.mean(list(rates.values()))), "min": min(rates.values()),
                                         "weakest_sensor": min(rates, key=rates.get), "per_sensor": rates}
    return {"alpha": alpha, "min_drifted_sensors": 1, "unit": "per-engine mean over the window",
            "false_alarm_by_alpha": {str(a): v for a, v in fa.items()},
            "false_alarm_by_alpha_and_dataset": {str(a): v for a, v in fa_ds.items()},
            "rejected_method": "reading-level KS (each reading as an independent sample): false alarms on clean "
                               "validation streams 9.9% at alpha 0.05 and still 2.6% at 0.0005, all in FD003 runs where "
                               "about 19 engines each gave 3 near-identical readings",
            "false_alarm_chosen": fa[alpha], "false_alarm_chosen_by_dataset": per_ds, "runs": len(runs),
            "window_shifts": M.WINDOW_SHIFTS, "min_engines": M.MIN_ENGINES, "power": power}


def loeo_bounds(v: pd.DataFrame) -> pd.DataFrame:
    """Rule-E bounds for each validation engine with k learned on the other engines."""
    parts = []
    for u in sorted(v.unit.unique()):
        oth, me = v[v.unit != u], v[v.unit == u]
        kn, kf = zone_scales(oth.RUL, oth["MC mean"], oth["MC std"], oth.cycle, oth.RUL_true)
        lo, hi = zone_interval(me["MC mean"], me["MC std"], kn, kf)
        parts.append(me.assign(low=lo, high=hi))
    return pd.concat(parts)


def calibrate_health() -> dict:
    rng = np.random.default_rng(SEED)
    out, ratios, misses = {}, {}, {}
    for ds, f in VAL.items():
        v = loeo_bounds(pd.read_csv(f, **CSV))
        tl = v[test_like(v.cycle, v.RUL_true)]
        ref = float(np.sqrt(np.mean((tl.LSTM - tl.RUL) ** 2)))
        by_unit = [g for _, g in tl.groupby("unit")]
        r_, m_ = np.empty(N_BOOT), np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = [g.iloc[rng.integers(len(g))] for g in (by_unit[i] for i in rng.integers(len(by_unit), size=M.HEALTH_WINDOW))]
            w = pd.DataFrame(pick)
            r_[b] = np.sqrt(np.mean((w.LSTM - w.RUL) ** 2)) / ref
            m_[b] = np.mean(w.RUL_true < w.low)
        ratios[ds], misses[ds] = r_, m_
        out[ds] = {"reference_rmse": ref, "loeo_miss_rate_testlike": float(np.mean(tl.RUL_true < tl.low)),
                   "loeo_coverage_testlike": float(np.mean((tl.low <= tl.RUL) & (tl.RUL <= tl.high))),
                   "rmse_ratio_quantiles": {q: float(np.quantile(r_, q)) for q in (0.5, 0.9, 0.95, 0.99)}}
    fa_m = {m: {ds: float(np.mean(ratios[ds] > m)) for ds in VAL} for m in MULTIPLIERS}
    fa_l = {l: {ds: float(np.mean(misses[ds] > l)) for ds in VAL} for l in MISS_LIMITS}
    mult = next(m for m in MULTIPLIERS if max(fa_m[m].values()) <= BUDGET["rmse"])
    lim = next(l for l in MISS_LIMITS if max(fa_l[l].values()) <= BUDGET["miss"])
    return {"rmse_multiplier": mult, "miss_rate_limit": lim, "min_labelled_engines": M.HEALTH_WINDOW,
            "window_engines": M.HEALTH_WINDOW, "reference": out,
            "false_alarm_rmse_chosen": fa_m[mult], "false_alarm_rmse_at_1.30": fa_m[1.3],
            "false_alarm_miss_chosen": fa_l[lim]}


def main() -> int:
    d, h = calibrate_drift(), calibrate_health()
    fa = [d["false_alarm_chosen"], max(h["false_alarm_rmse_chosen"].values()), max(h["false_alarm_miss_chosen"].values())]
    rule = {"pre_registered": "chosen on validation and training engines only; no test file was read",
            "label_rule": "an engine's label is available from shift L = last observed cycle + true RUL (L <= shift)",
            "budgets": BUDGET, "drift": d, "health": h,
            "retraining_rule": "flag when >= 1 sensor drifted (Holm-adjusted p < alpha) OR rolling RMSE of the last 20 "
                               "labelled engines > multiplier x validation RMSE OR their lower-bound miss rate > limit; "
                               "the health parts need >= 20 labelled engines",
            "combined_false_alarm_upper_estimate": 1 - float(np.prod([1 - x for x in fa]))}
    M.RULE_FILE.write_text(json.dumps(rule, indent=2), encoding="utf-8")
    print(f"drift: alpha {d['alpha']} -> false alarms {d['false_alarm_chosen']:.2%} of {d['runs']} clean runs "
          f"({ {k: f'{v:.2%}' for k, v in d['false_alarm_chosen_by_dataset'].items()} }); by alpha "
          f"{ {k: f'{v:.1%}' for k, v in d['false_alarm_by_alpha'].items()} }")
    for k, v in d["power"].items():
        print(f"   power {k:28s} mean {v['mean']:.0%}  min {v['min']:.0%} ({v['weakest_sensor']})")
    print(f"health: RMSE multiplier {h['rmse_multiplier']} (FA {h['false_alarm_rmse_chosen']}; at 1.30: "
          f"{h['false_alarm_rmse_at_1.30']}), miss limit {h['miss_rate_limit']} (FA {h['false_alarm_miss_chosen']})")
    for ds, r in h["reference"].items():
        print(f"   {ds}: reference RMSE {r['reference_rmse']:.2f}, LOEO miss rate {r['loeo_miss_rate_testlike']:.1%}, "
              f"coverage {r['loeo_coverage_testlike']:.1%}")
    print(f"combined false-alarm estimate {rule['combined_false_alarm_upper_estimate']:.2%}; wrote {M.RULE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
