"""Phase 7b checks on the fresh fleets FD002/FD004 with the PRE-REGISTERED rule (pipeline/monitoring_rule_v2.json).

Run:  python -m pipeline.check_phase7b       (after the fresh synchronized replay; needs the database)

C2 (the guide's test): in a throwaway schema, clean shifts 1-9 of the synchronized fleet, then shift 10 with +5% on
s4 of every engine (s4, as in Phase 7; shift 10 so that every engine reports). Pass: needs_retraining is set in shift
10 with a reason naming s4, and the clean shifts before it set nothing. Runs the deployed model (predictor "model"),
so the shifted readings that pass validation also change the model's predictions, as they would in the plant.
C1 is read from the four clean replays (results/phase7b/<scenario>/phase7_health.csv).
Writes results/phase7b/checks.json.
"""
import json
import sys
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import text

OUT = Path("results/phase7b")
FRESH = ("FD002", "FD004")


def c1() -> dict:
    out = {}
    for sc in ("fresh_synchronized", "fresh_staggered", "posthoc_synchronized", "posthoc_staggered"):
        h = pd.read_csv(OUT / sc / "phase7_health.csv")
        for ds, g in h.groupby("dataset"):
            out[f"{ds} {sc.split('_')[1]}"] = {"runs": len(g), "flagged": int(g.needs_retraining.sum()),
                                               "rate": float(g.needs_retraining.mean()),
                                               "kind": "confirmation" if sc.startswith("fresh") else "post hoc"}
    return out


def c2(ds: str, sensor: str = "s4", shift: int = 10) -> dict:
    from prefect.testing.utilities import prefect_test_harness
    from pipeline import flow as F
    from pipeline.db import get_engine
    schema, engine = f"c2_{uuid.uuid4().hex[:6]}", get_engine()
    kw = dict(predictor="model", db_schema=schema, datasets=[ds], fleet="synchronized", monitor="v2")
    try:
        with prefect_test_harness():
            clean = [F.shift_pipeline(shift=k, **kw)[ds] for k in range(1, shift)]
            hit = F.shift_pipeline(shift=shift, inject={"dataset": ds, "sensor": sensor, "pct": 0.05, "shifts": [shift]}, **kw)[ds]
        with engine.begin() as c:
            why = c.execute(text(f'SELECT retraining_reasons FROM "{schema}".fact_model_health WHERE shift = :s'), {"s": shift}).scalar()
    finally:
        with engine.begin() as c:
            c.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    passed = bool(hit["needs_retraining"]) and sensor in (why or "") and not any(r["needs_retraining"] for r in clean)
    return {"dataset": ds, "sensor": sensor, "shift": shift, "readings": hit["readings"], "quarantined": hit["quarantined"],
            "drifted": hit["drifted"], "needs_retraining": bool(hit["needs_retraining"]), "reasons": why,
            "clean_runs_flagged": sum(r["needs_retraining"] for r in clean), "passed": passed}


def main() -> int:
    path = OUT / "checks.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["C1"] = c1()
    res["C2"] = {ds: c2(ds) for ds in FRESH}
    path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    for k, v in res["C1"].items():
        print(f"C1 {k}: {v['flagged']}/{v['runs']} = {v['rate']:.1%} ({v['kind']})")
    for ds, v in res["C2"].items():
        print(f"C2 {ds}: passed {v['passed']}; {v['quarantined']}/{v['readings']} quarantined, drifted {v['drifted']}, reasons: {v['reasons']}")
    return 0


if __name__ == "__main__" and not {"--closed-loop", "--diagnosis"} & set(sys.argv):
    sys.exit(main())


def closed_loop() -> dict:
    """Censoring in the closed loop (results/phase7b/fresh_closed_loop, synchronized). Labels arrive only for engines
    that failed before their service. A serviced engine is censored, but not uninformative: it survived until its
    service, so its true remaining life at its last forecast is at least (service cycle - forecast cycle). If that is
    already >= the forecast's lower bound, it cannot have been a lower-bound miss. This gives bounds on the miss rate
    over ALL engines that reached an outcome, checked here against NASA's truth (which a plant would not have)."""
    from pipeline import flow as F
    d = OUT / "fresh_closed_loop"
    p = pd.read_csv(d / "phase7_predictions.csv", float_precision="round_trip")
    a = pd.read_csv(d / "phase7_alerts.csv")
    out = {}
    for ds in FRESH:
        L = F.failure_log(ds)
        pv = p[(p.dataset == ds) & p.valid].sort_values(["engine_id", "cycle"])
        orders = a[a.dataset == ds].set_index("engine_id")
        failed_lab = miss_f = cens = sure_ok = unknown = true_miss = 0
        for e, g in pv.groupby("engine_id"):
            e = int(e)
            service = int(orders.loc[e, "scheduled_day"]) * 2 if e in orders.index else None
            if service is not None and service < L[e]:                       # serviced first: censored
                last = g[g.cycle <= service].iloc[-1]
                cens += 1
                if service - last.cycle >= last.rul_low:
                    sure_ok += 1
                else:
                    unknown += 1
            else:                                                            # failed: label available
                last = g.iloc[-1]
                failed_lab += 1
                miss_f += int(L[e] - last.cycle < last.rul_low)
            true_miss += int(L[e] - last.cycle < last.rul_low)               # NASA truth, for checking only
        n = failed_lab + cens
        out[ds] = {"engines": n, "failed_labelled": failed_lab, "censored": cens, "censored_certainly_not_miss": sure_ok,
                   "censored_undetermined": unknown, "miss_rate_from_labels_only": miss_f / max(failed_lab, 1),
                   "miss_rate_bounds_all_engines": [miss_f / n, (miss_f + unknown) / n],
                   "true_miss_rate_same_forecasts": true_miss / n}
    return out


if __name__ == "__main__" and "--closed-loop" in sys.argv:
    path = OUT / "checks.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["closed_loop"] = closed_loop()
    path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    for ds, v in res["closed_loop"].items():
        print(ds, json.dumps(v))


def diagnosis() -> dict:
    """After the fact, changes no rule. (1) The spread defect: readings whose standardised residual exceeds 20 under the
    pre-registered spread model (v2) vs the post-hoc log-scale fix (v2.1), and the smallest predicted spread.
    (2) FD004's flagged windows: their mean residual pattern vs the end-of-life residual signature of the two groups
    of FD004 training engines split by the sign of s20's residual in their last 50 cycles (a proxy for the two fault
    modes, which NASA does not label)."""
    import numpy as np
    from features.build import load_json
    from pipeline import flow as F
    from pipeline import monitoring2 as M2
    from pipeline.calibrate_monitoring2 import load_dataset
    from pipeline.conditions import CONDITION_DATASETS, normalise, normalised_train, stats_file
    from pipeline.config import SENSORS
    src = {"FD001": "posthoc_synchronized", "FD003": "posthoc_synchronized", "FD002": "fresh_synchronized", "FD004": "fresh_synchronized"}
    out = {"spread": {}}
    for ds, sc in src.items():
        p = pd.read_csv(OUT / sc / "phase7_predictions.csv", float_precision="round_trip")
        p = p[(p.dataset == ds) & p.valid].rename(columns={"engine_id": "unit", "rul_point": "rhat"})[["unit", "cycle", "rhat"]]
        t = F.test_file(ds)
        t = normalise(t, load_json(stats_file(ds))) if ds in CONDITION_DATASETS else t
        df = t.merge(p, on=["unit", "cycle"])
        X = M2._inputs(df.rhat, df.cycle)
        row = {"readings": len(df)}
        for v in ("v2", "v2.1"):
            m = F.v2_models(ds, v)
            z = m["level"].z(df)
            row[v] = {"readings_abs_z_over_20": int((z[SENSORS].abs() > 20).any(axis=1).sum()),
                      "max_abs_z": float(z[SENSORS].abs().max().max()),
                      "smallest_predicted_spread": float(min(m["level"].mad(s, X).min() for s in SENSORS))}
        out["spread"][ds] = row
    m = F.v2_models("FD004", "v2.1")
    d = load_dataset("FD004", "models/val_predictions_FD004.csv", "models/oof_predictions_FD004.csv", normalised_train("FD004"))
    tr = d["tr"]
    z = m["level"].z(tr)
    z["rul"] = tr.groupby("unit").cycle.transform("max") - tr.cycle
    late = z[z.rul <= 50].groupby("unit")[SENSORS].mean()
    mode = np.sign(late["s20"])
    sig = {int(k): late[mode == k][SENSORS].mean() for k in (-1, 1)}
    h = pd.read_csv(OUT / "v21_fresh_synchronized" / "phase7_health.csv")
    shifts = sorted(h[(h.dataset == "FD004") & h.needs_retraining]["shift"])
    p = pd.read_csv(OUT / "fresh_synchronized" / "phase7_predictions.csv", float_precision="round_trip")
    p = p[(p.dataset == "FD004") & p.valid].rename(columns={"engine_id": "unit", "rul_point": "rhat"})[["unit", "cycle", "rhat"]]
    tz = m["level"].z(normalise(F.test_file("FD004"), load_json(stats_file("FD004"))).merge(p, on=["unit", "cycle"]))
    fz = tz[tz.cycle.isin(shifts)][SENSORS].mean()
    out["fd004"] = {"flagged_shifts_v21_synchronized": [int(shifts[0]), int(shifts[-1])] if shifts else None,
                    "flagged_runs": len(shifts), "training_engines_by_proxy_mode": {str(int(k)): int(v) for k, v in mode.value_counts().items()},
                    "flagged_window_mean_z": {s: float(v) for s, v in fz.items()},
                    "correlation_with_mode_signature": {str(k): float(np.corrcoef(fz, sig[k])[0, 1]) for k in sig}}
    return out


if __name__ == "__main__" and "--diagnosis" in sys.argv:
    path = OUT / "checks.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["diagnosis"] = diagnosis()
    path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res["diagnosis"]["fd004"]["correlation_with_mode_signature"]),
          {ds: (v["v2"]["readings_abs_z_over_20"], v["v2.1"]["readings_abs_z_over_20"]) for ds, v in res["diagnosis"]["spread"].items()})
