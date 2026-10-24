"""Phase 7 guide checks (Steps 7.6 and 7.7) on TEST data, after the replay. Measures only; nothing is tuned here.

Run:  python -m pipeline.check_phase7      (after python -m pipeline.flow --replay-all; needs the database)

1. Guide check through the real flow: in a throwaway schema, clean shifts 1-9, then shift 10 with +5% added to one
   sensor of every FD001 engine (the guide's test). That sensor must be flagged and the retraining flag set; the
   clean shifts must set nothing. Repeated for one in-range shift (+1 training sd) to show what the validator lets
   through. Uses the committed flow predictions (predictor "recorded"), so it runs anywhere.
2. Power on test windows: every 10th shift of the replay, the newest shift's readings of one sensor at a time are
   shifted (+1%, +2%, +5%, +0.5 sd, +1 sd) and the drift test is re-run on that window (~100 engines).
3. False alarms on the clean test replay, as recorded by the flow (out of sample; the rule was chosen on validation).
Writes pipeline/phase7_checks.json.
"""
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from pipeline import monitoring as M
from pipeline.config import FLOW_DATASETS, SENSORS

OUT = Path("pipeline/phase7_checks.json")
SIZES = (("pct", 0.01), ("pct", 0.02), ("pct", 0.05), ("sd", 0.5), ("sd", 1.0))


def label(kind, size):
    return f"+{size:.0%}" if kind == "pct" else f"+{size:g} sd"


def flow_check(sensor="s4", ds="FD001") -> dict:
    from prefect.testing.utilities import prefect_test_harness
    from pipeline import flow as F
    from pipeline.db import get_engine
    schema = f"phase7_check_{uuid.uuid4().hex[:8]}"
    engine = get_engine()
    out = {}
    try:
        with prefect_test_harness():
            runs = [F.shift_pipeline(shift=k, predictor="recorded", db_schema=schema) for k in range(1, 10)]
            five = F.shift_pipeline(shift=10, predictor="recorded", db_schema=schema,
                                    inject={"dataset": ds, "sensor": sensor, "pct": 0.05, "shifts": [10]})
            one_sd = F.shift_pipeline(shift=11, predictor="recorded", db_schema=schema,
                                      inject={"dataset": ds, "sensor": sensor, "sd": 1.0, "shifts": [11]})
        clean = [r[d] for r in runs for d in FLOW_DATASETS]
        out = {"dataset": ds, "sensor": sensor,
               "clean_runs": len(clean), "clean_runs_flagged": sum(r["needs_retraining"] for r in clean),
               "clean_runs_drifted": sum(bool(r["drifted"]) for r in clean),
               "plus5pct": {"quarantined": five[ds]["quarantined"], "readings": five[ds]["readings"],
                            "drifted": five[ds]["drifted"], "needs_retraining": five[ds]["needs_retraining"],
                            "other_dataset_flagged": five[[d for d in FLOW_DATASETS if d != ds][0]]["needs_retraining"]},
               "plus1sd": {"quarantined": one_sd[ds]["quarantined"], "readings": one_sd[ds]["readings"],
                           "drifted": one_sd[ds]["drifted"], "needs_retraining": one_sd[ds]["needs_retraining"]}}
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    return out


def test_window_power(alpha: float) -> dict:
    from pipeline import flow as F
    res = {}
    for kind, size in SIZES:
        hits, n = {s: 0 for s in SENSORS}, 0
        for ds in FLOW_DATASETS:
            test, train = F.test_file(ds), F.training_engines(ds)
            for k in range(10, int(test.cycle.max()) + 1, 10):
                lo, hi = M.window_cycles(k)
                rec = test[(test.cycle >= lo) & (test.cycle <= hi)]
                ref = M.reference_for(train, k)
                if min(rec.unit.nunique(), ref.unit.nunique()) < M.MIN_ENGINES or not (rec.cycle == k).any():
                    continue
                base = M.drift(rec, ref, alpha=1.0).p_value.to_numpy(float)
                n += 1
                newest = (rec.cycle == k).to_numpy()
                for j, s in enumerate(SENSORS):
                    x = rec[s].to_numpy(float).copy()
                    x[newest] += x[newest] * size if kind == "pct" else train[s].std() * size
                    d = M.drift(rec.assign(**{s: x}), ref, alpha=1.0)
                    p = base.copy()
                    p[j] = d.p_value.iloc[j]
                    hits[s] += bool(M.holm(p)[j] < alpha)
        rates = {s: h / n for s, h in hits.items()}
        res[f"newest shift: {label(kind, size)}"] = {"windows": n, "mean": float(np.mean(list(rates.values()))),
                                                    "min": min(rates.values()), "weakest_sensor": min(rates, key=rates.get),
                                                    "per_sensor": rates}
    return res


def replay_false_alarms() -> dict:
    d = pd.read_csv("pipeline/phase7_drift.csv")
    h = pd.read_csv("pipeline/phase7_health.csv")
    tested = d[d.tested].groupby("run_id").drifted.any()
    out = {"runs_with_drift_test": int(len(tested)), "runs_with_a_drifted_sensor": int(tested.sum()),
           "drift_false_alarm_rate": float(tested.mean()),
           "by_dataset": {ds: float(tested[tested.index.str.startswith(ds)].mean()) for ds in FLOW_DATASETS},
           "drifted_runs": tested[tested].index.tolist(),
           "runs_flagged_for_retraining": int(h.needs_retraining.sum()), "runs": int(len(h))}
    return out


def summarize(checks: dict) -> dict:
    """pipeline/results_phase7.json: the replay in numbers, generated from the exported tables."""
    import platform
    import prefect
    import sqlalchemy
    import torch
    h = pd.read_csv("pipeline/phase7_health.csv", float_precision="round_trip")
    runs = pd.read_csv("pipeline/phase7_runs.csv")
    alerts = pd.read_csv("pipeline/phase7_alerts.csv")
    preds = pd.read_csv("pipeline/phase7_predictions.csv", usecols=["dataset", "valid"])
    out = {"shifts": int(runs["shift"].max()), "runs": len(runs), "forecasts": len(preds),
           "invalid_forecasts": int((~preds.valid).sum()), "work_orders": len(alerts),
           "drift_false_alarm_rate_test": checks["replay"]["drift_false_alarm_rate"], "datasets": {}}
    for ds in FLOW_DATASETS:
        d = h[h.dataset == ds].sort_values("shift")
        lab = d[d.n_labelled > 0]
        full = d[d.n_scored >= M.HEALTH_WINDOW]
        fl = d[d.needs_retraining]
        reasons = {}
        for r in fl.retraining_reasons:
            for part in str(r).split("; "):
                key = part.split(" ")[0] + (" " + part.split(" ")[1] if part.startswith(("rolling", "lower")) else "")
                reasons[key] = reasons.get(key, 0) + 1
        fin = d.iloc[-1]
        out["datasets"][ds] = {
            "runs": len(d), "first_label_shift": int(lab["shift"].min()), "all_labelled_shift": int(d[d.n_labelled == 100]["shift"].min()),
            "rule_active_from_shift": int(full["shift"].min()),
            "final_cum": {"rmse": float(fin.cum_rmse), "coverage_90": float(fin.cum_coverage_90),
                          "lower_bound_miss_rate": float(fin.cum_lower_bound_miss_rate)},
            "rolling_while_active": {"rmse_max": float(full.rmse.max()), "rmse_min": float(full.rmse.min()),
                                     "miss_rate_max": float(full.lower_bound_miss_rate.max()),
                                     "coverage_min": float(full.coverage_90.min())},
            "flagged_runs": len(fl), "flagged_shifts": [int(x) for x in fl["shift"]], "flag_reasons": reasons,
            "work_orders": int((alerts.dataset == ds).sum())}
    out["environment"] = {"os": sys.platform, "platform": platform.platform(), "python": platform.python_version(),
                          "prefect": prefect.__version__, "sqlalchemy": sqlalchemy.__version__, "torch": torch.__version__}
    return out


def drift_diagnosis() -> dict:
    """Why clean test runs were flagged: compare the wear stage (true remaining life) of the test engines in each
    window with the training engines of the reference, and how strongly each flagged sensor tracks wear.
    A diagnosis after the fact; it changes no rule."""
    from pipeline import flow as F
    d = pd.read_csv("pipeline/phase7_drift.csv")
    flagged = d[d.tested].groupby(["dataset", "shift"]).drifted.any().reset_index()
    out = {"by_dataset": {}}
    corr = {s: [] for s in SENSORS}
    for ds in FLOW_DATASETS:
        test, train = F.test_file(ds).copy(), F.training_engines(ds).copy()
        train["rul"] = train.groupby("unit").cycle.transform("max") - train.cycle
        test["rul"] = test.unit.map(F.failure_log(ds)) - test.cycle
        for s in SENSORS:
            corr[s].append(abs(float(np.corrcoef(train[s], train.rul.clip(upper=125))[0, 1])))
        rows = []
        for _, r in flagged[flagged.dataset == ds].iterrows():
            lo, hi = M.window_cycles(int(r["shift"]))
            a = test[(test.cycle >= lo) & (test.cycle <= hi)].groupby("unit").rul.min()
            b = train[(train.cycle >= lo) & (train.cycle <= hi)].groupby("unit").rul.min()
            rows.append({"drift": bool(r.drifted), "test_rul": a.median(), "ref_rul": b.median(),
                         "test_near": float((a <= 50).mean()), "ref_near": float((b <= 50).mean())})
        t = pd.DataFrame(rows)
        out["by_dataset"][ds] = {("drift_runs" if k else "clean_runs"): {
            "runs": len(g), "median_rul_test": float(g.test_rul.median()), "median_rul_reference": float(g.ref_rul.median()),
            "share_near_failure_test": float(g.test_near.mean()), "share_near_failure_reference": float(g.ref_near.mean())}
            for k, g in t.groupby("drift")}
    wear = {s: float(np.mean(v)) for s, v in corr.items()}
    ranked = sorted(wear, key=wear.get, reverse=True)
    drifted = d[d.drifted].sensor.value_counts().to_dict()
    out.update({"sensor_rul_correlation": wear, "sensors_by_wear": ranked, "drift_count_by_sensor": drifted,
                "all_drifted_sensors_in_top_wear": sorted(drifted) and set(drifted) <= set(ranked[:len(drifted)]),
                "top_wear_n": len(drifted)})
    return out


def health_diagnosis() -> dict:
    """Labels arrive in order of failure, so the first window holds the shortest-lived engines. Compare them with the
    rest (each engine's last prediction, as the health rule scores it). A diagnosis after the fact; it changes no rule."""
    from pipeline import flow as F
    p = pd.read_csv("pipeline/phase7_predictions.csv", float_precision="round_trip")
    out = {}
    for ds in FLOW_DATASETS:
        L = F.failure_log(ds)
        q = p[p.dataset == ds].sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).set_index("engine_id")
        q["L"] = q.index.map(L)
        q["err2"] = (q.rul_point - (q.L - q.cycle).clip(upper=125)) ** 2
        q = q.sort_values(["L"], kind="stable")
        first, rest = q.head(M.HEALTH_WINDOW), q.iloc[M.HEALTH_WINDOW:]
        out[ds] = {"first_window": {"median_life": float(first.L.median()), "rmse": float(np.sqrt(first.err2.mean()))},
                   "other_engines": {"median_life": float(rest.L.median()), "rmse": float(np.sqrt(rest.err2.mean()))}}
    return out


def main() -> int:
    rule = M.load_rule()
    res = {"flow_check": flow_check(), "test_window_power": test_window_power(rule["drift"]["alpha"]),
           "replay": replay_false_alarms(), "drift_diagnosis": drift_diagnosis(), "health_diagnosis": health_diagnosis()}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path("pipeline/results_phase7.json").write_text(json.dumps(summarize(res), indent=2), encoding="utf-8")
    fc = res["flow_check"]
    print(f"guide check ({fc['dataset']} {fc['sensor']}): clean runs flagged {fc['clean_runs_flagged']}/{fc['clean_runs']}; "
          f"+5%: {fc['plus5pct']['quarantined']}/{fc['plus5pct']['readings']} quarantined, drifted {fc['plus5pct']['drifted']}, "
          f"flag {fc['plus5pct']['needs_retraining']}; +1 sd: quarantined {fc['plus1sd']['quarantined']}, drifted "
          f"{fc['plus1sd']['drifted']}, flag {fc['plus1sd']['needs_retraining']}")
    for k, v in res["test_window_power"].items():
        print(f"   test-window power {k:22s} mean {v['mean']:.0%} min {v['min']:.0%} ({v['weakest_sensor']}) over {v['windows']} windows")
    r = res["replay"]
    print(f"clean replay: {r['runs_with_a_drifted_sensor']}/{r['runs_with_drift_test']} runs flagged drift "
          f"({r['drift_false_alarm_rate']:.2%}); retraining flag in {r['runs_flagged_for_retraining']}/{r['runs']} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
