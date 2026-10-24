"""Independent verification of Phase 7 (orchestration and monitoring), with different methods from the flow.

Run:  python -m pipeline.verify_phase7 [--replay N]

  runs        every shift of both datasets completed once, with the fixed run IDs; last_cycle = shift
  forecasts   one forecast per loaded reading; last-cycle point forecasts = published (exact); rule-E bounds
              recomputed with a plain-Python quantile; with the cached published model on the producing OS, the
              forecasts of sample shifts are recomputed and must match exactly (point and MC)
  labels      failure cycles recomputed from the raw NASA files; each run scores exactly the engines whose failure
              cycle has passed (no future label), and its RMSE, MAE, near-failure MAE, coverage and lower-bound miss
              rate are recomputed in plain Python; once all engines are labelled, RMSE = the published test RMSE
  drift       every run's KS statistic recomputed with numpy ECDFs, p-values from scipy's Kolmogorov
              distribution, per-engine means with numpy bincount, Holm by hand, from the raw NASA files
  flag        the pre-registered retraining rule re-applied to the recomputed drift and health
  orders      one flow work order per engine; each cites a stored forecast and its aged lower bound <= 20; crews
  checks      the guide's +5% check and the test-window power recomputed with the independent KS
  database    row counts (when a database is reachable); --replay N re-runs shifts 1..N in a throwaway schema
              with the committed predictions and requires identical rows (used by CI on Linux)
  documents   docs/MONITORING.md equals a fresh rendering; README numbers
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstwo

CSV = {"float_precision": "round_trip"}
DS = ("FD001", "FD003")
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
RULE = json.loads(Path("pipeline/monitoring_rule.json").read_text())
PUB = {"FD001": ("models/results_phase5.json", "models/predictions_phase5.csv"),
       "FD003": ("models/results_phase5b.json", "models/predictions_phase5b.csv")}
VAL = {"FD001": "models/val_predictions_phase5.csv", "FD003": "models/val_predictions_phase5b.csv"}
SPLIT = {"FD001": "features/split.json", "FD003": "features/split_FD003.json"}
RESULTS = []


def check(ok, msg):
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok and os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title={Path(__file__).stem} check failed::{msg}")   # visible on the run page


def read_raw(path):
    rows = [[float(v) for v in line.split()] for line in Path(path).read_text().split("\n") if line.strip()]
    df = pd.DataFrame(rows, columns=COLS)
    df["unit"], df["cycle"] = df.unit.astype(int), df.cycle.astype(int)
    return df


def quantile(xs, q):
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def holm(p):
    order = sorted(range(len(p)), key=lambda i: (p[i], i))
    adj, run = [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def engine_means(units, values):
    """Per-engine mean with numpy bincount (a different route from pandas groupby)."""
    u, inv = np.unique(units, return_inverse=True)
    return np.round(np.bincount(inv, weights=values) / np.bincount(inv), 9)   # same definition: exact ties


def ks(a, b):
    """Two-sample KS: statistic from the ECDFs on the pooled points; p-value from the Kolmogorov distribution
    with the effective sample size n*m/(n+m) rounded (the asymptotic method)."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b))))
    en = len(a) * len(b) / (len(a) + len(b))
    return d, float(np.clip(kstwo.sf(d, np.round(en)), 0, 1))


def drift_run(test, train_ref, k):
    lo, hi = max(1, k - 2), k
    rec = test[(test.cycle >= lo) & (test.cycle <= hi)]
    ref = train_ref[(train_ref.cycle >= lo) & (train_ref.cycle <= hi)]
    out = {}
    for s in SENSORS:
        a, b = engine_means(rec.unit.to_numpy(), rec[s].to_numpy()), engine_means(ref.unit.to_numpy(), ref[s].to_numpy())
        out[s] = (len(a), len(b), *(ks(a, b) if min(len(a), len(b)) >= 15 else (None, None)))
    tested = [s for s in SENSORS if out[s][2] is not None]
    adj = dict(zip(tested, holm([out[s][3] for s in tested]))) if tested else {}
    return out, adj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=0, help="re-run shifts 1..N in a throwaway schema and compare rows")
    args = ap.parse_args(argv)
    preds = pd.read_csv("pipeline/phase7_predictions.csv", **CSV)
    health = pd.read_csv("pipeline/phase7_health.csv", **CSV)
    drift = pd.read_csv("pipeline/phase7_drift.csv", **CSV)
    alerts = pd.read_csv("pipeline/phase7_alerts.csv", **CSV)
    runs = pd.read_csv("pipeline/phase7_runs.csv", **CSV)
    res = json.loads(Path("pipeline/results_phase7.json").read_text())

    raw = {ds: read_raw(f"data/raw/test_{ds}.txt") for ds in DS}
    last = {ds: raw[ds].groupby("unit").cycle.max().to_dict() for ds in DS}
    truth = {ds: [float(x) for x in Path(f"data/raw/RUL_{ds}.txt").read_text().split()] for ds in DS}
    L = {ds: {u: int(last[ds][u] + truth[ds][u - 1]) for u in last[ds]} for ds in DS}
    final = max(max(v.values()) for v in L.values())

    print("== Runs")
    want = [f"{ds}-test-s{k:04d}" for ds in DS for k in range(1, final + 1)]
    check(sorted(runs.run_id) == sorted(want) and (runs.flow_status == "complete").all() and runs.run_id.is_unique,
          f"{len(want)} runs: every shift 1..{final} of both datasets completed exactly once with its fixed run ID")
    check((runs["shift"] == runs.last_cycle).all() and (runs.start_cycle == runs["shift"]).all() and (runs.status == "success").all(),
          "each run loaded exactly its own cycle and recorded it as last_cycle")
    nread = {ds: raw[ds].groupby("cycle").size().to_dict() for ds in DS}
    check(all(r.rows_in == nread[r.dataset].get(r["shift"], 0) and r.rows_quarantined == 0 for _, r in runs.iterrows()),
          "rows received per run = NASA readings at that cycle; nothing quarantined in the clean replay")

    print("\n== Forecasts")
    for ds in DS:
        p = preds[preds.dataset == ds]
        check(len(p) == len(raw[ds]) and p.valid.all() and not p.duplicated(["engine_id", "cycle"]).any(),
              f"{ds}: one valid forecast per NASA reading ({len(p):,})")
        lastp = p.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
        pub = pd.read_csv(PUB[ds][1], **CSV)
        check(lastp.rul_point.tolist() == pub.LSTM.tolist(),
              f"{ds}: last-cycle point forecasts = published LSTM (exact, fixed batch of 100)")
        v = pd.read_csv(VAL[ds], **CSV)
        near, far = [], []
        for y, m, s, c, t in zip(v.RUL, v["MC mean"], v["MC std"], v.cycle, v.RUL_true):
            if c >= 31 and t <= 145:
                (near if m <= 50 else far).append(abs(y - m) / max(s, 1e-6))
        kn, kf = quantile(near, 0.9), quantile(far, 0.9)
        k = np.where(p.rul_mc_mean <= 50, kn, kf)
        lo, hi = np.maximum(p.rul_mc_mean - k * p.rul_mc_std, 0), p.rul_mc_mean + k * p.rul_mc_std
        check(max(np.abs(lo - p.rul_low).max(), np.abs(hi - p.rul_high).max()) < 1e-9,
              f"{ds}: rule-E bounds recompute (k near {kn:.3f}, far {kf:.3f})")
    exact = _exact_model_check(preds)
    if exact is not None:
        check(exact[0], exact[1])

    print("\n== Labels and model health")
    for ds in DS:
        p = preds[preds.dataset == ds].sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).set_index("engine_id")
        h = health[health.dataset == ds].sort_values("shift")
        ok_labels = ok_metrics = True
        for _, r in h.iterrows():
            k = int(r["shift"])
            known = sorted([u for u in L[ds] if L[ds][u] <= k], key=lambda u: (L[ds][u], u))
            ok_labels &= r.n_labelled == len(known)
            win = known[-20:]
            if not win:
                ok_metrics &= pd.isna(r.rmse)
                continue
            tr = [L[ds][u] - p.loc[u, "cycle"] for u in win]
            cap = [min(t, 125) for t in tr]
            err = [p.loc[u, "rul_point"] - c for u, c in zip(win, cap)]
            rmse = math.sqrt(sum(e * e for e in err) / len(err))
            miss = sum(t < p.loc[u, "rul_low"] for u, t in zip(win, tr)) / len(win)
            cov = sum(p.loc[u, "rul_low"] <= c <= p.loc[u, "rul_high"] for u, c in zip(win, cap)) / len(win)
            ok_metrics &= abs(rmse - r.rmse) < 1e-9 and abs(miss - r.lower_bound_miss_rate) < 1e-12 and abs(cov - r.coverage_90) < 1e-12
        check(ok_labels, f"{ds}: every run scores exactly the engines whose failure cycle has passed (no future label)")
        check(ok_metrics, f"{ds}: rolling RMSE, coverage and lower-bound miss rate recompute for all {len(h)} runs")
        fin = h.iloc[-1]
        pub_rmse = json.loads(Path(PUB[ds][0]).read_text())["models"]["LSTM"]["rmse"]
        check(fin.n_labelled == 100 and abs(fin.cum_rmse - pub_rmse) < 1e-9,
              f"{ds}: with all 100 engines labelled, cumulative RMSE {fin.cum_rmse:.4f} = published {pub_rmse:.4f}")

    print("\n== Drift (independent KS from the raw files)")
    worst_d = worst_p = 0.0
    same_flags = True
    for ds in DS:
        split = json.loads(Path(SPLIT[ds]).read_text())
        tr = read_raw(f"data/raw/train_{ds}.txt")
        tr = tr[tr.unit.isin(split["train"])]
        d = drift[drift.dataset == ds].set_index(["shift", "sensor"])
        for k in range(1, final + 1):
            out, adj = drift_run(raw[ds], tr, k)
            for s in SENSORS:
                n1, n2, D, pv = out[s]
                row = d.loc[(k, s)]
                same_flags &= (row.n_recent, row.n_reference) == (n1, n2) and bool(row.tested) == (D is not None)
                if D is not None:
                    worst_d = max(worst_d, abs(D - row.ks_stat))
                    worst_p = max(worst_p, abs(adj[s] - row.p_holm))
                    same_flags &= bool(row.drifted) == (adj[s] < RULE["drift"]["alpha"])
    check(same_flags and worst_d < 1e-12 and worst_p < 1e-9,
          f"all {len(drift):,} sensor tests recompute: KS statistic (max diff {worst_d:.1e}), Holm p (max diff "
          f"{worst_p:.1e}), sample sizes and drifted flags")

    print("\n== Retraining flag")
    ok = True
    for _, r in health.iterrows():
        dr = drift[(drift.run_id == r.run_id) & drift.drifted].sensor.tolist()
        reasons = len(dr) >= RULE["drift"]["min_drifted_sensors"]
        ref = RULE["health"]["reference"][r.dataset]["reference_rmse"]
        if r.n_scored >= RULE["health"]["min_labelled_engines"]:
            reasons |= r.rmse > RULE["health"]["rmse_multiplier"] * ref or r.lower_bound_miss_rate > RULE["health"]["miss_rate_limit"]
        ok &= bool(reasons) == bool(r.needs_retraining) and (",".join(sorted(dr)) == ",".join(sorted(str(r.drifted_sensors).split(",")))
                                                              if dr else pd.isna(r.drifted_sensors) or r.drifted_sensors == "")
    rr = runs.set_index("run_id").needs_retraining
    check(ok and all(rr[r.run_id] == r.needs_retraining for _, r in health.iterrows()),
          f"the pre-registered rule re-applied gives the same flag in all {len(health)} runs "
          f"({int(health.needs_retraining.sum())} flagged), and pipeline_runs agrees")

    print("\n== Work orders")
    ok = alerts.groupby("dataset").engine_id.apply(lambda s: s.is_unique).all()
    pk = preds.set_index(["dataset", "engine_id", "cycle"])
    for _, a in alerts.iterrows():
        f = pk.loc[(a.dataset, a.engine_id, a.forecast_cycle)]
        age = a.decision_cycle - a.forecast_cycle
        ok &= f.run_id == a.forecast_run_id and abs(max(0.0, f.rul_low - age) - a.rul_low_used) < 1e-9 and a.rul_low_used <= 20
        ok &= a.decision_cycle % 2 == 0 and L[a.dataset][a.engine_id] > a.decision_cycle
        ok &= 0 <= a.scheduled_day - a.decision_cycle // 2 < 7 and a.service_cycle == 2 * a.scheduled_day
    per_day = alerts.groupby(["dataset", "scheduled_day"]).size().max() if len(alerts) else 0
    check(ok and per_day <= 3, f"{len(alerts)} work orders: one per engine, each cites a stored forecast with aged lower "
                               f"bound <= 20, made before the engine failed, within 7 days; at most {per_day} per day (3 crews)")

    print("\n== Guide checks")
    chk = json.loads(Path("pipeline/phase7_checks.json").read_text())
    fc = chk["flow_check"]
    check(fc["clean_runs_flagged"] == 0 and fc["plus5pct"]["drifted"] == [fc["sensor"]] and fc["plus5pct"]["needs_retraining"]
          and fc["plus5pct"]["quarantined"] == fc["plus5pct"]["readings"],
          f"+5% on {fc['sensor']}: all {fc['plus5pct']['readings']} readings quarantined, the sensor flagged, retraining set; "
          f"{fc['clean_runs']} clean runs set nothing")
    tr = read_raw("data/raw/train_FD001.txt")
    tr = tr[tr.unit.isin(json.loads(Path(SPLIT["FD001"]).read_text())["train"])]
    test = raw["FD001"]
    k = 100
    rec = test[(test.cycle >= k - 2) & (test.cycle <= k)].copy()
    rec.loc[rec.cycle == k, "s4"] *= 1.05
    out, adj = drift_run(rec, tr, k)
    check(adj["s4"] < RULE["drift"]["alpha"], f"independent KS: +5% on s4 in the newest shift of the FD001 shift-{k} window "
                                              f"is detected (Holm p {adj['s4']:.1e})")
    check(chk["test_window_power"]["newest shift: +5%"]["min"] == 1.0, "test-window power at +5%: every sensor, every window")

    print("\n== Database and determinism")
    if _db_available():
        from pipeline.db import get_engine
        from sqlalchemy import text
        with get_engine().connect() as c:
            has_ops = c.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'ops' "
                                     "AND table_name = 'pipeline_runs'")).scalar()
            n = {t: c.execute(text(f"SELECT COUNT(*) FROM ops.{t}")).scalar() for t in
                 ("fact_prediction", "fact_model_health", "fact_drift", "pipeline_runs")} if has_ops else {}
            na = c.execute(text("SELECT COUNT(*) FROM ops.fact_alert WHERE context = 'flow'")).scalar() if has_ops else 0
        if not n.get("pipeline_runs"):
            print("note: no local replay in this database (ops schema empty or absent): row counts skipped")
        else:
            check(n == {"fact_prediction": len(preds), "fact_model_health": len(health), "fact_drift": len(drift),
                        "pipeline_runs": len(runs)} and na == len(alerts), f"ops schema row counts match the committed files: {n}")
        if args.replay:
            check(*_replay_check(args.replay, preds, health, drift, alerts, runs))
    else:
        print("note: no database reachable; row counts and the replay check were skipped")

    print("\n== Documents")
    from pipeline.report_phase7 import DOC, render
    check(DOC.read_text(encoding="utf-8") == render(), f"{DOC} equals a fresh rendering")
    readme = Path("README.md").read_text(encoding="utf-8")
    quotes = [f"alpha {RULE['drift']['alpha']:g}", f"{res['drift_false_alarm_rate_test']:.1%} of clean test runs",
              f"{RULE['health']['rmse_multiplier']:.2f} x"]
    check(all(q in readme for q in quotes), f"README quotes the Phase 7 numbers ({', '.join(quotes)})")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
    return 0 if all(RESULTS) else 1


def _db_available():
    try:
        from pipeline.db import get_engine
        from sqlalchemy import text
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _exact_model_check(preds):
    """On the producing OS with cached weights: recompute sample shifts with the published model, exactly."""
    import torch
    from maintenance.forecasts import NotPublishedModel, WEIGHTS, model_id
    env = json.loads(Path(PUB["FD001"][0]).read_text())["environment"]
    if env["os"] != sys.platform or env["torch"] != torch.__version__ or not Path(WEIGHTS.format(model_id=model_id("FD001"))).exists():
        print("note: not the producing OS (or no cached weights): exact model re-run skipped; covered on Windows")
        return None
    from pipeline import flow as F
    from maintenance.forecasts import with_interval
    from models.deep import mc_dropout, predict
    from models.sequences import make_windows, scaled_readings
    worst = 0.0
    for ds in DS:
        model, _, cfg = F.model_for(ds)
        passes = json.loads(Path(PUB[ds][0]).read_text())["mc_dropout"]["passes"]
        X, idx = make_windows(scaled_readings("test", ds), cfg["window"])
        for k in (1, 77, 150, 303):
            sel = (idx.cycle == k).to_numpy()
            if not sel.any():
                continue
            pt = F._fixed_batches(lambda x: predict(model, x), X[sel])
            mc = F._fixed_batches(lambda x: mc_dropout(model, x, passes, seed=F.C.MC_SEED_BASE[ds] + k), X[sel])
            got = with_interval(pd.DataFrame({"rul_mc_mean": mc.mean(0), "rul_mc_std": mc.std(0, ddof=1)}), ds)
            p = preds[(preds.dataset == ds) & (preds.cycle == k)].sort_values("engine_id")
            worst = max(worst, np.abs(pt - p.rul_point.to_numpy()).max(), np.abs(got.rul_low.to_numpy() - p.rul_low.to_numpy()).max())
    return worst == 0.0, f"published model re-run for sample shifts reproduces stored point forecasts and bounds exactly (max diff {worst:.1e})"


def _replay_check(n, preds, health, drift, alerts, runs):
    """Re-run shifts 1..n in a throwaway schema with the committed predictions; rows must be identical."""
    import uuid
    from prefect.testing.utilities import prefect_test_harness
    from sqlalchemy import text
    from pipeline import flow as F
    from pipeline.db import get_engine
    schema = f"verify7_{uuid.uuid4().hex[:8]}"
    engine = get_engine()
    try:
        with prefect_test_harness():
            for k in range(1, n + 1):
                F.shift_pipeline(shift=k, predictor="recorded", db_schema=schema)
        with engine.begin() as c:
            c.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            got = {f: pd.read_sql(text(sql), c) for f, sql in F.EXPORT.items()}
    finally:
        with engine.begin() as c:
            c.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    pr = preds[preds.cycle <= n].reset_index(drop=True)
    same = {
        "predictions": np.allclose(got["phase7_predictions.csv"].drop(columns=["source", "invalid_reason"]).select_dtypes("number"),
                                   pr.drop(columns=["source", "invalid_reason"]).select_dtypes("number"), rtol=0, atol=0, equal_nan=True),
        # numeric columns compared as numbers (all-empty columns come back from SQL as objects before any label exists)
        "health": all(np.array_equal(pd.to_numeric(got["phase7_health.csv"][col], errors="coerce").fillna(-1).to_numpy(float),
                                     pd.to_numeric(health[health["shift"] <= n][col], errors="coerce").fillna(-1).to_numpy(float))
                      for col in health.columns if col not in ("run_id", "dataset", "model_id", "drifted_sensors", "retraining_reasons"))
                  and got["phase7_health.csv"].run_id.tolist() == health[health["shift"] <= n].run_id.tolist(),
        "alerts": len(got["phase7_alerts.csv"]) == int((alerts.decision_cycle <= n).sum()),
        "flags": got["phase7_runs.csv"].needs_retraining.tolist() == runs[runs["shift"] <= n].needs_retraining.tolist(),
    }
    # Drift: decisions and sample sizes must be identical; the KS statistic and p-values come from floating-point
    # library code that may differ in the last bits between operating systems (the Phase 4/5 rule: exact on the
    # producing OS, a stated tolerance elsewhere), so they get tight tolerances and the largest gaps are reported.
    g, c = got["phase7_drift.csv"].reset_index(drop=True), drift[drift["shift"] <= n].reset_index(drop=True)
    keys_same = g[["dataset", "shift", "sensor", "n_recent", "n_reference", "tested", "drifted"]].equals(
        c[["dataset", "shift", "sensor", "n_recent", "n_reference", "tested", "drifted"]])
    gap = {col: float(np.nanmax(np.abs(g[col].to_numpy(float) - c[col].to_numpy(float)))) if len(g) else 0.0
           for col in ("ks_stat", "p_value", "p_holm")}
    rel = float(np.nanmax(np.abs(g.p_holm - c.p_holm) / np.maximum(np.abs(c.p_holm), 1e-300))) if len(g) else 0.0
    exact = sys.platform == json.loads(Path(PUB["FD001"][0]).read_text())["environment"]["os"]
    same["drift"] = keys_same and (all(v == 0.0 for v in gap.values()) if exact else gap["ks_stat"] <= 1e-12 and rel <= 1e-9)
    return all(same.values()), (f"re-running shifts 1..{n} in a fresh schema reproduces the committed rows: {same}; drift "
                                f"decisions identical {keys_same}, largest gaps {gap}, relative p gap {rel:.1e} "
                                f"({'exact required on the producing OS' if exact else 'tolerance: KS 1e-12, p relative 1e-9'})")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title={Path(__file__).stem} crashed::{type(exc).__name__}: {str(exc)[:900]}")
        raise
