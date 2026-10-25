"""Independent verification of Phase 7b, with different code from the code that produced the results.

Run:  python -m pipeline.verify_phase7b

  fresh models   test RMSE and interval coverage of FD002/FD004 recomputed from the raw NASA RUL files
  criteria       C1 recounted from every replay's health table; C2 consistency; C3/C4 and the reported faults
                 recomputed from the fault-suite rows; mirror self-check and equivalence
  rule           the K-of-M / sudden decision re-applied by hand to every drift history (every scenario), and the
                 retraining flag re-derived from drift and health with the rule file's limits
  labels         labelled-engine counts per run re-derived from the raw files and re-drawn staggered offsets
  drift          KS (own engine-weighted ECDF code) and Holm (own code) recomputed for sampled runs, from the raw test
                 readings, the stored forecasts and the shared expected-level model
  closed loop    censoring bounds recomputed from the closed-loop replay
  documents      docs/MONITORING_7B.md equals a fresh rendering; README numbers
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstwo

CSV = {"float_precision": "round_trip"}
OUT = Path("pipeline/phase7b")
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
SCEN = {"fresh_synchronized": ("FD002", "FD004", "synchronized", "v2"), "fresh_staggered": ("FD002", "FD004", "staggered", "v2"),
        "posthoc_synchronized": ("FD001", "FD003", "synchronized", "v2"), "posthoc_staggered": ("FD001", "FD003", "staggered", "v2"),
        "v21_fresh_synchronized": ("FD002", "FD004", "synchronized", "v2.1"), "v21_fresh_staggered": ("FD002", "FD004", "staggered", "v2.1"),
        "v21_posthoc_synchronized": ("FD001", "FD003", "synchronized", "v2.1"), "v21_posthoc_staggered": ("FD001", "FD003", "staggered", "v2.1")}
RESULTS = []


def check(ok, msg):
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok and os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title={Path(__file__).stem} check failed::{msg}")


def raw(path):
    cols = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.DataFrame([[float(v) for v in l.split()] for l in Path(path).read_text().split("\n") if l.strip()], columns=cols)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def failures(ds):
    t = raw(f"data/raw/test_{ds}.txt")
    last = t.groupby("unit").cycle.max().to_dict()
    truth = [float(x) for x in Path(f"data/raw/RUL_{ds}.txt").read_text().split()]
    return {u: int(last[u] + truth[u - 1]) for u in last}, t


def offsets(units, fleet):
    if fleet == "synchronized":
        return {u: 0 for u in units}
    return dict(zip(sorted(units), np.random.default_rng(7).integers(0, 151, size=len(units)).tolist()))


def holm(p):
    order = sorted(range(len(p)), key=lambda i: (p[i], i))
    adj, run = [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def ks_weighted(sample, ref_vals, ref_w, n_ref):
    a = np.sort(sample)
    pts = np.unique(np.concatenate([a, ref_vals]))
    fa = np.array([np.sum(a <= x) for x in pts]) / len(a)
    order = np.argsort(ref_vals, kind="stable")
    rv, rw = ref_vals[order], np.cumsum(ref_w[order])
    fb = np.array([rw[np.searchsorted(rv, x, side="right") - 1] if np.searchsorted(rv, x, side="right") else 0.0 for x in pts])
    d = float(np.max(np.abs(fa - fb)))
    return d, float(min(1.0, max(0.0, kstwo.sf(d, round(len(a) * n_ref / (len(a) + n_ref))))))


def main() -> int:
    rule = {"v2": json.loads(Path("pipeline/monitoring_rule_v2.json").read_text()),
            "v2.1": json.loads(Path("pipeline/monitoring_rule_v2_1.json").read_text())}
    chk = json.loads((OUT / "checks.json").read_text())
    fs = json.loads((OUT / "fault_suite.json").read_text())
    fail, tests = {}, {}
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        fail[ds], tests[ds] = failures(ds)

    print("== Fresh models (FD002, FD004)")
    for ds in ("FD002", "FD004"):
        r = json.loads(Path(f"models/results_{ds}.json").read_text())
        p = pd.read_csv(f"models/predictions_{ds}.csv", **CSV)
        truth = [min(125.0, float(x)) for x in Path(f"data/raw/RUL_{ds}.txt").read_text().split()]
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(p.LSTM, truth)) / len(truth))
        cov = sum(lo <= t <= hi for lo, t, hi in zip(p["MC 90% low (rule E)"], truth, p["MC 90% high (rule E)"])) / len(truth)
        check(abs(rmse - r["models"]["LSTM"]["rmse"]) < 1e-9 and abs(cov - r["mc_dropout"]["test_coverage"]) < 1e-12,
              f"{ds}: test RMSE {rmse:.4f} and interval coverage {cov:.0%} recompute from NASA's RUL file")

    print("\n== C1 and the retraining rule, every replay")
    for sc, (d1, d2, fleet, var) in SCEN.items():
        h = pd.read_csv(OUT / sc / "phase7_health.csv", **CSV)
        dr = pd.read_csv(OUT / sc / "phase7_drift.csv", **CSV)
        R = rule[var]
        ok_dec = ok_flag = ok_lab = True
        for ds in (d1, d2):
            hd = h[h.dataset == ds].set_index("shift")
            dd = dr[dr.dataset == ds]
            hist = []
            off = offsets(fail[ds].keys(), fleet)
            for k in sorted(hd.index):
                rows = dd[dd["shift"] == k].set_index("sensor")
                now = {s: rows.loc[s, "p_holm"] for s in SENSORS} if len(rows) == 14 and rows.tested.all() else {}
                hist.append(now)
                recent = hist[-R["drift"]["m"]:]
                if now:
                    per = {s for s in SENSORS if sum(1 for x in recent if x and x[s] < R["drift"]["alpha"]) >= R["drift"]["k"]}
                    sud = {s for s in SENSORS if now[s] < R["drift"]["alpha_strong"]} - per
                    want = per | sud
                else:
                    want = set()
                got = set(rows[rows.drifted].index) if len(rows) else set()
                ok_dec &= want == got
                r = hd.loc[k]
                p0 = R["datasets"][ds]["p0"]
                health = r.n_scored >= R["health"]["min_labelled_engines"] and (
                    r.error_ratio > R["health"]["error_ratio_limit"] or r.miss_p < R["health"]["miss_alpha"])
                ok_flag &= bool(r.needs_retraining) == (bool(want) or bool(health))
                ok_lab &= r.n_labelled == sum(1 for e, L in fail[ds].items() if L + off[e] <= k)
        check(ok_dec, f"{sc}: the K-of-M / sudden decision re-applied by hand gives every run's drifted sensors")
        check(ok_flag and ok_lab, f"{sc}: every retraining flag re-derived from drift and health; labelled counts from the raw files")
        if sc in ("fresh_synchronized", "fresh_staggered", "posthoc_synchronized", "posthoc_staggered"):
            for ds, g in h.groupby("dataset"):
                c = chk["C1"][f"{ds} {fleet}"]
                check(c["flagged"] == int(g.needs_retraining.sum()) and c["runs"] == len(g),
                      f"C1 {ds} {fleet}: {c['flagged']}/{c['runs']} = {c['rate']:.1%} recounted")

    print("\n== Drift statistics recomputed for sampled runs")
    from pipeline import flow as F
    from pipeline.conditions import CONDITION_DATASETS, normalise, stats_file
    from features.build import load_json
    worst = 0.0
    n_runs = 0
    for sc, ds in (("fresh_synchronized", "FD002"), ("fresh_staggered", "FD004")):
        fleet, var = SCEN[sc][2], SCEN[sc][3]
        m = F.v2_models(ds, var)
        dr = pd.read_csv(OUT / sc / "phase7_drift.csv", **CSV)
        dr = dr[(dr.dataset == ds) & dr.tested]
        pr = pd.read_csv(OUT / sc / "phase7_predictions.csv.gz", **CSV)
        pr = pr[(pr.dataset == ds) & pr.valid].set_index(["engine_id", "cycle"]).rul_point.to_dict()
        t = tests[ds]
        t = normalise(t, load_json(stats_file(ds))) if ds in CONDITION_DATASETS else t
        off = offsets(fail[ds].keys(), fleet)
        for k in sorted(dr["shift"].unique())[::40]:
            rows = []
            for u, g in t.groupby("unit"):
                for s in (k - 2, k - 1, k):
                    c = s - off[u]
                    if 1 <= c <= len(g):
                        rows.append(g.iloc[c - 1])
            w = pd.DataFrame(rows)
            rh = []
            for u, c in zip(w.unit, w.cycle):
                back = c
                while (u, back) not in pr and back > 1:
                    back -= 1
                rh.append(max(0.0, pr[(u, back)] - (c - back)) if (u, back) in pr else np.nan)
            w = w.assign(rhat=rh).dropna(subset=["rhat"])
            z = m["level"].z(w)
            full = z.groupby("unit").cycle.transform("size") == 3
            em = z[full].groupby("unit")[SENSORS].mean().round(9)
            ps = [ks_weighted(em[s].to_numpy(), *m["ref"][s], m["n_ref"])[1] for s in SENSORS]
            adj = holm(ps)
            stored = dr[dr["shift"] == k].set_index("sensor").p_holm
            worst = max(worst, max(abs(a - stored[s]) for a, s in zip(adj, SENSORS)))
            n_runs += 1
    check(worst < 1e-9, f"{n_runs} sampled runs: KS and Holm recomputed from raw readings with independent code (max p gap {worst:.1e})")

    print("\n== C2, fault suite, closed loop")
    for ds in ("FD002", "FD004"):
        c = chk["C2"][ds]
        check(c["passed"] == (c["needs_retraining"] and c["sensor"] in (c["reasons"] or "") and c["clean_runs_flagged"] == 0)
              and c["passed"], f"C2 {ds}: one +5% batch on {c['sensor']} set the flag; clean runs before it set nothing")
        for kind in ("step", "ramp", "stuck", "noise", "dropout"):
            rows = pd.DataFrame(fs[ds][kind]["rows"])
            tt = rows[rows.tested]
            ok = fs[ds][kind]["detected"] == int(tt.detected.sum()) and fs[ds][kind]["testable"] == len(tt) and len(rows) == 84
            check(ok, f"{ds} {kind}: {int(tt.detected.sum())}/{len(tt)} detected, recounted from the 84 injection rows")
        check(fs[ds]["mirror_self_check_max_diff"] == 0.0, f"{ds}: fault mirror re-predicts clean windows exactly")
    eq = fs["mirror_equivalence"]
    check(eq["same_flags"] and eq["max_p_gap"] < 1e-9, f"fault mirror equals the real flow on an injected run (p gap {eq['max_p_gap']:.1e})")
    cl = pd.read_csv(OUT / "fresh_closed_loop" / "phase7_predictions.csv.gz", **CSV)
    al = pd.read_csv(OUT / "fresh_closed_loop" / "phase7_alerts.csv")
    for ds in ("FD002", "FD004"):
        v = cl[(cl.dataset == ds) & cl.valid]
        orders = dict(zip(al[al.dataset == ds].engine_id, al[al.dataset == ds].scheduled_day * 2))
        lab = miss = sure = unk = 0
        for e, g in v.groupby("engine_id"):
            sv = orders.get(e)
            if sv is not None and sv < fail[ds][e]:
                last = g[g.cycle <= sv].sort_values("cycle").iloc[-1]
                sure += sv - last.cycle >= last.rul_low
                unk += sv - last.cycle < last.rul_low
            else:
                last = g.sort_values("cycle").iloc[-1]
                lab += 1
                miss += fail[ds][e] - last.cycle < last.rul_low
        n = lab + sure + unk
        want = chk["closed_loop"][ds]
        check(lab == want["failed_labelled"] and abs(miss / n - want["miss_rate_bounds_all_engines"][0]) < 1e-12
              and abs((miss + unk) / n - want["miss_rate_bounds_all_engines"][1]) < 1e-12,
              f"closed loop {ds}: {lab} labels, miss-rate bounds {miss / n:.1%}-{(miss + unk) / n:.1%} recompute")

    print("\n== Documents")
    from pipeline.report_phase7b import DOC, render
    check(DOC.read_text(encoding="utf-8") == render(), f"{DOC} equals a fresh rendering")
    readme = Path("README.md").read_text(encoding="utf-8")
    c1 = chk["C1"]
    quotes = [f"{c1['FD004 synchronized']['rate']:.1%}", f"{c1['FD004 staggered']['rate']:.1%}", f"{c1['FD002 staggered']['rate']:.1%}"]
    check(all(q in readme for q in quotes), f"README quotes the Phase 7b C1 results ({', '.join(quotes)})")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title={Path(__file__).stem} crashed::{type(exc).__name__}: {str(exc)[:900]}")
        raise
