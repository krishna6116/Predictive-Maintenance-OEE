"""Independent verification of Phase 6, with different methods from the code that produced the results.

Run:  python -m maintenance.verify_phase6          (after run_phase6, report_phase6 and, with a database, store)

  forecasts   rule-E scale factors with a plain-Python quantile; intervals recomputed; rolling last-cycle
              forecasts equal the published Phase 5 / 5b ones exactly; cycles 1..last match the raw NASA files
  alerts      flagged set and risk levels recomputed from the bounds
  schedule    constraints re-checked; OPTIMALITY of each snapshot plan proved by solving the same problem as an
              assignment problem (Hungarian algorithm, scipy), a different method from CP-SAT
  replay      every work order re-derived from the forecasts: created on the first day the aged lower bound
              reached the threshold, within the horizon, crews respected, outcome consistent with the true failure
              point; machines without orders never reached the threshold; a re-run gives identical orders
  policies    run-to-failure and fixed-interval outcomes recomputed from the raw files
  OEE         recomputed per machine and per fleet with one formula, good / (ideal rate x planned hours), and
              checked against Availability x Performance x Quality
  documents   the generated doc equals a fresh rendering; README numbers; database rows (when DATABASE_URL set)
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from maintenance import config as C

CSV = {"float_precision": "round_trip"}
R = json.loads(Path("maintenance/results_phase6.json").read_text())
SNAP = pd.read_csv("maintenance/snapshot_decisions_phase6.csv", **CSV)
WO = pd.read_csv("maintenance/work_orders_phase6.csv", **CSV)
MACH = pd.read_csv("maintenance/machines_phase6.csv", **CSV)
SRC = {"FD001": ("models/predictions_phase5.csv", "models/val_predictions_phase5.csv"),
       "FD003": ("models/predictions_phase5b.csv", "models/val_predictions_phase5b.csv")}
RESULTS = []


def check(ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + msg)


def quantile(xs, q):
    """Linear-interpolation quantile (the textbook definition), in plain Python."""
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def read_raw(path):
    rows = {}
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            v = line.split()
            rows.setdefault(int(v[0]), []).append(int(v[1]))
    return rows


def k_rule_e(val):
    near, far = [], []
    for y, m, s, c, t in zip(val.RUL, val["MC mean"], val["MC std"], val.cycle, val.RUL_true):
        if c >= 31 and t <= 145:
            (near if m <= 50 else far).append(abs(y - m) / max(s, 1e-6))
    return quantile(near, 0.9), quantile(far, 0.9)


def bounds(mean, std, kn, kf):
    k = kn if mean <= 50 else kf
    return max(mean - k * std, 0.0), mean + k * std


def good_and_planned(life, run, event, p=C.PLANT):
    deg = max(0, run - max(0, life - p.degraded_cycles))
    per = p.hours_per_cycle * p.units_per_hour
    good = per * ((run - deg) * (1 - p.normal_scrap) + deg * (1 - p.degraded_speed_loss) * (1 - p.degraded_scrap))
    down = {"service": p.service_hours, "failure": p.failure_hours, "none": 0.0}[event]
    return good, run * p.hours_per_cycle + down


def main() -> int:
    from scipy.optimize import linear_sum_assignment
    roll = {}
    for ds in ("FD001", "FD003"):
        print(f"\n== {ds}: forecasts")
        pub, val = (pd.read_csv(f, **CSV) for f in SRC[ds])
        kn, kf = k_rule_e(val)
        last = {u: max(c) for u, c in read_raw(f"data/raw/test_{ds}.txt").items()}
        truth = [float(x) for x in Path(f"data/raw/RUL_{ds}.txt").read_text().split()]
        s = SNAP[SNAP.dataset == ds].sort_values("engine_id")
        check(s.engine_id.tolist() == list(range(1, 101)) and s.cycle.tolist() == [last[u] for u in range(1, 101)]
              and s.true_rul.tolist() == truth, "100 engines; last observed cycle and true RUL match the raw NASA files")
        check(s.rul_point.tolist() == pub.LSTM.tolist() and s.rul_mc_mean.tolist() == pub["MC mean"].tolist(),
              "snapshot forecasts are the published LSTM point and MC mean, unchanged")
        worst = max(max(abs(a - b) for a, b in zip(bounds(m, sd, kn, kf), (lo, hi)))
                    for m, sd, lo, hi in zip(s.rul_mc_mean, s.rul_mc_std, s.rul_low, s.rul_high))
        check(worst < 1e-9, f"rule-E k (near {kn:.3f}, far {kf:.3f}) and every interval recompute (max diff {worst:.1e})")
        r = pd.read_csv(f"maintenance/forecasts_rolling_{ds}.csv", **CSV)
        roll[ds] = r
        g = r.groupby("engine_id")
        check(all(g.get_group(u).cycle.tolist() == list(range(1, last[u] + 1)) for u in range(1, 101)),
              f"rolling forecasts: {len(r):,} rows, cycles 1..last for every engine")
        lastrow = r.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
        check(lastrow.rul_point.tolist() == pub.LSTM.tolist(), "rolling last-cycle point forecasts = published (exact)")
        worst = max(max(abs(a - b) for a, b in zip(bounds(m, sd, kn, kf), (lo, hi)))
                    for m, sd, lo, hi in zip(r.rul_mc_mean, r.rul_mc_std, r.rul_low, r.rul_high))
        fin = np.isfinite(r[["rul_point", "rul_mc_mean", "rul_mc_std", "rul_low", "rul_high"]].to_numpy()).all()
        check(fin and worst < 1e-9 and r.rul_low.min() >= 0, f"rolling intervals finite and recompute (max diff {worst:.1e})")

        print(f"== {ds}: alerts and snapshot schedule")
        flag = (s.rul_low <= C.ALERT_THRESHOLD).tolist()
        risk = ["critical" if lo <= 5 else "high" if lo <= 10 else "medium" for lo in s.rul_low]
        check(s.flagged.tolist() == flag and all(rk == r_ for rk, r_, f in zip(s.risk, risk, flag) if f),
              f"flagged set ({sum(flag)}) and risk levels recompute from the lower bounds")
        fl = s[s.flagged]
        plan = [(int(e), int(d)) for e, d in zip(fl.engine_id, fl.service_day) if not pd.isna(d)]
        per_day = [sum(1 for _, d in plan if d == k) for k in range(C.HORIZON_DAYS)]
        check(max(per_day) <= C.CREWS_PER_DAY and len({e for e, _ in plan}) == len(plan) and not s[~s.flagged].service_day.notna().any(),
              f"snapshot plan: {len(plan)} services, per day {per_day} <= {C.CREWS_PER_DAY} crews, only flagged engines")
        low = dict(zip(fl.engine_id.astype(int), fl.rul_low))
        cost = lambda e, d: C.LATE_WEIGHT * max(0, d * C.CYCLES_PER_DAY - math.floor(low[e])) + \
            C.EARLY_WEIGHT * max(0, math.floor(low[e]) - d * C.CYCLES_PER_DAY)
        eng = sorted(low)
        slots = [d for d in range(C.HORIZON_DAYS) for _ in range(C.CREWS_PER_DAY)]
        big = 10 ** 9
        M = np.full((len(eng), len(slots) + len(eng)), big, dtype=float)
        for i, e in enumerate(eng):
            for j, d in enumerate(slots):
                M[i, j] = cost(e, d)
            M[i, len(slots) + i] = C.UNSERVED_WEIGHT
        ri, ci = linear_sum_assignment(M)
        best = int(M[ri, ci].sum())
        mine = sum(cost(e, d) for e, d in plan) + C.UNSERVED_WEIGHT * (len(eng) - len(plan))
        snap_r = R["datasets"][ds]["snapshot"]
        check(best == mine == snap_r["objective"] and snap_r["schedule_status"] == "OPTIMAL",
              f"CP-SAT says OPTIMAL with cost {snap_r['objective']}; the Hungarian algorithm's optimum is {best}")
        served = {e: d for e, d in plan}
        ok = all((("serviced in time" if served[e] * 2 < t else "serviced too late (failed first)") == o) if e in served
                 else True for e, t, o in zip(s.engine_id, s.true_rul, s.outcome))
        missed = sum(1 for f, t in zip(s.flagged, s.true_rul) if not f and t <= C.HORIZON_DAYS * C.CYCLES_PER_DAY)
        check(ok and missed == sum(s.outcome.str.startswith("missed")),
              f"snapshot outcomes recompute against the true RUL (missed: {missed})")

        print(f"== {ds}: policies and OEE")
        lives = {u: last[u] + int(truth[u - 1]) for u in range(1, 101)}
        tl = read_raw(f"data/raw/train_{ds}.txt")
        n = math.floor(quantile([max(c) for c in tl.values()], C.FIXED_INTERVAL_PERCENTILE / 100))
        check(n == R["datasets"][ds]["fixed_interval_n"], f"fixed interval N = {n} (10th percentile of training lifetimes)")
        m = MACH[MACH.dataset == ds]
        expect = {"Run to failure": {u: (L, "failure") for u, L in lives.items()},
                  "Fixed interval": {u: ((n, "service") if n < L else (L, "failure")) for u, L in lives.items()}}
        for pol, exp in expect.items():
            mm = m[m.policy == pol]
            check(all(exp[u] == (rc, ev) and L == lives[u] for u, rc, ev, L in zip(mm.machine, mm.run_cycles, mm.event, mm.life)),
                  f"{pol}: every machine's outcome recomputes from the raw files")
        for pol in ("Run to failure", "Fixed interval", "Predictive"):
            mm = m[m.policy == pol]
            gp = [good_and_planned(L, rc, ev) for L, rc, ev in zip(mm.life, mm.run_cycles, mm.event)]
            per_ok = all(abs(g - a) < 1e-6 and abs(p - b) < 1e-9 for (g, p), a, b in zip(gp, mm.good_units, mm.planned_hours))
            fleet = sum(g for g, _ in gp) / (C.PLANT.units_per_hour * sum(p for _, p in gp))
            s_ = R["datasets"][ds]["policies"][pol]
            apq = s_["availability"] * s_["performance"] * s_["quality"]
            check(per_ok and abs(fleet - s_["oee"]) < 1e-12 and abs(apq - s_["oee"]) < 1e-12
                  and s_["unplanned_failures"] == int((mm.event == "failure").sum()),
                  f"{pol}: OEE {fleet:.4%} = good / (rate x planned) = A x P x Q; {s_['unplanned_failures']} failures")

        print(f"== {ds}: predictive replay (work orders)")
        w = WO[WO.dataset == ds]
        fc = {u: grp.sort_values("cycle") for u, grp in r.groupby("engine_id")}
        def low_at(u, c):
            f = fc[u]
            lc = len(f)
            return float(f.rul_low.iloc[c - 1]) if c <= lc else max(0.0, float(f.rul_low.iloc[lc - 1]) - (c - lc))
        pm = m[m.policy == "Predictive"].set_index("machine")
        orders = w.set_index("engine_id")
        good = True
        for u in range(1, 101):
            L = lives[u]
            # decisions happen at the start of each day (after c = 2, 4, ... cycles) while the machine still runs (c < L)
            first = next((c for c in range(2, L, 2) if low_at(u, c) <= C.ALERT_THRESHOLD), None)
            if u in orders.index:
                o = orders.loc[u]
                good &= first == o.created_cycle and abs(low_at(u, o.created_cycle) - o.rul_low) < 1e-9
                good &= 0 <= o.day - o.created_day < C.HORIZON_DAYS and o.service_cycle == o.day * C.CYCLES_PER_DAY
                ev, rc = pm.loc[u, "event"], pm.loc[u, "run_cycles"]
                good &= (o.status == "done" and ev == "service" and rc == o.service_cycle < L) or \
                        (o.status.startswith("cancelled") and ev == "failure" and L <= o.service_cycle)
            else:
                good &= first is None and pm.loc[u, "event"] == "failure"
        done = w[w.status == "done"]
        check(good, f"{len(w)} work orders: each created the first day its lower bound reached {C.ALERT_THRESHOLD}, served "
                    f"in the horizon or cancelled by a failure; machines without orders never reached it")
        check(w.engine_id.is_unique and done.groupby("day").size().max() <= C.CREWS_PER_DAY,
              f"no duplicate work orders; at most {done.groupby('day').size().max()} services a day (limit {C.CREWS_PER_DAY})")
        from maintenance.simulate import forecast_table, predictive
        from maintenance.forecasts import rolling
        _, w2, _ = predictive(lives, forecast_table(rolling(ds)))
        cols = ["engine_id", "created_day", "day", "status"]
        check(w2[cols].reset_index(drop=True).equals(w[cols].reset_index(drop=True)), "re-running the replay gives identical work orders")

    print("\n== Pooled, gaps and documents")
    for pol in ("Run to failure", "Fixed interval", "Predictive"):
        mm = MACH[MACH.policy == pol]
        fleet = mm.good_units.sum() / (C.PLANT.units_per_hour * mm.planned_hours.sum())
        check(abs(fleet - R["pooled"][pol]["oee"]) < 1e-12, f"pooled {pol} OEE {fleet:.4%} recomputes over 200 machines")
    for k, gap in R["pooled_gaps"].items():
        a, b = k.split(" - ")
        fa, fb = (MACH[MACH.policy == p] for p in (a, b))
        d = fa.good_units.sum() / (10 * fa.planned_hours.sum()) - fb.good_units.sum() / (10 * fb.planned_hours.sum())
        check(abs(d - gap["diff"]) < 1e-12 and gap["ci"][0] <= d <= gap["ci"][1],
              f"{k}: {d:+.2%} (95% interval {gap['ci'][0]:+.2%} to {gap['ci'][1]:+.2%}) recomputes")
    from maintenance.report_phase6 import DOC, render
    check(DOC.read_text(encoding="utf-8") == render(), f"{DOC} equals a fresh rendering of the results")
    readme = Path("README.md").read_text(encoding="utf-8")
    p = R["pooled"]
    quotes = [f"{p['Predictive']['oee']:.1%}", f"{p['Run to failure']['oee']:.1%}", f"{p['Fixed interval']['oee']:.1%}",
              f"{p['Predictive']['unplanned_failures']} unplanned failures", f"({p['Run to failure']['unplanned_failures']} failures)",
              f"({p['Fixed interval']['unplanned_failures']} failures, and "
              f"{p['Fixed interval']['wasted_cycles'] / p['Predictive']['wasted_cycles']:.1f} times as much useful life"]
    check(all(q in readme for q in quotes) and "simulat" in readme.lower(), f"README quotes the pooled OEE ({', '.join(quotes)}) as simulated")
    if os.environ.get("DATABASE_URL") or Path(".env").exists():
        from maintenance.store import counts
        from pipeline.db import get_engine
        n = counts(get_engine(), R["run_id"])
        want = {"fact_prediction": sum(len(x) for x in roll.values()), "fact_oee": len(MACH),
                "fact_alert": int(SNAP.flagged.sum()) + len(WO)}
        check(n == want, f"database rows for run {R['run_id']}: {n}")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
