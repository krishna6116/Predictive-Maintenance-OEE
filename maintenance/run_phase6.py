"""Phase 6: forecasts -> alerts -> CP-SAT schedule -> simulated plant -> OEE, for FD001 and FD003 (200 test engines).

Run:  python -m maintenance.forecasts     (once; rebuilds the published LSTMs, writes the rolling forecasts)
      python -m maintenance.run_phase6    (about 2-4 minutes)

Two analyses, both on the published LSTM with its rule-E 90% interval:
 A. Decision snapshot (guide 6.2-6.3 as written): at each test engine's last observed cycle (the official
    evaluation point) flag engines on the lower bound and schedule them over 7 days with 3 crews. Scored against
    NASA's true remaining life: serviced in time, too late, left unserved, or missed (not flagged, fails within
    the horizon).
 B. Policy simulation (guide 6.4-6.5): run to failure, fixed interval, predictive, on the same 200 machines.
    SIMULATED plant; the guide's assumptions; the gaps it leaves are listed in maintenance.config.ASSUMPTIONS.
Every number is fixed by the guide or the config before running; nothing is tuned on these results. The
sensitivity runs only show how the comparison moves when the assumptions change.

Outputs: maintenance/results_phase6.json, snapshot_decisions_phase6.csv, work_orders_phase6.csv,
machines_phase6.csv (one row per machine per policy: the fact_oee rows).
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import ortools
import pandas as pd

from maintenance import config as C
from maintenance.alerts import flag_engines
from maintenance.forecasts import ROLLING, SOURCES, rolling, snapshot
from maintenance.schedule import check_plan, plan_cost, schedule
from maintenance.simulate import (fixed_interval, fixed_interval_n, forecast_table, oee, predictive, production,
                                  run_to_failure)
from pipeline.check_raw_data import load
from pipeline.config import RAW

OUT = Path("maintenance")
DATASETS = ("FD001", "FD003")
POLICIES = ("Run to failure", "Fixed interval", "Predictive")
N_BOOT = 2000


def run_id() -> str:
    """Deterministic: the same inputs and settings give the same run ID, so re-runs replace rather than duplicate."""
    h = hashlib.sha256()
    for ds in DATASETS:
        for f in (SOURCES[ds]["preds"], SOURCES[ds]["val"], ROLLING.format(ds=ds)):
            h.update(Path(f).read_bytes())
    h.update(Path("maintenance/config.py").read_bytes())
    return "phase6-" + h.hexdigest()[:12]


def evaluate_snapshot(snap: pd.DataFrame, plan, flagged: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts, alerts and the plan, and score each engine against its true remaining life."""
    day = dict(plan)
    fl = flagged.set_index("engine_id")
    rows = []
    for _, r in snap.iterrows():
        e = int(r.engine_id)
        is_flag = e in fl.index and fl.loc[e, "risk"] != "invalid forecast"
        d = day.get(e)
        used = None if d is None else d * C.CYCLES_PER_DAY
        if d is not None:
            outcome = "serviced in time" if used < r.true_rul else "serviced too late (failed first)"
        elif is_flag:
            outcome = "flagged, no crew in horizon" + (" (fails inside horizon)" if r.true_rul <= C.HORIZON_DAYS * C.CYCLES_PER_DAY else "")
        elif not r.valid:
            outcome = "invalid forecast: manual inspection"
        else:
            outcome = "missed: fails inside horizon" if r.true_rul <= C.HORIZON_DAYS * C.CYCLES_PER_DAY else "not flagged, healthy"
        rows.append({**r.to_dict(), "flagged": is_flag, "risk": fl.loc[e, "risk"] if e in fl.index else "",
                     "reason": fl.loc[e, "reason"] if e in fl.index else "", "service_day": d, "service_after_cycles": used,
                     "wasted_cycles": (r.true_rul - used) if d is not None and used < r.true_rul else None,
                     "outcome": outcome})
    return pd.DataFrame(rows)


def bootstrap_gap(a: pd.DataFrame, b: pd.DataFrame, n: int = N_BOOT, seed: int = 0) -> dict:
    """Paired bootstrap over machines of fleet OEE(a) - OEE(b). OEE of a fleet = good units / (ideal rate x
    planned hours), which equals A x P x Q. Machines are resampled whole; under the predictive policy machines
    share crews, so treating their outcomes as fixed makes this interval approximate."""
    rate = C.PLANT.units_per_hour
    ga, pa = a.good_units.to_numpy(), a.planned_hours.to_numpy()
    gb, pb = b.good_units.to_numpy(), b.planned_hours.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ga), size=(n, len(ga)))
    diffs = ga[idx].sum(1) / (rate * pa[idx].sum(1)) - gb[idx].sum(1) / (rate * pb[idx].sum(1))
    point = ga.sum() / (rate * pa.sum()) - gb.sum() / (rate * pb.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


def simulate_dataset(ds: str, threshold=C.ALERT_THRESHOLD, crews=C.CREWS_PER_DAY, late_w=C.LATE_WEIGHT, plant=C.PLANT):
    snap = snapshot(ds)
    lives = {int(e): int(c + t) for e, c, t in zip(snap.engine_id, snap.cycle, snap.true_rul)}
    n = fixed_interval_n(load(RAW / f"train_{ds}.txt").groupby("unit").cycle.max())
    pred_out, wo, log = predictive(lives, forecast_table(rolling(ds)), threshold=threshold, crews=crews, late_w=late_w)
    outs = {"Run to failure": run_to_failure(lives), "Fixed interval": fixed_interval(lives, n), "Predictive": pred_out}
    rows = {p: pd.DataFrame([production(o, plant) for o in outs[p]]).assign(dataset=ds, policy=p) for p in POLICIES}
    return snap, n, rows, wo, log


def main() -> int:
    t0 = time.perf_counter()
    rid = run_id()
    res = {"run_id": rid, "simulation_notice": "SIMULATED plant: production, downtime and OEE are not measured data",
           "plant": C.PLANT.as_dict(), "assumptions": C.ASSUMPTIONS,
           "settings": {"alert_threshold": C.ALERT_THRESHOLD, "risk_bins": list(C.RISK_BINS), "horizon_days": C.HORIZON_DAYS,
                        "crews_per_day": C.CREWS_PER_DAY, "cycles_per_day": C.CYCLES_PER_DAY,
                        "weights": {"late": C.LATE_WEIGHT, "early": C.EARLY_WEIGHT, "unserved": C.UNSERVED_WEIGHT},
                        "fixed_interval_percentile": C.FIXED_INTERVAL_PERCENTILE},
           "datasets": {}}
    snaps, machines, orders = [], [], []
    for ds in DATASETS:
        # ---- A. decision snapshot
        snap = snapshot(ds)
        flagged = flag_engines(snap)
        sched = flagged[flagged.risk != "invalid forecast"]
        status, plan, obj = schedule(sched)
        problems = check_plan(sched, plan)
        if status not in ("OPTIMAL", "FEASIBLE") or problems or plan_cost(sched, plan) != obj:
            raise SystemExit(f"{ds}: snapshot schedule {status}; violations {problems}")
        ev = evaluate_snapshot(snap, plan, flagged).assign(run_id=rid)
        snaps.append(ev)
        weights = {}
        lows = dict(zip(sched.engine_id.astype(int), sched.rul_low))
        def plan_stats(pl):
            used = {e: d * C.CYCLES_PER_DAY for e, d in pl}
            return {"served": sorted(used), "late_cycles": int(sum(max(0, u - int(np.floor(lows[e]))) for e, u in used.items())),
                    "early_cycles": int(sum(max(0, int(np.floor(lows[e])) - u) for e, u in used.items())),
                    "unserved": len(lows) - len(used)}
        base_stats = plan_stats(plan)
        for name, lw, ew in (("guide (100 late / 1 early)", 100, 1), ("late 10", 10, 1), ("late 1000", 1000, 1), ("early 10", 100, 10)):
            s2, p2, o2 = schedule(sched, late_w=lw, early_w=ew)
            st = plan_stats(p2)
            weights[name] = {"status": s2, "plan": [[int(e), int(d)] for e, d in p2], "objective": o2, **st,
                             "same_engines_served": st["served"] == base_stats["served"],
                             "same_days": p2 == plan}
        # ---- B. policy simulation
        snap_, n, rows, wo, log = simulate_dataset(ds)
        for p in POLICIES:
            machines.append(rows[p])
        orders.append(wo.assign(dataset=ds, run_id=rid))
        summary = {p: oee(rows[p]) for p in POLICIES}
        gaps = {"Predictive - Run to failure": bootstrap_gap(rows["Predictive"], rows["Run to failure"]),
                "Predictive - Fixed interval": bootstrap_gap(rows["Predictive"], rows["Fixed interval"])}
        sens = {}
        for label, kw in (("threshold 10", {"threshold": 10}), ("threshold 30", {"threshold": 30}),
                          ("2 crews", {"crews": 2}), ("4 crews", {"crews": 4}), ("late weight 10", {"late_w": 10}),
                          ("failure downtime 24 h", {"plant": C.Plant(failure_hours=24.0)}),
                          ("failure downtime 96 h", {"plant": C.Plant(failure_hours=96.0)})):
            _, _, r2, _, _ = simulate_dataset(ds, **kw)
            sens[label] = {p: oee(r2[p]) for p in POLICIES}
        res["datasets"][ds] = {
            "model_id": snap.model_id.iloc[0], "forecast_source": snap.source.iloc[0],
            "snapshot": {"engines": len(snap), "invalid_forecasts": int((~snap.valid).sum()),
                         "flagged": int(ev.flagged.sum()), "risk_counts": ev[ev.flagged].risk.value_counts().to_dict(),
                         "schedule_status": status, "objective": obj, "scheduled": len(plan),
                         "per_day": {d: sum(1 for _, dd in plan if dd == d) for d in range(C.HORIZON_DAYS)},
                         "outcomes": ev.outcome.value_counts().to_dict(),
                         "wasted_cycles_mean": float(ev.wasted_cycles.dropna().mean()) if ev.wasted_cycles.notna().any() else None,
                         "weight_sensitivity": weights},
            "fixed_interval_n": n,
            "policies": summary, "oee_gaps": gaps,
            "predictive_replay": {"days": int(log.day.max()) if not log.empty else 0,
                                  "solves": int((log.event == "schedule").sum()) if not log.empty else 0,
                                  "all_optimal": bool((log[log.event == "schedule"].status == "OPTIMAL").all()) if not log.empty else True,
                                  "max_unserved_in_a_day": int(log.unserved.max()) if "unserved" in log else 0,
                                  "max_flagged_in_a_day": int(log.flagged.max()) if "flagged" in log else 0,
                                  "work_orders": len(wo), "order_status": wo.status.value_counts().to_dict() if len(wo) else {},
                                  "invalid_forecast_events": int((log.event == "invalid forecast").sum()) if not log.empty else 0},
            "sensitivity": sens}
    m_all = pd.concat(machines, ignore_index=True)
    res["pooled"] = {p: oee(m_all[m_all.policy == p]) for p in POLICIES}
    res["pooled_gaps"] = {
        "Predictive - Run to failure": bootstrap_gap(m_all[m_all.policy == "Predictive"], m_all[m_all.policy == "Run to failure"]),
        "Predictive - Fixed interval": bootstrap_gap(m_all[m_all.policy == "Predictive"], m_all[m_all.policy == "Fixed interval"])}
    res["environment"] = {"os": sys.platform, "platform": platform.platform(), "python": platform.python_version(),
                          "ortools": ortools.__version__, "solver_workers": C.SOLVER_WORKERS, "solver_seed": C.SOLVER_SEED}
    res["seconds_total"] = time.perf_counter() - t0
    pd.concat(snaps, ignore_index=True).to_csv(OUT / "snapshot_decisions_phase6.csv", index=False)
    pd.concat(orders, ignore_index=True).to_csv(OUT / "work_orders_phase6.csv", index=False)
    m_all.assign(run_id=rid).to_csv(OUT / "machines_phase6.csv", index=False)
    (OUT / "results_phase6.json").write_text(json.dumps(res, indent=2, default=int), encoding="utf-8")
    for ds in DATASETS:
        d = res["datasets"][ds]
        print(f"{ds}: snapshot flagged {d['snapshot']['flagged']}, scheduled {d['snapshot']['scheduled']} "
              f"({d['snapshot']['schedule_status']}); outcomes {d['snapshot']['outcomes']}")
        for p in POLICIES:
            s = d["policies"][p]
            print(f"   {p:15s} OEE {s['oee']:.1%}  A {s['availability']:.1%} P {s['performance']:.1%} Q {s['quality']:.1%}  "
                  f"failures {s['unplanned_failures']:3d} services {s['planned_services']:3d} downtime {s['downtime_hours']:6.0f} h "
                  f"wasted {s['wasted_cycles']}")
    print("pooled:", {p: round(res["pooled"][p]["oee"], 4) for p in POLICIES}, res["pooled_gaps"])
    print(f"run {rid}; {res['seconds_total']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
