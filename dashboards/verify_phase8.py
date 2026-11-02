"""Phase 8 independent verification: prints the scorecard (dashboards/phase8_scorecard.json) with pass/fail per check.

Run:  python -m dashboards.verify_phase8              all checks this environment can run (app + database if reachable)
      python -m dashboards.verify_phase8 --local      also D1.5/D2.1 on the local database; writes dashboards/local_checks.json
      python -m dashboards.verify_phase8 --clean-env  D2.2: must run in an environment with ONLY app/requirements.txt
      python -m dashboards.verify_phase8 --no-app     skip the AppTest checks (faster; D1.1, D3-D5 code checks not run)
      python -m dashboards.verify_phase8 --snapshots DIR   verify another snapshot folder (used by the planted-bug tests)

Independent by construction: raw results files are read with the csv module and every number is recomputed with plain
Python (math.fsum), not with the exporter's pandas code or the project's oee() function. Writes dashboards/phase8_results.json
(only for the committed snapshot folder and a full run).
"""
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "dashboards" / "phase8_spec.json").read_text(encoding="utf-8"))
CARD = json.loads((ROOT / "dashboards" / "phase8_scorecard.json").read_text(encoding="utf-8"))
SNAP = ROOT / "dashboards" / "snapshots"
LOCAL = ROOT / "dashboards" / "local_checks.json"
RESULTS = ROOT / "dashboards" / "phase8_results.json"
EVIDENCE = ROOT / "docs" / "figures" / "phase8"
CHECKLIST = ROOT / "docs" / "PHASE8_CHECKLIST.md"
TOL, TOL_RATIO = 1e-9, 1e-12
MODEL_COLS = {"Fleet average": "Fleet average", "Age only": "Age only (cycle count)", "Ridge (168 features)": "Ridge, all 168 features",
              "XGBoost (168 features)": "XGBoost, all 168 features", "LSTM": "LSTM", "CNN": "CNN"}


# ---------- independent readers and arithmetic ----------

def rows(path) -> list[dict]:
    with open(ROOT / path if not Path(path).is_absolute() else path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fl(x) -> float:
    return float("nan") if x in ("", None) else float(x)


def tf(x) -> bool:
    return str(x) == "True"


def mean(xs) -> float:
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else float("nan")


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def points(x, d=2) -> str:
    return f"{100 * x:+.{d}f} pts"


def close(a, b, tol=TOL) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def norm_sha(path) -> str:
    return sha((ROOT / path).read_bytes().replace(b"\r\n", b"\n"))


# ---------- the independent truth, from the raw results files ----------

def truth() -> dict:
    T = {}
    near = int(re.search(r"^NEAR = (\d+)", (ROOT / "models/calibration.py").read_text(), re.M).group(1))
    cfg = (ROOT / "maintenance/config.py").read_text()
    c = {k: int(re.search(rf"^{k} = (\d+)", cfg, re.M).group(1)) for k in ("ALERT_THRESHOLD", "HORIZON_DAYS", "CREWS_PER_DAY")}
    T["settings"] = c
    r5b = jload("models/results_phase5b.json")
    mc = r5b["mc_dropout"]
    kk = {"FD001": (mc["FD001_post_hoc"]["k_near"], mc["FD001_post_hoc"]["k_far"]), "FD003": (mc["FD003"]["k_near"], mc["FD003"]["k_far"])}
    preds = {"FD001": rows("models/predictions_phase5.csv"), "FD003": rows("models/predictions_phase5b.csv")}

    # fleet: bounds rebuilt with rule E from the published MC mean/sd (lower edge clipped at 0)
    fleet = {}
    for ds, pr in preds.items():
        kn, kf = kk[ds]
        eng = {}
        for r in pr:
            m, s = fl(r["MC mean"]), fl(r["MC std"])
            k = kn if m <= near else kf
            lo, hi = max(0.0, m - k * s), m + k * s
            y = fl(r["RUL"])
            eng[int(r["unit"])] = {"point": fl(r["LSTM"]), "mean": m, "sd": s, "k": k, "low": lo, "high": hi, "true": y,
                                   "flagged": lo <= c["ALERT_THRESHOLD"], "inside": lo <= y <= hi}
        fleet[ds] = eng
    T["fleet"] = fleet
    T["coverage_stated"] = {"FD001": mc["FD001_post_hoc"]["test_coverage"], "FD003": mc["FD003"]["test_coverage"]}

    # published model: re-apply the pre-registered rule (H1 Holm-significant with LSTM better, pooled near-failure MAE not worse)
    # Holm redone here from the raw p-values of the whole family (step-down: p x tests left, running maximum)
    alpha = float(re.search(r"def holm\(.*alpha: float = ([\d.]+)", (ROOT / "models/metrics.py").read_text()).group(1))
    tests = r5b["tests"]
    order = sorted(tests, key=lambda t: tests[t]["p"])
    adj, run = {}, 0.0
    for rank, t in enumerate(order):
        run = max(run, min(1.0, tests[t]["p"] * (len(order) - rank)))
        adj[t] = run
    h1_name = "H1 LSTM vs XGBoost, RMSE, FD001+FD003"
    h1 = tests[h1_name]
    pooled = r5b["pooled_fd001_fd003"]
    lstm_wins = adj[h1_name] < alpha and h1["diff"] < 0 and pooled["LSTM"]["near_failure_mae"] <= pooled["XGBoost, all 168 features"]["near_failure_mae"]
    T["published_model"] = "LSTM" if lstm_wins else "XGBoost, all 168 features"

    # model table: RMSE and near-failure MAE recomputed from the prediction files
    src = {"FD001": {**{k: rows("models/predictions_phase4.csv") for k in ("Fleet average", "Age only", "Ridge (168 features)", "XGBoost (168 features)")},
                     "LSTM": preds["FD001"], "CNN": preds["FD001"]},
           "FD003": {k: preds["FD003"] for k in MODEL_COLS}}
    models = {}
    for ds, per in src.items():
        for name, pr in per.items():
            col = MODEL_COLS[name]
            err = [fl(r[col]) - fl(r["RUL"]) for r in pr]
            nf = [abs(fl(r[col]) - fl(r["RUL"])) for r in pr if fl(r["RUL"]) <= near]
            models[(ds, name)] = {"rmse": math.sqrt(mean(e * e for e in err)), "near_failure_mae": mean(nf), "engines": len(pr),
                                  "near_failure_engines": len(nf), "published": name == T["published_model"]}
    T["models"] = models

    # maintenance plan, from the Phase 6 snapshot decisions
    dec = rows("maintenance/snapshot_decisions_phase6.csv")
    T["decisions"] = {(r["dataset"], int(r["engine_id"])): r for r in dec}
    plan = {}
    for ds in ("FD001", "FD003"):
        f = [r for r in dec if r["dataset"] == ds and tf(r["flagged"])]
        booked = [r for r in f if r["service_day"] != ""]
        plan[ds] = {"flagged": len(f), "scheduled": len(booked),
                    "unscheduled": sum(r["outcome"] == "flagged, no crew in horizon" for r in f),
                    "per_day": Counter(str(int(float(r["service_day"]))) for r in booked),
                    "risk": Counter(r["risk"] for r in f), "capacity": c["CREWS_PER_DAY"] * c["HORIZON_DAYS"]}
    T["plan"] = plan
    T["work_orders"] = Counter((r["dataset"], r["status"]) for r in rows("maintenance/work_orders_phase6.csv"))

    # OEE from the simulated machines: A = run/planned hours, P = actual/ideal units, Q = good/actual units
    mach = rows("maintenance/machines_phase6.csv")
    T["machines"] = mach
    oee = {}
    for scope in ("Pooled", "FD001", "FD003"):
        for pol in ("Run to failure", "Fixed interval", "Predictive"):
            g = [r for r in mach if r["policy"] == pol and (scope == "Pooled" or r["dataset"] == scope)]
            s = {k: math.fsum(fl(r[k]) for r in g) for k in ("run_hours", "planned_hours", "ideal_units", "actual_units", "good_units", "downtime_hours")}
            a, p, q = s["run_hours"] / s["planned_hours"], s["actual_units"] / s["ideal_units"], s["good_units"] / s["actual_units"]
            oee[(scope, pol)] = {"availability": a, "performance": p, "quality": q, "oee": a * p * q,
                                 "unplanned_failures": sum(r["event"] == "failure" for r in g),
                                 "planned_services": sum(r["event"] == "service" for r in g), "downtime_hours": s["downtime_hours"],
                                 "wasted_cycles": sum(int(r["wasted_cycles"]) for r in g), "machines": len(g)}
    T["oee"] = oee
    r6 = jload("maintenance/results_phase6.json")
    T["gaps"] = {g: {"diff": oee[("Pooled", "Predictive")]["oee"] - oee[("Pooled", g.split(" - ")[1])]["oee"],
                     "ci": r6["pooled_gaps"][g]["ci"]} for g in ("Predictive - Fixed interval", "Predictive - Run to failure")}
    T["r6"] = r6

    # monitoring: Phase 7 drift per run (a sensor counts only in a tested run), retraining flags, Phase 7b C1
    drift = defaultdict(lambda: [False, False])
    for r in rows("results/phase7/phase7_drift.csv"):
        t = tf(r["tested"])
        drift[r["run_id"]][0] |= t
        drift[r["run_id"]][1] |= t and tf(r["drifted"])
    tested = [v for v in drift.values() if v[0]]
    T["drift"] = dict(drift)
    T["p7_false_alarm"] = (sum(v[1] for v in tested), len(tested))
    h7 = rows("results/phase7/phase7_health.csv")
    T["p7_flags"] = (sum(tf(r["needs_retraining"]) for r in h7), len(h7))
    T["health"] = {"phase7_v1": h7}
    crit = jload("pipeline/phase7b_criteria.json")
    T["c1_limit"] = float(re.search(r"<= (\d+(?:\.\d+)?)%", crit["criteria"]["C1_clean_false_alarms"]).group(1)) / 100
    T["p7_target"] = float(re.search(r"target (\d+(?:\.\d+)?)%", crit["why"]).group(1)) / 100
    c1 = {}
    for fleet_ in ("synchronized", "staggered"):
        h = rows(f"results/phase7b/fresh_{fleet_}/phase7_health.csv")
        T["health"][f"phase7b_v2_{fleet_}"] = h
        for ds in ("FD002", "FD004"):
            g = [r for r in h if r["dataset"] == ds]
            c1[(ds, fleet_)] = (sum(tf(r["needs_retraining"]) for r in g), len(g))
    T["c1"] = c1

    # Monte Carlo noise: published vs rolling forecast at the last cycle
    roll = {}
    for ds in ("FD001", "FD003"):
        for r in rows(f"maintenance/forecasts_rolling_{ds}.csv"):
            key = (ds, int(r["engine_id"]))
            if key not in roll or int(r["cycle"]) > int(roll[key]["cycle"]):
                roll[key] = r
    T["rolling"] = roll
    diffs = [abs(fl(d["rul_mc_mean"]) - fl(roll[k]["rul_mc_mean"])) for k, d in T["decisions"].items()]
    flips = sum((fl(d["rul_low"]) <= c["ALERT_THRESHOLD"]) != (fl(roll[k]["rul_low"]) <= c["ALERT_THRESHOLD"]) for k, d in T["decisions"].items())
    T["mc_noise"] = (max(diffs), flips)
    return T


# ---------- what each page should display ----------

def expected_metrics(T, page: str, view: dict) -> dict:
    out = {}
    if page == "fleet":
        dss = ["FD001", "FD003"] if view["dataset"] == "Both" else [view["dataset"]]
        for ds in dss:
            e = T["fleet"][ds].values()
            out[f"Engines ({ds})"] = str(len(e))
            out[f"Flagged ({ds})"] = str(sum(x["flagged"] for x in e))
            out[f"Interval coverage on these engines ({ds})"] = pct(sum(x["inside"] for x in e) / len(e))
    elif page == "plan":
        dss = ["FD001", "FD003"] if view["dataset"] == "Both" else [view["dataset"]]
        for ds in dss:
            p = T["plan"][ds]
            out.update({f"Crew slots in the horizon ({ds})": str(p["capacity"]), f"Engines flagged ({ds})": str(p["flagged"]),
                        f"Scheduled ({ds})": str(p["scheduled"]), f"Flagged, no crew in horizon ({ds})": str(p["unscheduled"]),
                        f"Solver status ({ds})": T["r6"]["datasets"][ds]["snapshot"]["schedule_status"]})
    elif page == "oee":
        for pol in ("Run to failure", "Fixed interval", "Predictive"):
            out[f"OEE, {pol} ({view['scope']}, simulated)"] = pct(T["oee"][(view["scope"], pol)]["oee"])
        for g, v in T["gaps"].items():
            out[f"OEE gap: {g.replace(' - ', ' minus ')} (simulated)"] = points(v["diff"])
    elif page == "monitoring":
        k, n = T["p7_false_alarm"]
        out["Phase 7 drift false-alarm rate on clean test runs"] = pct(k / n)
        out["Phase 7 runs flagged for retraining"] = f"{T['p7_flags'][0]} of {T['p7_flags'][1]}"
    elif page == "about":
        out["Snapshot data checksum"] = None  # filled by the caller from the recomputed checksum
        out["Largest change in MC mean between two draw sets (cycles)"] = f"{T['mc_noise'][0]:.2f}"
        out["Engines whose flag differs between the two draw sets"] = str(T["mc_noise"][1])
    return out


def expected_tables(T, page: str, view: dict) -> list[tuple[str, list]]:
    """(description, [(row key, column, expected string)]) for the tables whose numbers the page shows."""
    checks = []
    if page == "fleet":
        dss = ["FD001", "FD003"] if view["dataset"] == "Both" else [view["dataset"]]
        cells = []
        for ds in dss:
            for name in MODEL_COLS:
                m = T["models"][(ds, name)]
                label = name + (" (published)" if m["published"] else "")
                cells += [((ds, label), "RMSE (cycles)", f"{m['rmse']:.2f}"), ((ds, label), "Near-failure MAE (cycles)", f"{m['near_failure_mae']:.2f}"),
                          ((ds, label), "Test engines", str(m["engines"])), ((ds, label), "Near-failure engines", str(m["near_failure_engines"]))]
        checks.append(("model table", cells))
    elif page == "oee":
        cells = []
        for pol in ("Run to failure", "Fixed interval", "Predictive"):
            o = T["oee"][(view["scope"], pol)]
            cells += [((pol,), "Availability", pct(o["availability"])), ((pol,), "Performance", pct(o["performance"])),
                      ((pol,), "Quality", pct(o["quality"])), ((pol,), "OEE", pct(o["oee"])),
                      ((pol,), "Unplanned failures", str(o["unplanned_failures"])), ((pol,), "Planned services", str(o["planned_services"])),
                      ((pol,), "Downtime (hours)", str(int(o["downtime_hours"]))), ((pol,), "Wasted life (cycles)", str(o["wasted_cycles"]))]
        checks.append(("policy table", cells))
    elif page == "monitoring":
        cells = []
        for (ds, fleet_), (k, n) in T["c1"].items():
            cells += [((ds, fleet_), "Flagged runs", f"{k} of {n}"), ((ds, fleet_), "Rate", pct(k / n)),
                      ((ds, fleet_), "Result", "PASS" if k / n <= T["c1_limit"] else "FAIL")]
        checks.append(("C1 table", cells))
    elif page == "plan":
        dss = ["FD001", "FD003"] if view["dataset"] == "Both" else [view["dataset"]]
        cells = [((ds, st_), "Work orders", str(n)) for (ds, st_), n in T["work_orders"].items() if ds in dss]
        checks.append(("work-order table", cells))
    return checks


def table_lookup(df, key: tuple, col: str):
    keycols = {1: ["Policy"], 2: ["Dataset", "Model"] if "Model" in df.columns else (["Dataset", "Fleet"] if "Fleet" in df.columns else ["Dataset", "Status"])}[len(key)]
    hit = df
    for kc, kv in zip(keycols, key):
        if kc not in hit.columns:
            return None
        hit = hit[hit[kc].astype(str) == str(kv)]
    if len(hit) != 1 or col not in hit.columns:
        return None
    v = hit[col].iloc[0]
    return f"{v:.2f}" if isinstance(v, float) and col.endswith("(cycles)") and "Wasted" not in col else str(v)


# ---------- checks ----------

class Card:
    def __init__(self):
        self.res = {}

    def set(self, cid, ok, detail):
        prev = self.res.get(cid)
        if prev and prev[0] is False:
            return  # a failure is never overwritten
        self.res[cid] = (None if ok is None else bool(ok), detail)  # None = pending: evidence not produced yet


def check_manifest(snap: Path, card: Card):
    man = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    probs = []
    for name, f in man["files"].items():
        b = (snap / name).read_bytes()
        if sha(b) != f["sha256"]:
            probs.append(f"{name} sha256")
        if name.endswith(".csv"):
            rs = rows(snap / name)
            if len(rs) != f["rows"]:
                probs.append(f"{name} rows {len(rs)} != {f['rows']}")
            keys = [tuple(r[k] for k in f["key"]) for r in rs]
            if len(set(keys)) != len(keys):
                probs.append(f"{name} key not unique")
            typed = [tuple(int(x) if x.lstrip("-").isdigit() else x for x in k) for k in keys]
            if typed != sorted(typed):
                probs.append(f"{name} not sorted by key")
            ids = sorted({r["run_id"] for r in rs if r.get("run_id")}) if rs and "run_id" in rs[0] else []
            if len(ids) != f["run_ids_count"] or (ids and sha("\n".join(ids).encode()) != f["run_ids_sha256"]):
                probs.append(f"{name} run IDs")
    data = "".join(f"{n}:{man['files'][n]['sha256']}\n" for n in sorted(man["files"]))
    if sha(data.encode()) != man["data_checksum"]:
        probs.append("data_checksum")
    card.set("D1.2", not probs, f"{len(man['files'])} files match the manifest (rows, sha256, sorted unique keys, run IDs)" if not probs else "; ".join(probs))
    stale = [s for s, h in man["sources"].items() if norm_sha(s) != h]
    return man, stale


def check_cells(T, snap: Path, card: Card):
    probs = []
    th = T["settings"]["ALERT_THRESHOLD"]
    # fleet: copied cells exact vs the decisions file; bounds vs the independent rebuild; rolling vs the rolling files
    for r in rows(snap / "fleet.csv"):
        key = (r["dataset"], int(r["engine_id"]))
        d, e, ro = T["decisions"][key], T["fleet"][key[0]][key[1]], T["rolling"][key]
        for c in ("cycle", "rul_point", "rul_mc_mean", "rul_mc_std", "k", "rul_low", "rul_high", "true_rul_capped", "risk", "outcome", "flagged"):
            if r[c] != d[c] and not (c not in ("risk", "outcome", "flagged") and fl(r[c]) == fl(d[c])):
                probs.append(f"fleet {key} {c}")
        if not (close(fl(r["rul_low"]), e["low"]) and close(fl(r["rul_high"]), e["high"]) and close(fl(r["rul_point"]), e["point"])
                and close(fl(r["true_rul_capped"]), e["true"]) and tf(r["flagged"]) == e["flagged"] and tf(r["inside_interval"]) == e["inside"]):
            probs.append(f"fleet {key} vs rebuilt interval")
        if tf(r["flagged"]) != (fl(r["rul_low"]) <= th):
            probs.append(f"fleet {key} flag vs lower bound")
        if r["cycle"] != ro["cycle"] or fl(r["rolling_mc_mean"]) != fl(ro["rul_mc_mean"]) or fl(r["rolling_rul_low"]) != fl(ro["rul_low"]) or fl(r["rolling_rul_high"]) != fl(ro["rul_high"]):
            probs.append(f"fleet {key} rolling columns")
    if len(rows(snap / "fleet.csv")) != len(T["decisions"]):
        probs.append("fleet row count vs decisions")
    # models
    seen = set()
    for r in rows(snap / "models.csv"):
        m = T["models"].get((r["dataset"], r["model"]))
        seen.add((r["dataset"], r["model"]))
        if m is None or not (close(fl(r["rmse"]), m["rmse"]) and close(fl(r["near_failure_mae"]), m["near_failure_mae"])
                             and int(r["engines"]) == m["engines"] and int(r["near_failure_engines"]) == m["near_failure_engines"]
                             and tf(r["published"]) == m["published"]):
            probs.append(f"models {r['dataset']} {r['model']}")
    if seen != set(T["models"]):
        probs.append("models: missing or extra rows")
    # plan
    plan = rows(snap / "plan.csv")
    flagged = {k: d for k, d in T["decisions"].items() if tf(d["flagged"])}
    if {(r["dataset"], int(r["engine_id"])) for r in plan} != set(flagged):
        probs.append("plan: engines differ from the flagged decisions")
    for r in plan:
        d = flagged.get((r["dataset"], int(r["engine_id"])))
        if d is None:
            continue
        sd = "" if d["service_day"] == "" else str(int(float(d["service_day"])))
        if not (r["decision_cycle"] == d["cycle"] and fl(r["rul_low_used"]) == fl(d["rul_low"]) and r["risk"] == d["risk"]
                and r["reason"] == d["reason"] and (r["scheduled_day"] == "" and sd == "" or sd != "" and int(float(r["scheduled_day"])) == int(sd))
                and r["status"] == d["outcome"]):
            probs.append(f"plan {r['dataset']} {r['engine_id']}")
    for ds in ("FD001", "FD003"):
        per = Counter(str(int(float(r["scheduled_day"]))) for r in plan if r["dataset"] == ds and r["scheduled_day"] != "")
        snap6 = T["r6"]["datasets"][ds]["snapshot"]
        if per != T["plan"][ds]["per_day"] or {k: v for k, v in snap6["per_day"].items() if v} != dict(per) or max(per.values()) > T["settings"]["CREWS_PER_DAY"]:
            probs.append(f"plan {ds} per day")
        if snap6["flagged"] != T["plan"][ds]["flagged"] or snap6["scheduled"] != T["plan"][ds]["scheduled"] or snap6["risk_counts"] != dict(T["plan"][ds]["risk"]):
            probs.append(f"plan {ds} vs results_phase6.json")
    # work orders
    wo = {(r["dataset"], int(r["engine_id"]), r["created_cycle"]): r for r in rows("maintenance/work_orders_phase6.csv")}
    snapwo = rows(snap / "work_orders.csv")
    if len(snapwo) != len(wo):
        probs.append(f"work orders: {len(snapwo)} rows vs {len(wo)}")
    for r in snapwo:
        w = wo.get((r["dataset"], int(r["engine_id"]), r["decision_cycle"]))
        if w is None or not (r["scheduled_day"] == w["day"] and r["service_cycle"] == w["service_cycle"] and fl(r["rul_low_used"]) == fl(w["rul_low"])
                             and r["status"] == w["status"] and r["forecast_cycle"] == w["forecast_cycle"] and r["risk"] == w["risk"]):
            probs.append(f"work order {r['dataset']} {r['engine_id']} {r['decision_cycle']}")
    # OEE machines and policies
    mach = {(r["dataset"], r["policy"], r["machine"]): r for r in T["machines"]}
    om = rows(snap / "oee_machines.csv")
    if len(om) != len(mach):
        probs.append(f"oee_machines: {len(om)} rows vs {len(mach)}")
    ren = {"machine_id": "machine", "life_cycles": "life"}
    for r in om:
        m = mach.get((r["dataset"], r["policy"], r["machine_id"]))
        if m is None or any(fl(r[c]) != fl(m[ren.get(c, c)]) for c in ("life_cycles", "run_cycles", "wasted_cycles", "degraded_cycles", "planned_hours",
                                                                        "run_hours", "downtime_hours", "ideal_units", "actual_units", "good_units")) or r["event"] != m["event"]:
            probs.append(f"oee machine {r['dataset']} {r['policy']} {r['machine_id']}")
    for r in rows(snap / "oee_policies.csv"):
        if r["kind"] == "policy":
            o = T["oee"][(r["scope"], r["policy"])]
            j = T["r6"]["pooled"][r["policy"]] if r["scope"] == "Pooled" else T["r6"]["datasets"][r["scope"]]["policies"][r["policy"]]
            for c in ("availability", "performance", "quality", "oee"):
                if not (close(fl(r[c]), o[c], TOL_RATIO) and close(fl(r[c]), j[c], TOL_RATIO)):
                    probs.append(f"oee policy {r['scope']} {r['policy']} {c}")
            for c in ("unplanned_failures", "planned_services", "wasted_cycles", "machines"):
                if int(float(r[c])) != o[c] or o[c] != j[c]:
                    probs.append(f"oee policy {r['scope']} {r['policy']} {c}")
            if not close(fl(r["downtime_hours"]), o["downtime_hours"]):
                probs.append(f"oee policy {r['scope']} {r['policy']} downtime")
        else:
            g = T["gaps"][r["policy"]]
            if not (close(fl(r["oee_gap"]), g["diff"], TOL_RATIO) and [fl(r["ci_low"]), fl(r["ci_high"])] == g["ci"]):
                probs.append(f"oee gap {r['policy']}")
    # monitoring
    mon = rows(snap / "monitoring_runs.csv")
    for sc, h in T["health"].items():
        got = {r["shift"] + r["dataset"]: r for r in mon if r["scenario"] == sc}
        if len(got) != len(h):
            probs.append(f"monitoring {sc}: {len(got)} runs vs {len(h)}")
        for r in h:
            g = got.get(r["shift"] + r["dataset"])
            if g is None or any((g[c] != r[c]) and not (fl(g[c]) == fl(r[c]) if c in ("rmse", "coverage_90") else False)
                                for c in ("run_id", "n_labelled", "n_scored", "rmse", "coverage_90", "drifted_sensors", "needs_retraining", "retraining_reasons")):
                probs.append(f"monitoring {sc} {r['run_id']}")
            if g is not None and sc == "phase7_v1":
                t, d = T["drift"][r["run_id"]]
                if tf(g["drift_tested"]) != t or tf(g["any_drifted"]) != d:
                    probs.append(f"monitoring drift {r['run_id']}")
    probs = sorted(set(probs))
    card.set("D1.3", not probs, "every snapshot cell equals the independent files (fleet, models, plan, work orders, OEE, monitoring)"
             if not probs else f"{len(probs)} differences: {probs[:4]}")
    # cross-checks against the published summaries
    k, n = T["p7_false_alarm"]
    p7 = jload("results/phase7/results_phase7.json")
    c1j = jload("results/phase7b/checks.json")["C1"]
    extra = []
    if not close(k / n, p7["drift_false_alarm_rate_test"]):
        extra.append("Phase 7 false-alarm rate vs results_phase7.json")
    for (ds, fleet_), (kk, nn) in T["c1"].items():
        j = c1j[f"{ds} {fleet_}"]
        if j["flagged"] != kk or j["runs"] != nn:
            extra.append(f"C1 {ds} {fleet_} vs checks.json")
    for ds in ("FD001", "FD003"):
        cov = sum(x["inside"] for x in T["fleet"][ds].values()) / len(T["fleet"][ds])
        if not close(cov, T["coverage_stated"][ds]):
            extra.append(f"{ds} coverage {cov} vs stated {T['coverage_stated'][ds]}")
    if T["published_model"] != jload("models/results_phase5b.json")["verdict"]["published_winner"]:
        extra.append("published model rule")
    if extra:
        card.set("D1.3", False, "; ".join(extra))
    return probs + extra


def check_facts(T, snap: Path, card: Card):
    f = {k: v["value"] for k, v in json.loads((snap / "facts.json").read_text(encoding="utf-8")).items()}
    want = {"published_model": T["published_model"], "alert_threshold": T["settings"]["ALERT_THRESHOLD"],
            "crews_per_day": T["settings"]["CREWS_PER_DAY"], "horizon_days": T["settings"]["HORIZON_DAYS"],
            "coverage_stated_FD001": T["coverage_stated"]["FD001"], "coverage_stated_FD003": T["coverage_stated"]["FD003"],
            "c1_limit": T["c1_limit"], "p7_drift_target": T["p7_target"], "definitions": SPEC["definitions"],
            "solver_status_FD001": T["r6"]["datasets"]["FD001"]["snapshot"]["schedule_status"],
            "solver_status_FD003": T["r6"]["datasets"]["FD003"]["snapshot"]["schedule_status"]}
    bad = [k for k, v in want.items() if f.get(k) != v]
    if bad:
        card.set("D1.3", False, f"facts.json differs from its sources: {bad}")


def check_app(T, card: Card, data_checksum: str):
    from dashboards import app_checks as A
    for cid, ok, detail in A.check_all():
        card.set(cid, ok, detail)
    probs, n = [], 0
    for page, views in A.rendered_numbers().items():
        for view_s, got in views.items():
            view = json.loads(view_s) or {}
            exp = expected_metrics(T, page, view) if page != "about" else {**expected_metrics(T, page, view), "Snapshot data checksum": data_checksum[:16]}
            if page == "plan" and "dataset" not in view:
                continue
            for label, value in got["metrics"].items():
                n += 1
                if label not in exp:
                    probs.append(f"{page} {view}: unexpected metric '{label}'")
                elif exp[label] != value:
                    probs.append(f"{page} {view}: '{label}' shows {value}, expected {exp[label]}")
            for label in exp:
                if label not in got["metrics"]:
                    probs.append(f"{page} {view}: missing metric '{label}'")
            for desc, cells in expected_tables(T, page, view):
                tabs = got["tables"]
                for key, col, want in cells:
                    n += 1
                    vals = [v for v in (table_lookup(t, key, col) for t in tabs) if v is not None]
                    if want not in vals:
                        probs.append(f"{page} {view} {desc} {key} {col}: shows {vals[:1]}, expected {want}")
    card.set("D1.1", not probs, f"{n} displayed values in every filter view equal the independent recomputation" if not probs
             else f"{len(probs)} mismatches: {probs[:3]}")


def check_db(snap: Path, card: Card, schemas: str):
    """D1.4 (public) and D1.5 (ops, ops7b_*): snapshots vs PostgreSQL cell for cell, with queries written here."""
    try:
        from sqlalchemy import create_engine, text
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        return False
    eng = create_engine(url)
    run6 = T_RUN6()
    probs = []
    with eng.connect() as c:
        def q(sql):
            res = c.execute(text(sql))
            cols = list(res.keys())
            return [dict(zip(cols, r)) for r in res]
        if not q("SELECT 1 AS x FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'fact_alert'") or \
           not q(f"SELECT 1 AS x FROM public.fact_alert WHERE run_id = '{run6}' LIMIT 1"):
            return False
        same = lambda a, b: (a is None and b == "") or str(a) == b or (isinstance(a, (int, float)) and b != "" and float(a) == float(b)) or (isinstance(a, bool) and str(a) == b)
        alerts = {(r["dataset"], r["engine_id"], r["context"], r["decision_cycle"]): r for r in q(f"SELECT * FROM public.fact_alert WHERE run_id = '{run6}'")}
        for fname, ctx in (("plan.csv", "snapshot"), ("work_orders.csv", "replay")):
            s = rows(snap / fname)
            if len(s) != sum(k[2] == ctx for k in alerts):
                probs.append(f"{fname}: {len(s)} rows vs {sum(k[2] == ctx for k in alerts)} in fact_alert")
            for r in s:
                a = alerts.get((r["dataset"], int(r["engine_id"]), ctx, int(r["decision_cycle"])))
                cols = [c_ for c_ in r if c_ not in ("dataset", "engine_id", "decision_cycle")]
                if a is None or not all(same(a[c_], r[c_]) for c_ in cols):
                    probs.append(f"{fname} {r['dataset']} {r['engine_id']} {r['decision_cycle']}")
        oee = {(r["dataset"], r["policy"], r["machine_id"]): r for r in q(f"SELECT * FROM public.fact_oee WHERE run_id = '{run6}'")}
        s = rows(snap / "oee_machines.csv")
        if len(s) != len(oee):
            probs.append(f"oee_machines: {len(s)} rows vs {len(oee)} in fact_oee")
        for r in s:
            a = oee.get((r["dataset"], r["policy"], int(r["machine_id"])))
            if a is None or not all(same(a[c_], r[c_]) for c_ in r):
                probs.append(f"oee_machines {r['dataset']} {r['policy']} {r['machine_id']}")
        last = {(r["dataset"], r["engine_id"]): r for r in q(f"""SELECT p.* FROM public.fact_prediction p JOIN (SELECT dataset, engine_id, max(cycle) AS c
                FROM public.fact_prediction WHERE run_id = '{run6}' GROUP BY 1, 2) m ON p.dataset = m.dataset AND p.engine_id = m.engine_id AND p.cycle = m.c
                WHERE p.run_id = '{run6}'""")}
        for r in rows(snap / "fleet.csv"):
            a = last.get((r["dataset"], int(r["engine_id"])))
            if a is None or not (int(r["cycle"]) == a["cycle"] and float(r["rul_point"]) == a["rul_point"] and float(r["rolling_mc_mean"]) == a["rul_mc_mean"]
                                 and float(r["rolling_rul_low"]) == a["rul_low"] and float(r["rolling_rul_high"]) == a["rul_high"]):
                probs.append(f"fleet {r['dataset']} {r['engine_id']} vs fact_prediction")
        card.set("D1.4", not probs, "plan, work orders, OEE machines and fleet forecasts equal public.fact_alert / fact_oee / fact_prediction cell for cell"
                 if not probs else f"{len(probs)} differences: {probs[:3]}")
        if schemas == "all":
            p5 = []
            mon = rows(snap / "monitoring_runs.csv")
            schema_of = {"phase7_v1": "ops", "phase7b_v2_synchronized": "ops7b_fresh_synchronized", "phase7b_v2_staggered": "ops7b_fresh_staggered"}
            for sc, schema in schema_of.items():
                db = {(r["dataset"], r["shift"]): r for r in q(f'SELECT * FROM "{schema}".fact_model_health')}
                s = [r for r in mon if r["scenario"] == sc]
                if len(s) != len(db):
                    p5.append(f"{sc}: {len(s)} rows vs {len(db)}")
                for r in s:
                    a = db.get((r["dataset"], int(r["shift"])))
                    cols = ("run_id", "n_labelled", "n_scored", "rmse", "coverage_90", "drifted_sensors", "needs_retraining", "retraining_reasons")
                    if a is None or not all(same(a[c_], r[c_]) for c_ in cols):
                        p5.append(f"{sc} {r['run_id']}")
            card.set("D1.5", not p5, f"{len(mon)} monitoring runs equal ops / ops7b_* fact_model_health cell for cell" if not p5 else f"{len(p5)} differences: {p5[:3]}")
    return True


def T_RUN6() -> str:
    return jload("maintenance/results_phase6.json")["run_id"]


def check_docs(T, card: Card):
    doc = ROOT / "docs" / "DASHBOARDS.md"
    if not doc.exists():
        card.set("D1.6", None, "docs/DASHBOARDS.md not generated yet")
        return
    try:
        from dashboards.report_phase8 import render
        same = render() == doc.read_text(encoding="utf-8").replace("\r\n", "\n")
    except Exception as e:  # report needs results; any failure is a failed check
        same, e_ = False, e
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pooled = {p: pct(T["oee"][("Pooled", p)]["oee"]) for p in ("Predictive", "Fixed interval", "Run to failure")}
    need = [*pooled.values(), "Simulated", "streamlit.app"]
    miss = [s for s in need if s not in readme]
    card.set("D1.6", same and not miss, "docs/DASHBOARDS.md equals its generator; README Phase 8 numbers present" if same and not miss
             else f"doc matches generator: {same}; README missing {miss}")


def check_powerbi(T, card: Card):
    exp_dir = EVIDENCE / "powerbi_exports"
    files = sorted(exp_dir.glob("*.csv")) if exp_dir.exists() else []
    if not files:
        card.set("D1.7", None, "no Power BI exports yet (docs/figures/phase8/powerbi_exports/)")
        return
    from dashboards.powerbi_checks import check_exports
    ok, detail = check_exports(T, files)
    card.set("D1.7", ok, detail)


def check_ci_and_local(card: Card, man: dict, stale: list, clean_env: bool):
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    m = re.search(r"\n  dashboards:\n(.*?)(?=\n  \w[\w-]*:\n|\Z)", ci, re.S)
    job = m.group(1) if m else ""
    clean_job = bool(job) and "pip install -r app/requirements.txt" in job and "requirements-lock" not in job and "verify_phase8 --clean-env" in job
    if clean_env:
        heavy = [p for p in ("torch", "sqlalchemy", "xgboost", "prefect", "pytest") if importlib.util.find_spec(p)]
        card.set("D2.2", clean_job and not heavy, "running with only app/requirements.txt; CI job 'dashboards' installs nothing else" if clean_job and not heavy
                 else f"CI job present: {clean_job}; non-app packages installed: {heavy}")
    else:
        card.set("D2.2", clean_job, "CI job 'dashboards' installs only app/requirements.txt and runs verify_phase8 --clean-env (run there)" if clean_job
                 else "no CI job installs only app/requirements.txt")
    linux = "runs-on: ubuntu" in job and clean_job
    card.set("D2.3", linux and not stale, "CI runs verify_phase8 on Linux; snapshot sources unchanged since export" if linux and not stale
             else f"CI job on Linux: {linux}; stale sources: {stale}")
    if LOCAL.exists():
        loc = json.loads(LOCAL.read_text(encoding="utf-8"))
        current = loc.get("data_checksum") == man["data_checksum"] and loc.get("sources") == man["sources"]
        for cid in ("D1.5", "D2.1"):
            if cid in loc["checks"] and cid not in card.res:
                ok, detail = loc["checks"][cid]
                card.set(cid, ok and current, detail + ("" if current else " (evidence is for an older snapshot: re-run --local)"))
    for cid in ("D1.5", "D2.1"):
        if cid not in card.res:
            card.set(cid, None, "not run: needs python -m dashboards.verify_phase8 --local on the machine with the database")


def check_manual(card: Card):
    items = {"D2.4": ["deployed_about.png"], "D5.5": [f"phone_{p}.png" for p in ("fleet", "plan", "oee", "monitoring", "about")],
             "D5.6": ["powerbi_report.pdf"]}
    text_ = CHECKLIST.read_text(encoding="utf-8") if CHECKLIST.exists() else ""
    for cid, files in items.items():
        missing = [f for f in files if not (EVIDENCE / f).exists()]
        ticked = bool(re.search(rf"- \[x\] \*\*{re.escape(cid)}\*\*", text_))
        state = True if not missing and ticked else (None if len(missing) == len(files) and not ticked else False)
        card.set(cid, state, f"evidence {files} committed and checklist ticked" if state
                 else f"missing evidence {missing}; checklist ticked: {ticked}")


def local_checks(snap: Path, card: Card, man: dict):
    """D2.1: rebuild twice, byte-identical and equal to the committed files. D1.5 is set by check_db(schemas='all')."""
    from dashboards.export_snapshots import export
    a, b = export(), export()
    same = a == b and all((snap / n).read_bytes() == v for n, v in a.items())
    card.set("D2.1", same, "two rebuilds are byte-identical and equal to the committed snapshots" if same else "rebuild differs")
    out = {"note": "Evidence for checks that need the local database; written by python -m dashboards.verify_phase8 --local",
           "data_checksum": man["data_checksum"], "sources": man["sources"],
           "checks": {c: list(card.res[c]) for c in ("D1.5", "D2.1") if c in card.res}}
    LOCAL.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def _edit_csv(path: Path, change) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields, rs = rd.fieldnames, list(rd)
    rs = change(rs)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rs)


def _resign(d: Path) -> None:
    """Make the manifest agree with the edited files, so only the numeric checks can catch the planted bug."""
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for n, f in man["files"].items():
        f["sha256"] = sha((d / n).read_bytes())
        if n.endswith(".csv"):
            f["rows"] = len(rows(d / n))
    man["data_checksum"] = sha("".join(f"{n}:{man['files'][n]['sha256']}\n" for n in sorted(man["files"])).encode())
    (d / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")


def _set(rs, i, **kv):
    rs[i] = {**rs[i], **kv}
    return rs


PLANTED = {
    "wrong OEE (pooled predictive +0.1 pt)": ("oee_policies.csv", lambda rs: [
        {**r, "oee": repr(float(r["oee"]) + 0.001)} if r["scope"] == "Pooled" and r["policy"] == "Predictive" else r for r in rs]),
    "off-by-one shift (last Phase 7 run)": ("monitoring_runs.csv", lambda rs: _set(rs, max(i for i, r in enumerate(rs) if r["scenario"] == "phase7_v1"),
                                                                                    shift=str(int(rs[max(i for i, r in enumerate(rs) if r["scenario"] == "phase7_v1")]["shift"]) + 1))),
    "dropped engine (FD003 engine 1)": ("fleet.csv", lambda rs: [r for r in rs if not (r["dataset"] == "FD003" and r["engine_id"] == "1")]),
    "swapped interval bounds (FD001 engine 1)": ("fleet.csv", lambda rs: [
        {**r, "rul_low": r["rul_high"], "rul_high": r["rul_low"]} if r["dataset"] == "FD001" and r["engine_id"] == "1" else r for r in rs]),
    "wrong flagged status (first FD001 flagged engine)": ("fleet.csv", lambda rs: _set(rs, next(i for i, r in enumerate(rs) if r["flagged"] == "True"), flagged="False")),
    "service booked a day late (first planned engine)": ("plan.csv", lambda rs: _set(rs, next(i for i, r in enumerate(rs) if r["scheduled_day"]),
                                                                                      scheduled_day=str(int(float(rs[next(i for i, r in enumerate(rs) if r["scheduled_day"])]["scheduled_day"])) + 1))),
    "stale snapshot (a source changed after export)": None,
}


def planted_bugs(snap: Path = SNAP) -> list[tuple[str, bool, str]]:
    """Each planted bug in a re-signed copy of the snapshots must make D1.2, D1.3 or D2.3 fail."""
    import shutil
    import tempfile
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        T = truth()
        for i, (name, bug) in enumerate(PLANTED.items()):
            d = Path(tmp) / f"bug{i}"
            shutil.copytree(snap, d)
            if bug is None:
                man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                man["sources"]["maintenance/machines_phase6.csv"] = "0" * 64   # what the export saw differs from today's file
                (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
            else:
                _edit_csv(d / bug[0], bug[1])
                _resign(d)
            card = Card()
            man, stale = check_manifest(d, card)
            if stale:
                card.set("D2.3", False, f"stale: {stale}")
            try:
                check_cells(T, d, card)
            except Exception as e:  # a crash on bad data also counts as caught, but say so
                card.set("D1.3", False, f"verifier stopped: {type(e).__name__}")
            caught = [c for c, (ok, _) in card.res.items() if not ok]
            out.append((name, bool(caught), ", ".join(caught) or "NOT CAUGHT"))
    return out


def verify(snap: Path = SNAP, app: bool = True, local: bool = False, clean_env: bool = False, db: bool = True) -> Card:
    card = Card()
    T = truth()
    man, stale = check_manifest(snap, card)
    check_cells(T, snap, card)
    check_facts(T, snap, card)
    if stale:
        card.set("D2.3", False, f"stale snapshot: sources changed since export: {stale}")
    if db and not clean_env:
        if not check_db(snap, card, "all" if local else "public"):
            card.set("D1.4", None, "not run: no database with the Phase 6 tables here")
    if local:
        local_checks(snap, card, man)
    if app:
        check_app(T, card, man["data_checksum"])
    if snap == SNAP:
        pb = planted_bugs(snap)
        missed = [n for n, ok, _ in pb if not ok]
        card.set("D3.3", len(pb) - len(missed) >= 5 and not missed,
                 f"{len(pb) - len(missed)}/{len(pb)} planted bugs caught: " + "; ".join(f"{n} -> {c}" for n, _, c in pb))
    check_docs(T, card)
    check_powerbi(T, card)
    check_ci_and_local(card, man, stale, clean_env)
    check_manual(card)
    return card


def report(card: Card, app: bool) -> dict:
    dims = []
    for d in CARD["dimensions"]:
        checks = [(c["id"], c["kind"], card.res.get(c["id"])) for c in d["checks"]]
        passed = all(r is not None and r[0] is True for _, _, r in checks)
        dims.append({"id": d["id"], "name": d["name"], "passed": passed,
                     "checks": [{"id": cid, "kind": k, "passed": None if r is None else r[0], "detail": "not run in this mode" if r is None else r[1]}
                                for cid, k, r in checks]})
    return {"score": sum(d["passed"] for d in dims), "out_of": len(dims), "dimensions": dims}


def main() -> int:
    args = sys.argv[1:]
    snap = Path(args[args.index("--snapshots") + 1]) if "--snapshots" in args else SNAP
    app, local, clean = "--no-app" not in args, "--local" in args, "--clean-env" in args
    card = verify(snap, app=app, local=local, clean_env=clean)
    rep = report(card, app)
    for d in rep["dimensions"]:
        state = "PASS" if d["passed"] else ("FAIL" if any(c["passed"] is False for c in d["checks"]) else "PENDING")
        print(f"{d['id']} {d['name']}: {state}")
        for c in d["checks"]:
            mark = "PASS" if c["passed"] else ("FAIL" if c["passed"] is False else ("PEND" if "not run in this mode" not in c["detail"] else "----"))
            print(f"   {mark}  {c['id']} [{c['kind']}] {c['detail']}")
    print(f"Phase 8 score: {rep['score']}/{rep['out_of']} (5 only if every check passes)")
    if os.environ.get("GITHUB_ACTIONS"):
        for d in rep["dimensions"]:
            for c in d["checks"]:
                if c["passed"] is False:
                    print(f"::error title=Phase 8 {c['id']}::{c['detail']}")
    if snap == SNAP and app and not clean:
        RESULTS.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    pending = sum(c["passed"] is None and "not run in this mode" not in c["detail"] for d in rep["dimensions"] for c in d["checks"])
    print(f"{pending} checks pending (evidence not produced yet); pending counts as not passed")
    return 1 if any(c["passed"] is False for d in rep["dimensions"] for c in d["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
