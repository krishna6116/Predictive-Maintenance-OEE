"""Phase 8 D1.7: Power BI 'Export data' CSVs (docs/figures/phase8/powerbi_exports/) vs the independent recomputation.

One table visual per page is exported with these exact column names (docs/DASHBOARDS.md walks through building them):
  fleet_summary.csv       Dataset | Engines | Flagged | Interval coverage
  plan_summary.csv        Dataset | Engines flagged | Scheduled | No crew in horizon
  oee_policies.csv        Policy | OEE | Unplanned failures | Planned services          (pooled, simulated)
  monitoring_summary.csv  Scenario | Dataset | Runs | Flagged runs | Flag rate
Counts must match exactly; ratios at the precision shown in the report (0.1 percentage point), whether Power BI
exports them as 0.982, 98.2 or "98.2%".
"""
import csv
import math

EXPECTED_FILES = ("fleet_summary.csv", "plan_summary.csv", "oee_policies.csv", "monitoring_summary.csv")


def _num(s: str) -> float:
    s = s.strip().replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100
    return float(s)


def _ratio_ok(shown: str, truth: float, percent_column: bool) -> bool:
    """percent_column: the whole column is in percent (some value > 1), decided once so 0.9 is never ambiguous."""
    v = _num(shown)
    if percent_column and not shown.strip().endswith("%"):
        v /= 100                                    # exported as a percentage number, e.g. 98.2
    return round(v, 3) == round(truth, 3)


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def expected(T) -> dict:
    fleet = {ds: {"Engines": len(e), "Flagged": sum(x["flagged"] for x in e.values()),
                  "Interval coverage": sum(x["inside"] for x in e.values()) / len(e)} for ds, e in T["fleet"].items()}
    plan = {ds: {"Engines flagged": p["flagged"], "Scheduled": p["scheduled"], "No crew in horizon": p["unscheduled"]} for ds, p in T["plan"].items()}
    oee = {pol: {"OEE": T["oee"][("Pooled", pol)]["oee"], "Unplanned failures": T["oee"][("Pooled", pol)]["unplanned_failures"],
                 "Planned services": T["oee"][("Pooled", pol)]["planned_services"]} for pol in ("Run to failure", "Fixed interval", "Predictive")}
    mon = {}
    for sc, h in T["health"].items():
        for ds in sorted({r["dataset"] for r in h}):
            g = [r for r in h if r["dataset"] == ds]
            k = sum(r["needs_retraining"] == "True" for r in g)
            mon[(sc, ds)] = {"Runs": len(g), "Flagged runs": k, "Flag rate": k / len(g)}
    return {"fleet_summary.csv": (["Dataset"], fleet), "plan_summary.csv": (["Dataset"], plan),
            "oee_policies.csv": (["Policy"], oee), "monitoring_summary.csv": (["Scenario", "Dataset"], mon)}


def check_exports(T, files) -> tuple[bool, str]:
    exp = expected(T)
    by_name = {f.name: f for f in files}
    probs, n = [], 0
    for name in EXPECTED_FILES:
        if name not in by_name:
            probs.append(f"{name} missing")
            continue
        keys, want = exp[name]
        got = {}
        for r in _read(by_name[name]):
            missing_cols = [k for k in keys if k not in r]
            if missing_cols:
                probs.append(f"{name}: columns {missing_cols} not found (has {list(r)})")
                break
            if all(r[k] for k in keys):          # skip a Power BI 'Total' row (blank key)
                got[tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]] = r
        pct_cols = {c: any(not r.get(c, "").endswith("%") and r.get(c) and _num(r[c]) > 1 for r in got.values())
                    for c in ("Interval coverage", "OEE", "Flag rate")}
        for key, cols in want.items():
            r = got.get(key)
            if r is None:
                probs.append(f"{name}: row {key} missing")
                continue
            for col, truth in cols.items():
                n += 1
                if col not in r:
                    probs.append(f"{name}: column '{col}' missing")
                elif isinstance(truth, float) and not float(truth).is_integer() or col in ("Interval coverage", "OEE", "Flag rate"):
                    if not _ratio_ok(r[col], truth, pct_cols[col]):
                        probs.append(f"{name} {key} {col}: {r[col]} vs {truth:.4f}")
                elif not math.isclose(_num(r[col]), truth):
                    probs.append(f"{name} {key} {col}: {r[col]} vs {truth}")
    if probs:
        return False, f"{len(probs)} problems: {probs[:4]}"
    return True, f"{n} Power BI values in {len(EXPECTED_FILES)} exported visuals equal the independent recomputation at displayed precision"
