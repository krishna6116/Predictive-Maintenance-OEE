"""v1.1 independent verification: prints the v1.1 scorecard (dashboards/v11_scorecard.json) with pass/fail per check.

Run:  python -m dashboards.verify_v11                 code checks (all 200 engines, 7 cost input sets, planted bugs)
      python -m dashboards.verify_v11 --local         also V5.2: two snapshot rebuilds (needs the local database)
      python -m dashboards.verify_v11 --prior         also re-run the Phase 8 and Phase 9 verifiers for V1.5
      python -m dashboards.verify_v11 --live URL      V2.1, V4.1, V4.3 and the V5.5 live smoke check on a deployed app
                                                      (needs Playwright, not a project dependency); writes the evidence
      python -m dashboards.verify_v11 --record-ci     V5.4: record the CI conclusion for HEAD (needs the GitHub CLI)

Independent by construction: the expected values come from the raw files read with the csv module and plain Python
(verify_phase8.truth(): rule-E bounds rebuilt from the published MC mean and sd, OEE counts re-aggregated from
maintenance/machines_phase6.csv, the plan from maintenance/snapshot_decisions_phase6.csv), never from the app's pandas
code, app/costmodel.py or the exporter. What the app shows is read from the rendered pages under Streamlit's AppTest.
Writes dashboards/v11_results.json after a full code run on the committed snapshots.
"""
import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from dashboards.verify_phase8 import Card, _edit_csv, _resign, expected_charts, expected_tables, fl, jload, pct, rows, sha, \
    table_lookup, tf, truth

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "dashboards" / "v11_spec.json").read_text(encoding="utf-8"))
CARD = json.loads((ROOT / "dashboards" / "v11_scorecard.json").read_text(encoding="utf-8"))
SNAP = ROOT / "dashboards" / "snapshots"
APP = ROOT / "app"
EVIDENCE = ROOT / "dashboards" / "v11_evidence.json"
LOCAL = ROOT / "dashboards" / "v11_local_checks.json"
RESULTS = ROOT / "dashboards" / "v11_results.json"
FIGS = ROOT / "docs" / "figures" / "v11"
PREVIEW_URL = SPEC["preview_app"]["url"]
REAL_URL = (ROOT / "dashboards" / "app_url.txt").read_text(encoding="utf-8").strip().rstrip("/")
DATASETS = ("FD001", "FD003")
POLICIES = ("Predictive", "Fixed interval", "Run to failure")
COST_KEYS = ("c_hour", "c_service", "c_failure", "c_cycle")
DEFAULTS = {k: v["default"] for k, v in SPEC["cost_page"]["inputs"].items()}
HISTORY_COLS = SPEC["engine_page"]["snapshot"]["columns"]
SENSOR_COLS = [c for c in HISTORY_COLS if re.fullmatch(r"s\d+", c)]
FORECAST_COLS = ["rul_mc_mean", "rul_low", "rul_high"]
MAX_SECONDS = 5.0
TIMEOUT = 90
PHONE_PAGES = {"home": "", "cost": "cost", "engine": "engine"}
UNITS = {"degR": "°R"}  # the spec writes degrees Rankine in ASCII; the page shows the symbol


class V11Card(Card):
    """Phase 8's scorecard holder plus notes collected across checks (timings, chart problems)."""

    def __init__(self):
        super().__init__()
        self.notes = {}


# ---------- running pages (optionally from a planted copy of the app or the snapshots) ----------

def _purge() -> None:
    """The pages import `common` and `costmodel` from their own folder; forget them so a planted copy is really used."""
    for m in ("common", "costmodel"):
        sys.modules.pop(m, None)


@contextmanager
def app_env(snap: Path):
    old = os.environ.get("PDM_SNAPSHOT_DIR")
    os.environ["PDM_SNAPSHOT_DIR"] = str(snap)
    _purge()
    try:
        yield
    finally:
        _purge()
        if old is None:
            os.environ.pop("PDM_SNAPSHOT_DIR", None)
        else:
            os.environ["PDM_SNAPSHOT_DIR"] = old


def new_test(app: Path, page: str):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(app / "views" / f"{page}.py"), default_timeout=TIMEOUT)


def timed_run(at) -> float:
    t0 = time.perf_counter()
    at.run()
    return time.perf_counter() - t0


def metrics(at) -> dict:
    return {m.label: str(m.value) for m in at.metric}


def errors(at) -> list[str]:
    return [str(e.value)[:300] for e in at.exception]


def warm_up(app: Path, snap: Path) -> None:
    """The first AppTest run in a process pays for importing Streamlit, pandas and Altair; time the pages after that."""
    with app_env(snap):
        new_test(app, "about").run()


# ---------- independent expectations ----------

def rolling(T) -> dict:
    """Rolling forecasts per (dataset, engine), sorted by cycle, with the rule-E interval rebuilt from mean and sd."""
    near = int(re.search(r"^NEAR = (\d+)", (ROOT / "models/calibration.py").read_text(), re.M).group(1))
    mc = jload("models/results_phase5b.json")["mc_dropout"]
    kk = {"FD001": (mc["FD001_post_hoc"]["k_near"], mc["FD001_post_hoc"]["k_far"]), "FD003": (mc["FD003"]["k_near"], mc["FD003"]["k_far"])}
    out = {}
    for ds in DATASETS:
        for r in rows(f"maintenance/forecasts_rolling_{ds}.csv"):
            m, s = fl(r["rul_mc_mean"]), fl(r["rul_mc_std"])
            k = kk[ds][0] if m <= near else kk[ds][1]
            r = {**r, "_low": max(0.0, m - k * s), "_high": m + k * s, "_k": k}
            out.setdefault((ds, int(r["engine_id"])), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r["cycle"]))
    return out


def default_engine(T, ds: str) -> int:
    """Flagged at the last cycle (rule-E lower bound <= threshold), lowest lower bound, ties lowest engine id."""
    f = T["fleet"][ds]
    return min((e for e, x in f.items() if x["flagged"]), key=lambda e: (f[e]["low"], e))


def expected_engine(T, roll: dict, ds: str, eng: int) -> dict:
    thr = T["settings"]["ALERT_THRESHOLD"]
    rs = roll[(ds, eng)]
    last = rs[-1]
    cross = [int(r["cycle"]) for r in rs if r["_low"] <= thr]
    nom = pct(jload("models/results_phase5.json")["mc_dropout"]["nominal"], 0)
    d = T["decisions"][(ds, eng)]
    day = int(fl(d["service_day"])) if d["service_day"] else None
    service = ("Not flagged" if not tf(d["flagged"]) else "No crew slot in the horizon" if day is None
               else "Service today" if day == 0 else f"Service day {day}")
    return {"Forecast remaining life at the last reading (cycles)": f"{fl(last['rul_mc_mean']):.1f}",
            f"{nom} interval at the last reading (cycles)": f"{last['_low']:.1f} to {last['_high']:.1f}",
            "Observed cycles": str(len(rs)),
            f"First cycle the lower bound reached {thr} cycles": str(cross[0]) if cross else "Never",
            "Flagged for service": "Yes" if T["fleet"][ds][eng]["flagged"] else "No",
            "Maintenance plan": service}


def expected_service_caption(T, ds: str, eng: int) -> str | None:
    d, crews = T["decisions"][(ds, eng)], T["settings"]["CREWS_PER_DAY"]
    if not tf(d["flagged"]) or not d["service_day"]:
        return None
    day = int(fl(d["service_day"]))
    return (f"Service today (day 0, one of {crews} crew slots)" if day == 0
            else f"Service day {day} (one of {crews} crew slots that day)")


def expected_costs(T, costs: dict) -> dict:
    """Own arithmetic, exact (fractions) from the re-aggregated machine counts, then shown in whole dollars."""
    tot = {}
    for p in POLICIES:
        o = T["oee"][("Pooled", p)]
        tot[p] = (Fraction(o["downtime_hours"]) * costs["c_hour"] + Fraction(o["planned_services"]) * costs["c_service"]
                  + Fraction(o["unplanned_failures"]) * costs["c_failure"] + Fraction(o["wasted_cycles"]) * costs["c_cycle"])
    money = lambda x: f"${float(x):,.0f}"  # noqa: E731
    out = {f"Total cost, {p.lower()} (illustrative)": money(tot[p]) for p in POLICIES}
    out["Saving of predictive vs fixed interval (illustrative)"] = money(tot["Fixed interval"] - tot["Predictive"])
    out["Saving of predictive vs run to failure (illustrative)"] = money(tot["Run to failure"] - tot["Predictive"])
    return out


def cost_input_sets() -> dict[str, dict]:
    sets = {"defaults": dict(DEFAULTS), "all zeros": dict.fromkeys(COST_KEYS, 0)}
    for k in COST_KEYS:
        sets[f"only {k}"] = {**dict.fromkeys(COST_KEYS, 0), k: DEFAULTS[k]}
    sets["very large (1e9 each)"] = dict.fromkeys(COST_KEYS, 10 ** 9)
    return sets


# ---------- code checks ----------

def check_home(T, card: Card, app: Path = APP, snap: Path = SNAP) -> None:
    from dashboards import app_checks as A
    with app_env(snap):
        at = new_test(app, "home")
        secs = timed_run(at)
        errs, shown, labels, tabs, text = errors(at), metrics(at), A.chart_labels(at), A.frames(at), A.texts(at)
        specs = A.charts(at)
    # V1.1: every bar label and every impact-table cell vs the recomputation from machines_phase6.csv
    probs, n = [], 0
    for title, want in expected_charts(T, "home").items():
        for pol, label in want.items():
            n += 1
            if labels.get(title, {}).get(pol) != label:
                probs.append(f"chart '{title}' {pol}: shows {labels.get(title, {}).get(pol)}, expected {label}")
    for _, cells in expected_tables(T, "home", {}):
        for key, col, want in cells:
            n += 1
            vals = [v for v in (table_lookup(t, key, col) for t in tabs) if v is not None]
            if want not in vals:
                probs.append(f"impact {key[0]} / {col}: shows {vals[:1]}, expected {want}")
    pooled = {r["policy"]: r for r in rows(snap / "oee_policies.csv") if r["scope"] == "Pooled" and r["kind"] == "policy"}
    for p in POLICIES:  # the snapshot the charts are drawn from, exact
        o = T["oee"][("Pooled", p)]
        for col in ("unplanned_failures", "downtime_hours", "planned_services", "wasted_cycles"):
            n += 1
            if p not in pooled or fl(pooled[p][col]) != o[col]:
                probs.append(f"oee_policies.csv {p} {col}")
    card.set("V1.1", not probs and not errs, f"{n} Home chart labels, impact cells and pooled counts equal the recomputation"
             if not probs and not errs else f"{len(probs)} mismatches {probs[:3]}; exceptions {errs[:1]}")
    # V2.2: headline labels (metric tiles and the impact table's row labels)
    heads = list(shown) + [str(v) for t in tabs if "Impact (simulated)" in t.columns for v in t["Impact (simulated)"]]
    bad = [h for h in heads if re.search(r"FD00\d", h) or len(h) > 60]
    card.set("V2.2", bool(heads) and not bad, f"{len(heads)} Home headline labels: no dataset code, each <= 60 characters"
             if heads and not bad else f"labels with a dataset code or over 60 characters: {bad}")
    # V3.1: 'simulated' on every chart title and on the impact table
    titles = [t for t in expected_charts(T, "home")]
    shown_titles = list(labels)
    miss = [t for t in shown_titles if "simulated" not in t.lower()] + [t for t in titles if t not in shown_titles]
    has_table = any("Impact (simulated)" in t.columns for t in tabs)
    notice = "Simulated plant: production, downtime and OEE are not measured data"  # the warning box (review fix 2026-11-15)
    card.set("V3.1", not miss and has_table and notice in text,
             f"Home: {len(shown_titles)} chart titles and the impact table say 'simulated', plus the simulated notice"
             if not miss and has_table and notice in text else
             f"Home: chart titles without 'simulated' or missing {miss}; impact header present {has_table}")
    _chart_and_time(card, "home", specs, secs)


def _chart_and_time(card: Card, page: str, specs: list, secs: float) -> None:
    from dashboards import app_checks as A
    probs = [f"{page}: {p}" for s in specs for p in A.chart_problems(s)]
    card.notes.setdefault("V4.4", []).extend(probs)
    card.notes.setdefault("V4.4 charts", []).append(len(specs))
    card.notes.setdefault("V4.2", []).append((page, secs))


def check_cost(T, card: Card, app: Path = APP, snap: Path = SNAP) -> None:
    from dashboards import app_checks as A
    probs, errs, n = [], [], 0
    with app_env(snap):
        at = new_test(app, "cost")
        secs = timed_run(at)
        text, specs = A.texts(at), A.charts(at)
        for name, costs in cost_input_sets().items():
            for k, v in costs.items():
                at.number_input(key=k).set_value(v)
            at.run()
            errs += [f"{name}: {e}" for e in errors(at)]
            got = metrics(at)
            for label, want in expected_costs(T, costs).items():
                n += 1
                if got.get(label) != want:
                    probs.append(f"{name}: '{label}' shows {got.get(label)}, expected {want}")
    tests = run_cost_tests()
    ok = not probs and not errs and tests[0] is not False
    card.set("V1.2", ok if tests[0] is not None or not ok else None,
             (f"{n} totals and savings over {len(cost_input_sets())} input sets equal the exact recomputation; {tests[1]}" if ok
              else f"{len(probs)} mismatches {probs[:3]}; exceptions {errs[:1]}; {tests[1]}"))
    card.set("V5.1 cost", not errs, "cost page renders at all-zero and very large inputs" if not errs else f"cost exceptions: {errs[:2]}")
    # V3.2: the honesty texts
    need = {"banner": SPEC["cost_page"]["banner"], "formula": "**Formula:** total = downtime hours",
            "defaults note": "Illustrative defaults, change them", "no double counting": "already include"}
    miss = [k for k, s in need.items() if s not in text]
    card.set("V3.2", not miss, "cost page shows the banner, the formula, the defaults note and the no-double-counting sentence"
             if not miss else f"cost page is missing: {miss}")
    _chart_and_time(card, "cost", specs, secs)


def run_cost_tests() -> tuple[bool | None, str]:
    """tests/test_cost.py (hand-computed cases incl. the no-double-counting case); needs pytest, absent in the app-only env."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        return None, "tests/test_cost.py not run here (no pytest; it runs in the CI test job)"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_cost.py"], cwd=ROOT,
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["no output"])[-1]
    return r.returncode == 0, f"tests/test_cost.py: {last}"


def check_engines(T, card: Card, app: Path = APP, snap: Path = SNAP, only: dict | None = None) -> None:
    """V1.3 (all 200 engines), V2.4 (default engine, sensor names), V5.1 and V4.2 for the drill-down."""
    from dashboards import app_checks as A
    roll = rolling(T)
    ids = {ds: sorted(e for (d, e) in roll if d == ds) for ds in DATASETS}
    probs, errs, slow, specs_probs, n, engines, secs_all = [], [], [], [], 0, 0, []
    defaults, names_ok = {}, []
    with app_env(snap):
        at = new_test(app, "engine")
        first = timed_run(at)
        for ds in DATASETS:
            if ds != "FD001":
                at.radio(key="dataset").set_value(ds).run()
            defaults[ds] = at.selectbox(key=f"engine_{ds}").value
            if ds == "FD001":
                names_ok = sensor_name_problems(at)
            for eng in (only or ids)[ds] if ds in (only or ids) else []:
                at.selectbox(key=f"engine_{ds}").set_value(eng)
                s = timed_run(at)
                secs_all.append(s)
                engines += 1
                if s >= MAX_SECONDS:
                    slow.append(f"{ds} {eng} {s:.1f}s")
                e = errors(at)
                if e:
                    errs.append(f"{ds} {eng}: {e[0]}")
                    continue
                got, want = metrics(at), expected_engine(T, roll, ds, eng)
                for label, w in want.items():
                    n += 1
                    if got.get(label) != w:
                        probs.append(f"{ds} engine {eng} '{label}': shows {got.get(label)}, expected {w}")
                cap = expected_service_caption(T, ds, eng)
                caps = " ".join(str(c.value) for c in at.caption)
                n += 1
                if cap and cap not in caps:
                    probs.append(f"{ds} engine {eng}: service caption '{cap}' not shown")
                if (ds, eng) == ("FD001", defaults["FD001"]) or eng == ids[ds][0]:
                    specs_probs += [f"engine {ds} {eng}: {p}" for s_ in A.charts(at) for p in A.chart_problems(s_)]
    # the interval the app shows comes from the rolling file; it must also equal the rule-E rebuild there
    rebuild = [k for k, rs in roll.items() for r in rs
               if not (math.isclose(fl(r["rul_low"]), r["_low"], abs_tol=1e-9) and math.isclose(fl(r["rul_high"]), r["_high"], abs_tol=1e-9))]
    total = sum(len(v) for v in (only or ids).values())
    ok = not probs and not errs and not rebuild and engines == total
    card.set("V1.3", ok, f"{engines} engines: {n} drill-down values equal the independent recomputation; rolling intervals equal the "
             f"rule-E rebuild on every cycle" if ok else
             f"{len(probs)} mismatches {probs[:3]}; exceptions {errs[:1]}; rule-E rebuild differs for {sorted(set(rebuild))[:3]}; "
             f"{engines} of {total} engines run")
    want_default = {ds: default_engine(T, ds) for ds in DATASETS}
    d_ok = defaults == want_default and not names_ok
    card.set("V2.4", d_ok, f"opens on FD001 engine {defaults['FD001']} (FD003: {defaults['FD003']}), the pre-registered default; "
             "4 sensor names, codes and units match v11_spec.json" if d_ok
             else f"default engines shown {defaults}, expected {want_default}; sensor problems {names_ok}")
    card.set("V5.1 engine", not errs and engines == total, f"drill-down renders for all {engines} engines with no exception"
             if not errs and engines == total else f"exceptions: {errs[:2]}; {engines} of {total} engines run")
    card.notes.setdefault("V4.4", []).extend(specs_probs)
    card.notes.setdefault("V4.4 charts", []).append(6)
    card.notes.setdefault("V4.2", []).append(("engine (first run)", first))
    if secs_all:
        card.notes["V4.2"].append((f"engine, slowest of {len(secs_all)}", max(secs_all)))


def sensor_name_problems(at) -> list[str]:
    """Each sensor: 'Name (code)' heading and an axis titled 'symbol (unit)', as in v11_spec.json engine_page.sensors.names."""
    from dashboards import app_checks as A
    heads = " ".join(str(m.value) for m in at.markdown)
    axes = {l.get("encoding", {}).get("y", {}).get("title") for s in A.charts(at) for l in A.layers(s)}
    probs = []
    for code, full in SPEC["engine_page"]["sensors"]["names"].items():
        m = re.fullmatch(r"(.+) \((\w+), (.+)\)", full)
        name, symbol, unit = m.group(1), m.group(2), UNITS.get(m.group(3), m.group(3))
        if f"**{name} ({code})**" not in heads:
            probs.append(f"heading '{name} ({code})'")
        if f"{symbol} ({unit})" not in axes:
            probs.append(f"axis '{symbol} ({unit})'")
    return probs


def check_history(T, card: Card, snap: Path = SNAP) -> None:
    """V1.4: engine_history.csv vs the NASA test files and the rolling forecasts, cell for cell; manifest entry current."""
    man = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    entry = man["files"].get("engine_history.csv", {})
    b = (snap / "engine_history.csv").read_bytes()
    hist = rows(snap / "engine_history.csv")
    probs = []
    if entry.get("sha256") != sha(b) or entry.get("rows") != len(hist):
        probs.append("manifest entry (rows, sha256) is not current")
    if hist and list(hist[0]) != HISTORY_COLS:
        probs.append(f"columns {list(hist[0])} != {HISTORY_COLS}")
    raw_files = [ROOT / "data" / "raw" / f"test_{ds}.txt" for ds in DATASETS]
    if not all(p.exists() for p in raw_files):
        card.set("V1.4", False if probs else None, "; ".join(probs) if probs else
                 "not run here: needs the pinned NASA data (CI verify job runs it)")
        return
    want = []
    roll = {(ds, int(r["engine_id"]), int(r["cycle"])): r for ds in DATASETS for r in rows(f"maintenance/forecasts_rolling_{ds}.csv")}
    for ds, p in zip(DATASETS, raw_files):
        for line in p.read_text().split("\n"):
            v = line.split()
            if not v:
                continue
            e, c = int(v[0]), int(v[1])
            want.append((ds, e, c, {s: float(v[4 + int(s[1:])]) for s in SENSOR_COLS}, roll.get((ds, e, c))))
    if len(hist) != len(want):
        probs.append(f"{len(hist)} rows, the raw files have {len(want)}")
    cells = 0
    for r, (ds, e, c, sens, f) in zip(hist, want):
        if (r["dataset"], int(r["engine_id"]), int(r["cycle"])) != (ds, e, c):
            probs.append(f"row order/key {r['dataset']} {r['engine_id']} {r['cycle']} != {ds} {e} {c}")
            break
        for s in SENSOR_COLS:
            cells += 1
            if float(r[s]) != sens[s]:
                probs.append(f"{ds} {e} cycle {c} {s}: {r[s]} != raw {sens[s]}")
        for col in FORECAST_COLS:
            cells += 1
            if f is None or float(r[col]) != float(f[col]):
                probs.append(f"{ds} {e} cycle {c} {col}: {r[col]} != rolling {None if f is None else f[col]}")
    card.set("V1.4", not probs, f"{len(hist):,} rows, {cells:,} cells equal the NASA test files and the rolling forecasts; manifest current"
             if not probs else f"{len(probs)} problems: {probs[:3]}")


def check_plain_help(card: Card, snap: Path = SNAP) -> None:
    """V2.3: 'Term: sentence' (app/common.py PLAIN) is rendered, as a caption or tooltip, on at least one page with the term."""
    import importlib.util
    from dashboards import app_checks as A
    spec = importlib.util.spec_from_file_location("_v11_common", APP / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = {t: [] for t in SPEC["plain_english_help"]["terms"]}
    errs = []
    with app_env(snap):
        for page in A.PAGES:
            at = new_test(APP, page)
            at.run()
            errs += [f"{page}: {e}" for e in errors(at)]
            text = A.texts(at)
            for t in found:
                sentence = f"{t}: {mod.PLAIN.get(t, '<missing>')}"
                if t in mod.PLAIN and sentence in text and "\n" not in mod.PLAIN[t]:
                    found[t].append(page)
    miss = [t for t, p in found.items() if not p]
    card.set("V2.3", not miss and not errs, "; ".join(f"{t}: {', '.join(p)}" for t, p in found.items())
             if not miss and not errs else f"no one-line help shown for {miss}; exceptions {errs[:1]}")


def check_honesty_docs(card: Card) -> None:
    """V3.3: no currency figure in README, the case study or any file written by writeup.report_phase9 (prose only)."""
    from writeup.report_phase9 import render_all
    files = {"README.md", "docs/CASE_STUDY.md", *render_all()}
    pat = re.compile(r"\$\s?\d|\d[\d,.]*\s?(?:USD|dollars?)\b", re.I)
    hits = []
    for p in sorted(files):
        path = ROOT / p
        if not path.exists():
            continue
        if p.endswith(".ipynb"):  # cells only: outputs hold base64 images
            text = "\n".join("".join(c.get("source", [])) for c in json.loads(path.read_text(encoding="utf-8"))["cells"])
        else:
            text = path.read_text(encoding="utf-8")
        hits += [f"{p}: '{m.group(0)}'" for m in pat.finditer(text)]
    card.set("V3.3", not hits, f"no currency figure in {len(files)} files (README, case study, generated docs and notebook cells)"
             if not hits else f"currency figures: {hits[:5]}")


def check_monitoring_misses(card: Card, snap: Path = SNAP) -> None:
    from dashboards import app_checks as A
    with app_env(snap):
        at = new_test(APP, "monitoring")
        at.run()
        text = A.texts(at)
    ok = "not met" in text and "FAIL" in text
    card.set("V3.4", ok, "monitoring page still shows the Phase 7 'not met' and the FD004 C1 'FAIL'" if ok
             else f"'not met' shown {'not met' in text}; 'FAIL' shown {'FAIL' in text}")


def check_prior(card: Card, rerun: bool) -> None:
    """V1.5: Phase 8 still 5/5 and Phase 9 still 10 of 10 (CI runs both; --prior re-runs them here)."""
    notes = []
    if rerun:
        for mod in ("dashboards.verify_phase8", "writeup.verify_phase9"):
            r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT, capture_output=True, text=True)
            notes.append(f"{mod} exit {r.returncode}")
    p8 = json.loads((ROOT / "dashboards" / "phase8_results.json").read_text(encoding="utf-8"))
    p9 = json.loads((ROOT / "writeup" / "phase9_results.json").read_text(encoding="utf-8"))
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    in_ci = all(s in ci for s in ("dashboards.verify_phase8", "writeup.verify_phase9"))
    # as the check says: Phase 9 code checks now; its network, CI and manual checks are re-run after the merge
    kinds = {c["id"]: c["kind"] for c in json.loads((ROOT / "writeup" / "phase9_scorecard.json").read_text(encoding="utf-8"))["checks"]}
    code9 = {k: v["passed"] for k, v in p9["checks"].items() if kinds.get(k) == "code"}
    rest9 = sorted(k for k, v in p9["checks"].items() if kinds.get(k) != "code" and not v["passed"])
    p8_ok = p8["score"] == p8["out_of"] == 5
    code_ok = bool(code9) and all(code9.values())
    reran = all(n.endswith("exit 0") for n in notes)
    detail = (f"Phase 8 {p8['score']}/{p8['out_of']}; Phase 9 {p9['passed']}/{p9['out_of']}, code checks {sum(code9.values())}/{len(code9)}"
              + (f", not passing yet (network/CI/manual, re-run after the merge): {rest9}" if rest9 else "")
              + f" (results files{', re-run here: ' + '; '.join(notes) if notes else ''}); both verifiers run in CI: {in_ci}")
    if not (p8_ok and code_ok and in_ci and reran):
        card.set("V1.5", False, detail)
    else:
        card.set("V1.5", True if p9["passed"] == p9["out_of"] == 10 else None, detail)


def check_timing_and_charts(card: Card) -> None:
    t = card.notes.get("V4.2", [])
    slow = [f"{p} {s:.1f}s" for p, s in t if s >= MAX_SECONDS]
    card.set("V4.2", bool(t) and not slow, "under 5 s under AppTest: " + ", ".join(f"{p} {s:.2f}s" for p, s in t)
             if t and not slow else f"slow: {slow}")
    probs = card.notes.get("V4.4", [])
    card.set("V4.4", not probs, f"{sum(card.notes.get('V4.4 charts', []))} new charts: Okabe-Ito only, colour paired with a "
             "second channel, every axis titled with units" if not probs else f"chart problems: {probs[:4]}")


def check_robust(card: Card) -> None:
    parts = [card.res.pop(k) for k in ("V5.1 engine", "V5.1 cost") if k in card.res]
    if parts:
        card.set("V5.1", all(p[0] for p in parts), "; ".join(p[1] for p in parts))


def check_local(card: Card, run: bool) -> None:
    """V5.2: two rebuilds of the snapshots are byte-identical (incl. engine_history.csv) and equal the committed files."""
    man = json.loads((SNAP / "manifest.json").read_text(encoding="utf-8"))
    if run:
        from dashboards.export_snapshots import export
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            a, b = export(), export()
        finally:
            os.chdir(cwd)
        same = a == b and all((SNAP / n).read_bytes() == v for n, v in a.items()) and "engine_history.csv" in a
        detail = (f"two rebuilds byte-identical ({len(a)} files incl. engine_history.csv) and equal to the committed snapshots"
                  if same else "rebuilds differ from each other or from the committed snapshots")
        LOCAL.write_text(json.dumps({"note": "V5.2 evidence; written by python -m dashboards.verify_v11 --local (needs the local database)",
                                     "data_checksum": man["data_checksum"], "checked_utc": now(),
                                     "checks": {"V5.2": [same, detail]}}, indent=2) + "\n", encoding="utf-8")
    if LOCAL.exists():
        loc = json.loads(LOCAL.read_text(encoding="utf-8"))
        ok, detail = loc["checks"]["V5.2"]
        current = loc["data_checksum"] == man["data_checksum"]
        card.set("V5.2", ok and current, detail + ("" if current else " (evidence is for older snapshots: re-run --local)"))
    else:
        card.set("V5.2", None, "pending: run python -m dashboards.verify_v11 --local on the machine with the database")


# ---------- planted bugs (V5.3) ----------

def _plant_file(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"planted bug could not be placed: '{old}' not in {path.name}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _shift_engine(rs: list, ds: str, eng: int) -> list:
    idx = [i for i, r in enumerate(rs) if r["dataset"] == ds and int(r["engine_id"]) == eng]
    vals = [{c: rs[i][c] for c in SENSOR_COLS + FORECAST_COLS} for i in idx]
    for j, i in enumerate(idx[1:], start=1):
        rs[i] = {**rs[i], **vals[j - 1]}  # every reading moved one cycle later
    return rs


def planted_bugs(T) -> list[tuple[str, bool, str]]:
    eng = default_engine(T, "FD001")
    bugs = [
        ("wrong cost formula term (failure downtime counted twice)", "V1.2",
         lambda a, s: _plant_file(a / "costmodel.py", "c.unplanned_failures * c_failure",
                                  "c.unplanned_failures * (c_failure + 48 * c_hour)"),
         lambda T, c, a, s: check_cost(T, c, a, s)),
        (f"shifted engine-history rows (FD001 engine {eng}, one cycle late)", "V1.3/V1.4",
         lambda a, s: (_edit_csv(s / "engine_history.csv", lambda rs: _shift_engine(rs, "FD001", eng)), _resign(s)),
         lambda T, c, a, s: (check_history(T, c, s), check_engines(T, c, a, s, only={"FD001": [eng]}))),
        ("wrong default engine (highest lower bound instead of lowest)", "V2.4",
         lambda a, s: _plant_file(a / "views" / "engine.py", "return int(f.engine_id.iloc[0])", "return int(f.engine_id.iloc[-1])"),
         lambda T, c, a, s: check_engines(T, c, a, s, only={"FD001": [], "FD003": []})),
        ("dropped 'simulated' label on a Home chart", "V3.1",
         lambda a, s: _plant_file(a / "views" / "home.py", '"Downtime (simulated, hours)"', '"Downtime (hours)"'),
         lambda T, c, a, s: check_home(T, c, a, s)),
    ]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, target, plant, check) in enumerate(bugs):
            a, s = Path(tmp) / f"bug{i}" / "app", Path(tmp) / f"bug{i}" / "snapshots"
            shutil.copytree(APP, a, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(SNAP, s)
            plant(a, s)
            c = V11Card()
            try:
                check(T, c, a, s)
            except Exception as e:  # a crash on bad input also counts as caught, but say so
                c.set(target, False, f"verifier stopped: {type(e).__name__}")
            caught = sorted(cid for cid, (ok, _) in c.res.items() if ok is False and not cid.startswith("V5.1"))
            out.append((name, bool(caught), ", ".join(caught) or "NOT CAUGHT"))
    return out


# ---------- live checks (V2.1, V4.1, V4.3, V5.5 live) ----------

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def live(url: str) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("--live needs Playwright (pip install playwright==1.55.0; python -m playwright install chromium)")
    url = url.rstrip("/")
    local = "localhost" in url or "127.0.0.1" in url
    page_url = lambda slug: f"{url}/{slug}" if local else f"{url}/~/+/{slug}"  # noqa: E731
    FIGS.mkdir(parents=True, exist_ok=True)
    rec = {"url": url, "captured_utc": now()}

    def load(pg, slug: str) -> float:
        t0 = time.perf_counter()
        pg.goto(page_url(slug), wait_until="domcontentloaded", timeout=120_000)
        pg.locator("h1").first.wait_for(state="visible", timeout=120_000)
        s = time.perf_counter() - t0
        pg.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000)
        pg.wait_for_timeout(3_000)  # charts finish drawing
        return round(s, 2)

    with sync_playwright() as p:
        b = p.chromium.launch()
        desk = b.new_context(viewport={"width": 1280, "height": 800}, color_scheme="light")
        pg = desk.new_page()
        if not local:  # wake a sleeping Community Cloud app
            pg.goto(url, wait_until="domcontentloaded", timeout=120_000)
            btn = pg.get_by_role("button", name=re.compile("get this app back up", re.I))
            if btn.count():
                btn.first.click()
                pg.wait_for_timeout(60_000)
        # V5.5 live: every page loads with no exception
        smoke = {}
        for page in ("home", "fleet", "engine", "plan", "oee", "cost", "monitoring", "about"):
            load(pg, "" if page == "home" else page)
            exc = pg.locator("[data-testid='stException']")
            smoke[page] = [exc.nth(i).inner_text()[:300] for i in range(exc.count())]
        rec["V5.5"] = {"ok": all(not v for v in smoke.values()), "pages": smoke}
        # V2.1: Home at 1280 x 800, no scrolling: headline tiles, the three policy charts and the impact table
        load(pg, "")
        box = pg.evaluate("""() => {
            const b = sel => [...document.querySelectorAll(sel)].map(e => Math.round(e.getBoundingClientRect().bottom));
            return {viewport: window.innerHeight, metrics: b('[data-testid="stMetric"]'),
                    charts: b('[data-testid="stVegaLiteChart"]').slice(0, 3), table: b('[data-testid="stTable"], [data-testid="stDataFrame"]').slice(0, 1)};
        }""")
        pg.screenshot(path=str(FIGS / "home_1280x800.png"))
        parts = box["metrics"][:2] + box["charts"] + box["table"]
        ok21 = len(box["metrics"]) >= 2 and len(box["charts"]) == 3 and len(box["table"]) == 1 and max(parts) <= box["viewport"]
        rec["V2.1"] = {"ok": ok21, "viewport": box["viewport"], "bottoms": box, "screenshot": "docs/figures/v11/home_1280x800.png",
                       "detail": f"lowest required element ends at {max(parts) if parts else None} px of {box['viewport']}"}
        # V4.3: each new page's title within 5 s of navigation, three times, median counts
        t43 = {}
        for page in ("cost", "engine"):
            t43[page] = [load(pg, page) for _ in range(3)]
        rec["V4.3"] = {"ok": all(statistics.median(v) < MAX_SECONDS for v in t43.values()), "seconds": t43,
                       "median": {k: statistics.median(v) for k, v in t43.items()}}
        desk.close()
        # V4.1: phone width 375 px (2x pixels, touch, dark system theme on purpose so the pinned light theme is tested),
        # a fresh browser context per page so an opened menu cannot carry over; no horizontal scroll; the menu opens
        res41 = {}
        for page, slug in PHONE_PAGES.items():
            phone = b.new_context(viewport={"width": 375, "height": 667}, device_scale_factor=2, is_mobile=True, has_touch=True,
                                  color_scheme="dark", user_agent=p.devices["iPhone SE"]["user_agent"])
            pg = phone.new_page()
            load(pg, slug)
            width = pg.evaluate("""() => {
                const els = [document.documentElement, ...document.querySelectorAll('*')];
                const sc = els.filter(e => e.scrollHeight > e.clientHeight + 1 &&
                    ['auto', 'scroll'].includes(getComputedStyle(e).overflowY)).sort((a, b) => b.scrollHeight - a.scrollHeight)[0]
                    || document.documentElement;
                return {viewport: window.innerWidth, scroll_width: Math.max(document.documentElement.scrollWidth, sc.scrollWidth),
                        content_height: sc.scrollHeight};
            }""")
            # screenshot first, as a visitor first sees the page (Streamlit remembers an opened menu, even across a reload)
            shot = FIGS / f"phone_{page}.png"
            pg.set_viewport_size({"width": 375, "height": max(667, int(width["content_height"]) + 80)})
            pg.mouse.move(1, 1)
            pg.wait_for_timeout(1_500)
            pg.screenshot(path=str(shot), full_page=True)
            pg.set_viewport_size({"width": 375, "height": 667})
            pg.wait_for_timeout(1_000)
            # the menu: Streamlit's sidebar starts collapsed on a phone; open it and look for the page links
            opener = pg.locator("[data-testid='stExpandSidebarButton'], [data-testid='stSidebarCollapsedControl'] button").first
            # Streamlit keeps a collapsed sidebar in the page, moved off-screen with zero width: judge it by its position
            on_screen = lambda: pg.evaluate("""() => { const s = document.querySelector('[data-testid="stSidebar"]');
                const r = s && s.getBoundingClientRect(); const l = document.querySelector('[data-testid="stSidebarNavLink"]');
                const q = l && l.getBoundingClientRect();
                return !!(r && r.width > 0 && r.right > 0 && q && q.left >= 0 && q.right <= window.innerWidth && q.width > 0); }""")  # noqa: E731
            closed_first = not on_screen()
            menu = False
            if closed_first and opener.count():
                opener.click()
                pg.wait_for_timeout(1_500)
                menu = on_screen()
            res41[page] = {"viewport_width": width["viewport"], "scroll_width": width["scroll_width"],
                           "menu_starts_closed": closed_first, "menu_opens": menu, "screenshot": f"docs/figures/v11/phone_{page}.png"}
            phone.close()
        rec["V4.1"] = {"ok": all(v["viewport_width"] == 375 and v["scroll_width"] <= 375 and v["menu_opens"] for v in res41.values()),
                       "pages": res41}
        b.close()
    return rec


def record_live(rec: dict) -> None:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {
        "note": "v1.1 live evidence; written by python -m dashboards.verify_v11 --live URL and --record-ci", "runs": {}}
    ev["runs"][rec["url"]] = rec
    EVIDENCE.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_ci() -> int:
    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    r = subprocess.run([gh, "run", "list", "--commit", head, "--workflow", "CI", "--json", "conclusion,status,headSha,url"],
                       cwd=ROOT, capture_output=True, text=True)
    runs = json.loads(r.stdout or "[]")
    if not runs or runs[0]["status"] != "completed":
        print(f"no completed CI run for {head[:7]} yet")
        return 1
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {"note": "v1.1 live evidence", "runs": {}}
    ev["ci"] = {"commit": head, "conclusion": runs[0]["conclusion"], "url": runs[0]["url"], "recorded_utc": now()}
    EVIDENCE.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CI for {head[:7]}: {runs[0]['conclusion']}")
    return 0 if runs[0]["conclusion"] == "success" else 1


def check_live_evidence(card: Card) -> None:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {"runs": {}}
    runs = {u.rstrip("/"): r for u, r in ev.get("runs", {}).items()}
    for cid in ("V2.1", "V4.1", "V4.3"):
        states = {name: runs.get(u, {}).get(cid) for name, u in (("preview", PREVIEW_URL), ("real app", REAL_URL))}
        if cid == "V4.1":
            missing = [f"phone_{p}.png" for p in PHONE_PAGES if not (FIGS / f"phone_{p}.png").exists()]
            if missing and any(states.values()):  # screenshots are written by the first live run
                states["screenshots"] = {"ok": False}
        text = "; ".join(f"{n}: {'pass' if s and s['ok'] else 'FAIL' if s else 'not measured yet'}" for n, s in states.items())
        if any(s and not s["ok"] for s in states.values()):
            card.set(cid, False, text)
        elif all(states.values()):
            card.set(cid, True, text)
        else:
            card.set(cid, None, text + " (live checks run on the preview before the merge and on the real app after it)")
    ci = ev.get("ci")
    if not ci:
        card.set("V5.4", None, "pending: python -m dashboards.verify_v11 --record-ci after CI finishes on the merge candidate")
    else:
        changed = subprocess.run(["git", "diff", "--name-only", ci["commit"], "HEAD"], cwd=ROOT, capture_output=True, text=True)
        later = [f for f in changed.stdout.split() if f != "dashboards/v11_evidence.json"]
        if changed.returncode != 0:
            card.set("V5.4", None, f"CI {ci['conclusion']} on {ci['commit'][:7]}; this checkout cannot compare it with HEAD")
        else:
            card.set("V5.4", ci["conclusion"] == "success" and not later,
                     f"CI {ci['conclusion']} on {ci['commit'][:7]}" + (f"; files changed since: {later[:3]}" if later else ""))
    real = runs.get(REAL_URL, {}).get("V5.5")
    card.notes["V5.5 live"] = real


def check_smoke(card: Card) -> None:
    """V5.5: every page loads from the committed tree (HEAD), CI runs it, the planted missing title is tested; live after the merge."""
    from dashboards import smoke_v11
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    in_ci = "dashboards.smoke_v11" in ci and (ROOT / "tests" / "test_smoke_v11.py").exists()
    try:
        res = smoke_v11.committed("HEAD")
        tree = f"committed tree {res['sha'][:7]}: {sum(not v for v in res['pages'].values())}/{len(res['pages'])} pages load"
        ok = res["ok"]
    except Exception as e:  # no git history here (e.g. an archive)
        ok, tree = None, f"committed tree not checked here ({type(e).__name__})"
    real = card.notes.get("V5.5 live")
    live_text = "live real app: " + ("pass" if real and real["ok"] else "FAIL" if real else "after the merge")
    if ok is False or (real and not real["ok"]) or not in_ci:
        card.set("V5.5", False, f"{tree}; in CI with planted-title test: {in_ci}; {live_text}")
    else:
        card.set("V5.5", True if ok and real else None, f"{tree}; in CI with planted-title test: {in_ci}; {live_text}")


# ---------- scorecard ----------

def verify(local: bool = False, prior: bool = False, planted: bool = True) -> Card:
    card = V11Card()
    T = truth()
    warm_up(APP, SNAP)
    check_home(T, card)
    check_cost(T, card)
    check_engines(T, card)
    check_history(T, card)
    check_plain_help(card)
    check_honesty_docs(card)
    check_monitoring_misses(card)
    check_prior(card, prior)
    check_timing_and_charts(card)
    check_robust(card)
    check_local(card, local)
    check_live_evidence(card)
    check_smoke(card)
    if planted:
        pb = planted_bugs(T)
        missed = [n for n, ok, _ in pb if not ok]
        card.set("V5.3", len(pb) - len(missed) >= 3 and not missed,
                 f"{len(pb) - len(missed)}/{len(pb)} planted bugs caught: " + "; ".join(f"{n} -> {c}" for n, _, c in pb))
    return card


def report(card: Card) -> dict:
    dims = []
    for d in CARD["dimensions"]:
        checks = [(c["id"], c["kind"], card.res.get(c["id"])) for c in d["checks"]]
        dims.append({"id": d["id"], "name": d["name"], "passed": all(r is not None and r[0] is True for _, _, r in checks),
                     "checks": [{"id": cid, "kind": k, "passed": None if r is None else r[0],
                                 "detail": "not run in this mode" if r is None else r[1]} for cid, k, r in checks]})
    return {"score": sum(d["passed"] for d in dims), "out_of": len(dims), "dimensions": dims}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="V5.2: rebuild the snapshots twice (needs the local database)")
    ap.add_argument("--prior", action="store_true", help="V1.5: re-run the Phase 8 and Phase 9 verifiers")
    ap.add_argument("--live", metavar="URL", help="live checks on a deployed app; writes dashboards/v11_evidence.json")
    ap.add_argument("--record-ci", action="store_true", help="V5.4: record the CI conclusion for HEAD")
    ap.add_argument("--no-planted", action="store_true", help="skip the planted bugs (faster)")
    a = ap.parse_args()
    if a.record_ci:
        return record_ci()
    if a.live:
        rec = live(a.live)
        record_live(rec)
        for cid in ("V5.5", "V2.1", "V4.3", "V4.1"):
            print(f"{cid} on {rec['url']}: {'PASS' if rec[cid]['ok'] else 'FAIL'}  {json.dumps({k: v for k, v in rec[cid].items() if k != 'ok'})[:400]}")
        return 0 if all(rec[c]["ok"] for c in ("V5.5", "V2.1", "V4.3", "V4.1")) else 1
    card = verify(local=a.local, prior=a.prior, planted=not a.no_planted)
    rep = report(card)
    for d in rep["dimensions"]:
        state = "PASS" if d["passed"] else ("FAIL" if any(c["passed"] is False for c in d["checks"]) else "PENDING")
        print(f"{d['id']} {d['name']}: {state}")
        for c in d["checks"]:
            mark = "PASS" if c["passed"] else ("FAIL" if c["passed"] is False else "PEND")
            print(f"   {mark}  {c['id']} [{c['kind']}] {c['detail']}")
    print(f"v1.1 score: {rep['score']}/{rep['out_of']} (5 only if every check passes; pending counts as not passed)")
    if os.environ.get("GITHUB_ACTIONS"):
        for d in rep["dimensions"]:
            for c in d["checks"]:
                if c["passed"] is False:
                    print(f"::error title=v1.1 {c['id']}::{c['detail']}")
    if not a.no_planted and not os.environ.get("GITHUB_ACTIONS"):  # CI checks that the run changes no committed file
        RESULTS.write_text(json.dumps({**rep, "checked_utc": now()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if any(c["passed"] is False for d in rep["dimensions"] for c in d["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
