"""Phase 8 app checks (scorecard D3.1, D3.2, D4.1-D4.5, D5.1-D5.4) through Streamlit's AppTest.

Run:  python -m dashboards.app_checks
Needs only app/requirements.txt (no pytest, no database), so CI runs it in a clean environment. Also collects what each
page renders (metrics and tables) for verify_phase8's comparison with the independent recomputation (D1.1).
"""
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "dashboards" / "phase8_spec.json").read_text(encoding="utf-8"))
PAGES = ["fleet", "plan", "oee", "monitoring", "about"]
OKABE_ITO = {"#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"}
SIMULATED_PAGES = {"oee"}             # pages that show OEE or simulated plant values
TIMEOUT = 60
MAX_SECONDS = 5.0


def run_page(page: str, snap: Path | None = None, widgets: dict | None = None) -> AppTest:
    """Run a page script; `widgets` maps widget key -> value, applied before a second run."""
    with snapshot_env(snap):
        at = AppTest.from_file(str(ROOT / "app" / "views" / f"{page}.py"), default_timeout=TIMEOUT).run()
        if widgets:
            for k, v in widgets.items():
                el = at.radio(key=k) if k in ("dataset", "scope", "scenario") else at.multiselect(key=k)
                el.set_value(v)
            at.run()
    return at


@contextmanager
def snapshot_env(snap: Path | None):
    old = os.environ.get("PDM_SNAPSHOT_DIR")
    if snap is not None:
        os.environ["PDM_SNAPSHOT_DIR"] = str(snap)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PDM_SNAPSHOT_DIR", None)
        else:
            os.environ["PDM_SNAPSHOT_DIR"] = old


def combos(page: str) -> list[dict]:
    """Every filter combination the spec lists, plus every subset of the multiselects (empty included)."""
    if page in ("fleet",):
        return [{"dataset": d} for d in SPEC_FILTER(page, "dataset")]
    if page == "plan":
        risks = ["critical", "high", "medium"]
        subsets = [list(s) for n in range(len(risks) + 1) for s in itertools.combinations(risks, n)]
        return [{"dataset": d, "risks": r} for d in SPEC_FILTER(page, "dataset") for r in subsets]
    if page == "oee":
        return [{"scope": s} for s in SPEC_FILTER(page, "scope")]
    if page == "monitoring":
        out = []
        for sc in SPEC_FILTER(page, "scenario"):
            ds = ["FD001", "FD003"] if sc.startswith("Phase 7:") else ["FD002", "FD004"]
            out += [{"scenario": sc, "datasets": list(s)} for n in range(3) for s in itertools.combinations(ds, n)]
        return out
    return [{}]


def SPEC_FILTER(page: str, name: str) -> list:
    return next(p for p in SPEC["pages"] if p["id"] == page)["filters"][name]


def texts(at: AppTest) -> str:
    """All visible text of a rendered page: titles, markdown, captions, alerts, metrics (label, value, help), tables, charts."""
    out = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success"):
        out += [str(e.value) for e in getattr(at, kind)]
    for m in at.metric:
        out += [m.label, str(m.value), m.help or ""]
    for d in at.dataframe:
        out += [" ".join(map(str, d.value.columns)), d.value.to_string()]
    for c in charts(at):
        out.append(json.dumps(c))
    for e in at.get("expander"):
        out.append(str(e.label))
    return "\n".join(out)


def charts(at: AppTest) -> list[dict]:
    specs = []

    def walk(node):
        for e in getattr(node, "children", {}).values():
            if getattr(e, "type", None) == "vega_lite_chart":
                specs.append(json.loads(e.proto.spec))
            walk(e)
    walk(at.main)
    return specs


def layers(spec: dict) -> list[dict]:
    out = []
    for l in spec.get("layer", [spec]):
        out += layers(l) if "layer" in l and l is not spec else [l]
    return out


def colours(obj) -> set[str]:
    """Every literal colour in a chart spec (scale ranges, mark colours, value encodings)."""
    found = set()
    if isinstance(obj, str):
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", obj):
            found.add(obj.upper())
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= colours(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= colours(v)
    return found


def chart_problems(spec: dict) -> list[str]:
    probs = []
    bad = colours({k: v for k, v in spec.items() if k != "config"}) - OKABE_ITO
    if bad:
        probs.append(f"non Okabe-Ito colours {sorted(bad)}")
    ls = layers(spec)
    for l in ls:
        enc = l.get("encoding", {})
        cf = enc.get("color", {}).get("field")
        if cf:
            paired = any(enc.get(ch, {}).get("field") == cf for ch in ("shape", "strokeDash", "x", "y"))
            paired |= any(o.get("encoding", {}).get("text", {}).get("field") == cf for o in ls)
            if not paired:
                probs.append(f"colour encodes '{cf}' with no second channel")
    for ch in ("x", "y"):
        encs = [l.get("encoding", {}).get(ch, {}) for l in ls]
        encs = [e for e in encs if e.get("field")]
        if not encs:
            continue
        titled = [e for e in encs if e.get("title")]
        if not titled:
            probs.append(f"{ch} axis has no title")
        for e in titled:
            if e.get("type") == "quantitative" and not re.search(r"\(.+\)", e["title"]):
                probs.append(f"{ch} axis title '{e['title']}' names no unit")
    return probs


def check_all() -> list[tuple[str, bool, str]]:
    import streamlit as st
    results, rendered = [], {}
    p7_rate = json.loads((ROOT / "results" / "phase7" / "results_phase7.json").read_text())["drift_false_alarm_rate_test"]
    p7_shown = f"{100 * p7_rate:.1f}%"

    # D3.1 every page x every filter combination, no exception; keep the renders for the other checks
    failures = []
    for page in PAGES:
        for w in combos(page):
            at = run_page(page, widgets=w or None)
            if at.exception:
                failures.append(f"{page} {w}: {at.exception[0].value[:120]}")
            rendered.setdefault(page, []).append((w, at))
    n = sum(len(v) for v in rendered.values())
    results.append(("D3.1", not failures, f"{n} page renders, {len(failures)} with exceptions" + (f": {failures[:3]}" if failures else "")))

    # D3.2 missing / corrupt / unparsable snapshot and an empty filter: clear message, no exception
    probs = []
    with tempfile.TemporaryDirectory() as tmp:
        src = ROOT / "dashboards" / "snapshots"
        cases = {"missing file": lambda d: (d / "models.csv").unlink(),
                 "missing manifest": lambda d: (d / "manifest.json").unlink(),
                 "corrupt bytes": lambda d: (d / "fleet.csv").write_bytes((d / "fleet.csv").read_bytes()[:-50]),
                 "unparsable": lambda d: _unparsable(d, "oee_policies.csv")}
        for case, damage in cases.items():
            d = Path(tmp) / case.replace(" ", "_")
            shutil.copytree(src, d)
            damage(d)
            for page in PAGES:
                at = run_page(page, snap=d)
                if at.exception:
                    probs.append(f"{case}/{page}: exception")
            hit = {"missing file": "fleet", "missing manifest": "fleet", "corrupt bytes": "fleet", "unparsable": "oee"}[case]
            at = run_page(hit, snap=d)
            if not any("Data unavailable" in str(e.value) for e in at.error):
                probs.append(f"{case}/{hit}: no clear message")
    for page, w in (("plan", {"dataset": "FD001", "risks": []}), ("monitoring", {"scenario": SPEC_FILTER("monitoring", "scenario")[0], "datasets": []})):
        at = run_page(page, widgets=w)
        if at.exception or not any("No " in str(e.value) and "filter" in str(e.value) for e in at.info):
            probs.append(f"empty filter on {page}: no clear message")
    results.append(("D3.2", not probs, "missing, corrupt, unparsable snapshot and empty filters all show a message" if not probs else "; ".join(probs)))

    default = {p: rendered[p][0][1] for p in PAGES}
    page_text = {p: texts(default[p]) for p in PAGES}

    # D4.1 'Simulated' on every page that shows OEE or simulated plant values
    miss = [p for p in SIMULATED_PAGES if "Simulated" not in page_text[p]]
    results.append(("D4.1", not miss, f"'Simulated' on {sorted(SIMULATED_PAGES)}" + (f"; missing on {miss}" if miss else "")))

    # D4.2 intervals with nominal and observed coverage, FD001 post hoc (default view shows FD001; 'Both' shows both)
    both = texts(run_page("fleet", widgets={"dataset": "Both"}))
    ok = all(s in both for s in ("Nominal 90%", "Interval coverage on these engines (FD001)", "Interval coverage on these engines (FD003)", "post hoc"))
    results.append(("D4.2", ok, "nominal and observed coverage for FD001 and FD003; FD001 marked post hoc" if ok else "coverage labels missing"))

    # D4.3 FD004 failure and Phase 7 target miss shown, with rates, under every monitoring filter
    bad = [w for w, at in rendered["monitoring"] if not all(s in texts(at) for s in ("FD004", "FAIL", "not met", p7_shown))]
    results.append(("D4.3", not bad, f"FD004 FAIL and Phase 7 'not met' visible in all {len(rendered['monitoring'])} monitoring views" if not bad else f"missing in {bad[:2]}"))

    # D4.4 every metric has help text; every required_text present
    no_help = [(p, m.label) for p in PAGES for _, at in rendered[p] for m in at.metric if not (m.help or "").strip()]
    missing = [(p["id"], t) for p in SPEC["pages"] for t in p.get("required_text", []) if t not in page_text[p["id"]]]
    results.append(("D4.4", not no_help and not missing, f"metrics without help: {sorted(set(no_help))[:3]}; missing text: {missing}"
                    if no_help or missing else "every metric has its definition; every required text present"))

    # D4.5 published model and all baselines in one table per dataset
    want = {"Fleet average", "Age only", "Ridge (168 features)", "XGBoost (168 features)", "LSTM (published)", "CNN"}
    tables = [d.value for d in run_page("fleet", widgets={"dataset": "Both"}).dataframe if "Model" in d.value.columns]
    ok = bool(tables) and all(want <= set(tables[0][tables[0].Dataset == ds].Model) for ds in ("FD001", "FD003"))
    results.append(("D4.5", ok, "LSTM (published) and 5 other models in one table, same engines, per dataset" if ok else "model table incomplete"))

    # D5.1 titles are the spec's questions
    bad = [p["id"] for p in SPEC["pages"] if [t.value for t in default[p["id"]].title][:1] != [p["title"]]]
    results.append(("D5.1", not bad, "every page title is its question" if not bad else f"wrong titles: {bad}"))

    # D5.2 / D5.3 palette, second channel, axis titles with units (every chart in every render)
    probs52, probs53, nch = [], [], 0
    for p in PAGES:
        for w, at in rendered[p]:
            for spec in charts(at):
                nch += 1
                for pr in chart_problems(spec):
                    (probs53 if "axis" in pr else probs52).append(f"{p} {w}: {pr}")
    results.append(("D5.2", not probs52, f"{nch} charts: Okabe-Ito only, colour always paired" if not probs52 else "; ".join(sorted(set(probs52))[:3])))
    results.append(("D5.3", not probs53, f"{nch} charts: every axis titled, units on quantitative axes" if not probs53 else "; ".join(sorted(set(probs53))[:3])))

    # D5.4 each page script < 5 s with a cold cache
    times = {}
    for p in PAGES:
        st.cache_data.clear()
        t = time.perf_counter()
        run_page(p)
        times[p] = time.perf_counter() - t
    slow = {p: t for p, t in times.items() if t >= MAX_SECONDS}
    results.append(("D5.4", not slow, "cold-cache script time: " + ", ".join(f"{p} {t:.2f}s" for p, t in times.items())))
    return results


def _unparsable(d: Path, name: str) -> None:
    """Replace a file with bytes pandas cannot read, and update the manifest so only the parse step can catch it."""
    import hashlib
    b = b"\xff\xfe\x00garbage\x00"
    (d / name).write_bytes(b)
    man = json.loads((d / "manifest.json").read_text())
    man["files"][name]["sha256"] = hashlib.sha256(b).hexdigest()
    (d / "manifest.json").write_text(json.dumps(man))


def rendered_numbers() -> dict:
    """What the pages show, per filter view: {page: {view: {metric label: value}}} plus the tables (for D1.1)."""
    out = {}
    for page in PAGES:
        for w in combos(page):
            at = run_page(page, widgets=w or None)
            out.setdefault(page, {})[json.dumps(w, sort_keys=True)] = {
                "metrics": {m.label: str(m.value) for m in at.metric},
                "tables": [d.value for d in at.dataframe],
                "text": texts(at)}
    return out


def main() -> int:
    res = check_all()
    for cid, ok, detail in res:
        print(f"{'PASS' if ok else 'FAIL'}  {cid}  {detail}")
    if os.environ.get("GITHUB_ACTIONS"):
        for cid, ok, detail in res:
            if not ok:
                print(f"::error title=Phase 8 {cid}::{detail}")
    return 0 if all(ok for _, ok, _ in res) else 1


if __name__ == "__main__":
    sys.exit(main())
