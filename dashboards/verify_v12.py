"""v1.2 verifier: the How it works page and the author section (dashboards/v12_scorecard.json, W1-W4).

Run:  python -m dashboards.verify_v12            code checks (CI), planted bugs included
      python -m dashboards.verify_v12 --live URL  W4.3 on a running app (needs Playwright; local server before the merge,
                                                  the real URL after it); evidence in dashboards/v12_evidence.json
      python -m dashboards.verify_v12 --record-ci W4.4: record the CI result of HEAD (after CI finishes on the merge candidate)

Every number the How page shows is recomputed here from the results files (through verify_phase8.truth(), which never
reads the snapshots), not from the snapshots the page reads. The author facts are compared with the README's Author
section, which writeup/report_phase9.py generates.
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dashboards.verify_phase8 import Card, _resign, fl, jload, rows, truth
from dashboards.verify_v11 import app_env, errors, new_test, now, timed_run, warm_up

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "dashboards" / "v12_spec.json").read_text(encoding="utf-8"))
CARD = json.loads((ROOT / "dashboards" / "v12_scorecard.json").read_text(encoding="utf-8"))
SNAP = ROOT / "dashboards" / "snapshots"
APP = ROOT / "app"
EVIDENCE = ROOT / "dashboards" / "v12_evidence.json"
REAL_URL = (ROOT / "dashboards" / "app_url.txt").read_text(encoding="utf-8").strip().rstrip("/")
LOCAL_URL = "http://localhost:8501"
HOW = SPEC["how_page"]
AUTHOR = SPEC["author_section"]["content"]
OKABE_ITO = {"#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"}
MAX_SECONDS = 5.0
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w.])|\+\d{1,3}[\s-]?\d{5,}")


# ---------- rendering ----------

def render(page: str, app: Path = APP, snap: Path = SNAP) -> tuple[object, float]:
    with app_env(snap):
        at = new_test(app, page)
        return at, timed_run(at)


def page_text(at) -> str:
    """Visible text: title, markdown, captions, alerts, expander labels (the How page has no metrics or tables)."""
    out = []
    for kind in ("title", "subheader", "markdown", "caption", "info", "warning", "error"):
        out += [str(e.value) for e in getattr(at, kind)]
    out += [str(e.label) for e in at.get("expander")]
    return "\n".join(out)


def dot_of(at) -> str:
    found = []

    def walk(node):
        for e in getattr(node, "children", {}).values():
            if getattr(e, "type", None) == "graphviz_chart":
                found.append(e.proto.spec)
            walk(e)
    walk(at.main)
    return found[0] if len(found) == 1 else ""


def parse_dot(dot: str) -> tuple[dict, list]:
    nodes = {m.group(1): {"label": m.group(2).split("\\n")[0], "attrs": m.group(0)}
             for m in re.finditer(r'^\s*(n\d+) \[label="([^"]*)"[^\n]*\];', dot, re.M)}
    edges = [(a, b, attrs or "") for a, b, attrs in re.findall(r"^\s*(n\d+) -> (n\d+)(?: \[([^\]]*)\])?;", dot, re.M)]
    return nodes, edges


def stage_blocks(at) -> dict:
    """{stage: the markdown of its four fields}, from the '**i. Stage**' heading and the markdown that follows it."""
    md = [str(m.value) for m in at.markdown]
    out = {}
    for i, v in enumerate(md):
        m = re.fullmatch(r"\*\*(\d+)\. (.+)\*\*", v.strip())
        if m and i + 1 < len(md):
            out[m.group(2)] = {"n": int(m.group(1)), "body": md[i + 1]}
    return out


# ---------- independent expectations ----------

def expected_numbers(T) -> dict:
    """number id -> the exact phrases that must appear on the page, built from the results files."""
    pub = T["published_model"]
    pub_row = "LSTM" if pub == "LSTM" else "XGBoost (168 features)"
    clipped = {fl(r["RUL"]) for r in rows(ROOT / "models/predictions_phase5b.csv") if fl(r["RUL_true"]) > fl(r["RUL"])}
    cap = int(clipped.pop()) if len(clipped) == 1 else None
    nominal = jload(ROOT / "models/results_phase5.json")["mc_dropout"]["nominal"]
    passes = jload(ROOT / "models/results_phase5b.json")["mc_dropout"]["passes"]
    r6 = jload(ROOT / "maintenance/results_phase6.json")["datasets"]
    c, p, o = T["settings"], T["plan"], T["oee"]
    pooled = {pol: o[("Pooled", pol)] for pol in ("Predictive", "Fixed interval", "Run to failure")}
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    sched = []
    for ds in ("FD001", "FD003"):
        tail = " flagged engines booked, " if ds == "FD001" else " booked, "
        sched.append(f"{ds}: {p[ds]['scheduled']} of {p[ds]['flagged']}{tail}{p[ds]['unscheduled']} left")
        sched.append(f"(solver {r6[ds]['snapshot']['schedule_status']})")
    return {
        "how.rul_cap": [f"capped at {cap} cycles"],
        "how.forecast_rmse": [f"RMSE {T['models'][('FD003', pub_row)]['rmse']:.2f} cycles for the {pub} against "
                              f"{T['models'][('FD003', 'XGBoost (168 features)')]['rmse']:.2f} for XGBoost",
                              f"On {T['models'][('FD003', pub_row)]['engines']} engines it never saw (NASA FD003)"],
        "how.interval": [f"runs it {passes} times to give a {100 * nominal:.0f}% interval"],
        "how.flagged": [f"at or below {c['ALERT_THRESHOLD']} cycles",
                        f"{p['FD001']['flagged']} in FD001 and {p['FD003']['flagged']} in FD003"],
        "how.schedule": [f"onto {c['CREWS_PER_DAY']} crews a day over a {c['HORIZON_DAYS']}-day horizon", *sched],
        "how.oee": [f"pooled over {pooled['Predictive']['machines']} machines: OEE {pct(pooled['Predictive']['oee'])} predictive, "
                    f"{pct(pooled['Fixed interval']['oee'])} fixed interval, {pct(pooled['Run to failure']['oee'])} run to failure",
                    f"failures {pooled['Predictive']['unplanned_failures']}, {pooled['Fixed interval']['unplanned_failures']} "
                    f"and {pooled['Run to failure']['unplanned_failures']}"],
    }


def readme_author() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Author\n(.*?)(?=^## )", text, re.M | re.S)
    return m.group(1) if m else ""


# ---------- checks ----------

def check_how(T, card: Card, app: Path = APP, snap: Path = SNAP) -> None:
    at, secs = render("how", app, snap)
    exc = errors(at)
    text = page_text(at)
    dot = dot_of(at)
    card.notes["how"] = {"seconds": round(secs, 2), "exceptions": exc}

    # W1.1 every pre-registered number, recomputed
    want = expected_numbers(T)
    ids = {n["id"] for n in HOW["numbers"]}
    missing = [f"{k}: '{s}'" for k, ss in want.items() for s in ss if s not in text]
    card.set("W1.1", not exc and set(want) == ids and not missing,
             f"{sum(map(len, want.values()))} phrases for {len(want)} pre-registered numbers found on the page" if not missing
             else f"{len(missing)} not found: {missing[:3]}")

    # W2.1 menu position, title, Home's reading list
    nav = re.findall(r'st\.Page\("views/(\w+)\.py", title="([^"]+)"', (app / "streamlit_app.py").read_text(encoding="utf-8"))
    title = [str(t.value) for t in at.title][:1]
    home_at, _ = render("home", app, snap)
    reading = next((str(m.value) for m in home_at.markdown if "How to read" not in str(m.value) and "- **" in str(m.value)
                    and "Fleet health" in str(m.value)), "")
    first = re.match(r"- \*\*([^*:]+):\*\*", reading)
    ok21 = (len(nav) > 1 and nav[1] == ("how", HOW["nav_title"]) and title == [HOW["title"]]
            and first is not None and first.group(1) == HOW["nav_title"])
    card.set("W2.1", ok21, f"menu: {[t for _, t in nav][:3]}; title {title}; Home list starts with "
                           f"{first.group(1) if first else None}")

    # W2.2 diagram stages, order, retrain edge; a section per stage with the four fields and an existing page
    nodes, edges = parse_dot(dot)
    order = [nodes[f"n{i}"]["label"] for i in range(len(nodes))] if all(f"n{i}" in nodes for i in range(len(nodes))) else []
    chain = [(f"n{i}", f"n{i + 1}") for i in range(len(order) - 1)]
    retrain = [e for e in edges if "retrain flag" in e[2] and "dashed" in e[2]]
    blocks = stage_blocks(at)
    menu = {t for _, t in nav}
    probs = []
    if order != HOW["diagram"]["stages_in_order"]:
        probs.append(f"diagram stages {order}")
    if [e[:2] for e in edges if "retrain" not in e[2]] != chain:
        probs.append("chain edges")
    if len(retrain) != 1 or nodes.get(retrain[0][0], {}).get("label") != "Monitoring" or nodes.get(retrain[0][1], {}).get("label") != "Forecast":
        probs.append("retrain edge")
    for i, stage in enumerate(HOW["diagram"]["stages_in_order"], 1):
        b = blocks.get(stage)
        page = HOW["stage_sections"]["see_it"][stage]
        if not b or b["n"] != i:
            probs.append(f"section {stage} missing or out of order")
            continue
        for field in ("What happens", "Why it matters", "Result", "See it"):
            if f"**{field}:**" not in b["body"]:
                probs.append(f"{stage}: no '{field}'")
        if f"*{page}*" not in b["body"] or page not in menu:
            probs.append(f"{stage}: page '{page}' not named or not in the menu")
    card.set("W2.2", not probs, "7 stages in order, chain + dashed retrain edge, 7 sections with 4 fields and a menu page"
             if not probs else "; ".join(probs[:4]))

    # W2.3 plain-English lines
    sys.path.insert(0, str(app))
    with app_env(snap):
        import common
        plain = {t: common.plain(t) for t in ("CP-SAT", "OEE", "MC dropout", "drift")}
    gone = [t for t, s in plain.items() if s not in text]
    card.set("W2.3", not gone, "plain-English lines for CP-SAT, OEE, MC dropout, drift shown" if not gone else f"missing: {gone}")

    # W3.1 simulated stage marked by line style and text; W3.2 the misses are stated
    oee = next((n for n in nodes.values() if n["label"].startswith("OEE")), None)
    oee_block = blocks.get("OEE (simulated)", {}).get("body", "")
    ok31 = bool(oee) and "(simulated)" in oee["label"] and "dashed" in oee["attrs"] and "Simulated" in oee_block
    card.set("W3.1", ok31, "OEE stage dashed and labelled '(simulated)'; its section says Simulated" if ok31
             else f"OEE node {oee}; 'Simulated' in section: {'Simulated' in oee_block}")
    mon = blocks.get("Monitoring", {}).get("body", "")
    card.set("W3.2", "FD004" in mon and "not met" in mon, "Monitoring stage states the FD004 failure and the 'not met' target"
             if "FD004" in mon and "not met" in mon else "Monitoring stage hides a miss")
    card.notes["how_text"] = text

    # W4.1 (How part): no exception, fast, Okabe-Ito only
    bad = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}", dot)} - OKABE_ITO
    card.notes["how_ok"] = (not exc and secs < MAX_SECONDS and bool(dot) and not bad,
                            f"How: {secs:.2f} s, exceptions {len(exc)}, colours outside Okabe-Ito {sorted(bad)}")


def check_about(card: Card, app: Path = APP, snap: Path = SNAP) -> None:
    at, secs = render("about", app, snap)
    exc = errors(at)
    text = page_text(at)
    sub = [str(s.value) for s in at.subheader]
    readme = readme_author()
    links = {"LinkedIn": "linkedin.com/in/", "GitHub": "github.com/Krishna6116)", "Case study": "docs/CASE_STUDY.md",
             "Source code": "github.com/krishna6116/Predictive-Maintenance-OEE)"}
    probs = []
    if not sub or sub[0] != "About the author":
        probs.append(f"first section is {sub[:1]}")
    for key in ("name", "program", "background", "roles"):
        if AUTHOR[key] not in text:
            probs.append(f"page lacks {key}")
        if AUTHOR[key] not in readme:
            probs.append(f"README Author section lacks {key}")
    for name, frag in links.items():
        if f"[{name}](" not in text or frag not in text:
            probs.append(f"link {name}")
    li = re.search(r"\[LinkedIn\]\((https://[^)]+)\)", text)
    if not li or li.group(1) not in readme:
        probs.append("LinkedIn URL differs from the README")
    card.set("W1.2", not exc and not probs, "author section first on About; name, program, background, roles and 4 links equal "
             "the spec and the README" if not probs else "; ".join(probs[:4]))

    # W3.3 no email or phone: app source and the two rendered pages
    hits = []
    for f in sorted(app.rglob("*")):
        if f.is_file() and f.suffix in (".py", ".toml", ".txt", ".md", ".json", ".csv") and "__pycache__" not in f.parts:
            t = f.read_text(encoding="utf-8", errors="ignore")
            hits += [f"{f.relative_to(app)}: {m.group(0)}" for m in (*EMAIL.finditer(t), *PHONE.finditer(t))]
    for name, t in (("About page", text), ("How page", card.notes.get("how_text", ""))):
        hits += [f"{name}: {m.group(0)}" for m in (*EMAIL.finditer(t), *PHONE.finditer(t))]
    card.set("W3.3", not hits, "no email address or phone number in app/ or on the two pages" if not hits else f"found {hits[:3]}")

    how_ok, how_detail = card.notes.get("how_ok", (False, "How not rendered"))
    from dashboards import smoke_v11
    smoke = "how" in smoke_v11.PAGES
    card.set("W4.1", how_ok and not exc and secs < MAX_SECONDS and smoke,
             f"{how_detail}; About: {secs:.2f} s, exceptions {len(exc)}; smoke_v11 covers How: {smoke}")


def check_prior(card: Card) -> None:
    """W1.3: the snapshot data checksum equals the one deployed before v1.2 (Phase 8 capture log); the earlier scorecards'
    page lists include How; the Phase 8 / v1.1 / Phase 9 verifiers themselves run as their own CI steps."""
    man = json.loads((SNAP / "manifest.json").read_text(encoding="utf-8"))
    deployed = json.loads((ROOT / "docs/figures/phase8/capture_log.json").read_text(encoding="utf-8"))["manifest_data_checksum"]
    p8 = json.loads((ROOT / "dashboards/phase8_spec.json").read_text(encoding="utf-8"))
    p8_how = next((p for p in p8["pages"] if p["id"] == "how"), None)
    from dashboards import app_checks
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    in_ci = all(s in ci for s in ("dashboards.verify_phase8", "dashboards.verify_v11", "writeup.verify_phase9", "dashboards.verify_v12"))
    ok = man["data_checksum"] == deployed and p8_how == SPEC["phase8_spec_pages"]["how"] and "how" in app_checks.PAGES and in_ci
    card.set("W1.3", ok, f"snapshot checksum unchanged ({deployed[:12]}): {man['data_checksum'] == deployed}; Phase 8 spec and "
                         f"checks include How: {p8_how == SPEC['phase8_spec_pages']['how'] and 'how' in app_checks.PAGES}; "
                         f"Phase 8, v1.1, Phase 9 and v1.2 verifiers in CI: {in_ci}")


def check_live_evidence(card: Card) -> None:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {"runs": {}}
    runs = {u.rstrip("/"): r for u, r in ev.get("runs", {}).items()}
    local = next((r for u, r in runs.items() if "localhost" in u or "127.0.0.1" in u), None)
    states = {"local server": local, "real app": runs.get(REAL_URL)}
    text = "; ".join(f"{n}: {'pass' if s and s['ok'] else 'FAIL' if s else 'not measured yet'}" for n, s in states.items())
    if any(s and not s["ok"] for s in states.values()):
        card.set("W4.3", False, text)
    else:
        card.set("W4.3", True if all(states.values()) else None, text)
    ci = ev.get("ci")
    if not ci:
        card.set("W4.4", None, "pending: python -m dashboards.verify_v12 --record-ci after CI finishes on the merge candidate")
    else:
        card.set("W4.4", ci["conclusion"] == "success", f"CI {ci['conclusion']} on {ci['commit'][:7]} ({ci['url']})")


# ---------- planted bugs (W4.2) ----------

def _copy_app(tmp: Path) -> Path:
    dst = tmp / "app"
    shutil.copytree(APP, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _replace(path: Path, old: str, new: str) -> None:
    t = path.read_text(encoding="utf-8")
    assert old in t, f"planted bug anchor not found in {path.name}: {old!r}"
    path.write_text(t.replace(old, new, 1), encoding="utf-8")


def planted_bugs(T) -> list[tuple[str, bool, str]]:
    out = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # 1. wrong RMSE row: the FD003 published model's RMSE is changed in the snapshot (manifest re-signed)
        snap = tmp / "snap"
        shutil.copytree(SNAP, snap)
        rs = rows(snap / "models.csv")
        pub = "LSTM" if T["published_model"] == "LSTM" else "XGBoost (168 features)"
        i = next(i for i, r in enumerate(rs) if r["dataset"] == "FD003" and r["model"] == pub)
        rs[i]["rmse"] = repr(fl(rs[i]["rmse"]) + 0.5)
        with open(snap / "models.csv", "w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=list(rs[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rs)
        _resign(snap)
        c = Card()
        c.notes = {}
        check_how(T, c, APP, snap)
        out.append(("wrong RMSE row", c.res["W1.1"][0] is False, "W1.1 fails"))

        # 2. '(simulated)' dropped from the diagram
        app = _copy_app(tmp / "a2")
        _replace(app / "views" / "how.py", '"OEE (simulated)": "3 policies replayed"', '"OEE": "3 policies replayed"')
        _replace(app / "views" / "how.py", '"Features", "Forecast", "Alert", "Schedule", "OEE (simulated)", "Monitoring"]',
                 '"Features", "Forecast", "Alert", "Schedule", "OEE", "Monitoring"]')
        c = Card()
        c.notes = {}
        check_how(T, c, app, SNAP)
        out.append(("'(simulated)' dropped from the diagram", c.res["W3.1"][0] is False, "W3.1 fails"))

        # 3. an email address added to the author section
        app = _copy_app(tmp / "a3")
        _replace(app / "common.py", '"roles": "Open to', '"roles": "Write to someone@example.com. Open to')
        c = Card()
        c.notes = {}
        check_about(c, app, SNAP)
        out.append(("email address in the author section", c.res["W3.3"][0] is False, "W3.3 fails"))
    return out


# ---------- live (W4.3) ----------

def live(url: str) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("--live needs Playwright (pip install playwright==1.55.0; python -m playwright install chromium)")
    url = url.rstrip("/")
    local = "localhost" in url or "127.0.0.1" in url
    page_url = lambda slug: f"{url}/{slug}" if local else f"{url}/~/+/{slug}"  # noqa: E731
    rec = {"url": url, "captured_utc": now()}

    def load(pg, slug: str) -> float:
        t0 = time.perf_counter()
        pg.goto(page_url(slug), wait_until="domcontentloaded", timeout=120_000)
        pg.locator("h1").first.wait_for(state="visible", timeout=120_000)
        s = time.perf_counter() - t0
        pg.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000)
        pg.wait_for_timeout(3_000)
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
        smoke = {}
        for page in ("home", "how", "fleet", "engine", "plan", "oee", "cost", "monitoring", "about"):
            load(pg, "" if page == "home" else page)
            exc = pg.locator("[data-testid='stException']")
            smoke[page] = [exc.nth(i).inner_text()[:300] for i in range(exc.count())]
        times = {page: [load(pg, page) for _ in range(3)] for page in ("how", "about")}
        # extension 2026-11-16: the 'Open <page>' links under the stages exist only in the running app (a page run on
        # its own has no menu to link into), so they are counted here: one per stage
        load(pg, "how")
        page_links = pg.locator("[data-testid='stPageLink']").count()
        desk.close()
        phone = {}
        for page in ("how", "about"):
            ctx = b.new_context(viewport={"width": 375, "height": 667}, device_scale_factor=2, is_mobile=True, has_touch=True,
                                color_scheme="dark", user_agent=p.devices["iPhone SE"]["user_agent"])
            pg = ctx.new_page()
            load(pg, page)
            # as v1.1 V4.1: the page's main scroll container, not inner boxes such as a code block that scrolls by itself
            phone[page] = pg.evaluate("""() => {
                const els = [document.documentElement, ...document.querySelectorAll('*')];
                const sc = els.filter(e => e.scrollHeight > e.clientHeight + 1 &&
                    ['auto', 'scroll'].includes(getComputedStyle(e).overflowY)).sort((a, b) => b.scrollHeight - a.scrollHeight)[0]
                    || document.documentElement;
                return {viewport_width: window.innerWidth, scroll_width: Math.max(document.documentElement.scrollWidth, sc.scrollWidth)};
            }""")
            ctx.close()
        b.close()
    med = {k: statistics.median(v) for k, v in times.items()}
    stages = len(HOW["diagram"]["stages_in_order"])
    ok = (all(not v for v in smoke.values()) and all(m < MAX_SECONDS for m in med.values()) and page_links == stages
          and all(v["viewport_width"] == 375 and v["scroll_width"] <= 375 for v in phone.values()))
    rec.update(ok=ok, exceptions=smoke, title_seconds=times, median=med, phone_375=phone, how_page_links=page_links)
    return rec


def record(key: str, value) -> None:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {
        "note": "v1.2 evidence; written by python -m dashboards.verify_v12 --live URL and --record-ci", "runs": {}}
    if key == "ci":
        ev["ci"] = value
    else:
        ev["runs"][key] = value
    EVIDENCE.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_ci() -> int:
    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    r = subprocess.run([gh, "run", "list", "--commit", head, "--workflow", "CI", "--json", "conclusion,status,url"],
                       cwd=ROOT, capture_output=True, text=True)
    runs = json.loads(r.stdout or "[]")
    if not runs or runs[0]["status"] != "completed":
        print(f"no completed CI run for {head[:7]} yet")
        return 1
    record("ci", {"commit": head, "conclusion": runs[0]["conclusion"], "url": runs[0]["url"], "recorded_utc": now()})
    print(f"CI for {head[:7]}: {runs[0]['conclusion']}")
    return 0 if runs[0]["conclusion"] == "success" else 1


# ---------- scorecard ----------

def verify(planted: bool = True) -> Card:
    card = Card()
    card.notes = {}
    T = truth()
    warm_up(APP, SNAP)
    check_how(T, card)
    check_about(card)
    check_prior(card)
    check_live_evidence(card)
    if planted:
        pb = planted_bugs(T)
        missed = [n for n, ok, _ in pb if not ok]
        card.set("W4.2", len(pb) >= 3 and not missed,
                 f"{len(pb) - len(missed)}/{len(pb)} planted bugs caught: " + "; ".join(f"{n} -> {c}" for n, _, c in pb))
    return card


def report(card: Card) -> int:
    fails = 0
    passed_dims = 0
    for dim in CARD["dimensions"]:
        states = [card.res.get(c["id"], (None, "not run"))[0] for c in dim["checks"]]
        dim_ok = all(s is True for s in states)
        passed_dims += dim_ok
        print(f"{dim['id']} {dim['name']}: {'PASS' if dim_ok else 'FAIL' if False in states else 'pending'}")
        for c in dim["checks"]:
            ok, detail = card.res.get(c["id"], (None, "not run"))
            fails += ok is False
            print(f"  {c['id']} [{'pass' if ok else 'FAIL' if ok is False else 'pending'}] {detail}")
    print(f"v1.2 scorecard: {passed_dims}/{len(CARD['dimensions'])} dimensions pass; {fails} failed checks")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", metavar="URL")
    ap.add_argument("--record-ci", action="store_true")
    ap.add_argument("--no-planted", action="store_true")
    a = ap.parse_args()
    if a.record_ci:
        return record_ci()
    if a.live:
        rec = live(a.live)
        record(rec["url"], rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "exceptions"}, indent=2))
        print("exceptions:", {k: len(v) for k, v in rec["exceptions"].items()})
        return 0 if rec["ok"] else 1
    return report(verify(planted=not a.no_planted))


if __name__ == "__main__":
    sys.exit(main())
