"""Phase 9 verifier: the 10 pre-registered write-up checks (writeup/phase9_scorecard.json), with independent code.

Run:  python -m writeup.verify_phase9              code checks (CI); network checks reported from writeup/phase9_evidence.json
      python -m writeup.verify_phase9 --network    also runs the network checks (links, live app, GitHub release) and
                                                   rewrites writeup/phase9_evidence.json
      python -m writeup.verify_phase9 --planted    the planted-bug self-test only

Independence: the number check (W1) does not use writeup.facts. It re-reads every results file's numeric leaves directly
(the file list is the pre-registered SOURCES list) and accepts a displayed number only if it equals one of them, or one
of the pre-registered derived values, at the precision shown. Interpretations of the spec made here (reported):
  - a number followed by 'percentage points' is a percent form (compared as 100 x value), like a number followed by '%';
  - criterion and hypothesis IDs (C1-C4, H1-H8) are section numbers, hence structural.
"""
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "writeup" / "phase9_spec.json").read_text(encoding="utf-8"))
CARD = json.loads((ROOT / "writeup" / "phase9_scorecard.json").read_text(encoding="utf-8"))
EVIDENCE = ROOT / "writeup" / "phase9_evidence.json"
RESULTS = ROOT / "writeup" / "phase9_results.json"
RESULTS_DOCS = ["README.md", "docs/CASE_STUDY.md", "docs/RESULTS.md"]
GENERATED_W1 = ["README.md", "docs/CASE_STUDY.md", "docs/RESULTS.md", "docs/README.md"]
STATUS_DOCS = ["README.md", "docs/README.md", "docs/CASE_STUDY.md", "docs/RESULTS.md", "docs/ARCHITECTURE.md",
               "docs/DECISIONS.md", "docs/GLOSSARY.md", "CLAUDE.md"]
PHASE_IDS = ["0", "1", "2", "3", "4", "5", "5b", "6", "7", "7b", "8", "9"]


def read(root: Path, rel: str) -> str:
    p = root / rel
    return p.read_text(encoding="utf-8").replace("\r\n", "\n") if p.exists() else ""


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    extra = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.md")
             if ".venv" not in p.parts and "data" not in p.parts[:1]] if root != ROOT else []
    return sorted(set(out) | set(extra))


# ---------------------------------------------------------------- markdown helpers (independent of the renderer)
FENCE = re.compile(r"```.*?```", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
INLINE = re.compile(r"`[^`\n]*`")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)[^)]*\)")
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)[^)]*\)")
BARE = re.compile(r"https?://[^\s)>\]|]+")


def links(md: str) -> list[str]:
    body = INLINE.sub("", COMMENT.sub("", FENCE.sub("", md)))
    found = [m.group(2) for m in IMAGE.finditer(body)] + [m.group(2) for m in LINK.finditer(body)]
    stripped = LINK.sub(" ", IMAGE.sub(" ", body))
    return found + BARE.findall(stripped)


def prose(md: str) -> str:
    """Text a reader sees, without code, comments, link targets or URLs."""
    t = INLINE.sub(" ", COMMENT.sub(" ", FENCE.sub(" ", md)))
    t = IMAGE.sub(lambda m: m.group(1), t)
    t = LINK.sub(lambda m: m.group(1), t)
    return BARE.sub(" ", t)


def blocks(md: str) -> list[str]:
    """Paragraphs (blank-line separated) and table rows; heading lines dropped (exempt)."""
    out = []
    for para in re.split(r"\n\s*\n", prose(md)):
        lines = [l for l in para.split("\n") if not l.lstrip().startswith("#")]
        rows = [l for l in lines if l.lstrip().startswith("|")]
        rest = [l for l in lines if not l.lstrip().startswith("|")]
        out += rows
        if "".join(rest).strip():
            out.append("\n".join(rest))
    return out


# ---------------------------------------------------------------- W1 numbers
STRUCTURAL = [
    r"\b(?:N-)?C-?MAPSS\b", r"\bFD00[1-4]\b",
    r"\bPhases? \d+b?(?:\s*[-–]\s*\d+b?)?", r"\|\s*\d+b?\.\s", r"(?m)^\s*\d+\.\s",
    r"\bv\d+\.\d+\.\d+\b", r"\bPython \d\.\d+\b", r"\b3\.1\d\b", r"\b(?:19|20)\d\d(?:-\d\d-\d\d)?\b",
    r"\b[CH]\d\b", r"\b[\w./-]+\.(?:md|py|json|csv|ipynb|sql|yml|txt|png)\b",
    r"\b(?:30 seconds|2 minutes|3 minutes|10 minutes)\b",
]
TOKEN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and math.isfinite(o):
        yield float(o)


def allowed_values(root: Path) -> set[float]:
    from writeup.facts import SOURCES  # the pre-registered file list only, not the loader
    vals = set()
    for rel in SOURCES:
        vals |= set(leaves(json.loads(read(root, rel))))
    p6 = json.loads(read(root, "maintenance/results_phase6.json"))["pooled"]
    f = {k: p6[k]["unplanned_failures"] for k in p6}
    w = {k: p6[k]["wasted_cycles"] for k in p6}
    vals |= {f["Fixed interval"] - f["Predictive"], f["Run to failure"] - f["Predictive"],
             w["Fixed interval"] / w["Predictive"]}  # the three pre-registered derived values
    return vals | {abs(v) for v in vals}


def unexplained_numbers(text: str, vals: set[float]) -> list[str]:
    t = text
    for pat in STRUCTURAL:
        t = re.sub(pat, " ", t)
    bad = []
    for m in TOKEN.finditer(t):
        tok = m.group(0)
        pct_form = tok.endswith("%") or t[m.end():m.end() + 20].lstrip().startswith("percentage point")
        num = tok.rstrip("%").replace(",", "").lstrip("+")
        d = len(num.split(".")[1]) if "." in num else 0
        v = float(num)
        ok = any(abs(round((100 * x if pct_form else x), d) - v) < 1e-9 or abs(round((100 * x if pct_form else x), d) + v) < 1e-9
                 for x in vals)
        if not ok:
            bad.append(tok)
    return bad


def home_text() -> str:
    from dashboards.app_checks import run_page, texts
    at = run_page("home")
    if at.exception:
        raise RuntimeError(f"Home page raised {[e.value for e in at.exception]}")
    return texts(at)


def w1(root: Path, home: str | None) -> tuple[bool, str]:
    from writeup.report_phase9 import render_all
    files = render_all()
    stale = [p for p in GENERATED_W1 if read(root, p) != files[p]]
    vals = allowed_values(root)
    bad = {p: unexplained_numbers(prose(read(root, p)), vals) for p in RESULTS_DOCS}
    if home is not None:
        bad["Home page"] = unexplained_numbers(home, vals)
    bad = {k: v for k, v in bad.items() if v}
    ok = not stale and not bad
    return ok, ("generated docs equal a fresh render; every number in README, CASE_STUDY, RESULTS"
                + (" and the Home page" if home is not None else "") + " is a results-file value at its displayed precision"
                if ok else f"stale generated files {stale}; unexplained numbers {dict((k, v[:5]) for k, v in bad.items())}")


# ---------------------------------------------------------------- W2 stale text and phase status
def w2(root: Path) -> tuple[bool, str]:
    probs = []
    for rel in STATUS_DOCS:
        t = read(root, rel)
        for pat in SPEC["stale_patterns"]:
            for m in re.finditer(pat, t, re.I):
                probs.append(f"{rel}: '{m.group(0)[:40]}'")
    progress = read(root, "README.md").split("## Progress", 1)[-1].split("\n## ", 1)[0]
    for name, md in (("README progress", progress), ("docs/README.md", read(root, "docs/README.md"))):
        status = {m.group(1): m.group(2).strip() for m in re.finditer(r"^\|\s*(\d+b?)\.[^|]*\|\s*([^|]+)\|", md, re.M)}
        for pid in PHASE_IDS:
            if not status.get(pid, "").startswith("Done"):
                probs.append(f"{name}: phase {pid} is '{status.get(pid, 'missing')}'")
    claude = read(root, "CLAUDE.md")
    for pid in PHASE_IDS:
        line = next((l for l in claude.splitlines() if re.match(rf"- Phase {pid}\b", l)), "")
        if "done" not in line.lower():
            probs.append(f"CLAUDE.md: Phase {pid} not marked done")
    return (not probs, "no stale phrase; phases 0-9 Done in the README, docs/README.md and CLAUDE.md" if not probs
            else f"{len(probs)} problems: {probs[:5]}")


# ---------------------------------------------------------------- W3 links (code part) + network part
def md_files(root: Path) -> list[str]:
    return [p for p in tracked(root) if p.endswith(".md") and (root / p).exists()]


def w3_local(root: Path) -> tuple[bool, str, list[str]]:
    files = set(tracked(root))
    broken, external = [], set()
    for rel in md_files(root):
        for target in links(read(root, rel)):
            if target.startswith(("http://", "https://")):
                external.add(target.rstrip(".,;"))
                continue
            if target.startswith(("#", "mailto:")):
                continue
            path = urllib.parse.unquote(target.split("#")[0])
            full = (Path(rel).parent / path).as_posix()
            norm = str(Path(full)).replace("\\", "/")
            parts = []
            for seg in norm.split("/"):
                if seg == "..":
                    parts = parts[:-1]
                elif seg not in ("", "."):
                    parts.append(seg)
            resolved = "/".join(parts)
            if resolved not in files and not any(f.startswith(resolved + "/") for f in files):
                broken.append(f"{rel} -> {target}")
    return not broken, (f"every relative link and image in {len(md_files(root))} Markdown files resolves" if not broken
                        else f"{len(broken)} broken: {broken[:5]}"), sorted(external)


def check_external(urls: list[str]) -> dict:
    import http.cookiejar
    import urllib.request
    ex = SPEC["links"]["exceptions"]
    out = {}
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (compatible; pdm-link-check)")]
    for u in urls:
        rule = next((e for e in ex if e["match"] in u), None)
        if rule:
            out[u] = {"ok": u.rstrip("/") == SPEC["author"]["linkedin"], "how": f"exception: {rule['rule']}"}
            continue
        try:
            with opener.open(urllib.request.Request(u, method="GET"), timeout=30) as r:
                out[u] = {"ok": r.status == 200, "how": f"HTTP {r.status}"}
        except Exception as e:  # noqa: BLE001 - any failure is a failed link
            out[u] = {"ok": False, "how": f"{type(e).__name__}: {str(e)[:80]}"}
    return out


# ---------------------------------------------------------------- W4 30-second test
def w4(root: Path) -> tuple[bool, str]:
    t = read(root, "README.md")
    first = t.split("\n## ", 1)[0]
    paras = [p.strip() for p in re.split(r"\n\s*\n", first) if p.strip() and not p.strip().startswith("#")]
    problem = paras[0] if paras else ""
    acronyms = [a for a in re.findall(r"\b[A-Z]{2,}\b", problem)]
    r6 = json.loads(read(root, "maintenance/results_phase6.json"))["pooled"]
    m5 = json.loads(read(root, "models/results_phase5b.json"))["models"]
    need = {
        "problem line": bool(problem) and len(problem) <= 200 and not acronyms and "](" not in problem,
        "live app link": SPEC["release"]["homepage"] in first,
        "forecast result": f"{m5['LSTM']['rmse']:.2f}" in first and f"{m5['XGBoost, all 168 features']['rmse']:.2f}" in first,
        "OEE result": all(f"{100 * r6[p]['oee']:.1f}%" in first for p in r6) and "imulated" in first,
        "failures result": all(re.search(rf"\b{r6[p]['unplanned_failures']}\b", first) for p in r6),
        "tech stack": "Tech stack" in first,
        "image": bool(IMAGE.search(first)),
    }
    miss = [k for k, v in need.items() if not v]
    return not miss, ("first screen has the problem line, live app link, three headline results, tech stack and an image"
                      if not miss else f"first screen lacks {miss}")


# ---------------------------------------------------------------- W5 simulated labelling
def w5(root: Path, home: str | None) -> tuple[bool, str]:
    terms = [re.compile(p, re.I) for p in SPEC["simulated"]["terms"]]
    probs = []
    sources = {p: read(root, p) for p in RESULTS_DOCS}
    for rel, md in sources.items():
        for b in blocks(md):
            if any(t.search(b) for t in terms) and "simulated" not in b.lower():
                probs.append(f"{rel}: '{b.strip()[:60]}'")
    if home is not None:
        for b in home.split("\n"):
            if any(t.search(b) for t in terms) and "simulated" not in b.lower():
                probs.append(f"Home: '{b.strip()[:60]}'")
    return (not probs, "every paragraph and table row mentioning OEE, downtime, failures, hours or wasted life says simulated"
            if not probs else f"{len(probs)} unlabelled: {probs[:4]}")


# ---------------------------------------------------------------- W6 honesty
def w6(root: Path) -> tuple[bool, str]:
    p7 = json.loads(read(root, "results/phase7/results_phase7.json"))["drift_false_alarm_rate_test"]
    c1 = json.loads(read(root, "results/phase7b/checks.json"))["C1"]
    lim = json.loads(read(root, "dashboards/snapshots/facts.json"))["c1_limit"]["value"]
    tgt = json.loads(read(root, "dashboards/snapshots/facts.json"))["p7_drift_target"]["value"]
    probs = []
    for rel in ("README.md", "docs/CASE_STUDY.md"):
        t = read(root, rel)
        need = [f"{100 * p7:.1f}%", f"{100 * tgt:.0f}%", "not met", "C1", "FD004", f"{100 * c1['FD004 synchronized']['rate']:.1f}%",
                f"{100 * c1['FD004 staggered']['rate']:.1f}%", f"{100 * lim:.0f}%"]
        probs += [f"{rel} lacks '{n}'" for n in need if n not in t]
        if not re.search(r"\bfail", t, re.I):
            probs.append(f"{rel} does not say FD004 failed")
        lim_sec = re.search(r"^#{2,3} [^\n]*Limitations[^\n]*\n(.*?)(?=^#{1,3} |\Z)", t, re.S | re.M)
        if not lim_sec:
            probs.append(f"{rel} has no Limitations section")
            continue
        for key in SPEC["honesty"]["limitations"]:
            if key.lower() not in lim_sec.group(1).lower():
                probs.append(f"{rel} Limitations do not name '{key}'")
    return not probs, ("README and case study state both misses with their numbers and name every limitation" if not probs
                       else f"{len(probs)} problems: {probs[:4]}")


# ---------------------------------------------------------------- W7 reproducibility (consistency; execution in CI)
def readme_block(t: str, lang: str) -> list[str]:
    sec = t.split("## Run locally", 1)[-1].split("\n## ", 1)[0]
    m = re.search(rf"```{lang}\n(.*?)```", sec, re.S)
    return [l.strip() for l in m.group(1).strip().splitlines()] if m else []


def ci_job_commands(root: Path, job: str) -> tuple[list[str], str]:
    ci = read(root, ".github/workflows/ci.yml")
    m = re.search(rf"\n  {job}:\n(.*?)(?=\n  [\w-]+:\n|\Z)", ci, re.S)
    if not m:
        return [], ""
    body = m.group(1)
    run = re.search(r"run: \|\n((?:\s{10}.*\n?)+)", body)
    py = re.search(r'python-version: "([\d.]+)"', body)
    return ([l.strip() for l in run.group(1).strip().splitlines()] if run else []), (py.group(1) if py else "")


def w7(root: Path) -> tuple[bool, str]:
    t = read(root, "README.md")
    block = readme_block(t, "bash")
    cmds, py = ci_job_commands(root, "writeup-repro")
    probs = []
    if not block:
        probs.append("README has no bash block under Run locally")
    if cmds != block:
        probs.append("CI job writeup-repro does not run the README block verbatim")
    if not py or f"Python {py}" not in t:
        probs.append(f"README does not state the CI Python version ({py or 'none found'})")
    if "requirements-lock.txt" not in "\n".join(block):
        probs.append("README block does not install requirements-lock.txt")
    return not probs, ("CI job writeup-repro runs the README block verbatim on Python " + py + "; execution is proven by that job"
                       if not probs else "; ".join(probs))


# ---------------------------------------------------------------- W9 docs index
def w9(root: Path) -> tuple[bool, str]:
    idx = read(root, "docs/README.md")
    files = [p for p in tracked(root) if (p.startswith("docs/") and p.count("/") == 1 and p != "docs/README.md")
             or p.startswith("notebooks/")]
    probs = []
    for p in files:
        name = Path(p).name
        line = next((l for l in idx.splitlines() if re.search(rf"\[{re.escape(name)}\]\(", l)), None)
        if line is None:
            probs.append(f"{p} not listed")
            continue
        desc = re.split(r"\]\([^)]*\)\s*[:—-]?", line, maxsplit=1)[-1]
        if len(re.findall(r"[A-Za-z]+", desc)) < 5:
            probs.append(f"{p} description under 5 words")
    return not probs, (f"all {len(files)} docs and notebooks are listed with a description" if not probs
                       else f"{len(probs)} problems: {probs[:5]}")


# ---------------------------------------------------------------- network checks (W3 external, W7 CI run, W8, W10)
API = "https://api.github.com/repos/krishna6116/Predictive-Maintenance-OEE"


def gh(path: str):
    import urllib.request
    req = urllib.request.Request(f"{API}{path}", headers={"Accept": "application/vnd.github+json", "User-Agent": "pdm-verify"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def live_app() -> dict:
    from playwright.sync_api import sync_playwright
    sys.path.insert(0, str(ROOT / "app"))
    from common import TITLES
    url = SPEC["release"]["homepage"]
    man = json.loads(read(ROOT, "dashboards/snapshots/manifest.json"))
    res = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(url, wait_until="domcontentloaded", timeout=90_000)
        wake = pg.get_by_role("button", name=re.compile("get this app back up", re.I))
        if wake.count():
            wake.first.click()
            pg.wait_for_timeout(90_000)
        pg.goto(f"{url}/~/+/", wait_until="domcontentloaded", timeout=90_000)
        pg.locator("h1").first.wait_for(timeout=90_000)
        res["home_title"] = pg.locator("h1").first.inner_text().strip()
        pg.goto(f"{url}/~/+/about", wait_until="domcontentloaded", timeout=90_000)
        pg.locator("h1").first.wait_for(timeout=90_000)
        pg.wait_for_timeout(3_000)
        found = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", pg.locator("body").inner_text())))
        b.close()
    res.update(checksums=found, ok=res["home_title"] == TITLES["home"] and found == [man["data_checksum"]])
    return res


def release_state() -> dict:
    rel = SPEC["release"]
    out = {}
    try:
        repo = gh("")
        out["description"] = repo.get("description") == rel["description"]
        out["homepage"] = (repo.get("homepage") or "").rstrip("/") == rel["homepage"]
        out["topics"] = sorted(repo.get("topics", [])) == sorted(rel["topics"])
        ref = gh(f"/git/ref/tags/{rel['tag']}")
        sha = ref["object"]["sha"]
        if ref["object"]["type"] == "tag":
            sha = gh(f"/git/tags/{sha}")["object"]["sha"]
        out["tag_on_main"] = gh(f"/compare/{sha}...main")["status"] in ("identical", "ahead")
        runs = gh(f"/commits/{sha}/check-runs?per_page=100")["check_runs"]
        out["ci_green_on_tag"] = bool(runs) and all(r["conclusion"] == "success" for r in runs)
        from writeup.report_phase9 import render_all
        body = gh(f"/releases/tags/{rel['tag']}").get("body", "").replace("\r\n", "\n").strip()
        out["release_notes"] = body == render_all()["docs/RELEASE_NOTES.md"].strip()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    out["ok"] = all(v is True for k, v in out.items() if k != "error") and "error" not in out
    return out


def ci_repro_ran() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        runs = gh(f"/commits/{sha}/check-runs?per_page=100")["check_runs"]
        job = [r for r in runs if r["name"] == "writeup-repro"]
        return {"commit": sha, "ok": bool(job) and all(r["conclusion"] == "success" for r in job)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}


def checklist_ticked(root: Path) -> bool:
    text = read(root, "docs/PHASE9_CHECKLIST.md").split("## Not scored", 1)[0]
    items = re.findall(r"^- \[( |x)\]", text, re.M)
    return bool(items) and all(i == "x" for i in items)


# ---------------------------------------------------------------- run
def code_checks(root: Path, home: str | None) -> dict:
    ok3, d3, _ = w3_local(root)
    return {"W1": w1(root, home), "W2": w2(root), "W3_local": (ok3, d3), "W4": w4(root), "W5": w5(root, home),
            "W6": w6(root), "W7_consistency": w7(root), "W9": w9(root)}


def planted_bugs() -> list[tuple[str, str, bool]]:
    """Each bug is planted in a fresh copy of the write-up; the named check must then fail."""
    bugs = [
        ("stale phrase", "W2", lambda r: _edit(r, "docs/GLOSSARY.md", lambda t: t + "\nMore results coming soon.\n")),
        ("wrong number", "W1", lambda r: _edit(r, "docs/RESULTS.md", lambda t: t.replace("12.66", "12.61", 1))),
        ("broken link", "W3_local", lambda r: _edit(r, "docs/CASE_STUDY.md", lambda t: t.replace("(RESULTS.md)", "(RESULT.md)", 1))),
        ("missing 'simulated'", "W5", lambda r: _edit(r, "README.md", lambda t: t.replace(
            "| Downtime, hours (simulated) |", "| Downtime, hours |"))),
        ("doc missing from the index", "W9", lambda r: _edit(r, "docs/README.md", lambda t: "\n".join(
            l for l in t.splitlines() if "[GLOSSARY.md]" not in l))),
        ("a miss left out", "W6", lambda r: _edit(r, "docs/CASE_STUDY.md", lambda t: t.replace("not met", "met"))),
        ("live app link dropped from the first screen (TL;DR line and badge)", "W4", lambda r: _edit(r, "README.md", lambda t: (
            t.split("\n## ", 1)[0].replace(SPEC["release"]["homepage"], "#") + "\n## " + t.split("\n## ", 1)[1]))),
        ("README block differs from CI", "W7_consistency", lambda r: _edit(r, "README.md", lambda t: t.replace(
            "pytest -q\n```\n\nThe full", "pytest -q -x\n```\n\nThe full", 1))),
    ]
    out = []
    for name, cid, plant in bugs:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in tracked(ROOT):
                src = ROOT / rel
                if src.is_file() and (rel.endswith((".md", ".json", ".yml")) or rel.startswith("docs/")):
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, root / rel)
            plant(root)
            res = code_checks(root, None)
            out.append((name, cid, res[cid][0] is False))
    return out


def _edit(root: Path, rel: str, change) -> None:
    p = root / rel
    t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    new = change(t)
    assert new != t, f"planted bug did not change {rel}"
    p.write_text(new, encoding="utf-8")


def main() -> int:
    network = "--network" in sys.argv
    if "--links" in sys.argv:  # external links only (the non-blocking CI step)
        _, _, ext = w3_local(ROOT)
        res = check_external(ext)
        for u, r in res.items():
            print(f"{'ok  ' if r['ok'] else 'FAIL'}  {u}  ({r['how']})")
        return 0 if all(r["ok"] for r in res.values()) else 1
    if "--planted" in sys.argv:
        res = planted_bugs()
        for name, cid, caught in res:
            print(f"{'CAUGHT' if caught else 'MISSED'}  {name} -> {cid}")
        return 0 if all(c for _, _, c in res) else 1
    home = home_text()
    res = code_checks(ROOT, home)
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {}
    if network:
        _, _, ext = w3_local(ROOT)
        ev = {"external_links": check_external(ext), "live_app": live_app(), "ci_repro": ci_repro_ran(),
              "release": release_state()}
        EVIDENCE.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    ext = ev.get("external_links", {})
    bad_ext = [u for u, r in ext.items() if not r["ok"]]
    _, _, current_ext = w3_local(ROOT)
    unchecked = [u for u in current_ext if u not in ext]
    final = {
        "W1": res["W1"], "W2": res["W2"],
        "W3": (res["W3_local"][0] and bool(ext) and not bad_ext and not unchecked,
               f"{res['W3_local'][1]}; external: " + ("not checked yet (run --network)" if not ext else
                                                      f"{len(ext) - len(bad_ext)}/{len(ext)} ok" + (f", failing {bad_ext[:3]}" if bad_ext else "")
                                                      + (f", unchecked {unchecked[:3]}" if unchecked else ""))),
        "W4": res["W4"], "W5": res["W5"], "W6": res["W6"],
        "W7": (res["W7_consistency"][0] and ev.get("ci_repro", {}).get("ok", False),
               res["W7_consistency"][1] + ("; CI run confirmed" if ev.get("ci_repro", {}).get("ok") else "; CI run not confirmed yet")),
        "W8": (ev.get("live_app", {}).get("ok", False),
               f"live app: {ev.get('live_app') or 'not checked yet (run --network)'}; Home renders under AppTest with no exception"),
        "W9": res["W9"],
        "W10": (ev.get("release", {}).get("ok", False) and checklist_ticked(ROOT),
                f"release: {ev.get('release') or 'not checked yet'}; checklist ticked: {checklist_ticked(ROOT)}"),
    }
    names = {c["id"]: c["name"] for c in CARD["checks"]}
    for cid, (ok, detail) in final.items():
        print(f"{'PASS' if ok else 'FAIL'}  {cid} {names[cid]}: {detail}")
    passed = sum(ok for ok, _ in final.values())
    print(f"Phase 9: {passed} of {len(final)} checks pass")
    RESULTS.write_text(json.dumps({"checks": {k: {"passed": ok, "detail": d} for k, (ok, d) in final.items()},
                                   "passed": passed, "out_of": len(final)}, indent=2) + "\n", encoding="utf-8")
    code_ids = ["W1", "W2", "W4", "W5", "W6", "W9"]
    return 0 if all(final[c][0] for c in code_ids) and res["W3_local"][0] and res["W7_consistency"][0] else 1


if __name__ == "__main__":
    sys.exit(main())
