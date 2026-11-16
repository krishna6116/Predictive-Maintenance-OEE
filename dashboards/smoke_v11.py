"""v1.1 smoke check (scorecard V5.5): every app page loads with no exception, from the COMMITTED tree, not the working copy.

Why: a page can work locally while the deployed app crashes, because the working copy has changes that were never
committed or pushed. This check exports a git ref (default: the pushed branch, @{u}, which is what the preview app
deploys; HEAD when there is no upstream, as in CI) into a temporary folder with `git archive`, and runs every page there
under Streamlit's AppTest in a fresh Python process, so nothing from the working copy or an earlier import can leak in.

    python -m dashboards.smoke_v11                 # pushed branch (or HEAD)
    python -m dashboards.smoke_v11 --ref HEAD      # last local commit
    python -m dashboards.smoke_v11 --live URL      # every page of a deployed app in a browser (needs Playwright)

Exit code 0 only if every page loads with no exception. --live does not need the repository's Python packages
beyond Playwright, which is not a project dependency (install it separately, as for the Phase 8 screenshots).
"""
import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "how", "fleet", "engine", "plan", "oee", "cost", "monitoring", "about"]
SLUGS = {"home": ""}  # st.navigation serves Home at the root; the others at their file name

# Runs inside the exported tree, in a fresh interpreter: no module from the working copy is importable there.
CHILD = r"""
import json, sys
from streamlit.testing.v1 import AppTest
out = {}
for page in sys.argv[1:]:
    path = "app/streamlit_app.py" if page == "_nav" else f"app/views/{page}.py"
    try:
        at = AppTest.from_file(path, default_timeout=90).run()
        out[page] = [e.value for e in at.exception]
    except Exception as e:  # a crash in AppTest itself counts as a failure too
        out[page] = [f"{type(e).__name__}: {e}"]
print(json.dumps(out))
"""


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def resolve(ref: str | None) -> str:
    if ref:
        return ref
    try:
        return git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}").decode().strip()
    except subprocess.CalledProcessError:
        return "HEAD"


def committed(ref: str) -> dict:
    sha = git("rev-parse", ref).decode().strip()
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", sha)), mode="r:") as tar:
            tar.extractall(tmp, filter="data")
        return {"ref": ref, "sha": sha, **run_tree(Path(tmp))}


def run_tree(folder: Path) -> dict:
    """Run every page of the app in `folder` in a fresh, isolated interpreter (-I: no working-copy imports)."""
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, "_nav", *PAGES], cwd=folder, capture_output=True, text=True,
                       timeout=1800)
    if r.returncode != 0 or not r.stdout.strip():
        return {"ok": False, "pages": {}, "error": r.stderr[-2000:]}
    pages = json.loads(r.stdout.strip().splitlines()[-1])
    return {"ok": all(not v for v in pages.values()), "pages": pages}


def unpushed_warning(ref: str) -> str:
    """Local changes the committed check cannot see (they would not reach the deployed app either)."""
    notes = []
    if git("status", "--porcelain", "--", "app", "dashboards/snapshots").strip():
        notes.append("uncommitted changes under app/ or dashboards/snapshots/")
    if ref != "HEAD" and git("rev-parse", "HEAD") != git("rev-parse", ref):
        notes.append(f"HEAD differs from {ref} (commits not pushed)")
    return "; ".join(notes)


def live(url: str) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("--live needs Playwright (pip install playwright; python -m playwright install chromium)")
    url, pages = url.rstrip("/"), {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        for page in PAGES:
            slug = SLUGS.get(page, page)
            target = f"{url}/{slug}" if "localhost" in url or "127.0.0.1" in url else f"{url}/~/+/{slug}"
            pg.goto(target, wait_until="domcontentloaded", timeout=120_000)
            pg.locator("h1, [data-testid='stException']").first.wait_for(timeout=120_000)
            pg.wait_for_timeout(5_000)  # let the page finish running
            exc = pg.locator("[data-testid='stException']")
            pages[page] = [exc.nth(i).inner_text()[:300] for i in range(exc.count())]
        b.close()
    return {"url": url, "ok": all(not v for v in pages.values()), "pages": pages}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", help="git ref to export (default: the pushed branch @{u}, else HEAD)")
    ap.add_argument("--live", metavar="URL", help="load every page of a deployed app instead")
    a = ap.parse_args()
    if a.live:
        res, where = live(a.live), a.live
    else:
        ref = resolve(a.ref)
        res, where = committed(ref), f"{ref} ({git('rev-parse', '--short', ref).decode().strip()}, committed tree)"
        if warn := unpushed_warning(ref):
            print(f"note: {warn}; this check sees only {ref}")
    print(f"V5.5 smoke check on {where}")
    if "error" in res:
        print("  the page runner failed:\n" + res["error"])
    for page, errs in res["pages"].items():
        print(f"  {'FAIL' if errs else 'ok  '}  {page}" + (f": {errs[0][:200]}" if errs else ""))
    print("V5.5:", "PASS" if res["ok"] else "FAIL")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
