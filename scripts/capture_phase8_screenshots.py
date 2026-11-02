"""Phase 8 evidence: screenshots of the DEPLOYED app for D5.5 (phone width) and D2.4 (About page checksum).

Run:  python scripts/capture_phase8_screenshots.py [app_url]
Needs Playwright (not an app or project dependency):  pip install playwright==1.55.0 && python -m playwright install chromium

D5.5: each page in a 375 px wide phone emulation (iPhone SE, 2x pixels), with the system theme set to DARK on purpose,
      so the app's pinned light theme is tested too. Records the time until the page title is visible and the
      page's scroll width (must be <= 375: no horizontal page scroll).
D2.4: About page at 1280 px with the sidebar closed; the 64-hex checksum is read from the page text and compared
      with dashboards/snapshots/manifest.json.
Everything measured is written to docs/figures/phase8/capture_log.json next to the images.

--readme: instead, three desktop screenshots (1280 px, light theme) for the README in docs/figures/readme/:
      Fleet health, Maintenance plan and OEE policies, each cropped below the page's first chart and reduced to a
      256-colour palette to stay small.
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "phase8"
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
URL = (ARGS[0] if ARGS else (ROOT / "dashboards" / "app_url.txt").read_text().strip()).rstrip("/")
PAGES = {"fleet": "", "plan": "plan", "oee": "oee", "monitoring": "monitoring", "about": "about"}
PHONE_WIDTH = 375


def app_url(path: str) -> str:
    return f"{URL}/~/+/{path}"  # the app itself, without Community Cloud's outer frame


def wake(page) -> None:
    """A sleeping Community Cloud app shows a wake-up button on its outer page."""
    page.goto(URL, wait_until="domcontentloaded")
    btn = page.get_by_role("button", name=re.compile("get this app back up", re.I))
    if btn.count():
        btn.first.click()
        page.wait_for_timeout(60_000)


def load(page, path: str) -> float:
    """Open a page; seconds until its title (h1) is visible, then wait until the script has finished running."""
    t0 = time.perf_counter()
    page.goto(app_url(path), wait_until="domcontentloaded")
    page.locator("h1").first.wait_for(state="visible", timeout=60_000)
    seconds = time.perf_counter() - t0
    page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000)
    page.wait_for_timeout(2_500)  # let the charts finish drawing
    return round(seconds, 2)


def measure(page) -> dict:
    return page.evaluate("""() => {
        const els = [document.documentElement, ...document.querySelectorAll('*')];
        const scroller = els.filter(e => e.scrollHeight > e.clientHeight + 1 &&
            ['auto', 'scroll'].includes(getComputedStyle(e).overflowY)).sort((a, b) => b.scrollHeight - a.scrollHeight)[0]
            || document.documentElement;
        const app = document.querySelector('[data-testid="stApp"]') || document.body;
        return {scroll_width: Math.max(document.documentElement.scrollWidth, scroller.scrollWidth),
                content_height: scroller.scrollHeight, background: getComputedStyle(app).backgroundColor};
    }""")


def full_page_shot(page, path: Path, width: int, height: int) -> None:
    """Streamlit scrolls inside a container, so grow the viewport to the content before the screenshot."""
    page.set_viewport_size({"width": width, "height": height})
    page.mouse.move(1, 1)  # a resting pointer over a table shows its hover toolbar, which a phone never does untapped
    page.wait_for_timeout(1_500)
    page.screenshot(path=str(path), full_page=True)


def main() -> int:
    manifest = json.loads((ROOT / "dashboards" / "snapshots" / "manifest.json").read_text(encoding="utf-8"))
    log = {"app_url": URL, "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "manifest_data_checksum": manifest["data_checksum"], "phone": {}, "about_desktop": {}}
    ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch()
        phone = browser.new_context(**{**p.devices["iPhone SE"], "color_scheme": "dark"})
        page = phone.new_page()
        wake(page)
        for name, path in PAGES.items():
            secs = load(page, path)
            m = measure(page)
            full_page_shot(page, OUT / f"phone_{name}.png", PHONE_WIDTH, m["content_height"])
            page.set_viewport_size({"width": PHONE_WIDTH, "height": 667})
            row = {"title_visible_s": secs, **m, "no_horizontal_scroll": m["scroll_width"] <= PHONE_WIDTH,
                   "under_5_s": secs < 5}
            log["phone"][name] = row
            ok &= row["no_horizontal_scroll"] and row["under_5_s"]
            print(f"phone_{name}.png  {secs:.2f} s  scrollWidth {m['scroll_width']}  background {m['background']}")
        phone.close()

        desk = browser.new_context(viewport={"width": 1280, "height": 900}, color_scheme="dark")
        page = desk.new_page()
        load(page, "about")
        collapse = page.locator('[data-testid="stSidebarCollapseButton"] button')
        if collapse.count():
            page.locator('[data-testid="stSidebar"]').hover()  # the button only shows while the sidebar is hovered
            collapse.first.click()
            page.wait_for_timeout(800)
        found = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", page.locator("body").inner_text())))
        page.screenshot(path=str(OUT / "deployed_about.png"))
        match = found == [manifest["data_checksum"]]
        log["about_desktop"] = {"checksums_on_page": found, "equals_manifest": match,
                                "sidebar_closed": not page.locator('[data-testid="stSidebar"][aria-expanded="true"]').count()}
        ok &= match
        print(f"deployed_about.png  checksum on page {found}  equals manifest: {match}")
        browser.close()
    (OUT / "capture_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return 0 if ok else 1


def readme() -> int:
    from PIL import Image

    out = ROOT / "docs" / "figures" / "readme"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900}, color_scheme="light")
        wake(page)
        for name in ("fleet", "plan", "oee"):
            load(page, PAGES[name])
            chart = page.locator('[data-testid="stVegaLiteChart"], .vega-embed').first
            chart.wait_for(state="visible", timeout=60_000)
            m = measure(page)
            full_page_shot(page, out / f"app_{name}.png", 1280, m["content_height"])
            bottom = chart.bounding_box()["y"] + chart.bounding_box()["height"] + 6
            page.set_viewport_size({"width": 1280, "height": 900})
            img = Image.open(out / f"app_{name}.png").convert("RGB")
            img = img.crop((0, 0, img.width, min(img.height, int(bottom))))
            img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out / f"app_{name}.png", optimize=True)
            kb = (out / f"app_{name}.png").stat().st_size // 1024
            print(f"app_{name}.png  {img.width}x{img.height}  {kb} KB  title: {page.locator('h1').first.inner_text()}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(readme() if "--readme" in sys.argv else main())
