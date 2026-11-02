"""Keep the Streamlit Community Cloud demo awake: one visit to the public page, waking it if it sleeps.

Run:  python scripts/keep_awake.py [app_url]      (used by .github/workflows/keep_awake.yml, not by the CI test jobs)
Needs Playwright:  pip install playwright==1.55.0 && python -m playwright install chromium

Logs whether the app was awake or asleep. Exit 1 if the app was asleep and still has not loaded 120 s after clicking
the wake-up button, or if neither the app's title nor the sleep page appears (no app detected at that URL).
No sign-in, no app interaction: it opens the public URL like any visitor.
"""
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = (sys.argv[1] if len(sys.argv) > 1 else (ROOT / "dashboards" / "app_url.txt").read_text().strip()).rstrip("/")
WAKE_TIMEOUT_S = 120


def app_title(page) -> str | None:
    """The app's page title (h1) from the app's own frame (path /~/+/); the outer page also embeds unrelated frames,
    such as a status widget with its own h1."""
    for frame in page.frames:
        if "/~/+/" not in frame.url:
            continue
        h1 = frame.locator("h1")
        try:
            if h1.count() and h1.first.is_visible():
                return h1.first.inner_text().strip()
        except Exception:  # a frame can detach while the app boots
            continue
    return None


def wait_for_app(page, seconds: float) -> str | None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        title = app_title(page)
        if title:
            return title
        page.wait_for_timeout(2_000)
    return None


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(5_000)
        wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
        if wake.count():
            print(f"{URL}: asleep; clicking the wake-up button")
            wake.first.click()
            title = wait_for_app(page, WAKE_TIMEOUT_S)
            browser.close()
            if title:
                print(f"awake after waking: '{title}'")
                return 0
            print(f"FAILED: app did not load within {WAKE_TIMEOUT_S} s of waking")
            return 1
        title = wait_for_app(page, 60)
        browser.close()
        if not title:
            print(f"FAILED: {URL}: app not detected: no title and no sleep page within 60 s")
            return 1
        print(f"{URL}: awake: '{title}'")
        return 0


if __name__ == "__main__":
    sys.exit(main())
