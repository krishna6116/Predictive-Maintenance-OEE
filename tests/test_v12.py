"""v1.2: the How page entry in the Phase 8 spec equals its pre-registered definition; the app's author facts match the spec."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V12 = json.loads((ROOT / "dashboards" / "v12_spec.json").read_text(encoding="utf-8"))


def test_phase8_spec_how_page_equals_its_v12_definition():
    p8 = {p["id"]: p for p in json.loads((ROOT / "dashboards" / "phase8_spec.json").read_text(encoding="utf-8"))["pages"]}
    assert p8["how"] == V12["phase8_spec_pages"]["how"]


def test_how_page_is_second_in_the_menu():
    nav = re.findall(r'st\.Page\("views/(\w+)\.py", title="([^"]+)"', (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8"))
    assert nav[1] == ("how", V12["how_page"]["nav_title"])


def test_author_facts_in_the_app_match_the_spec_and_hold_no_contact_details():
    src = (ROOT / "app" / "common.py").read_text(encoding="utf-8")
    flat = re.sub(r'"\s*\n\s*"', "", src)  # join implicitly concatenated string literals
    for key in ("program", "background", "roles"):
        assert V12["author_section"]["content"][key] in flat, key
    assert not re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", src)


def test_planted_bug_anchors_exist():
    how = (ROOT / "app" / "views" / "how.py").read_text(encoding="utf-8")
    assert '"OEE (simulated)": "3 policies replayed"' in how
    assert '"Features", "Forecast", "Alert", "Schedule", "OEE (simulated)", "Monitoring"]' in how
    assert '"roles": "Open to' in (ROOT / "app" / "common.py").read_text(encoding="utf-8")
