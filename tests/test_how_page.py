"""How it works page: loads, has its question as the title, labels the simulated part, adds no hand-typed results,
keeps contact details out of the repository, and handles a missing snapshot with the app's usual message."""
import json
import re
import shutil
from pathlib import Path

from dashboards.app_checks import run_page, texts

ROOT = Path(__file__).resolve().parents[1]
FACTS = json.loads((ROOT / "dashboards" / "snapshots" / "facts.json").read_text(encoding="utf-8"))


def fact(name):
    return FACTS[name]["value"]


def test_page_loads_with_its_question_as_title():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from common import TITLES
    at = run_page("how")
    assert not at.exception, at.exception
    assert [t.value for t in at.title][:1] == [TITLES["how"]]


def test_eight_steps_and_simulated_label():
    at = run_page("how")
    labels = [str(e.label) for e in at.get("expander")]
    steps = [l for l in labels if re.match(r"\d\. ", l)]
    assert [s.split(".")[0] for s in steps] == [str(i) for i in range(1, 9)]
    page = texts(at)
    assert "Simulated" in page and "(simulated)" in page


def test_every_number_comes_from_the_snapshot_facts():
    """The page is a map, not a results page: apart from step numbers, years and names (C-MAPSS, 1D-CNN, CP-SAT), the
    only numbers it may show are the facts it reads, so a changed fact can never leave a stale number here."""
    at = run_page("how")
    page = texts(at)
    # the page's one table has Part as its index, so it shows no row numbers; app_checks.frames() turns that index back
    # into a column, which puts pandas' own row numbers (0-5) into the text scanned here (CI, PR #15)
    assert [t.value.index.name for t in at.table] == ["Part"]
    allowed = {str(i) for i in range(0, 9)} | {"2027", "2028"}
    allowed |= {str(fact(k)) for k in ("rul_cap", "alert_threshold", "crews_per_day", "horizon_days", "interval_passes")}
    allowed.add(str(round(100 * fact("interval_nominal"))))
    shown = set(re.findall(r"(?<![\w.#-])\d+(?![\w.])", re.sub(r"1D-CNN", "", page)))
    assert shown <= allowed, sorted(shown - allowed)


def test_no_email_or_phone_and_links_are_the_author_constants():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    import common as C
    src = (ROOT / "app" / "views" / "how.py").read_text(encoding="utf-8")
    assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", src)               # phase9_spec author.forbidden_in_repo
    assert not re.search(r"\+?\d[\d ()-]{8,}\d", src)
    at = run_page("how")
    urls = {b.proto.url for b in at.get("link_button")}
    assert {C.LINKEDIN, C.REPO, f"{C.DOCS}/CASE_STUDY.md"} <= urls


def test_missing_facts_shows_the_usual_message(tmp_path):
    d = tmp_path / "snap"
    shutil.copytree(ROOT / "dashboards" / "snapshots", d)
    (d / "facts.json").unlink()
    at = run_page("how", snap=d)
    assert not at.exception
    assert any("Data unavailable" in str(e.value) for e in at.error)
