"""V5.5: the committed-tree smoke check passes on the app as committed and catches a planted crash."""
import shutil
from pathlib import Path

from dashboards import smoke_v11 as S

ROOT = Path(__file__).resolve().parents[1]


def copy_app(tmp: Path) -> Path:
    shutil.copytree(ROOT / "app", tmp / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "dashboards" / "snapshots", tmp / "dashboards" / "snapshots")
    return tmp


def test_every_page_loads(tmp_path):
    res = S.run_tree(copy_app(tmp_path))
    assert res["ok"], res
    assert set(res["pages"]) == {"_nav", *S.PAGES}


def test_planted_missing_title_is_caught(tmp_path):
    """The preview crash of 2026-11-15: common.py without the 'engine' entry while the page exists."""
    d = copy_app(tmp_path)
    c = d / "app" / "common.py"
    src = c.read_text(encoding="utf-8")
    line = '    "engine": "What happened to one engine, cycle by cycle?",\n'
    assert line in src
    c.write_text(src.replace(line, ""), encoding="utf-8")
    res = S.run_tree(d)
    assert not res["ok"]
    assert res["pages"]["engine"] and any("engine" in e for e in res["pages"]["engine"])


def test_in_place_update_reloads_common(tmp_path):
    """The preview crash replayed in one process: the app runs with an old common.py, the file is updated in place,
    then the engine page opens. app/streamlit_app.py must re-load common.py, so the engine page finds its title."""
    import sys

    from streamlit.testing.v1 import AppTest

    d = copy_app(tmp_path)
    c = d / "app" / "common.py"
    new = c.read_text(encoding="utf-8")
    line = '    "engine": "What happened to one engine, cycle by cycle?",\n'
    saved = {k: sys.modules.get(k) for k in ("common", "costmodel")}
    try:
        c.write_text(new.replace(line, ""), encoding="utf-8")      # the old deployment
        at = AppTest.from_file(str(d / "app" / "streamlit_app.py"), default_timeout=90).run()
        assert not at.exception
        assert "engine" not in sys.modules["common"].TITLES         # the old module is loaded
        c.write_text(new + "\n", encoding="utf-8")                   # the update lands in place (size differs)
        at = AppTest.from_file(str(d / "app" / "streamlit_app.py"), default_timeout=90).run()
        at.switch_page("views/engine.py").run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.title[0].value == "What happened to one engine, cycle by cycle?"
    finally:
        for k, m in saved.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
