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
