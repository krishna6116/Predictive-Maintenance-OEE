"""Phase 9 write-up: generated docs are current, the code checks pass, and every planted bug is caught."""
from writeup import verify_phase9 as V
from writeup.facts import facts
from writeup.report_phase9 import render_all


def test_every_fact_has_a_source_and_unit():
    F = facts()
    assert F and all(f.source and f.unit for f in F.values())


def test_render_is_deterministic_and_committed_files_are_current():
    a, b = render_all(), render_all()
    assert a == b
    for path, text in a.items():
        assert V.read(V.ROOT, path) == text, f"{path} differs from a fresh render (run python -m writeup.report_phase9)"


def test_numbers_must_match_a_results_value_at_displayed_precision():
    vals = {0.9816694048241356, 12.656307522914739, 8.0, 0.003736898280448764}
    assert V.unexplained_numbers("OEE 98.2% and RMSE 12.66 with 8 failures", vals) == []
    assert V.unexplained_numbers("a gap of +0.37 percentage points", vals) == []
    assert V.unexplained_numbers("RMSE 12.61", vals) == ["12.61"]
    assert V.unexplained_numbers("98.3%", vals) == ["98.3%"]


def test_structural_numbers_are_not_metrics():
    assert V.unexplained_numbers("FD001 and FD004 in Phase 7b, Phases 3-6, criterion C1, v1.0.0, Python 3.14, 2026", set()) == []


def test_code_checks_pass_on_the_repository():
    res = V.code_checks(V.ROOT, None)
    for cid in ("W1", "W3_local", "W4", "W5", "W6", "W7_consistency", "W9"):
        assert res[cid][0], f"{cid}: {res[cid][1]}"


def test_every_planted_bug_is_caught():
    missed = [(name, cid) for name, cid, caught in V.planted_bugs() if not caught]
    assert not missed, f"planted bugs not caught: {missed}"
