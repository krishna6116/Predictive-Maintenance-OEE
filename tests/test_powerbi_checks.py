"""Phase 8 D1.7: the Power BI export check accepts correct exports (any percent format) and rejects a wrong value."""
import csv

from dashboards import powerbi_checks as P
from dashboards import verify_phase8 as V


def write_exports(tmp_path, pct_style, tamper=False):
    T = V.truth()
    out = []
    for name, (keys, want) in P.expected(T).items():
        cols = keys + list(next(iter(want.values())))
        with open(tmp_path / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for key, vals in want.items():
                key = key if isinstance(key, tuple) else (key,)
                cells = [pct_style(v) if c in ("Interval coverage", "OEE", "Flag rate") else str(v) for c, v in vals.items()]
                w.writerow(list(key) + cells)
            if name == "oee_policies.csv":
                w.writerow(["", "", "", ""])       # a blank Total row is ignored
        out.append(tmp_path / name)
    if tamper:
        rows = list(csv.reader(open(tmp_path / "plan_summary.csv")))
        rows[1][2] = str(int(rows[1][2]) + 1)
        with open(tmp_path / "plan_summary.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
    return T, out


def test_correct_exports_pass_in_every_percent_format(tmp_path):
    for style in (lambda v: f"{v:.4f}", lambda v: f"{100 * v:.1f}", lambda v: f"{100 * v:.1f}%"):
        T, files = write_exports(tmp_path, style)
        ok, detail = P.check_exports(T, files)
        assert ok, detail


def test_wrong_value_fails(tmp_path):
    T, files = write_exports(tmp_path, lambda v: f"{v:.4f}", tamper=True)
    ok, detail = P.check_exports(T, files)
    assert not ok and "plan_summary.csv" in detail
