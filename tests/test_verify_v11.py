"""v1.1 verifier: its own cost arithmetic by hand, and a planted cost-formula bug is caught (V1.2, V5.3)."""
import shutil

from dashboards import verify_v11 as V
from dashboards.verify_phase8 import truth


def test_cost_arithmetic_by_hand():
    T = {"oee": {("Pooled", p): {"downtime_hours": h, "planned_services": 2, "unplanned_failures": 1, "wasted_cycles": 5}
                 for p, h in (("Predictive", 10), ("Fixed interval", 20), ("Run to failure", 100))}}
    got = V.expected_costs(T, dict(c_hour=500, c_service=2000, c_failure=10000, c_cycle=20))
    # predictive 10 x 500 + 2 x 2000 + 1 x 10000 + 5 x 20 = 19,100; fixed adds 10 h x 500 = 24,100
    assert got["Total cost, predictive (illustrative)"] == "$19,100"
    assert got["Saving of predictive vs fixed interval (illustrative)"] == "$5,000"
    assert set(V.cost_input_sets()) >= {"defaults", "all zeros", "very large (1e9 each)"}


def test_planted_cost_bug_is_caught(tmp_path):
    app = tmp_path / "app"
    shutil.copytree(V.APP, app, ignore=shutil.ignore_patterns("__pycache__"))
    V._plant_file(app / "costmodel.py", "c.unplanned_failures * c_failure", "c.unplanned_failures * (c_failure + 48 * c_hour)")
    T = truth()
    good, bad = V.V11Card(), V.V11Card()
    V.check_cost(T, good)
    V.check_cost(T, bad, app=app)
    assert good.res["V1.2"][0] in (True, None) and bad.res["V1.2"][0] is False
