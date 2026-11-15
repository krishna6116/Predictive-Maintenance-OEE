"""v1.1 cost what-if: hand-computed cases for app/costmodel.py (V1.2)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from costmodel import DEFAULTS, POLICIES, Counts, compare, total  # noqa: E402

TOY = {"Predictive": Counts(10, 2, 1, 5), "Fixed interval": Counts(20, 3, 2, 50), "Run to failure": Counts(100, 0, 10, 0)}


def test_each_term_by_hand():
    c = Counts(downtime_hours=10, planned_services=2, unplanned_failures=1, wasted_cycles=5)
    assert total(c, c_hour=1, c_service=0, c_failure=0, c_cycle=0) == 10
    assert total(c, c_hour=0, c_service=1, c_failure=0, c_cycle=0) == 2
    assert total(c, c_hour=0, c_service=0, c_failure=1, c_cycle=0) == 1
    assert total(c, c_hour=0, c_service=0, c_failure=0, c_cycle=1) == 5
    # 10 x 500 + 2 x 2000 + 1 x 10000 + 5 x 20 = 5000 + 4000 + 10000 + 100
    assert total(c, **DEFAULTS) == 19100


def test_savings_by_hand():
    r = compare(TOY, c_hour=100, c_service=10, c_failure=1000, c_cycle=1)
    # predictive 1000 + 20 + 1000 + 5 = 2025; fixed 2000 + 30 + 2000 + 50 = 4080; run to failure 10000 + 0 + 10000 + 0 = 20000
    assert r["totals"] == {"Predictive": 2025, "Fixed interval": 4080, "Run to failure": 20000}
    assert r["saving_vs_fixed"] == 2055 and r["saving_vs_rtf"] == 17975


def test_failure_downtime_is_not_counted_twice():
    """A failure's lost hours are already in downtime_hours: c_failure adds only the extra repair cost."""
    no_failure = Counts(downtime_hours=8, planned_services=1, unplanned_failures=0, wasted_cycles=0)
    with_failure = Counts(downtime_hours=8 + 48, planned_services=1, unplanned_failures=1, wasted_cycles=0)
    costs = dict(c_hour=100, c_service=0, c_failure=5000, c_cycle=0)
    # the failure adds its 48 downtime hours once (48 x 100) plus the extra repair cost once (5000): nothing more
    assert total(with_failure, **costs) - total(no_failure, **costs) == 48 * 100 + 5000


def test_zero_and_large_inputs():
    zero = compare(TOY, c_hour=0, c_service=0, c_failure=0, c_cycle=0)
    assert zero["totals"] == dict.fromkeys(POLICIES, 0) and zero["saving_vs_fixed"] == 0 == zero["saving_vs_rtf"]
    big = compare(TOY, c_hour=1e9, c_service=1e9, c_failure=1e9, c_cycle=1e9)
    assert big["totals"]["Run to failure"] == 110e9


def test_negative_cost_is_refused():
    with pytest.raises(ValueError):
        total(Counts(1, 1, 1, 1), c_hour=-1, c_service=0, c_failure=0, c_cycle=0)
