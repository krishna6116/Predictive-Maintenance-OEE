"""Phase 6: alerts, schedule, simulated plant and OEE, with hand-checkable numbers."""
import math

import numpy as np
import pandas as pd
import pytest

from maintenance import config as C
from maintenance.alerts import flag_engines
from maintenance.forecasts import validate
from maintenance.schedule import check_plan, day_cost, plan_cost, schedule
from maintenance.simulate import (Outcome, fixed_interval, fixed_interval_n, forecast_at, forecast_table, oee,
                                  predictive, production, run_to_failure)


def preds(lows, points=None, highs=None):
    lows = np.asarray(lows, float)
    points = lows + 5 if points is None else np.asarray(points, float)
    highs = lows + 10 if highs is None else np.asarray(highs, float)
    df = pd.DataFrame({"dataset": "T", "engine_id": np.arange(1, len(lows) + 1), "cycle": 50, "rul_point": points,
                       "rul_mc_mean": points, "rul_mc_std": 1.0, "rul_low": lows, "rul_high": highs})
    return validate(df)


# ---- forecasts and alerts
def test_validate_marks_bad_forecasts_with_reasons():
    df = preds([10, 10, 10, 10])
    df.loc[0, "rul_point"] = np.nan
    df.loc[1, "rul_low"] = 30            # above the upper bound (20)
    df.loc[2, "rul_low"] = -1
    v = validate(df.drop(columns=["valid", "invalid_reason"]))
    assert v.valid.tolist() == [False, False, False, True]
    assert v.invalid_reason.tolist()[:3] == ["non-finite value", "lower bound above upper bound", "negative lower bound"]
    dup = validate(pd.concat([df.iloc[[3]], df.iloc[[3]]]).drop(columns=["valid", "invalid_reason"]))
    assert not dup.valid.any() and set(dup.invalid_reason) == {"duplicate forecast"}


def test_alert_threshold_boundary_and_risk_bins():
    f = flag_engines(preds([0, 5, 5.01, 10, 10.5, 20, 20.01, 80]))
    assert f.engine_id.tolist() == [1, 2, 3, 4, 5, 6]               # 20 is flagged, 20.01 is not; sorted by rul_low
    assert f.risk.tolist() == ["critical", "critical", "high", "high", "medium", "medium"]
    assert all("lower 90% bound" in r for r in f.reason)


def test_invalid_forecast_goes_to_manual_inspection_not_scheduler():
    df = preds([3, 15, 50])
    df.loc[0, "rul_mc_std"] = np.inf
    f = flag_engines(validate(df.drop(columns=["valid", "invalid_reason"])))
    assert f.set_index("engine_id").loc[1, "risk"] == "invalid forecast"
    assert f.set_index("engine_id").loc[1, "reason"].startswith("manual inspection")
    assert f[f.risk != "invalid forecast"].engine_id.tolist() == [2]


def test_alert_bins_shrink_for_low_threshold():
    f = flag_engines(preds([2, 7, 9]), threshold=8)
    assert f.risk.tolist() == ["critical", "high"]


# ---- schedule
def test_day_cost_hand_values():
    assert day_cost(7.9, 5) == 100 * 3       # floor(7.9)=7, 10 cycles used -> 3 late
    assert day_cost(7.0, 2) == 3             # 4 used -> 3 early
    assert day_cost(4.0, 2) == 0


def test_schedule_respects_crews_and_serves_most_critical_first():
    f = flag_engines(preds([0, 1, 2, 15, 18]))
    status, plan, obj = schedule(f, days=2, crews_per_day=1)
    assert status == "OPTIMAL" and check_plan(f, plan, days=2, crews_per_day=1) == []
    assert len(plan) == 2 and {e for e, _ in plan} <= {1, 2, 3}     # only the most urgent get the 2 slots
    assert obj == plan_cost(f, plan)


def test_schedule_is_deterministic():
    f = flag_engines(preds(np.linspace(0, 20, 25)))
    assert schedule(f) == schedule(f)


def test_schedule_rejects_duplicates_and_nonfinite():
    f = flag_engines(preds([3, 4]))
    with pytest.raises(ValueError):
        schedule(pd.concat([f, f]))
    g = f.copy()
    g.loc[0, "rul_low"] = np.nan
    with pytest.raises(ValueError):
        schedule(g)


def test_overbooked_capacity_is_reported_infeasible():
    f = flag_engines(preds([3]))
    status, plan, obj = schedule(f, days=1, crews_per_day=1, booked={0: 2})
    assert status == "INFEASIBLE" and plan == [] and obj is None


def test_check_plan_catches_violations():
    f = flag_engines(preds([1, 2, 3]))
    bad = [(1, 0), (1, 1), (2, 0), (9, 0), (3, 7)]
    problems = " | ".join(check_plan(f, bad, days=7, crews_per_day=2))
    for text in ("more than once", "not flagged", "outside the 7-day horizon", "day 0: 3 services"):
        assert text in problems


def test_empty_flag_list():
    assert schedule(flag_engines(preds([90])).iloc[0:0]) == ("OPTIMAL", [], 0)


# ---- plant and OEE (guide 6.4 assumptions)
def test_run_to_failure_hand_example():
    r = production(Outcome(1, 100, 100, "failure"))
    assert (r["degraded_cycles"], r["run_hours"], r["planned_hours"]) == (20, 800, 848)
    assert math.isclose(r["actual_units"], 80 * 80 + 20 * 80 * 0.9)
    assert math.isclose(r["good_units"], 6400 * 0.99 + 1440 * 0.95)
    o = oee([r])
    assert math.isclose(o["availability"], 800 / 848) and math.isclose(o["performance"], 7840 / 8000)
    assert math.isclose(o["quality"], 7704 / 7840) and math.isclose(o["oee"], 7704 / 8480)
    assert math.isclose(o["oee"], o["availability"] * o["performance"] * o["quality"])


def test_early_service_hand_example():
    r = production(Outcome(1, 100, 50, "service"))
    assert r["degraded_cycles"] == 0 and r["wasted_cycles"] == 50 and r["planned_hours"] == 408
    assert math.isclose(oee([r])["oee"], 3960 / 4080)


def test_service_inside_degraded_zone():
    r = production(Outcome(1, 100, 90, "service"))      # cycles 81..90 are within the last 20
    assert r["degraded_cycles"] == 10 and r["wasted_cycles"] == 10


def test_guide_sanity_never_fails_never_serviced_is_full_availability():
    assert oee([production(Outcome(1, 1000, 100, "none"))])["availability"] == 1.0


def test_zero_production_time_is_undefined_not_an_error():
    o = oee([production(Outcome(1, 100, 0, "none"))])
    assert math.isnan(o["availability"]) and math.isnan(o["oee"])


def test_impossible_outcomes_are_rejected():
    with pytest.raises(ValueError):
        production(Outcome(1, 100, 100, "service"))
    with pytest.raises(ValueError):
        production(Outcome(1, 100, 90, "failure"))


def test_fixed_interval_boundary_and_n():
    out = fixed_interval({1: 150, 2: 155, 3: 200}, 155)
    assert [(o.event, o.run) for o in out] == [("failure", 150), ("failure", 155), ("service", 155)]
    assert fixed_interval_n([100, 200, 300, 400]) == int(np.floor(np.percentile([100, 200, 300, 400], 10)))
    assert all(o.event == "failure" for o in run_to_failure({1: 10, 2: 20}))


# ---- predictive replay
def rolling_frame(lives, last, margin=3.0, bias=0.0):
    """Synthetic forecasts: point = true RUL + bias, interval +/- margin."""
    rows = []
    for e, L in lives.items():
        for c in range(1, last[e] + 1):
            t = L - c
            rows.append({"dataset": "T", "engine_id": e, "cycle": c, "rul_point": t + bias, "rul_mc_mean": t + bias,
                         "rul_mc_std": 1.0, "rul_low": max(0.0, t + bias - margin), "rul_high": t + bias + margin})
    return validate(pd.DataFrame(rows))


def test_forecast_ages_after_data_ends():
    fc = forecast_table(rolling_frame({1: 100}, {1: 60}))
    f = forecast_at(fc, 1, 70)
    assert (f["forecast_cycle"], f["age"]) == (60, 10) and f["rul_low"] == pytest.approx(100 - 60 - 3 - 10)
    assert forecast_at(fc, 1, 200)["rul_low"] == 0.0


def test_predictive_with_good_forecasts_avoids_failures_and_respects_crews():
    lives = {e: 120 + 7 * e for e in range(1, 11)}
    fc = forecast_table(rolling_frame(lives, {e: L - 1 for e, L in lives.items()}))
    outs, wo, log = predictive(lives, fc, crews=2)
    assert all(o.event == "service" for o in outs)
    assert wo.engine_id.is_unique and set(wo.status) == {"done"}
    assert wo.groupby("day").size().max() <= 2
    for o in outs:   # every service traces to its work order and happens before the failure point
        w = wo.set_index("engine_id").loc[o.machine]
        assert o.run == w.service_cycle < o.life and w.rul_low <= C.ALERT_THRESHOLD
    a, b = predictive(lives, fc, crews=2), predictive(lives, fc, crews=2)
    assert a[0] == b[0] and a[1].equals(b[1])


def test_predictive_under_crew_shortage_records_failures():
    lives = {e: 100 for e in range(1, 7)}      # six machines due at once, one crew, late alerts
    fc = forecast_table(rolling_frame(lives, {e: 99 for e in lives}, margin=0.0, bias=+15))
    outs, wo, log = predictive(lives, fc, crews=1)
    assert sum(o.event == "failure" for o in outs) > 0
    assert wo.groupby("day").size().max() <= 1
    assert all(s in ("done", "cancelled: failed first") for s in wo.status)


def test_invalid_rolling_forecast_is_never_scheduled():
    lives = {1: 100, 2: 100}
    r = rolling_frame(lives, {1: 99, 2: 99})
    r.loc[r.engine_id == 1, "rul_mc_std"] = np.nan
    fc = forecast_table(validate(r.drop(columns=["valid", "invalid_reason"])))
    outs, wo, log = predictive(lives, fc)
    assert outs[0].event == "failure" and 1 not in set(wo.engine_id)
    assert (log.event == "invalid forecast").any()


# ---- end to end on the real files (skipped when the NASA data is not downloaded)
from pathlib import Path

HAVE_DATA = Path("data/raw/test_FD001.txt").exists() and Path("maintenance/results_phase6.json").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="NASA data or Phase 6 results not present")


@needs_data
def test_snapshot_forecasts_are_the_published_ones_and_valid():
    from maintenance.forecasts import rolling, snapshot
    for ds, f in (("FD001", "models/predictions_phase5.csv"), ("FD003", "models/predictions_phase5b.csv")):
        s, pub = snapshot(ds), pd.read_csv(f, float_precision="round_trip")
        assert s.valid.all() and s.engine_id.tolist() == list(range(1, 101))
        assert s.rul_point.tolist() == pub.LSTM.tolist()
        assert ((s.rul_low <= s.rul_point) & (s.rul_point <= s.rul_high)).all()
        last = rolling(ds).sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
        assert last.rul_point.tolist() == pub.LSTM.tolist() and last.cycle.tolist() == s.cycle.tolist()
    s3 = snapshot("FD003")
    assert np.array_equal(s3.rul_low, pd.read_csv("models/predictions_phase5b.csv", float_precision="round_trip")["MC 90% low (rule E)"])


@needs_data
def test_forecast_to_schedule_to_oee_matches_saved_results():
    import json
    from maintenance.forecasts import rolling, snapshot
    res = json.loads(Path("maintenance/results_phase6.json").read_text())
    saved = pd.read_csv("maintenance/snapshot_decisions_phase6.csv", float_precision="round_trip")
    s = snapshot("FD001")
    f = flag_engines(s)
    status, plan, obj = schedule(f)
    sv = saved[(saved.dataset == "FD001") & saved.service_day.notna()]
    assert status == "OPTIMAL" and obj == res["datasets"]["FD001"]["snapshot"]["objective"]
    assert sorted(plan) == sorted(zip(sv.engine_id.astype(int), sv.service_day.astype(int)))
    lives = {int(e): int(c + t) for e, c, t in zip(s.engine_id, s.cycle, s.true_rul)}
    outs, wo, _ = predictive(lives, forecast_table(rolling("FD001")))
    assert math.isclose(oee([production(o) for o in outs])["oee"], res["datasets"]["FD001"]["policies"]["Predictive"]["oee"])
    machines = pd.read_csv("maintenance/machines_phase6.csv", float_precision="round_trip")
    pm = machines[(machines.dataset == "FD001") & (machines.policy == "Predictive")]
    assert [(o.run, o.event) for o in outs] == list(zip(pm.run_cycles, pm.event))
