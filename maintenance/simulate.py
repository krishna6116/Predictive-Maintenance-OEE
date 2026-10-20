"""Simulated plant and OEE (guide 6.4) and the three maintenance policies (guide 6.5). SIMULATION, not measured data.

Each NASA test engine stands in for one production machine. Its true failure point is L = last observed cycle +
true remaining life (NASA's RUL file, uncapped). A machine's life ends in one of two events:
  service  after `run` cycles (run < L), costing PLANT.service_hours of planned downtime, or
  failure  after completing cycle L, costing PLANT.failure_hours of unplanned downtime.
Policies: run to failure (always fails at L); fixed interval (service after N cycles if N < L); predictive
(alerts on the forecast's lower bound + CP-SAT schedule under crew limits, replayed day by day over the fleet).
Assumptions the guide leaves open are listed in maintenance.config.ASSUMPTIONS.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintenance import config as C
from maintenance.alerts import flag_engines
from maintenance.schedule import check_plan, schedule


@dataclass(frozen=True)
class Outcome:
    machine: int
    life: int          # true failure point L (cycles)
    run: int           # cycles actually run
    event: str         # "service", "failure" or "none"

    @property
    def wasted(self) -> int:
        """Useful life thrown away by servicing early (0 for failures)."""
        return self.life - self.run if self.event == "service" else 0


def production(o: Outcome, plant: C.Plant = C.PLANT) -> dict:
    """Hours and units for one machine life. Degraded cycles are those run with true remaining life < 20."""
    if o.event == "service" and not 0 <= o.run < o.life:
        raise ValueError(f"machine {o.machine}: a service must come before the failure point ({o.run} vs {o.life})")
    if o.event == "failure" and o.run != o.life:
        raise ValueError(f"machine {o.machine}: a failure happens at the failure point")
    degraded = max(0, o.run - max(0, o.life - plant.degraded_cycles))
    normal = o.run - degraded
    per_cycle = plant.hours_per_cycle * plant.units_per_hour
    actual_normal = normal * per_cycle
    actual_degraded = degraded * per_cycle * (1 - plant.degraded_speed_loss)
    down = {"service": plant.service_hours, "failure": plant.failure_hours, "none": 0.0}[o.event]
    run_hours = o.run * plant.hours_per_cycle
    return {"machine": o.machine, "life": o.life, "run_cycles": o.run, "event": o.event, "wasted_cycles": o.wasted,
            "degraded_cycles": degraded, "run_hours": run_hours, "downtime_hours": down,
            "planned_hours": run_hours + down, "ideal_units": run_hours * plant.units_per_hour,
            "actual_units": actual_normal + actual_degraded,
            "good_units": actual_normal * (1 - plant.normal_scrap) + actual_degraded * (1 - plant.degraded_scrap)}


def _div(a: float, b: float) -> float:
    """Like Power BI's DIVIDE: undefined (NaN) instead of an error when there is nothing to divide by."""
    return a / b if b > 0 else math.nan


def oee(rows) -> dict:
    """Availability x Performance x Quality, from summed hours and units (one machine or a whole fleet)."""
    t = pd.DataFrame(list(rows) if not isinstance(rows, pd.DataFrame) else rows)
    s = t[["run_hours", "planned_hours", "ideal_units", "actual_units", "good_units", "downtime_hours"]].sum()
    a, p, q = _div(s.run_hours, s.planned_hours), _div(s.actual_units, s.ideal_units), _div(s.good_units, s.actual_units)
    return {"availability": a, "performance": p, "quality": q, "oee": a * p * q,
            "unplanned_failures": int((t.event == "failure").sum()), "planned_services": int((t.event == "service").sum()),
            "downtime_hours": float(s.downtime_hours), "wasted_cycles": int(t.wasted_cycles.sum()),
            "degraded_cycles": int(t.degraded_cycles.sum()), "machines": len(t)}


def run_to_failure(lives: dict) -> list[Outcome]:
    return [Outcome(m, L, L, "failure") for m, L in lives.items()]


def fixed_interval(lives: dict, n: int) -> list[Outcome]:
    return [Outcome(m, L, n, "service") if n < L else Outcome(m, L, L, "failure") for m, L in lives.items()]


def fixed_interval_n(train_lives) -> int:
    """Guide: N = 10th percentile of the TRAINING engines' lifetimes (floored to whole cycles)."""
    return int(np.floor(np.percentile(np.asarray(train_lives, float), C.FIXED_INTERVAL_PERCENTILE)))


def forecast_at(fc: dict, machine: int, cycle: int) -> dict:
    """The forecast the planner has after `cycle` cycles: the newest reading's forecast, or, once the machine's
    data has ended, its last forecast aged by the cycles run since (ASSUMPTIONS['after_data_ends'])."""
    f = fc[machine]
    last = f["last_cycle"]
    if cycle <= last:
        i = cycle - 1
        return {"engine_id": machine, "forecast_cycle": cycle, "age": 0, "rul_point": f["point"][i],
                "rul_low": f["low"][i], "rul_high": f["high"][i], "valid": f["valid"][i], "invalid_reason": f["why"][i]}
    i, age = last - 1, cycle - last
    return {"engine_id": machine, "forecast_cycle": last, "age": age, "rul_point": max(0.0, f["point"][i] - age),
            "rul_low": max(0.0, f["low"][i] - age), "rul_high": max(0.0, f["high"][i] - age), "valid": f["valid"][i],
            "invalid_reason": f["why"][i]}


def forecast_table(rolling: pd.DataFrame) -> dict:
    """Per-machine arrays from maintenance.forecasts.rolling (validated). Cycles must run 1..last without gaps."""
    out = {}
    for e, g in rolling.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
        if not np.array_equal(g.cycle.to_numpy(), np.arange(1, len(g) + 1)):
            raise ValueError(f"engine {e}: forecast cycles must run 1..last")
        out[int(e)] = {"last_cycle": len(g), "point": g.rul_point.to_numpy(float), "low": g.rul_low.to_numpy(float),
                       "high": g.rul_high.to_numpy(float), "valid": g.valid.to_numpy(bool),
                       "why": g.invalid_reason.astype(str).to_numpy()}
    return out


def predictive(lives: dict, fc: dict, threshold: float = C.ALERT_THRESHOLD, crews: int = C.CREWS_PER_DAY,
               days: int = C.HORIZON_DAYS, cpd: int = C.CYCLES_PER_DAY, late_w: int = C.LATE_WEIGHT,
               early_w: int = C.EARLY_WEIGHT, unserved_w: int = C.UNSERVED_WEIGHT):
    """Replay the fleet day by day. At the start of day k every running machine has completed c = k x cpd cycles:
    1) machines whose failure point was reached during the previous day fail (their open orders are cancelled);
    2) services booked for today are carried out;
    3) machines without an open order are scored, flagged on the lower bound, and the new alerts are scheduled
       with CP-SAT into the crews left over the next `days` days (day 0 = today, carried out immediately).
    Returns (outcomes, work_orders, solver_log)."""
    alive = {m: True for m in lives}
    orders, log, done = {}, [], {}
    for k in range(1, max(lives.values()) // cpd + 2):
        c = k * cpd
        for m in [m for m in lives if alive[m] and lives[m] <= c]:            # 1) failures
            alive[m] = False
            done[m] = Outcome(m, lives[m], lives[m], "failure")
            if m in orders and orders[m]["status"] == "open":
                orders[m]["status"] = "cancelled: failed first"
        def serve(m):
            alive[m] = False
            done[m] = Outcome(m, lives[m], c, "service")
            orders[m]["status"] = "done"
        for m, o in orders.items():                                           # 2) today's services
            if o["status"] == "open" and o["day"] == k and alive[m]:
                serve(m)
        if not any(alive.values()):
            break
        cand = pd.DataFrame([forecast_at(fc, m, c) for m in lives if alive[m] and m not in orders])  # 3) plan
        if cand.empty:
            continue
        flagged = flag_engines(cand, threshold)
        bad = flagged[flagged.risk == "invalid forecast"]
        for _, r in bad.iterrows():
            log.append({"day": k, "event": "invalid forecast", "engine_id": int(r.engine_id), "detail": r.reason})
        flagged = flagged[flagged.risk != "invalid forecast"]
        if flagged.empty:
            continue
        booked = {d: sum(1 for o in orders.values() if o["day"] == k + d and o["status"] in ("open", "done"))
                  for d in range(days)}
        status, plan, obj = schedule(flagged, days, crews, cpd, booked, late_w, early_w, unserved_w)
        problems = check_plan(flagged, plan, days, crews, booked)
        if status not in ("OPTIMAL", "FEASIBLE") or problems:
            raise RuntimeError(f"day {k}: schedule {status}, violations {problems}")
        log.append({"day": k, "event": "schedule", "status": status, "objective": obj, "flagged": len(flagged),
                    "booked": len(plan), "unserved": len(flagged) - len(plan)})
        info = flagged.set_index("engine_id")
        for m, d in plan:
            r = info.loc[m]
            orders[m] = {"engine_id": m, "created_day": k, "created_cycle": c, "day": k + d, "service_cycle": (k + d) * cpd,
                         "rul_low": float(r.rul_low), "rul_point": float(r.rul_point), "forecast_cycle": int(r.forecast_cycle),
                         "forecast_age": int(r.age), "risk": r.risk, "reason": r.reason, "status": "open"}
            if d == 0:
                serve(m)
    for m in lives:                                                           # anything left (cannot happen: all fail)
        if m not in done:
            raise RuntimeError(f"machine {m} neither failed nor was serviced")
    wo = pd.DataFrame(orders.values())
    if not wo.empty:
        wo = wo.assign(true_failure_cycle=wo.engine_id.map(lives)).sort_values(["created_day", "day", "engine_id"])
    return [done[m] for m in sorted(lives)], wo.reset_index(drop=True), pd.DataFrame(log)
