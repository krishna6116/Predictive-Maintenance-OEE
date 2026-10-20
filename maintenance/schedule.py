"""Maintenance schedule with OR-Tools CP-SAT (guide 6.3).

Decision: x[e, d] = 1 if flagged engine e is serviced on day d (d = 0 is today), d < HORIZON_DAYS.
Constraints: each engine at most once; at most CREWS_PER_DAY services a day, minus crews already booked.
Cost (guide weights): serviced on day d uses d x CYCLES_PER_DAY cycles of its pessimistic life rul_low;
  LATE_WEIGHT per cycle past rul_low + EARLY_WEIGHT per cycle of life left unused; UNSERVED_WEIGHT per engine
  left unscheduled. rul_low is floored to whole cycles (conservative).
One search worker and a fixed seed make the solver deterministic for identical inputs. `check_plan` re-checks
any plan independently of the solver.
"""
import math

import pandas as pd
from ortools.sat.python import cp_model

from maintenance import config as C


def day_cost(rul_low: float, day: int, cycles_per_day: int = C.CYCLES_PER_DAY,
             late_w: int = C.LATE_WEIGHT, early_w: int = C.EARLY_WEIGHT) -> int:
    low, used = int(math.floor(rul_low)), day * cycles_per_day
    return late_w * max(0, used - low) + early_w * max(0, low - used)


def schedule(flagged: pd.DataFrame, days: int = C.HORIZON_DAYS, crews_per_day: int = C.CREWS_PER_DAY,
             cycles_per_day: int = C.CYCLES_PER_DAY, booked: dict | None = None, late_w: int = C.LATE_WEIGHT,
             early_w: int = C.EARLY_WEIGHT, unserved_w: int = C.UNSERVED_WEIGHT):
    """flagged: engine_id, rul_low (one row per engine). booked: {day: crews already used}.
    Returns (status, plan, objective) with plan = [(engine_id, day), ...] sorted by day then engine."""
    booked = booked or {}
    if flagged.engine_id.duplicated().any():
        raise ValueError("an engine appears more than once in the flagged list")
    if not flagged.rul_low.map(math.isfinite).all():
        raise ValueError("non-finite rul_low reached the scheduler")
    if flagged.empty:
        return "OPTIMAL", [], 0
    m = cp_model.CpModel()
    eng = [int(e) for e in flagged.engine_id]
    low = dict(zip(eng, flagged.rul_low.astype(float)))
    x = {(e, d): m.NewBoolVar(f"x_{e}_{d}") for e in eng for d in range(days)}
    for e in eng:
        m.Add(sum(x[e, d] for d in range(days)) <= 1)
    for d in range(days):
        m.Add(sum(x[e, d] for e in eng) <= crews_per_day - booked.get(d, 0))
    cost = [x[e, d] * day_cost(low[e], d, cycles_per_day, late_w, early_w) for e in eng for d in range(days)]
    cost += [unserved_w * (1 - sum(x[e, d] for d in range(days))) for e in eng]
    m.Minimize(sum(cost))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = C.SOLVER_SECONDS
    s.parameters.num_search_workers = C.SOLVER_WORKERS
    s.parameters.random_seed = C.SOLVER_SEED
    status = s.StatusName(s.Solve(m))
    if status not in ("OPTIMAL", "FEASIBLE"):
        return status, [], None
    plan = sorted(((e, d) for (e, d), v in x.items() if s.Value(v)), key=lambda p: (p[1], p[0]))
    return status, plan, int(round(s.ObjectiveValue()))


def plan_cost(flagged: pd.DataFrame, plan, cycles_per_day: int = C.CYCLES_PER_DAY, late_w: int = C.LATE_WEIGHT,
              early_w: int = C.EARLY_WEIGHT, unserved_w: int = C.UNSERVED_WEIGHT) -> int:
    low = dict(zip(flagged.engine_id.astype(int), flagged.rul_low.astype(float)))
    served = {e for e, _ in plan}
    return (sum(day_cost(low[e], d, cycles_per_day, late_w, early_w) for e, d in plan)
            + unserved_w * len(set(low) - served))


def check_plan(flagged: pd.DataFrame, plan, days: int = C.HORIZON_DAYS, crews_per_day: int = C.CREWS_PER_DAY,
               booked: dict | None = None) -> list[str]:
    """Independent constraint check. Returns a list of violations (empty = the plan is valid)."""
    booked, problems = booked or {}, []
    known = set(flagged.engine_id.astype(int))
    engines = [e for e, _ in plan]
    if len(engines) != len(set(engines)):
        problems.append("an engine is scheduled more than once")
    if set(engines) - known:
        problems.append(f"scheduled engines that were not flagged: {sorted(set(engines) - known)}")
    for e, d in plan:
        if not 0 <= d < days:
            problems.append(f"engine {e} scheduled on day {d}, outside the {days}-day horizon")
    for d in range(days):
        n = sum(1 for _, dd in plan if dd == d) + booked.get(d, 0)
        if n > crews_per_day:
            problems.append(f"day {d}: {n} services for {crews_per_day} crews")
    return problems
