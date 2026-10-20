"""Every number Phase 6 uses, in one place, each tagged with where it comes from.

"guide"      = stated in the study guide (Phase 6, steps 6.2-6.5); used unchanged.
"assumption" = the guide leaves it undefined; chosen here, documented, and varied in the sensitivity runs.
"data"       = computed from the NASA training files.
The plant, its production and its OEE are a SIMULATION: nothing here is measured plant data.
"""
from dataclasses import asdict, dataclass

# ---- Alerts (guide 6.2)
ALERT_THRESHOLD = 20                       # guide: flag when the LOWER edge of the interval is <= 20 cycles
RISK_BINS = (-1, 5, 10, ALERT_THRESHOLD)   # guide: critical <= 5, high <= 10, medium <= 20
RISK_LABELS = ("critical", "high", "medium")

# ---- Schedule (guide 6.3)
HORIZON_DAYS = 7          # guide
CREWS_PER_DAY = 3         # guide
CYCLES_PER_DAY = 2        # guide
LATE_WEIGHT = 100         # guide: cost per cycle serviced after the pessimistic failure point (rul_low)
EARLY_WEIGHT = 1          # guide: cost per cycle of useful life thrown away
UNSERVED_WEIGHT = 1000    # guide: cost per flagged engine not serviced inside the horizon
SOLVER_SECONDS = 10       # guide
SOLVER_WORKERS = 1        # assumption: one search worker + fixed seed makes CP-SAT deterministic
SOLVER_SEED = 0


@dataclass(frozen=True)
class Plant:
    """Simulated plant (guide 6.4 table). Each test engine stands in for one production machine."""
    hours_per_cycle: float = 8.0          # guide: planned time per cycle
    units_per_hour: float = 10.0          # guide: ideal output per hour
    service_hours: float = 8.0            # guide: planned maintenance downtime
    failure_hours: float = 48.0           # guide: unplanned failure downtime
    degraded_cycles: int = 20             # guide: "last 20 cycles before failure"
    degraded_speed_loss: float = 0.10     # guide
    degraded_scrap: float = 0.05          # guide
    normal_scrap: float = 0.01            # guide

    def as_dict(self) -> dict:
        return asdict(self)


PLANT = Plant()

# ---- Fixed-interval policy (guide 6.5): service every N cycles, N = 10th percentile of training lifetimes
FIXED_INTERVAL_PERCENTILE = 10            # guide

# ---- Assumptions the guide does not define (all documented in docs/MAINTENANCE_OEE.md)
ASSUMPTIONS = {
    "renewal": "A service or a repair ends the machine's current engine life (the engine is replaced or "
               "overhauled to new). OEE is totalled over each machine's single life; by the renewal-reward "
               "theorem the fleet ratio equals the long-run ratio if lives repeat.",
    "calendar": "All machines start at cycle 1 on day 0 and run CYCLES_PER_DAY cycles a day (the Phase 2 replay "
                "convention). A machine fails when it completes cycle L (its true failure point, RUL 0). It runs "
                "degraded (speed loss, more scrap) in its last 20 cycles, i.e. while its TRUE remaining life is < 20.",
    "service_timing": "A service booked for day d happens at the start of day d, before that day's cycles run.",
    "after_data_ends": "NASA test files stop before failure. After a machine's last reading the predictive policy "
                       "has no new data, so it keeps using its last forecast, aged by one cycle per cycle run.",
    "crew_limit": "Only the predictive policy is crew-limited (as in the guide). Run-to-failure and fixed-interval "
                  "services are not, which favours the two baselines.",
    "work_orders": "A booked service is kept (no re-planning of booked orders). An engine with an open order is not "
                   "re-booked. If it fails before its service day, that is an unplanned failure and the order is "
                   "cancelled. Flagged engines left unserved are retried the next day.",
    "rounding": "rul_low is floored to whole cycles before scheduling (the guide's int()); flooring is conservative.",
}
