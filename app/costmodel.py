"""Cost what-if (v1.1): illustrative costs applied to the simulated counts per maintenance policy.

total(policy) = downtime_hours x c_hour + planned_services x c_service + unplanned_failures x c_failure
                + wasted_cycles x c_cycle

downtime_hours already include the downtime of failures and of services, so c_failure is only the EXTRA repair cost
of a failure (parts, call-out), never its lost production time: counting that again would double-count it.
Pure function, no Streamlit and no file reading, so tests/test_cost.py can check it with hand-computed cases.
"""
from dataclasses import dataclass

POLICIES = ("Predictive", "Fixed interval", "Run to failure")

# Illustrative defaults for the demo (dashboards/v11_spec.json cost_page.inputs): round example values, not data
DEFAULTS = {"c_hour": 500, "c_service": 2000, "c_failure": 10000, "c_cycle": 20}


@dataclass(frozen=True)
class Counts:
    downtime_hours: float
    planned_services: float
    unplanned_failures: float
    wasted_cycles: float


def total(c: Counts, c_hour: float, c_service: float, c_failure: float, c_cycle: float) -> float:
    for name, v in (("c_hour", c_hour), ("c_service", c_service), ("c_failure", c_failure), ("c_cycle", c_cycle)):
        if v < 0:
            raise ValueError(f"{name} must not be negative")
    return (c.downtime_hours * c_hour + c.planned_services * c_service + c.unplanned_failures * c_failure
            + c.wasted_cycles * c_cycle)


def compare(counts: dict[str, Counts], **costs) -> dict:
    """Totals per policy and the savings of predictive maintenance against the other two."""
    totals = {p: total(counts[p], **costs) for p in POLICIES}
    return {"totals": totals,
            "saving_vs_fixed": totals["Fixed interval"] - totals["Predictive"],
            "saving_vs_rtf": totals["Run to failure"] - totals["Predictive"]}
