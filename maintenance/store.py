"""Write Phase 6 results to PostgreSQL (fact_prediction, fact_alert, fact_oee).

Run:  python -m maintenance.store       (after python -m maintenance.run_phase6; needs DATABASE_URL in .env)

One transaction: delete everything under this run_id, then insert. The run_id is a hash of the inputs, so a
re-run replaces its rows instead of duplicating them, and a failure leaves the database unchanged.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from maintenance.forecasts import rolling
from pipeline.db import apply_schema, get_engine

CSV = {"float_precision": "round_trip"}
TABLES = ("fact_alert", "fact_oee", "fact_prediction")   # delete order (alerts reference predictions)


def frames(run_id: str) -> dict:
    preds = pd.concat([rolling(ds) for ds in ("FD001", "FD003")], ignore_index=True).assign(run_id=run_id)
    snap = pd.read_csv("maintenance/snapshot_decisions_phase6.csv", **CSV)
    snap = snap[snap.flagged | (snap.risk == "invalid forecast")]
    a1 = pd.DataFrame({"run_id": run_id, "dataset": snap.dataset, "context": "snapshot", "engine_id": snap.engine_id,
                       "decision_cycle": snap.cycle, "forecast_cycle": snap.cycle, "forecast_age": 0,
                       "rul_low_used": snap.rul_low, "risk": snap.risk, "reason": snap.reason,
                       "scheduled_day": snap.service_day, "service_cycle": snap.cycle + snap.service_after_cycles,
                       "status": snap.outcome})
    wo = pd.read_csv("maintenance/work_orders_phase6.csv", **CSV)
    a2 = pd.DataFrame({"run_id": run_id, "dataset": wo.dataset, "context": "replay", "engine_id": wo.engine_id,
                       "decision_cycle": wo.created_cycle, "forecast_cycle": wo.forecast_cycle,
                       "forecast_age": wo.forecast_age, "rul_low_used": wo.rul_low, "risk": wo.risk, "reason": wo.reason,
                       "scheduled_day": wo.day, "service_cycle": wo.service_cycle, "status": wo.status})
    m = pd.read_csv("maintenance/machines_phase6.csv", **CSV)
    oee = m.rename(columns={"machine": "machine_id", "life": "life_cycles"})[
        ["run_id", "dataset", "policy", "machine_id", "life_cycles", "run_cycles", "event", "wasted_cycles",
         "degraded_cycles", "planned_hours", "run_hours", "downtime_hours", "ideal_units", "actual_units", "good_units"]]
    alerts = pd.concat([a1, a2], ignore_index=True)
    for c in ("scheduled_day", "service_cycle"):
        alerts[c] = alerts[c].astype("Int64")
    return {"fact_prediction": preds.drop(columns=["valid"]).assign(valid=preds.valid.astype(bool)),
            "fact_alert": alerts, "fact_oee": oee}


def write(engine, run_id: str) -> dict:
    data = frames(run_id)
    with engine.begin() as conn:
        for t in TABLES:
            conn.execute(text(f"DELETE FROM {t} WHERE run_id = :r"), {"r": run_id})
        for t in ("fact_prediction", "fact_alert", "fact_oee"):
            data[t].to_sql(t, conn, if_exists="append", index=False, method="multi", chunksize=2000)
    return {t: len(df) for t, df in data.items()}


def counts(engine, run_id: str) -> dict:
    with engine.connect() as conn:
        return {t: conn.execute(text(f"SELECT COUNT(*) FROM {t} WHERE run_id = :r"), {"r": run_id}).scalar()
                for t in ("fact_prediction", "fact_alert", "fact_oee")}


def main() -> int:
    run_id = json.loads(Path("maintenance/results_phase6.json").read_text())["run_id"]
    engine = get_engine()
    apply_schema(engine)
    n = write(engine, run_id)
    print(f"run {run_id}: wrote {n}; in database {counts(engine, run_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
