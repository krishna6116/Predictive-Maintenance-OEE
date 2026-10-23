"""Phase 7: the shift pipeline as a Prefect flow. One flow run = one 8-hour shift = one engine cycle.

For each dataset (FD001, FD003), in order:
  ingest -> validate -> load -> features -> predict_and_store -> alerts -> schedule -> model health -> drift
  -> retraining flag
Run:
  python -m pipeline.flow --once           one shift (continues after the last completed shift)
  python -m pipeline.flow --serve          demo mode: a shift every SERVE_INTERVAL_SECONDS (pipeline/config.py)
  python -m pipeline.flow --replay-all     every shift until all failures are known, then export the results
  python -m pipeline.flow --reset          drop and recreate ONLY the replay schema (ops)

Shadow mode (decision D3): the NASA test data is recorded history, so recommended services are logged as work
orders but do not stop an engine's data. Phase 6 measured what executing them would achieve.
The replay lives in its own Postgres schema (FLOW_SCHEMA), apart from the Phase 2 demo loads.
Predictions come from the PUBLISHED seed-0 LSTM with rule-E intervals (maintenance.forecasts.published_model,
which proves the model reproduces the published predictions and MC passes exactly before it is used), always
in fixed batches of 100 windows so they are bit-identical to the published ones. With --predictor recorded
(CI, other operating systems) the flow replays the committed flow predictions instead of running the model.
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import text

from features.build import apply_scaler, load_json, scaler_file, split_file
from maintenance import config as MCFG
from maintenance.alerts import flag_engines
from maintenance.forecasts import COLUMNS, SOURCES, model_id, published_model, validate as validate_forecasts, with_interval
from maintenance.simulate import ACTIVE_ORDER, aged_forecast, plan_day
from pipeline import config as C
from pipeline import monitoring as MON
from pipeline.check_raw_data import load
from pipeline.db import apply_schema, get_engine
from pipeline.ingest import load_file, next_batch
from pipeline.load import last_received_cycles, load_batch
from pipeline.validate import load_limits, validate as check

OUT = Path("pipeline")
RECORDED = OUT / "phase7_predictions.csv"
BATCH = 100                                   # fixed prediction batch (bit-identical to the published predictions)
UNREADABLE = ("missing value", "duplicate cycle")   # quarantined rows the drift test cannot use (decision D1)
RETRY = {"retries": C.TASK_RETRIES, "retry_delay_seconds": C.TASK_RETRY_DELAY_SECONDS, "cache_policy": NO_CACHE}
PLAIN = {"cache_policy": NO_CACHE}


def run_id_for(ds: str, shift: int) -> str:
    return f"{ds}-test-s{shift:04d}"


# ---------------------------------------------------------------- cached, read-only inputs
@lru_cache(maxsize=None)
def test_file(ds: str) -> pd.DataFrame:
    return load_file(ds, "test")


@lru_cache(maxsize=None)
def failure_log(ds: str) -> dict:
    """The plant's failure log: engine -> true failure cycle L (last NASA reading + NASA's true RUL). The flow
    may only use an entry once its shift has come (MON.label_available)."""
    last = test_file(ds).groupby("unit").cycle.max().to_dict()
    truth = [float(x) for x in (C.RAW / f"RUL_{ds}.txt").read_text().split()]
    return MON.failures(last, {u: truth[u - 1] for u in last})


@lru_cache(maxsize=None)
def training_engines(ds: str) -> pd.DataFrame:
    """Readings of the training-split engines: the drift reference (validation engines are kept out)."""
    raw = load(C.RAW / f"train_{ds}.txt")
    return raw[raw.unit.isin(load_json(split_file(ds))["train"])]


@lru_cache(maxsize=None)
def model_for(ds: str):
    return published_model(ds)           # raises NotPublishedModel unless identity is proven


@lru_cache(maxsize=None)
def recorded(ds: str) -> pd.DataFrame:
    r = pd.read_csv(RECORDED, float_precision="round_trip")
    return r[r.dataset == ds].set_index(["engine_id", "cycle"])


def last_shift() -> int:
    return max(max(failure_log(ds).values()) for ds in C.FLOW_DATASETS)


def _sp(conn, schema):
    conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))


def replace_rows(engine, schema, table, run_id, rows: pd.DataFrame) -> None:
    """Idempotent write: delete this run's rows, insert the new ones, in one transaction."""
    with engine.begin() as conn:
        _sp(conn, schema)
        conn.execute(text(f"DELETE FROM {table} WHERE run_id = :r"), {"r": run_id})
        if len(rows):
            rows.to_sql(table, conn, schema=schema, if_exists="append", index=False, method="multi",
                        chunksize=max(1, 60_000 // max(1, rows.shape[1])))


def mark_run(engine, schema, run_id, **cols) -> None:
    sets = ", ".join(f"{k} = :{k}" for k in cols)
    with engine.begin() as conn:
        _sp(conn, schema)
        conn.execute(text(f"UPDATE pipeline_runs SET {sets} WHERE run_id = :r"), {**cols, "r": run_id})


# ---------------------------------------------------------------- tasks
@task(**RETRY)
def ingest(ds: str, shift: int, inject: dict | None = None) -> pd.DataFrame:
    """Readings that arrived during this shift (one cycle per engine still reporting)."""
    batch = next_batch(test_file(ds), shift, C.CYCLES_PER_SHIFT)
    if inject and inject.get("dataset") == ds and shift in inject.get("shifts", []):
        s = inject["sensor"]
        rows = batch.unit.isin(inject["engines"]) if "engines" in inject else slice(None)
        batch.loc[rows, s] = (batch.loc[rows, s] * (1 + inject["pct"]) if "pct" in inject
                              else batch.loc[rows, s] + inject["sd"] * training_engines(ds)[s].std())
    get_run_logger().info(f"{ds} shift {shift}: {len(batch)} readings from {batch.unit.nunique()} engines"
                          + (f" (injected {inject})" if inject and inject.get("dataset") == ds and shift in inject.get("shifts", []) else ""))
    return batch


@task(**RETRY)
def validate_batch(engine, schema, ds: str, batch: pd.DataFrame):
    good, bad = check(batch, load_limits(ds), last_received_cycles(engine, ds, "test", schema))
    get_run_logger().info(f"{ds}: {len(good)} rows passed, {len(bad)} quarantined"
                          + (f" ({bad.reason.value_counts().to_dict()})" if len(bad) else ""))
    return good, bad


@task(**RETRY)
def load_step(engine, schema, ds: str, shift: int, good, bad, flow_run: str | None) -> str:
    rid = run_id_for(ds, shift)
    load_batch(engine, good, bad, dataset=ds, split="test", start_cycle=shift, n_cycles=C.CYCLES_PER_SHIFT,
               schema=schema, run_id=rid)
    mark_run(engine, schema, rid, shift=shift, last_cycle=shift, flow_status="running", flow_run_id=flow_run)
    get_run_logger().info(f"{ds}: loaded as run {rid}")
    return rid


@task(**RETRY)
def features(engine, schema, ds: str, shift: int, good: pd.DataFrame, window: int):
    """Model inputs: the last `window` scaled readings of each engine that reported this shift, left-padded with
    its first reading (models.sequences.make_windows). An engine with a gap in its loaded history (a quarantined
    cycle) gets no window: its forecast is marked invalid instead of being computed on incomplete data."""
    from features.store import read_readings
    from models.sequences import make_windows
    units = sorted(good.unit.astype(int).unique())
    if not units:
        return np.empty((0, window, len(C.SENSORS)), np.float32), [], []
    hist = apply_scaler(read_readings(engine, ds, "test", units=units, schema=schema), load_json(scaler_file(ds)))
    xs, ok, invalid = [], [], []
    for u, g in hist.groupby("unit"):
        try:
            X, idx = make_windows(g, window, last_only=True)
        except ValueError:
            invalid.append(int(u))
            continue
        xs.append(X)
        ok.append(int(u))
    get_run_logger().info(f"{ds}: {len(ok)} windows built, {len(invalid)} engines with gaps in their history")
    X = np.concatenate(xs) if xs else np.empty((0, window, len(C.SENSORS)), np.float32)
    return X, ok, invalid


def _fixed_batches(fn, X):
    """Run fn on X in batches of exactly BATCH rows (zero-padded), so results do not depend on the batch size."""
    out = []
    for i in range(0, len(X), BATCH):
        part = X[i:i + BATCH]
        pad = np.zeros((BATCH - len(part),) + X.shape[1:], np.float32)
        out.append(fn(np.concatenate([part, pad]))[..., :len(part)])
    return np.concatenate(out, axis=-1) if out else np.empty((0,))


@task(**RETRY)
def predict_and_store(engine, schema, ds: str, shift: int, run_id: str, X, units, invalid, predictor: str) -> pd.DataFrame:
    log = get_run_logger()
    cols = ["dataset", "engine_id", "cycle", "rul_point", "rul_mc_mean", "rul_mc_std", "k", "rul_low", "rul_high"]
    if predictor == "recorded":
        rec = recorded(ds)
        keys = [(u, shift) for u in units]
        missing = [k for k in keys if k not in rec.index]
        if missing:
            raise RuntimeError(f"{ds}: no recorded prediction for {missing[:5]}")
        df = rec.loc[keys].reset_index()[cols] if keys else pd.DataFrame(columns=cols)
    else:
        from models.deep import mc_dropout, predict
        model, _, _ = model_for(ds)
        passes = json.loads(Path(SOURCES[ds]["results"]).read_text())["mc_dropout"]["passes"]
        if len(units):
            point = _fixed_batches(lambda x: predict(model, x), X)
            mc = _fixed_batches(lambda x: mc_dropout(model, x, passes, seed=C.MC_SEED_BASE[ds] + shift), X)
            df = pd.DataFrame({"dataset": ds, "engine_id": units, "cycle": shift, "rul_point": point,
                               "rul_mc_mean": mc.mean(0), "rul_mc_std": mc.std(0, ddof=1)})
            df = with_interval(df, ds)
        else:
            df = pd.DataFrame(columns=cols)
    if invalid:          # no forecast is computed (or replayed) for an engine with a gap in its history
        df = pd.concat([df, pd.DataFrame({"dataset": ds, "engine_id": invalid, "cycle": shift, "rul_point": np.nan,
                                          "rul_mc_mean": np.nan, "rul_mc_std": np.nan, "k": np.nan,
                                          "rul_low": np.nan, "rul_high": np.nan})], ignore_index=True)
    df = df.astype({c: float for c in cols[3:]}).assign(model_id=model_id(ds), source=f"flow:{run_id}")[COLUMNS]
    df = validate_forecasts(df)
    if invalid:
        df.loc[df.engine_id.isin(invalid), "invalid_reason"] = "gap in the engine's loaded history"
    df = df.assign(run_id=run_id, engine_id=df.engine_id.astype(int), cycle=df.cycle.astype(int))
    replace_rows(engine, schema, "fact_prediction", run_id, df)
    log.info(f"{ds}: stored {len(df)} forecasts ({int((~df.valid).sum())} invalid), model {model_id(ds)}")
    return df


@task(**RETRY)
def alerts(engine, schema, ds: str, shift: int):
    """Running engines without a work order, with the forecast the planner has now (newest, or aged), flagged on
    the lower bound. Engines whose failure is already in the log are excluded."""
    fail = failure_log(ds)
    with engine.begin() as conn:
        _sp(conn, schema)
        latest = pd.read_sql(text("""
            SELECT DISTINCT ON (engine_id) engine_id, cycle, rul_point, rul_low, rul_high, valid, invalid_reason
            FROM fact_prediction WHERE dataset = :d AND cycle <= :s ORDER BY engine_id, cycle DESC"""),
            conn, params={"d": ds, "s": shift})
        ordered = set(conn.execute(text("SELECT engine_id FROM fact_alert WHERE context = 'flow' AND dataset = :d "
                                        "AND decision_cycle < :s"), {"d": ds, "s": shift}).scalars())
    running = latest[[not MON.label_available(fail[int(e)], shift) and int(e) not in ordered for e in latest.engine_id]]
    cand = pd.DataFrame([aged_forecast(int(r.engine_id), int(r.cycle), r.rul_point, r.rul_low, r.rul_high, bool(r.valid),
                                       r.invalid_reason or "", shift) for r in running.itertuples()])
    flagged = flag_engines(cand) if len(cand) else cand
    get_run_logger().info(f"{ds}: {len(cand)} running engines without an order; {len(flagged)} flagged on the lower bound")
    return cand, flagged


@task(**RETRY)
def schedule_step(engine, schema, ds: str, shift: int, run_id: str, cand: pd.DataFrame) -> dict:
    """At the start of each day (every CYCLES_PER_DAY shifts) schedule the new alerts with the Phase 6 planning
    step (maintenance.simulate.plan_day: CP-SAT + independent plan check). Orders are kept as recommendations."""
    log = get_run_logger()
    if shift % MCFG.CYCLES_PER_DAY:
        replace_rows(engine, schema, "fact_alert", run_id, pd.DataFrame())
        log.info(f"{ds}: mid-day shift, planning happens at the start of the next day")
        return {"planned": False}
    day, fail = shift // MCFG.CYCLES_PER_DAY, failure_log(ds)
    with engine.begin() as conn:
        _sp(conn, schema)
        prev = pd.read_sql(text("SELECT engine_id, scheduled_day, service_cycle FROM fact_alert WHERE context = 'flow' "
                                "AND dataset = :d AND decision_cycle < :s"), conn, params={"d": ds, "s": shift})
    orders = {}
    for r in prev.itertuples():
        L = fail[int(r.engine_id)]
        status = ("cancelled: failed first" if MON.label_available(L, shift) and L <= r.service_cycle
                  else "due" if r.scheduled_day <= day else "open")
        orders[int(r.engine_id)] = {"day": int(r.scheduled_day), "status": status}
    new_rows, entries = [], []
    if len(cand):
        plan, entries = plan_day(day, shift, cand, orders)
        for e, d in plan:
            o = orders[e]
            new_rows.append({"run_id": run_id, "dataset": ds, "context": "flow", "engine_id": e, "decision_cycle": shift,
                             "forecast_cycle": o["forecast_cycle"], "forecast_age": o["forecast_age"],
                             "rul_low_used": o["rul_low"], "risk": o["risk"], "reason": o["reason"],
                             "scheduled_day": o["day"], "service_cycle": o["service_cycle"],
                             "status": "due" if d == 0 else "open",
                             "forecast_run_id": run_id_for(ds, o["forecast_cycle"])})
    replace_rows(engine, schema, "fact_alert", run_id, pd.DataFrame(new_rows))
    sched = next((x for x in entries if x["event"] == "schedule"), None)
    held = sum(1 for o in orders.values() if o["status"] in ACTIVE_ORDER)
    log.info(f"{ds}: day {day}: {len(new_rows)} new work orders"
             + (f" (CP-SAT {sched['status']}, {sched['unserved']} flagged left for tomorrow)" if sched else "")
             + f"; {held} orders active")
    return {"planned": True, "new_orders": len(new_rows), "status": sched["status"] if sched else None,
            "unserved": sched["unserved"] if sched else 0}


@task(**RETRY)
def model_health(engine, schema, ds: str, shift: int) -> dict:
    """Score each engine whose failure is already in the log, on its last valid prediction."""
    with engine.begin() as conn:
        _sp(conn, schema)
        last = pd.read_sql(text("""
            SELECT DISTINCT ON (engine_id) engine_id, cycle, rul_point, rul_low, rul_high FROM fact_prediction
            WHERE dataset = :d AND cycle <= :s AND valid ORDER BY engine_id, cycle DESC"""), conn, params={"d": ds, "s": shift})
    h = MON.health(last, failure_log(ds), shift)
    get_run_logger().info(f"{ds}: {h['n_labelled']} engines labelled; rolling RMSE {h['rmse']}, "
                          f"miss rate {h['lower_bound_miss_rate']}")
    return h


@task(**RETRY)
def drift_step(engine, schema, ds: str, shift: int, run_id: str, alpha: float) -> pd.DataFrame:
    """KS test on everything received in the window (loaded rows + readable quarantined rows) vs training engines."""
    lo, hi = MON.window_cycles(shift)
    with engine.begin() as conn:
        _sp(conn, schema)
        good = pd.read_sql(text(f"SELECT engine_id AS unit, cycle, {', '.join(C.SENSORS)} FROM fact_sensor_reading "
                                "WHERE dataset = :d AND split = 'test' AND cycle BETWEEN :lo AND :hi"),
                           conn, params={"d": ds, "lo": lo, "hi": hi})
        q = pd.read_sql(text("SELECT reason, raw FROM quarantine WHERE dataset = :d AND split = 'test' "
                             "AND cycle BETWEEN :lo AND :hi"), conn, params={"d": ds, "lo": lo, "hi": hi})
    if len(q):
        q = q[~q.reason.isin(UNREADABLE)]
        raw = pd.DataFrame([r if isinstance(r, dict) else json.loads(r) for r in q.raw])
        if len(raw):
            good = pd.concat([good, raw[["unit", "cycle"] + C.SENSORS]], ignore_index=True)
    d = MON.drift(good, MON.reference_for(training_engines(ds), shift), alpha)
    d[["ks_stat", "p_value", "p_holm"]] = d[["ks_stat", "p_value", "p_holm"]].astype(float)
    replace_rows(engine, schema, "fact_drift", run_id, d.assign(run_id=run_id, dataset=ds, shift=shift)[
        ["run_id", "dataset", "shift", "sensor", "n_recent", "n_reference", "ks_stat", "p_value", "p_holm", "tested", "drifted"]])
    get_run_logger().info(f"{ds}: drift tested on {int(d.tested.sum())} sensors; drifted: {d[d.drifted].sensor.tolist()}")
    return d


@task(**RETRY)
def retraining_flag(engine, schema, ds: str, shift: int, run_id: str, d: pd.DataFrame, h: dict, rule: dict) -> bool:
    log = get_run_logger()
    ref = rule["health"]["reference"][ds]["reference_rmse"]
    flag, reasons = MON.retraining(rule, d, h, ref)
    row = pd.DataFrame([{"run_id": run_id, "dataset": ds, "shift": shift, "model_id": model_id(ds),
                         "n_labelled": h["n_labelled"], "n_scored": h["n_scored"], "rmse": h["rmse"], "mae": h["mae"],
                         "near_failure_mae": h["near_failure_mae"], "coverage_90": h["coverage_90"],
                         "lower_bound_miss_rate": h["lower_bound_miss_rate"], "cum_rmse": h["cum_rmse"],
                         "cum_coverage_90": h["cum_coverage_90"], "cum_lower_bound_miss_rate": h["cum_lower_bound_miss_rate"],
                         "drifted_sensors": ",".join(d[d.drifted].sensor), "needs_retraining": flag,
                         "retraining_reasons": "; ".join(reasons)}])
    metrics = ["rmse", "mae", "near_failure_mae", "coverage_90", "lower_bound_miss_rate", "cum_rmse", "cum_coverage_90",
               "cum_lower_bound_miss_rate"]
    row[metrics] = row[metrics].astype(float)            # empty metrics (no labels yet) are NULL numbers, not text
    replace_rows(engine, schema, "fact_model_health", run_id, row)
    mark_run(engine, schema, run_id, needs_retraining=flag, retraining_reasons="; ".join(reasons), flow_status="complete")
    if flag:
        log.warning(f"{ds} shift {shift}: RETRAINING NEEDED: {'; '.join(reasons)}")
    else:
        log.info(f"{ds}: no retraining needed")
    return flag


# ---------------------------------------------------------------- flow
def next_shift(engine, schema) -> int:
    with engine.begin() as conn:
        _sp(conn, schema)
        rows = conn.execute(text("SELECT dataset, MAX(shift) FROM pipeline_runs WHERE flow_status = 'complete' "
                                 "GROUP BY dataset")).all()
    done = {d: s for d, s in rows}
    return min(done.get(ds, 0) for ds in C.FLOW_DATASETS) + 1


@flow(name="shift-pipeline")
def shift_pipeline(shift: int | None = None, predictor: str = "model", inject: dict | None = None,
                   db_schema: str = C.FLOW_SCHEMA, retry_delay_seconds: float | None = None) -> dict:
    from prefect.runtime import flow_run
    schema = db_schema
    log = get_run_logger()
    engine = get_engine()
    apply_schema(engine, schema)
    shift = shift or next_shift(engine, schema)
    if shift > last_shift():
        log.info(f"replay complete: every engine's failure is known by shift {last_shift()}")
        return {"shift": shift, "done": True}
    rule = MON.load_rule()
    opts = {} if retry_delay_seconds is None else {"retry_delay_seconds": retry_delay_seconds}
    t = {name: fn.with_options(**opts) for name, fn in (
        ("ingest", ingest), ("validate", validate_batch), ("load", load_step), ("features", features),
        ("predict", predict_and_store), ("alerts", alerts), ("schedule", schedule_step), ("health", model_health),
        ("drift", drift_step), ("flag", retraining_flag))}
    out = {"shift": shift, "done": False}
    for ds in C.FLOW_DATASETS:
        rid = run_id_for(ds, shift)
        try:
            batch = t["ingest"](ds, shift, inject)
            good, bad = t["validate"](engine, schema, ds, batch)
            t["load"](engine, schema, ds, shift, good, bad, str(flow_run.id) if flow_run.id else None)
            window = json.loads(Path(SOURCES[ds]["results"]).read_text())["models"]["LSTM"]["settings"]["window"]
            X, units, invalid = t["features"](engine, schema, ds, shift, good, window)
            t["predict"](engine, schema, ds, shift, rid, X, units, invalid, predictor)
            cand, flagged = t["alerts"](engine, schema, ds, shift)
            sched = t["schedule"](engine, schema, ds, shift, rid, cand)
            h = t["health"](engine, schema, ds, shift)
            d = t["drift"](engine, schema, ds, shift, rid, rule["drift"]["alpha"])
            flag = t["flag"](engine, schema, ds, shift, rid, d, h, rule)
            out[ds] = {"run_id": rid, "readings": len(batch), "quarantined": len(bad), "flagged": len(flagged),
                       "schedule": sched, "labelled": h["n_labelled"], "drifted": d[d.drifted].sensor.tolist(),
                       "needs_retraining": flag}
        except Exception:
            try:
                mark_run(engine, schema, rid, flow_status="failed")   # so the next run resumes this shift
            except Exception as mark_exc:
                log.error(f"could not mark {rid} as failed: {mark_exc}")
            raise                                                     # always report the original error
    return out


# ---------------------------------------------------------------- replay export (committed outputs)
EXPORT = {
    "phase7_predictions.csv": ("SELECT dataset, engine_id, cycle, rul_point, rul_mc_mean, rul_mc_std, k, rul_low, rul_high, "
                               "model_id, source, valid, invalid_reason, run_id FROM fact_prediction ORDER BY dataset, engine_id, cycle"),
    "phase7_health.csv": "SELECT * FROM fact_model_health ORDER BY dataset, shift",
    "phase7_drift.csv": "SELECT * FROM fact_drift ORDER BY dataset, shift, sensor",
    "phase7_alerts.csv": ("SELECT run_id, dataset, engine_id, decision_cycle, forecast_cycle, forecast_age, rul_low_used, risk, "
                          "reason, scheduled_day, service_cycle, status, forecast_run_id FROM fact_alert WHERE context = 'flow' "
                          "ORDER BY dataset, decision_cycle, engine_id"),
    "phase7_runs.csv": ("SELECT run_id, dataset, shift, last_cycle, start_cycle, n_cycles, rows_in, rows_loaded, rows_quarantined, "
                        "status, flow_status, needs_retraining, retraining_reasons FROM pipeline_runs WHERE shift IS NOT NULL "
                        "ORDER BY dataset, shift"),
}


def export(engine, schema=C.FLOW_SCHEMA) -> dict:
    """Write the replay's tables as committed CSVs (no timestamps, so they are deterministic)."""
    n = {}
    with engine.begin() as conn:
        _sp(conn, schema)
        for f, sql in EXPORT.items():
            df = pd.read_sql(text(sql), conn)
            df.to_csv(OUT / f, index=False)
            n[f] = len(df)
    return n


def reset(engine, schema=C.FLOW_SCHEMA) -> None:
    if schema == "public":
        raise ValueError("refusing to reset the public schema")
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    apply_schema(engine, schema)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true")
    g.add_argument("--serve", action="store_true")
    g.add_argument("--replay-all", action="store_true")
    g.add_argument("--reset", action="store_true")
    p.add_argument("--predictor", default="model", choices=["model", "recorded"])
    p.add_argument("--until", type=int, help="with --replay-all: stop after this shift")
    p.add_argument("--schema", default=C.FLOW_SCHEMA)
    a = p.parse_args(argv)
    if a.reset:
        reset(get_engine(), a.schema)
        print(f"schema {a.schema} reset")
    elif a.once:
        print(shift_pipeline(predictor=a.predictor, db_schema=a.schema))
    elif a.serve:
        shift_pipeline.serve(name="every-shift", interval=C.SERVE_INTERVAL_SECONDS,
                             parameters={"predictor": a.predictor, "db_schema": a.schema})
    else:
        stop = a.until or last_shift()
        while True:
            r = shift_pipeline(predictor=a.predictor, db_schema=a.schema)
            if r["done"] or r["shift"] >= stop:
                break
        if not a.until:
            print("exported", export(get_engine(), a.schema))
    return 0


if __name__ == "__main__":
    sys.exit(main())
