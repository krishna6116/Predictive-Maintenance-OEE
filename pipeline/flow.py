"""Phase 7: the shift pipeline as a Prefect flow. One flow run = one 8-hour shift = one engine cycle.

For each dataset, in order:
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
(CI, other operating systems) the flow replays recorded flow predictions instead of running the model.

Phase 7b scenarios (defaults reproduce Phase 7 exactly: FD001+FD003, synchronized, monitor v1, shadow mode):
  fleet="staggered"   engine e starts at shift 1 + offset_e (offsets: numpy default_rng(7), 0..150, engine order);
                      its cycle c arrives at shift offset_e + c, its failure at offset_e + L.
  monitor="v2"        the pre-registered Phase 7b rule (pipeline/monitoring2.py, pipeline/monitoring_rule_v2.json);
  monitor="v2.1"      the same rule with the post-hoc spread fix (monitoring_rule_v2_1.json; never a confirmation).
  closed_loop=True    recommended services are carried out: a serviced engine stops reporting after its service
                      shift and never produces a failure label (censored), unless it failed first.
Engine cycles and plant shifts coincide in a synchronized fleet; with offsets, forecasts and orders keep engine
cycles and everything that depends on the clock uses the run ID (one per plant shift).
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
from pipeline import monitoring2 as M2
from pipeline.check_raw_data import load
from pipeline.conditions import CONDITION_DATASETS, normalise, stats_file
from pipeline.db import apply_schema, get_engine
from pipeline.ingest import load_file
from pipeline.load import last_received_cycles, load_batch
from pipeline.validate import load_limits, validate as check

OUT = Path("results/phase7")
RECORDED = OUT / "phase7_predictions.csv"
BATCH = 100                                   # fixed prediction batch (bit-identical to the published predictions)
UNREADABLE = ("missing value", "duplicate cycle")   # quarantined rows the drift test cannot use (decision D1)
RETRY = {"retries": C.TASK_RETRIES, "retry_delay_seconds": C.TASK_RETRY_DELAY_SECONDS, "cache_policy": NO_CACHE}
MAX_OFFSET = 150


def run_id_for(ds: str, shift: int) -> str:
    return f"{ds}-test-s{shift:04d}"


# ---------------------------------------------------------------- cached, read-only inputs
@lru_cache(maxsize=None)
def test_file(ds: str) -> pd.DataFrame:
    return load_file(ds, "test")


@lru_cache(maxsize=None)
def failure_log(ds: str) -> dict:
    """The plant's failure log: engine -> true failure CYCLE L (last NASA reading + NASA's true RUL). The flow may
    only use an entry once its shift has come (on the plant clock: offset + L)."""
    last = test_file(ds).groupby("unit").cycle.max().to_dict()
    truth = [float(x) for x in (C.RAW / f"RUL_{ds}.txt").read_text().split()]
    return MON.failures(last, {u: truth[u - 1] for u in last})


@lru_cache(maxsize=None)
def offsets(ds: str, fleet: str) -> dict:
    units = sorted(test_file(ds).unit.unique())
    if fleet == "synchronized":
        return {int(u): 0 for u in units}
    if fleet != "staggered":
        raise ValueError(f"unknown fleet {fleet}")
    off = np.random.default_rng(7).integers(0, MAX_OFFSET + 1, size=len(units))
    return {int(u): int(o) for u, o in zip(units, off)}


@lru_cache(maxsize=None)
def training_engines(ds: str) -> pd.DataFrame:
    """Readings of the training-split engines: the Phase 7 drift reference (validation engines are kept out)."""
    raw = load(C.RAW / f"train_{ds}.txt")
    return raw[raw.unit.isin(load_json(split_file(ds))["train"])]


@lru_cache(maxsize=None)
def model_for(ds: str):
    return published_model(ds)           # raises NotPublishedModel unless identity is proven


@lru_cache(maxsize=None)
def recorded(ds: str, path: str = str(RECORDED)) -> pd.DataFrame:
    r = pd.read_csv(path, float_precision="round_trip")
    return r[r.dataset == ds].set_index(["engine_id", "cycle"])


VAL_FILES = {"FD001": "models/val_predictions_phase5.csv", "FD003": "models/val_predictions_phase5b.csv"}


@lru_cache(maxsize=None)
def v2_models(ds: str, variant: str = "v2") -> dict:
    """The Phase 7b rule's dataset-specific models, fitted on ALL training-split engines (the flow's deployment)."""
    from pipeline.calibrate_monitoring2 import fit_models, load_dataset
    from pipeline.conditions import normalised_train
    raw = normalised_train(ds) if ds in CONDITION_DATASETS else None
    d = load_dataset(ds, VAL_FILES.get(ds, f"models/val_predictions_{ds}.csv"), f"models/oof_predictions_{ds}.csv", raw)
    return fit_models(d["tr"], variant)


def scale(ds: str, df: pd.DataFrame) -> pd.DataFrame:
    """Model-input scaling: per operating condition for FD002/FD004, the Phase 3 scaler otherwise."""
    if ds in CONDITION_DATASETS:
        return normalise(df, load_json(stats_file(ds)))
    return apply_scaler(df, load_json(scaler_file(ds)))


def last_shift(datasets=C.FLOW_DATASETS, fleet: str = "synchronized") -> int:
    return max(max(L + offsets(ds, fleet)[e] for e, L in failure_log(ds).items()) for ds in datasets)


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


def service_shifts(engine, schema, ds: str, run_id: str) -> dict:
    """Closed loop: engine -> plant shift after which it is serviced (start of its scheduled day), from earlier runs."""
    with engine.begin() as conn:
        _sp(conn, schema)
        r = conn.execute(text("SELECT engine_id, scheduled_day FROM fact_alert WHERE context = 'flow' AND dataset = :d "
                              "AND run_id < :r"), {"d": ds, "r": run_id}).all()
    return {int(e): int(day) * MCFG.CYCLES_PER_DAY for e, day in r}


def label_shifts(ds: str, fleet: str, serviced: dict | None) -> dict:
    """engine -> {"at": plant shift its failure enters the log (inf if serviced first: censored), "life": L}."""
    off = offsets(ds, fleet)
    out = {}
    for e, L in failure_log(ds).items():
        at = L + off[e]
        if serviced and e in serviced and serviced[e] < at:
            at = float("inf")
        out[e] = {"at": at, "life": L}
    return out


def sensor_sd(ds: str, s: str, setting1: pd.Series) -> pd.Series:
    """'One training sd' of sensor s for each reading: the training engines' sd, within the reading's operating
    condition for FD002/FD004 (across conditions the sd mostly measures the gaps between conditions)."""
    if ds in CONDITION_DATASETS:
        stats = load_json(stats_file(ds))
        return setting1.round().astype(int).astype(str).map(lambda c: stats[c][s][1])
    return pd.Series(training_engines(ds)[s].std(), index=setting1.index)


def inject_fault(ds: str, batch: pd.DataFrame, inject: dict, shift: int) -> pd.DataFrame:
    """Test faults: {"sensor", "shifts", and "pct" (fraction of the reading) or "sd" (training sd) or
    "sd_by_shift" ({"<shift>": sd}), optional "engines"}. Fleet-wide unless engines are given."""
    batch = batch.copy()
    s = inject["sensor"]
    rows = batch.unit.isin(inject["engines"]) if "engines" in inject else pd.Series(True, index=batch.index)
    if "pct" in inject:
        batch.loc[rows, s] = batch.loc[rows, s] * (1 + inject["pct"])
    else:
        amount = inject["sd"] if "sd" in inject else inject["sd_by_shift"].get(str(shift), 0.0)
        batch.loc[rows, s] = batch.loc[rows, s] + amount * sensor_sd(ds, s, batch.loc[rows, "setting1"])
    return batch


# ---------------------------------------------------------------- tasks
@task(**RETRY)
def ingest(ds: str, shift: int, inject: dict | None = None, fleet: str = "synchronized", stopped: tuple = ()) -> pd.DataFrame:
    """Readings that arrived during this shift: cycle (shift - offset) of every engine still reporting."""
    df = test_file(ds)
    off = df.unit.map(offsets(ds, fleet))
    batch = df[(df.cycle == shift - off) & ~df.unit.isin(stopped)].reset_index(drop=True)
    if inject and inject.get("dataset") == ds and shift in inject.get("shifts", []):
        batch = inject_fault(ds, batch, inject, shift)
    get_run_logger().info(f"{ds} shift {shift}: {len(batch)} readings from {batch.unit.nunique()} engines"
                          + (f" (injected {inject})" if inject and inject.get("dataset") == ds and shift in inject.get("shifts", []) else "")
                          + (f"; {len(stopped)} engines serviced and removed" if stopped else ""))
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


def read_history(engine, schema, ds: str, units) -> pd.DataFrame:
    """Loaded readings of these engines (with setting1 for the operating condition), oldest first."""
    with engine.begin() as conn:
        _sp(conn, schema)
        return pd.read_sql(text(f"SELECT engine_id AS unit, cycle, setting1, {', '.join(C.SENSORS)} FROM fact_sensor_reading "
                                "WHERE dataset = :d AND split = 'test' AND engine_id = ANY(:u) ORDER BY engine_id, cycle"),
                           conn, params={"d": ds, "u": [int(u) for u in units]})


@task(**RETRY)
def features(engine, schema, ds: str, shift: int, good: pd.DataFrame, window: int):
    """Model inputs: the last `window` scaled readings of each engine that reported this shift, left-padded with
    its first reading (models.sequences.make_windows). An engine with a gap in its loaded history (a quarantined
    cycle) gets no window: its forecast is marked invalid instead of being computed on incomplete data."""
    from models.sequences import make_windows
    units = sorted(good.unit.astype(int).unique())
    if not units:
        return np.empty((0, window, len(C.SENSORS)), np.float32), [], []
    hist = scale(ds, read_history(engine, schema, ds, units))
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
def predict_and_store(engine, schema, ds: str, shift: int, run_id: str, X, units, invalid, predictor: str,
                      fleet: str = "synchronized", recorded_file: str = str(RECORDED)) -> pd.DataFrame:
    log = get_run_logger()
    off = offsets(ds, fleet)
    cyc = {u: shift - off[u] for u in list(units) + list(invalid)}
    cols = ["dataset", "engine_id", "cycle", "rul_point", "rul_mc_mean", "rul_mc_std", "k", "rul_low", "rul_high"]
    if predictor == "recorded":
        rec = recorded(ds, recorded_file)
        keys = [(u, cyc[u]) for u in units]
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
            df = pd.DataFrame({"dataset": ds, "engine_id": units, "cycle": [cyc[u] for u in units], "rul_point": point,
                               "rul_mc_mean": mc.mean(0), "rul_mc_std": mc.std(0, ddof=1)})
            df = with_interval(df, ds)
        else:
            df = pd.DataFrame(columns=cols)
    if invalid:          # no forecast is computed (or replayed) for an engine with a gap in its history
        df = pd.concat([df, pd.DataFrame({"dataset": ds, "engine_id": invalid, "cycle": [cyc[u] for u in invalid],
                                          "rul_point": np.nan, "rul_mc_mean": np.nan, "rul_mc_std": np.nan, "k": np.nan,
                                          "rul_low": np.nan, "rul_high": np.nan})], ignore_index=True)
    df = df.astype({c: float for c in cols[3:]}).assign(model_id=model_id(ds), source=f"flow:{run_id}")[COLUMNS]
    df = validate_forecasts(df)
    if invalid:
        df.loc[df.engine_id.isin(invalid), "invalid_reason"] = "gap in the engine's loaded history"
    df = df.assign(run_id=run_id, engine_id=df.engine_id.astype(int), cycle=df.cycle.astype(int))
    replace_rows(engine, schema, "fact_prediction", run_id, df)
    log.info(f"{ds}: stored {len(df)} forecasts ({int((~df.valid).sum())} invalid), model {model_id(ds)}")
    return df


def latest_forecasts(engine, schema, ds: str, run_id: str, valid_only: bool = False) -> pd.DataFrame:
    """Each engine's newest forecast stored by this run or earlier (run IDs follow the plant clock)."""
    with engine.begin() as conn:
        _sp(conn, schema)
        return pd.read_sql(text(f"""
            SELECT DISTINCT ON (engine_id) engine_id, cycle, rul_point, rul_low, rul_high, valid, invalid_reason
            FROM fact_prediction WHERE dataset = :d AND run_id <= :r {'AND valid' if valid_only else ''}
            ORDER BY engine_id, cycle DESC"""), conn, params={"d": ds, "r": run_id})


@task(**RETRY)
def alerts(engine, schema, ds: str, shift: int, fleet: str = "synchronized", serviced: dict | None = None):
    """Running engines without a work order, with the forecast the planner has now (newest, or aged), flagged on
    the lower bound. Engines whose failure is already in the log are excluded."""
    rid, off = run_id_for(ds, shift), offsets(ds, fleet)
    labels = label_shifts(ds, fleet, serviced)
    latest = latest_forecasts(engine, schema, ds, rid)
    with engine.begin() as conn:
        _sp(conn, schema)
        ordered = set(conn.execute(text("SELECT engine_id FROM fact_alert WHERE context = 'flow' AND dataset = :d "
                                        "AND run_id < :r"), {"d": ds, "r": rid}).scalars())
    running = latest[[labels[int(e)]["at"] > shift and int(e) not in ordered for e in latest.engine_id]]
    cand = pd.DataFrame([aged_forecast(int(r.engine_id), int(r.cycle), r.rul_point, r.rul_low, r.rul_high, bool(r.valid),
                                       r.invalid_reason or "", shift - off[int(r.engine_id)]) for r in running.itertuples()])
    flagged = flag_engines(cand) if len(cand) else cand
    get_run_logger().info(f"{ds}: {len(cand)} running engines without an order; {len(flagged)} flagged on the lower bound")
    return cand, flagged


@task(**RETRY)
def schedule_step(engine, schema, ds: str, shift: int, run_id: str, cand: pd.DataFrame, fleet: str = "synchronized",
                  serviced: dict | None = None) -> dict:
    """At the start of each day (every CYCLES_PER_DAY shifts) schedule the new alerts with the Phase 6 planning
    step (maintenance.simulate.plan_day: CP-SAT + independent plan check). Orders keep engine cycles."""
    log = get_run_logger()
    if shift % MCFG.CYCLES_PER_DAY:
        replace_rows(engine, schema, "fact_alert", run_id, pd.DataFrame())
        log.info(f"{ds}: mid-day shift, planning happens at the start of the next day")
        return {"planned": False}
    day, fail, off = shift // MCFG.CYCLES_PER_DAY, failure_log(ds), offsets(ds, fleet)
    labels = label_shifts(ds, fleet, serviced)
    with engine.begin() as conn:
        _sp(conn, schema)
        prev = pd.read_sql(text("SELECT engine_id, scheduled_day, service_cycle FROM fact_alert WHERE context = 'flow' "
                                "AND dataset = :d AND run_id < :r"), conn, params={"d": ds, "r": run_id})
    orders = {}
    for r in prev.itertuples():
        e = int(r.engine_id)
        status = ("cancelled: failed first" if labels[e]["at"] <= shift and fail[e] <= r.service_cycle
                  else "due" if r.scheduled_day <= day else "open")
        orders[e] = {"day": int(r.scheduled_day), "status": status}
    new_rows, entries = [], []
    if len(cand):
        plan, entries = plan_day(day, shift, cand, orders)
        for e, d in plan:
            o = orders[e]
            new_rows.append({"run_id": run_id, "dataset": ds, "context": "flow", "engine_id": e,
                             "decision_cycle": shift - off[e], "forecast_cycle": o["forecast_cycle"],
                             "forecast_age": o["forecast_age"], "rul_low_used": o["rul_low"], "risk": o["risk"],
                             "reason": o["reason"], "scheduled_day": o["day"], "service_cycle": o["service_cycle"] - off[e],
                             "status": "due" if d == 0 else "open",
                             "forecast_run_id": run_id_for(ds, o["forecast_cycle"] + off[e])})
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
    """Phase 7 rule: score each engine whose failure is already in the log, on its last valid prediction."""
    last = latest_forecasts(engine, schema, ds, run_id_for(ds, shift), valid_only=True)
    h = MON.health(last, failure_log(ds), shift)
    get_run_logger().info(f"{ds}: {h['n_labelled']} engines labelled; rolling RMSE {h['rmse']}, "
                          f"miss rate {h['lower_bound_miss_rate']}")
    return h


@task(**RETRY)
def model_health_v2(engine, schema, ds: str, shift: int, fleet: str, serviced: dict | None, p0: float,
                    variant: str = "v2") -> dict:
    """Phase 7b rule: life-matched error ratio and binomial miss-rate test on the last 20 labelled engines."""
    last = latest_forecasts(engine, schema, ds, run_id_for(ds, shift), valid_only=True)
    h = M2.health(last, label_shifts(ds, fleet, serviced), shift, v2_models(ds, variant)["error"], p0)
    get_run_logger().info(f"{ds}: {h['n_labelled']} engines labelled; error ratio {h['error_ratio']}, miss rate "
                          f"{h['miss_rate']} (p {h['miss_p']})")
    return h


def received_rows(engine, schema, ds: str, run_ids: list) -> pd.DataFrame:
    """Everything received in these runs: loaded rows plus readable quarantined rows (decision D1)."""
    with engine.begin() as conn:
        _sp(conn, schema)
        good = pd.read_sql(text(f"SELECT engine_id AS unit, cycle, setting1, {', '.join(C.SENSORS)} FROM fact_sensor_reading "
                                "WHERE dataset = :d AND split = 'test' AND run_id = ANY(:r)"), conn, params={"d": ds, "r": run_ids})
        q = pd.read_sql(text("SELECT reason, raw FROM quarantine WHERE dataset = :d AND split = 'test' AND run_id = ANY(:r)"),
                        conn, params={"d": ds, "r": run_ids})
    if len(q):
        q = q[~q.reason.isin(UNREADABLE)]
        raw = pd.DataFrame([r if isinstance(r, dict) else json.loads(r) for r in q.raw])
        if len(raw):
            good = pd.concat([good, raw[["unit", "cycle", "setting1"] + C.SENSORS]], ignore_index=True)
    return good


@task(**RETRY)
def drift_step(engine, schema, ds: str, shift: int, run_id: str, alpha: float) -> pd.DataFrame:
    """Phase 7 rule: KS on everything received in the window vs age-matched training engines (synchronized fleet)."""
    lo, hi = MON.window_cycles(shift)
    good = received_rows(engine, schema, ds, [run_id_for(ds, k) for k in range(lo, hi + 1)])
    d = MON.drift(good, MON.reference_for(training_engines(ds), shift), alpha)
    d[["ks_stat", "p_value", "p_holm"]] = d[["ks_stat", "p_value", "p_holm"]].astype(float)
    replace_rows(engine, schema, "fact_drift", run_id, d.assign(run_id=run_id, dataset=ds, shift=shift)[
        ["run_id", "dataset", "shift", "sensor", "n_recent", "n_reference", "ks_stat", "p_value", "p_holm", "tested", "drifted"]])
    get_run_logger().info(f"{ds}: drift tested on {int(d.tested.sum())} sensors; drifted: {d[d.drifted].sensor.tolist()}")
    return d


@task(**RETRY)
def drift_step_v2(engine, schema, ds: str, shift: int, run_id: str, rule: dict, variant: str = "v2"):
    """Phase 7b rule: wear-adjusted residuals of everything received in the last 3 shifts, KS vs the training
    engines' residuals, Holm; persistent (K of M runs) or sudden (one run below alpha_strong)."""
    m = v2_models(ds, variant)
    rows = received_rows(engine, schema, ds, [run_id_for(ds, k) for k in range(max(1, shift - M2.WINDOW_SHIFTS + 1), shift + 1)])
    if len(rows):
        if ds in CONDITION_DATASETS:
            rows = normalise(rows, load_json(stats_file(ds)))
        f = latest_forecasts_all(engine, schema, ds, run_id, rows.unit.unique())
        rows = attach_rhat(rows, f)
    d = M2.drift(m["level"].z(rows), m["ref"], m["n_ref"]) if len(rows) else \
        pd.DataFrame({"sensor": C.SENSORS, "n_recent": 0, "n_reference": m["n_ref"], "ks_stat": np.nan, "p_value": np.nan,
                      "tested": False, "p_holm": np.nan})
    with engine.begin() as conn:
        _sp(conn, schema)
        prev = pd.read_sql(text("SELECT run_id, sensor, p_holm, tested FROM fact_drift WHERE dataset = :d AND run_id = ANY(:r)"),
                           conn, params={"d": ds, "r": [run_id_for(ds, k) for k in range(max(1, shift - rule["m"] + 1), shift)]})
    history = []
    for k in range(max(1, shift - rule["m"] + 1), shift):
        p = prev[prev.run_id == run_id_for(ds, k)]
        history.append({r.sensor: r.p_holm for r in p.itertuples()} if len(p) and p.tested.all() else {})
    history.append(dict(zip(d.sensor, d.p_holm)) if d.tested.all() else {})
    persistent, sudden = M2.drift_decision(history, rule)
    d["drifted"] = d.sensor.isin(persistent + sudden)
    d[["ks_stat", "p_value", "p_holm"]] = d[["ks_stat", "p_value", "p_holm"]].astype(float)
    replace_rows(engine, schema, "fact_drift", run_id, d.assign(run_id=run_id, dataset=ds, shift=shift)[
        ["run_id", "dataset", "shift", "sensor", "n_recent", "n_reference", "ks_stat", "p_value", "p_holm", "tested", "drifted"]])
    get_run_logger().info(f"{ds}: drift tested on {int(d.tested.sum())} sensors; persistent {persistent}, sudden {sudden}")
    return d, persistent, sudden


def latest_forecasts_all(engine, schema, ds, run_id, units) -> pd.DataFrame:
    """All valid forecasts of these engines stored by this run or earlier (to give each reading its wear estimate)."""
    with engine.begin() as conn:
        _sp(conn, schema)
        return pd.read_sql(text("SELECT engine_id AS unit, cycle AS fcycle, rul_point FROM fact_prediction WHERE dataset = :d "
                                "AND run_id <= :r AND valid AND engine_id = ANY(:u) ORDER BY fcycle"),
                           conn, params={"d": ds, "r": run_id, "u": [int(u) for u in units]})


def attach_rhat(rows: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Each reading's wear estimate: the engine's forecast at that cycle, or its newest earlier forecast aged by
    the cycles since (quarantined readings have no forecast of their own). Readings with none are dropped."""
    r = rows.sort_values("cycle").reset_index(drop=True)
    out = pd.merge_asof(r, f.sort_values("fcycle"), left_on="cycle", right_on="fcycle", by="unit", direction="backward")
    out["rhat"] = np.maximum(0.0, out.rul_point - (out.cycle - out.fcycle))
    return out.dropna(subset=["rhat"]).sort_values(["unit", "cycle"]).reset_index(drop=True)


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


@task(**RETRY)
def retraining_flag_v2(engine, schema, ds: str, shift: int, run_id: str, d: pd.DataFrame, persistent, sudden,
                       h: dict, rule_ds: dict) -> bool:
    log = get_run_logger()
    flag, reasons = M2.retraining(rule_ds, persistent, sudden, h)
    row = pd.DataFrame([{"run_id": run_id, "dataset": ds, "shift": shift, "model_id": model_id(ds),
                         "n_labelled": h["n_labelled"], "n_scored": h["n_scored"], "rmse": h["rmse"],
                         "coverage_90": h["coverage_90"], "lower_bound_miss_rate": h["miss_rate"],
                         "error_ratio": h["error_ratio"], "miss_p": h["miss_p"],
                         "drifted_sensors": ",".join(d[d.drifted].sensor), "needs_retraining": flag,
                         "retraining_reasons": "; ".join(reasons)}])
    metrics = ["rmse", "coverage_90", "lower_bound_miss_rate", "error_ratio", "miss_p"]
    row[metrics] = row[metrics].astype(float)
    replace_rows(engine, schema, "fact_model_health", run_id, row)
    mark_run(engine, schema, run_id, needs_retraining=flag, retraining_reasons="; ".join(reasons), flow_status="complete")
    (log.warning if flag else log.info)(f"{ds} shift {shift}: " + (f"RETRAINING NEEDED: {'; '.join(reasons)}" if flag
                                                                    else "no retraining needed"))
    return flag


# ---------------------------------------------------------------- flow
def next_shift(engine, schema, datasets=C.FLOW_DATASETS) -> int:
    with engine.begin() as conn:
        _sp(conn, schema)
        rows = conn.execute(text("SELECT dataset, MAX(shift) FROM pipeline_runs WHERE flow_status = 'complete' "
                                 "GROUP BY dataset")).all()
    done = {d: s for d, s in rows}
    return min(done.get(ds, 0) for ds in datasets) + 1


@flow(name="shift-pipeline")
def shift_pipeline(shift: int | None = None, predictor: str = "model", inject: dict | None = None,
                   db_schema: str = C.FLOW_SCHEMA, retry_delay_seconds: float | None = None,
                   datasets: list[str] | None = None, fleet: str = "synchronized", monitor: str = "v1",
                   closed_loop: bool = False, recorded_file: str = str(RECORDED)) -> dict:
    from prefect.runtime import flow_run
    schema = db_schema
    datasets = list(datasets or C.FLOW_DATASETS)
    if monitor == "v1" and fleet != "synchronized":
        raise ValueError("the Phase 7 rule (monitor v1) is defined for a synchronized fleet only")
    log = get_run_logger()
    engine = get_engine()
    apply_schema(engine, schema)
    shift = shift or next_shift(engine, schema, datasets)
    if shift > last_shift(datasets, fleet):
        log.info(f"replay complete: every engine's failure is known by shift {last_shift(datasets, fleet)}")
        return {"shift": shift, "done": True}
    rule = MON.load_rule() if monitor == "v1" else M2.load_rule(monitor)
    opts = {} if retry_delay_seconds is None else {"retry_delay_seconds": retry_delay_seconds}
    t = {name: fn.with_options(**opts) for name, fn in (
        ("ingest", ingest), ("validate", validate_batch), ("load", load_step), ("features", features),
        ("predict", predict_and_store), ("alerts", alerts), ("schedule", schedule_step), ("health", model_health),
        ("health2", model_health_v2), ("drift", drift_step), ("drift2", drift_step_v2), ("flag", retraining_flag),
        ("flag2", retraining_flag_v2))}
    out = {"shift": shift, "done": False}
    for ds in datasets:
        rid = run_id_for(ds, shift)
        try:
            serviced = service_shifts(engine, schema, ds, rid) if closed_loop else None
            stopped = tuple(sorted(e for e, s in (serviced or {}).items() if s < shift))
            batch = t["ingest"](ds, shift, inject, fleet, stopped)
            good, bad = t["validate"](engine, schema, ds, batch)
            t["load"](engine, schema, ds, shift, good, bad, str(flow_run.id) if flow_run.id else None)
            window = json.loads(Path(SOURCES[ds]["results"]).read_text())["models"]["LSTM"]["settings"]["window"]
            X, units, invalid = t["features"](engine, schema, ds, shift, good, window)
            t["predict"](engine, schema, ds, shift, rid, X, units, invalid, predictor, fleet, recorded_file)
            cand, flagged = t["alerts"](engine, schema, ds, shift, fleet, serviced)
            sched = t["schedule"](engine, schema, ds, shift, rid, cand, fleet, serviced)
            if monitor == "v1":
                h = t["health"](engine, schema, ds, shift)
                d = t["drift"](engine, schema, ds, shift, rid, rule["drift"]["alpha"])
                flag = t["flag"](engine, schema, ds, shift, rid, d, h, rule)
                drifted = d[d.drifted].sensor.tolist()
            else:
                p0 = rule["datasets"][ds]["p0"]
                h = t["health2"](engine, schema, ds, shift, fleet, serviced, p0, monitor)
                d, persistent, sudden = t["drift2"](engine, schema, ds, shift, rid, rule["drift"], monitor)
                flag = t["flag2"](engine, schema, ds, shift, rid, d, persistent, sudden, h, {"health": {**rule["health"], "p0": p0}})
                drifted = d[d.drifted].sensor.tolist()
            out[ds] = {"run_id": rid, "readings": len(batch), "quarantined": len(bad), "flagged": len(flagged),
                       "schedule": sched, "labelled": h["n_labelled"], "drifted": drifted, "needs_retraining": flag}
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
    "phase7_health.csv": ("SELECT run_id, dataset, shift, model_id, n_labelled, n_scored, rmse, mae, near_failure_mae, coverage_90, "
                          "lower_bound_miss_rate, cum_rmse, cum_coverage_90, cum_lower_bound_miss_rate, drifted_sensors, "
                          "needs_retraining, retraining_reasons FROM fact_model_health ORDER BY dataset, shift"),
    "phase7_drift.csv": "SELECT * FROM fact_drift ORDER BY dataset, shift, sensor",
    "phase7_alerts.csv": ("SELECT run_id, dataset, engine_id, decision_cycle, forecast_cycle, forecast_age, rul_low_used, risk, "
                          "reason, scheduled_day, service_cycle, status, forecast_run_id FROM fact_alert WHERE context = 'flow' "
                          "ORDER BY dataset, decision_cycle, engine_id"),
    "phase7_runs.csv": ("SELECT run_id, dataset, shift, last_cycle, start_cycle, n_cycles, rows_in, rows_loaded, rows_quarantined, "
                        "status, flow_status, needs_retraining, retraining_reasons FROM pipeline_runs WHERE shift IS NOT NULL "
                        "ORDER BY dataset, shift"),
}
EXPORT_V2 = {**EXPORT,
             "phase7_health.csv": ("SELECT run_id, dataset, shift, model_id, n_labelled, n_scored, rmse, coverage_90, "
                                   "lower_bound_miss_rate, error_ratio, miss_p, drifted_sensors, needs_retraining, "
                                   "retraining_reasons FROM fact_model_health ORDER BY dataset, shift"),
             "phase7_alerts.csv": ("SELECT run_id, dataset, engine_id, decision_cycle, forecast_cycle, forecast_age, rul_low_used, "
                                   "risk, reason, scheduled_day, service_cycle, status, forecast_run_id FROM fact_alert WHERE "
                                   "context = 'flow' ORDER BY dataset, run_id, engine_id")}


def export(engine, schema=C.FLOW_SCHEMA, out_dir: Path = OUT, monitor: str = "v1") -> dict:
    """Write the replay's tables as committed CSVs (no timestamps, so they are deterministic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = {}
    with engine.begin() as conn:
        _sp(conn, schema)
        for f, sql in (EXPORT if monitor == "v1" else EXPORT_V2).items():
            df = pd.read_sql(text(sql), conn)
            df.to_csv(out_dir / f, index=False)
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
    p.add_argument("--interval", type=int, default=C.SERVE_INTERVAL_SECONDS,
                   help=f"with --serve: seconds between runs (default {C.SERVE_INTERVAL_SECONDS}, from pipeline/config.py)")
    p.add_argument("--datasets", nargs="+", default=list(C.FLOW_DATASETS))
    p.add_argument("--fleet", default="synchronized", choices=["synchronized", "staggered"])
    p.add_argument("--monitor", default="v1", choices=["v1", "v2", "v2.1"])
    p.add_argument("--closed-loop", action="store_true")
    p.add_argument("--out-dir", default=str(OUT), help="with --replay-all: where to export the results")
    p.add_argument("--recorded-file", default=str(RECORDED))
    a = p.parse_args(argv)
    scen = {"datasets": a.datasets, "fleet": a.fleet, "monitor": a.monitor, "closed_loop": a.closed_loop,
            "recorded_file": a.recorded_file}
    if a.reset:
        reset(get_engine(), a.schema)
        print(f"schema {a.schema} reset")
    elif a.once:
        print(shift_pipeline(predictor=a.predictor, db_schema=a.schema, **scen))
    elif a.serve:
        shift_pipeline.serve(name="every-shift", interval=a.interval,
                             parameters={"predictor": a.predictor, "db_schema": a.schema, **scen})
    else:
        stop = a.until or last_shift(a.datasets, a.fleet)
        while True:
            r = shift_pipeline(predictor=a.predictor, db_schema=a.schema, **scen)
            if r["done"] or r["shift"] >= stop:
                break
        if not a.until:
            print("exported", export(get_engine(), a.schema, Path(a.out_dir), a.monitor))
    return 0


if __name__ == "__main__":
    sys.exit(main())
