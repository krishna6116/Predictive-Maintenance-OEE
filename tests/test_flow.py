"""Phase 7 flow: idempotent shifts, retries, resume after failure, invalid forecasts, drift -> retraining flag.

Runs the real Prefect flow in Prefect's test harness (no server) against a throwaway Postgres schema, replaying
Phase 6's committed forecasts (predictor "recorded"), so neither the model nor a Prefect server is needed.
Skipped when no database or no NASA data is available.
"""
import uuid
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

HAVE_DATA = Path("data/raw/test_FD001.txt").exists() and Path("data/raw/test_FD003.txt").exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="NASA data not present")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from prefect.testing.utilities import prefect_test_harness
    from pipeline import flow as F
    from pipeline.db import get_engine
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no database available: {type(exc).__name__}")
    # recorded predictions: the Phase 6 rolling forecasts (same published model, committed)
    rec = pd.concat([pd.read_csv(f"maintenance/forecasts_rolling_{ds}.csv", float_precision="round_trip")
                     for ds in ("FD001", "FD003")])
    path = tmp_path_factory.mktemp("rec") / "recorded.csv"
    rec.to_csv(path, index=False)
    old = F.RECORDED
    F.RECORDED = path
    F.recorded.cache_clear()
    schema = f"flow_test_{uuid.uuid4().hex[:8]}"
    with prefect_test_harness():
        yield F, engine, schema
    F.RECORDED = old
    F.recorded.cache_clear()
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


def run(F, schema, shift=None, **kw):
    return F.shift_pipeline(shift=shift, predictor="recorded", db_schema=schema, retry_delay_seconds=0, **kw)


def q(engine, schema, sql):
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        return pd.read_sql(text(sql), conn)


def table_counts(engine, schema):
    return {t: int(q(engine, schema, f"SELECT COUNT(*) AS n FROM {t}").n[0]) for t in
            ("pipeline_runs", "fact_sensor_reading", "quarantine", "fact_prediction", "fact_model_health", "fact_drift")}


def test_shifts_run_in_order_and_a_rerun_changes_nothing(env):
    F, engine, schema = env
    for k in (1, 2, 3):
        assert run(F, schema)["shift"] == k                   # continues where the last run stopped
    before = table_counts(engine, schema)
    preds = q(engine, schema, "SELECT * FROM fact_prediction ORDER BY run_id, engine_id")
    run(F, schema, shift=2)                                   # re-run an earlier shift
    assert table_counts(engine, schema) == before
    assert q(engine, schema, "SELECT * FROM fact_prediction ORDER BY run_id, engine_id").equals(preds)
    runs = q(engine, schema, "SELECT run_id, shift, last_cycle, flow_status FROM pipeline_runs ORDER BY run_id")
    assert runs.run_id.tolist() == [f"{ds}-test-s{k:04d}" for ds in ("FD001", "FD003") for k in (1, 2, 3)]
    assert (runs.flow_status == "complete").all() and (runs["shift"] == runs.last_cycle).all()
    assert before["fact_sensor_reading"] == 600 and before["fact_drift"] == 6 * 14


def test_database_step_failing_once_is_retried(env, monkeypatch):
    F, engine, schema = env
    calls = {"n": 0}
    real = F.load_batch

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            from sqlalchemy.exc import OperationalError
            raise OperationalError("INSERT", {}, Exception("connection dropped (deliberate test failure)"))
        return real(*a, **k)

    monkeypatch.setattr(F, "load_batch", flaky)
    out = run(F, schema, shift=4)
    assert calls["n"] == 3                                     # FD001: fail + retry; FD003: first try
    assert out["FD001"]["run_id"] == "FD001-test-s0004"
    rows = q(engine, schema, "SELECT dataset, COUNT(*) AS n FROM fact_sensor_reading WHERE cycle = 4 GROUP BY dataset")
    assert dict(zip(rows.dataset, rows.n)) == {"FD001": 100, "FD003": 100}
    st = q(engine, schema, "SELECT flow_status FROM pipeline_runs WHERE shift = 4")
    assert (st.flow_status == "complete").all() and len(st) == 2


def test_a_failed_run_is_resumed_without_duplicates(env, monkeypatch):
    F, engine, schema = env

    def broken(*a, **k):
        raise RuntimeError("model host unreachable (deliberate test failure)")

    monkeypatch.setattr(F, "recorded", broken)                 # every attempt fails, retries included
    with pytest.raises(Exception):
        run(F, schema)                                         # shift 5
    st = q(engine, schema, "SELECT run_id, flow_status FROM pipeline_runs WHERE shift = 5")
    assert st.flow_status.tolist() == ["failed"]
    assert F.next_shift(engine, schema) == 5
    monkeypatch.undo()
    F.recorded.cache_clear()
    assert run(F, schema)["shift"] == 5                        # resumes the failed shift
    st = q(engine, schema, "SELECT run_id, flow_status FROM pipeline_runs WHERE shift = 5 ORDER BY run_id")
    assert st.flow_status.tolist() == ["complete", "complete"]
    assert q(engine, schema, "SELECT COUNT(*) AS n FROM fact_prediction WHERE cycle = 5").n[0] == 200


def test_quarantined_reading_makes_the_next_forecast_invalid(env):
    F, engine, schema = env
    run(F, schema, inject={"dataset": "FD001", "sensor": "s4", "pct": 0.05, "shifts": [6], "engines": [7]})
    qa = q(engine, schema, "SELECT engine_id, reason FROM quarantine WHERE dataset = 'FD001' AND cycle = 6")
    assert qa.engine_id.tolist() == [7] and "s4 out of range" in qa.reason[0]
    run(F, schema)                                             # shift 7: engine 7 now has a gap at cycle 6
    p = q(engine, schema, "SELECT engine_id, valid, invalid_reason, rul_low FROM fact_prediction "
                          "WHERE dataset = 'FD001' AND cycle = 7 AND engine_id = 7")
    assert not p.valid[0] and p.invalid_reason[0] == "gap in the engine's loaded history" and pd.isna(p.rul_low[0])
    assert q(engine, schema, "SELECT COUNT(*) AS n FROM fact_prediction WHERE dataset = 'FD001' AND cycle = 7 "
                             "AND valid").n[0] == 99


def test_guide_drift_check_shifted_batch_sets_the_flag(env):
    F, engine, schema = env
    clean = run(F, schema)                                     # shift 8, clean
    assert clean["FD003"]["drifted"] == [] and not clean["FD003"]["needs_retraining"]
    out = run(F, schema, inject={"dataset": "FD003", "sensor": "s11", "pct": 0.05, "shifts": [9]})
    assert out["FD003"]["quarantined"] == 100                  # the range validator stops every shifted row ...
    assert out["FD003"]["drifted"] == ["s11"]                  # ... and drift still sees what the sensor sent
    assert out["FD003"]["needs_retraining"] and not out["FD001"]["needs_retraining"]
    r = q(engine, schema, "SELECT needs_retraining, retraining_reasons FROM pipeline_runs WHERE run_id = 'FD003-test-s0009'")
    assert r.needs_retraining[0] and r.retraining_reasons[0] == "drift in s11"
    h = q(engine, schema, "SELECT drifted_sensors, needs_retraining FROM fact_model_health WHERE run_id = 'FD003-test-s0009'")
    assert h.drifted_sensors[0] == "s11" and h.needs_retraining[0]


def test_no_labels_before_any_failure_cycle(env):
    F, engine, schema = env
    h = q(engine, schema, "SELECT dataset, shift, n_labelled, rmse FROM fact_model_health")
    first = {ds: min(F.failure_log(ds).values()) for ds in ("FD001", "FD003")}
    assert all(s < first[d] for d, s in zip(h.dataset, h["shift"]))
    assert (h.n_labelled == 0).all() and h.rmse.isna().all()
