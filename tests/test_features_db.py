"""Feature store tests against PostgreSQL, in a throwaway schema. Skipped when no database is reachable."""
import uuid

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect, text

from features.build import feature_columns, fit_scaler
from features.store import build_all, build_for_run, read_readings
from pipeline.config import KEEP_COLS
from pipeline.db import apply_schema, get_engine
from pipeline.load import load_batch


@pytest.fixture(scope="module")
def db():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no database available: {type(exc).__name__}")
    schema = f"pdm_test_{uuid.uuid4().hex[:8]}"
    apply_schema(engine, schema)
    yield engine, schema
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


def batch(units, cycles, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"unit": u, "cycle": c, **{k: 50 + 0.2 * c + rng.normal() for k in KEEP_COLS}}
                         for u in units for c in cycles], columns=["unit", "cycle"] + KEEP_COLS)


def no_bad():
    return batch([], [], 0).assign(reason=[])


def features_in(engine, schema):
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        return pd.read_sql(text("SELECT * FROM fact_features ORDER BY engine_id, cycle"), conn)


def test_full_rebuild_is_idempotent_and_keyed(db):
    engine, schema = db
    load_batch(engine, batch([1, 2], range(1, 31), 1), no_bad(), dataset="FD001", split="test",
               start_cycle=1, n_cycles=30, schema=schema)
    scaler = fit_scaler(read_readings(engine, "FD001", "test", schema=schema), [1, 2])
    for _ in range(2):
        assert build_all(engine, "FD001", "test", scaler, schema=schema) == 60
    stored = features_in(engine, schema)
    assert len(stored) == 60 and not stored.duplicated(["engine_id", "cycle"]).any()
    with engine.connect() as conn:
        pk = inspect(conn).get_pk_constraint("fact_features", schema=schema)["constrained_columns"]
    assert pk == ["dataset", "split", "engine_id", "cycle"]


def test_incremental_build_matches_full_rebuild_in_db(db):
    engine, schema = db
    run = load_batch(engine, batch([1, 2], range(31, 36), 2), no_bad(), dataset="FD001", split="test",
                     start_cycle=31, n_cycles=5, schema=schema)
    scaler = fit_scaler(read_readings(engine, "FD001", "test", schema=schema), [1, 2])
    assert build_for_run(engine, run, scaler, schema=schema) == 10
    incremental = features_in(engine, schema).query("cycle > 30").reset_index(drop=True)
    build_all(engine, "FD001", "test", scaler, schema=schema)
    full = features_in(engine, schema).query("cycle > 30").reset_index(drop=True)
    assert len(full) == 10
    pd.testing.assert_frame_equal(incremental[feature_columns()], full[feature_columns()])


def test_unknown_run_is_rejected(db):
    engine, schema = db
    with pytest.raises(ValueError):
        build_for_run(engine, "no-such-run", {}, schema=schema)
