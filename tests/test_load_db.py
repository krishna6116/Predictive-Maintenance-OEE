"""Database tests. They run in a throwaway Postgres schema and are skipped when no database is reachable."""
import uuid

import pandas as pd
import pytest
from sqlalchemy import text

from pipeline.config import KEEP_COLS
from pipeline.db import apply_schema, get_engine
from pipeline.load import last_received_cycles, load_batch, next_start_cycle


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


def rows(units, cycles, **override):
    base = {c: 1.0 for c in KEEP_COLS}
    # Explicit columns, so an empty batch still has them, as validate() always returns
    return pd.DataFrame([{"unit": u, "cycle": c, **base, **override} for u in units for c in cycles],
                        columns=["unit", "cycle"] + KEEP_COLS + list(override))


def count(engine, schema, sql):
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        return conn.execute(text(sql)).scalar()


def test_rerun_is_idempotent(db):
    engine, schema = db
    good = rows([1, 2], [1, 2, 3])
    bad = rows([3], [1]).assign(reason="s4 out of range")
    for _ in range(2):
        load_batch(engine, good, bad, dataset="T", split="test", start_cycle=1, n_cycles=3, schema=schema)

    assert count(engine, schema, "SELECT COUNT(*) FROM fact_sensor_reading WHERE dataset = 'T'") == 6
    assert count(engine, schema, "SELECT COUNT(*) FROM quarantine WHERE dataset = 'T'") == 1
    assert count(engine, schema, "SELECT COUNT(*) FROM dim_engine WHERE dataset = 'T'") == 2
    assert count(engine, schema, "SELECT COUNT(*) FROM pipeline_runs WHERE dataset = 'T' AND status = 'success'") == 2


def test_later_delivery_replaces_earlier_one(db):
    engine, schema = db
    load_batch(engine, rows([1], [1]), rows([], []).assign(reason=[]), dataset="R", split="test",
               start_cycle=1, n_cycles=1, schema=schema)
    load_batch(engine, rows([], []), rows([1], [1]).assign(reason="missing value"), dataset="R", split="test",
               start_cycle=1, n_cycles=1, schema=schema)
    assert count(engine, schema, "SELECT COUNT(*) FROM fact_sensor_reading WHERE dataset = 'R'") == 0
    assert count(engine, schema, "SELECT COUNT(*) FROM quarantine WHERE dataset = 'R'") == 1


def test_progress_tracking(db):
    engine, schema = db
    load_batch(engine, rows([1], [1, 2, 3, 4, 5]), rows([], []).assign(reason=[]), dataset="P", split="test",
               start_cycle=1, n_cycles=5, schema=schema)
    load_batch(engine, rows([], []), rows([1], [6]).assign(reason="s2 out of range"), dataset="P", split="test",
               start_cycle=6, n_cycles=1, schema=schema)
    assert next_start_cycle(engine, "P", "test", schema=schema) == 7
    assert last_received_cycles(engine, "P", "test", schema=schema) == {1: 6}


def test_failed_batch_rolls_back_and_is_recorded(db):
    engine, schema = db
    broken = rows([1], [1]).drop(columns=["s2"])  # a missing column makes the insert fail
    with pytest.raises(Exception):
        load_batch(engine, broken, rows([], []).assign(reason=[]), dataset="F", split="test",
                   start_cycle=1, n_cycles=1, schema=schema)
    assert count(engine, schema, "SELECT COUNT(*) FROM fact_sensor_reading WHERE dataset = 'F'") == 0
    assert count(engine, schema, "SELECT status FROM pipeline_runs WHERE dataset = 'F'") == "failed"
