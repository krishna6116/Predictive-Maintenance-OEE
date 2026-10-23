"""Load a validated batch into PostgreSQL in one transaction, tagged with a run ID.

Idempotent: loading the same batch again leaves the same rows in fact_sensor_reading and quarantine
(the latest delivery of an engine-cycle replaces the earlier one), so a rerun after a failure never
duplicates data. Each run still gets its own row in pipeline_runs, which keeps the history.
"""
import datetime as dt
import uuid

import pandas as pd
from sqlalchemy import Connection, Engine, text

from pipeline.config import KEEP_COLS

KEYS = "SELECT * FROM unnest(CAST(:e AS int[]), CAST(:c AS int[]))"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _use_schema(conn: Connection, schema: str | None) -> None:
    if schema:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))


def last_received_cycles(engine: Engine, dataset: str, split: str, schema: str | None = None) -> dict[int, int]:
    """Last cycle received per engine, whether it was loaded or quarantined."""
    with engine.begin() as conn:
        _use_schema(conn, schema)
        rows = conn.execute(text("""
            SELECT engine_id, MAX(cycle) FROM (
                SELECT engine_id, cycle FROM fact_sensor_reading WHERE dataset = :d AND split = :s
                UNION ALL
                SELECT engine_id, cycle FROM quarantine WHERE dataset = :d AND split = :s AND cycle IS NOT NULL
            ) received GROUP BY engine_id"""), {"d": dataset, "s": split}).all()
    return {int(e): int(c) for e, c in rows}


def next_start_cycle(engine: Engine, dataset: str, split: str, schema: str | None = None) -> int:
    """The cycle after the last successful run's window, or 1 if nothing has been loaded."""
    with engine.begin() as conn:
        _use_schema(conn, schema)
        end = conn.execute(text("""
            SELECT MAX(start_cycle + n_cycles) FROM pipeline_runs
            WHERE dataset = :d AND split = :s AND status = 'success'"""), {"d": dataset, "s": split}).scalar()
    return int(end) if end else 1


def load_batch(engine: Engine, good: pd.DataFrame, bad: pd.DataFrame, *, dataset: str, split: str,
               start_cycle: int, n_cycles: int, note: str | None = None, schema: str | None = None,
               run_id: str | None = None) -> str:
    """Write one batch and return its run ID. Everything happens in one transaction: all or nothing.

    Without `run_id` each call gets a new random ID (Phases 2-3). The Phase 7 flow passes a fixed ID per shift,
    so re-running a shift updates its own pipeline_runs row instead of adding one."""
    run_id = run_id or uuid.uuid4().hex[:12]
    tag = {"d": dataset, "s": split}
    try:
        with engine.begin() as conn:
            _use_schema(conn, schema)
            conn.execute(text("""
                INSERT INTO pipeline_runs (run_id, started_at, status, dataset, split, start_cycle, n_cycles, rows_in, note)
                VALUES (:r, :t, 'running', :d, :s, :sc, :n, :rows, :note)
                ON CONFLICT (run_id) DO UPDATE SET started_at = EXCLUDED.started_at, status = 'running',
                    finished_at = NULL, rows_in = EXCLUDED.rows_in, note = EXCLUDED.note"""),
                {"r": run_id, "t": _now(), **tag, "sc": start_cycle, "n": n_cycles,
                 "rows": len(good) + len(bad), "note": note})

            # The latest delivery of each engine-cycle wins: clear earlier quarantine entries for every
            # key in this batch, and earlier good readings for keys that are now quarantined.
            all_keys = pd.concat([good[["unit", "cycle"]], bad[["unit", "cycle"]]]).dropna().drop_duplicates()
            bad_keys = bad[["unit", "cycle"]].dropna().drop_duplicates()
            for table, keys in (("quarantine", all_keys), ("fact_sensor_reading", bad_keys)):
                if len(keys):
                    conn.execute(text(f"DELETE FROM {table} WHERE dataset = :d AND split = :s "
                                      f"AND (engine_id, cycle) IN ({KEYS})"),
                                 {**tag, "e": keys.unit.astype(int).tolist(), "c": keys.cycle.astype(int).tolist()})

            if len(good):
                engines = [{**tag, "e": int(u), "r": run_id} for u in good.unit.unique()]
                conn.execute(text("""
                    INSERT INTO dim_engine (dataset, split, engine_id, first_run_id) VALUES (:d, :s, :e, :r)
                    ON CONFLICT DO NOTHING"""), engines)

                cols = ", ".join(KEEP_COLS)
                params = ", ".join(f":{c}" for c in KEEP_COLS)
                updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in KEEP_COLS + ["run_id"])
                rows = (good.rename(columns={"unit": "engine_id"})[["engine_id", "cycle"] + KEEP_COLS]
                        .assign(dataset=dataset, split=split, run_id=run_id))
                conn.execute(text(f"""
                    INSERT INTO fact_sensor_reading (dataset, split, engine_id, cycle, run_id, {cols})
                    VALUES (:dataset, :split, :engine_id, :cycle, :run_id, {params})
                    ON CONFLICT (dataset, split, engine_id, cycle) DO UPDATE SET {updates}"""),
                    rows.to_dict("records"))

            if len(bad):
                q = [{"r": run_id, **tag,
                      "e": None if pd.isna(row.unit) else int(row.unit),
                      "c": None if pd.isna(row.cycle) else int(row.cycle),
                      "why": row.reason,
                      "raw": row.drop("reason").to_json()} for _, row in bad.iterrows()]
                conn.execute(text("""
                    INSERT INTO quarantine (run_id, dataset, split, engine_id, cycle, reason, raw)
                    VALUES (:r, :d, :s, :e, :c, :why, CAST(:raw AS JSONB))"""), q)

            conn.execute(text("""
                UPDATE pipeline_runs SET status = 'success', finished_at = :t, rows_loaded = :g, rows_quarantined = :b
                WHERE run_id = :r"""), {"t": _now(), "g": len(good), "b": len(bad), "r": run_id})
    except Exception as exc:
        # The batch was rolled back; record the failure in its own transaction so it stays visible.
        with engine.begin() as conn:
            _use_schema(conn, schema)
            conn.execute(text("""
                INSERT INTO pipeline_runs (run_id, started_at, finished_at, status, dataset, split, start_cycle, n_cycles, note)
                VALUES (:r, :t, :t, 'failed', :d, :s, :sc, :n, :note)
                ON CONFLICT (run_id) DO UPDATE SET status = 'failed', finished_at = EXCLUDED.finished_at,
                    note = EXCLUDED.note"""),
                {"r": run_id, "t": _now(), **tag, "sc": start_cycle, "n": n_cycles, "note": f"{type(exc).__name__}: {exc}"[:500]})
        raise
    return run_id
