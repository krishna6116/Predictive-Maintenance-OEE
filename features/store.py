"""Read readings from PostgreSQL, build features, and write them to fact_features (Phase 3).

fact_features holds one row per engine per cycle: the keys, the run that delivered the reading,
and the feature columns from features.build.feature_columns(). It is created on first write.
Writes are idempotent: rows for the same (dataset, split, engine, cycle) are replaced, never duplicated.
"""
import pandas as pd
from sqlalchemy import Engine, inspect, text

from features.build import build_features, feature_columns
from pipeline.config import SENSORS

KEYS = ["dataset", "split", "engine_id", "cycle"]


def _search_path(conn, schema: str | None) -> None:
    if schema:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))


def read_readings(engine: Engine, dataset: str, split: str, units=None, schema: str | None = None) -> pd.DataFrame:
    """Clean readings (unit, cycle, run_id, sensors) for one dataset and split, optionally some engines only."""
    sql = f"SELECT engine_id AS unit, cycle, run_id, {', '.join(SENSORS)} FROM fact_sensor_reading " \
          "WHERE dataset = :d AND split = :s"
    params = {"d": dataset, "s": split}
    if units is not None:
        sql += " AND engine_id = ANY(:u)"
        params["u"] = [int(u) for u in units]
    with engine.begin() as conn:
        _search_path(conn, schema)
        return pd.read_sql(text(sql + " ORDER BY engine_id, cycle"), conn, params=params)


def write_features(engine: Engine, feats: pd.DataFrame, dataset: str, split: str, schema: str | None = None) -> int:
    """Replace-then-insert in one transaction, so reruns never duplicate rows."""
    rows = feats.rename(columns={"unit": "engine_id"}).assign(dataset=dataset, split=split)
    rows = rows[KEYS + ["run_id"] + feature_columns()]
    with engine.begin() as conn:
        _search_path(conn, schema)
        if inspect(conn).has_table("fact_features"):
            conn.execute(text("DELETE FROM fact_features WHERE dataset = :d AND split = :s "
                              "AND (engine_id, cycle) IN (SELECT * FROM unnest(CAST(:e AS int[]), CAST(:c AS int[])))"),
                         {"d": dataset, "s": split, "e": rows.engine_id.astype(int).tolist(),
                          "c": rows.cycle.astype(int).tolist()})
        # PostgreSQL allows at most 65,535 values per statement, so size chunks from the column count
        rows.to_sql("fact_features", conn, if_exists="append", index=False, method="multi",
                    chunksize=60_000 // rows.shape[1])
    return len(rows)


def build_all(engine: Engine, dataset: str, split: str, scaler: dict, schema: str | None = None) -> int:
    """Rebuild features for every reading of a dataset/split."""
    readings = read_readings(engine, dataset, split, schema=schema)
    feats = build_features(readings, scaler).merge(readings[["unit", "cycle", "run_id"]], on=["unit", "cycle"])
    return write_features(engine, feats, dataset, split, schema=schema)


def build_for_run(engine: Engine, run_id: str, scaler: dict, schema: str | None = None) -> int:
    """Incremental build: features only for the readings delivered by one pipeline run.

    The engines in the run are read with their full earlier history, so rolling windows and the EMA
    see exactly what a full rebuild would see; only the run's own rows are written.
    """
    with engine.begin() as conn:
        _search_path(conn, schema)
        info = conn.execute(text("SELECT dataset, split FROM pipeline_runs WHERE run_id = :r"), {"r": run_id}).first()
        if info is None:
            raise ValueError(f"unknown run_id {run_id}")
        dataset, split = info
        units = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT engine_id FROM fact_sensor_reading WHERE run_id = :r"), {"r": run_id})]
    if not units:
        return 0
    history = read_readings(engine, dataset, split, units=units, schema=schema)
    feats = build_features(history, scaler).merge(history[["unit", "cycle", "run_id"]], on=["unit", "cycle"])
    return write_features(engine, feats[feats.run_id == run_id], dataset, split, schema=schema)
