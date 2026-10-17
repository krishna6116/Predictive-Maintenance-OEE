-- Star schema for the predictive maintenance pipeline (Phase 2).
-- Safe to run more than once: every statement uses IF NOT EXISTS.

-- One row per pipeline run: what was loaded, when, and how it went.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           TEXT PRIMARY KEY,
    started_at       TIMESTAMPTZ NOT NULL,
    finished_at      TIMESTAMPTZ,
    status           TEXT NOT NULL,           -- running | success | failed
    dataset          TEXT NOT NULL,           -- FD001 ... FD004
    split            TEXT NOT NULL,           -- train | test
    start_cycle      INT,
    n_cycles         INT,
    rows_in          INT,
    rows_loaded      INT,
    rows_quarantined INT,
    note             TEXT
);

-- Dimension: one row per engine. Train and test engines reuse IDs, so split is part of the key.
CREATE TABLE IF NOT EXISTS dim_engine (
    dataset      TEXT NOT NULL,
    split        TEXT NOT NULL,
    engine_id    INT  NOT NULL,
    first_run_id TEXT REFERENCES pipeline_runs(run_id),
    PRIMARY KEY (dataset, split, engine_id)
);

-- Fact: one cleaned reading per engine per cycle.
CREATE TABLE IF NOT EXISTS fact_sensor_reading (
    dataset   TEXT NOT NULL,
    split     TEXT NOT NULL,
    engine_id INT  NOT NULL,
    cycle     INT  NOT NULL,
    run_id    TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    setting1 DOUBLE PRECISION, setting2 DOUBLE PRECISION,
    s2  DOUBLE PRECISION, s3  DOUBLE PRECISION, s4  DOUBLE PRECISION, s7  DOUBLE PRECISION,
    s8  DOUBLE PRECISION, s9  DOUBLE PRECISION, s11 DOUBLE PRECISION, s12 DOUBLE PRECISION,
    s13 DOUBLE PRECISION, s14 DOUBLE PRECISION, s15 DOUBLE PRECISION, s17 DOUBLE PRECISION,
    s20 DOUBLE PRECISION, s21 DOUBLE PRECISION,
    PRIMARY KEY (dataset, split, engine_id, cycle),
    FOREIGN KEY (dataset, split, engine_id) REFERENCES dim_engine(dataset, split, engine_id)
);

-- Rows that failed validation, kept with the reason instead of being dropped silently.
CREATE TABLE IF NOT EXISTS quarantine (
    id        BIGSERIAL PRIMARY KEY,
    run_id    TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    dataset   TEXT NOT NULL,
    split     TEXT NOT NULL,
    engine_id INT,
    cycle     INT,
    reason    TEXT NOT NULL,
    raw       JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_quarantine_run ON quarantine(run_id);
