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

-- ---- Phase 6: decisions. One run_id per Phase 6 run (a hash of its inputs, so re-runs replace, never duplicate).

-- Fact: every LSTM forecast the decisions could use (each observed test cycle), with its rule-E 90-percent interval.
CREATE TABLE IF NOT EXISTS fact_prediction (
    run_id         TEXT NOT NULL,
    dataset        TEXT NOT NULL,
    engine_id      INT  NOT NULL,
    cycle          INT  NOT NULL,
    model_id       TEXT NOT NULL,
    rul_point      DOUBLE PRECISION,
    rul_mc_mean    DOUBLE PRECISION,
    rul_mc_std     DOUBLE PRECISION,
    k              DOUBLE PRECISION,
    rul_low        DOUBLE PRECISION,
    rul_high       DOUBLE PRECISION,
    valid          BOOLEAN NOT NULL,
    invalid_reason TEXT,
    source         TEXT NOT NULL,
    PRIMARY KEY (run_id, dataset, engine_id, cycle)
);

-- Fact: engines flagged for maintenance. context = 'snapshot' (official evaluation point) or 'replay' (policy
-- simulation). One row per engine per context: the key itself rules out duplicate work orders.
CREATE TABLE IF NOT EXISTS fact_alert (
    run_id         TEXT NOT NULL,
    dataset        TEXT NOT NULL,
    context        TEXT NOT NULL,
    engine_id      INT  NOT NULL,
    decision_cycle INT  NOT NULL,
    forecast_cycle INT  NOT NULL,
    forecast_age   INT  NOT NULL,
    rul_low_used   DOUBLE PRECISION NOT NULL,
    risk           TEXT NOT NULL,
    reason         TEXT NOT NULL,
    scheduled_day  INT,
    service_cycle  INT,
    status         TEXT NOT NULL,
    forecast_run_id TEXT NOT NULL,   -- the run that stored the forecast this alert acted on
    PRIMARY KEY (run_id, dataset, context, engine_id),
    CONSTRAINT fact_alert_forecast_fkey FOREIGN KEY (forecast_run_id, dataset, engine_id, forecast_cycle)
        REFERENCES fact_prediction(run_id, dataset, engine_id, cycle)
);

-- Fact: SIMULATED production per machine per maintenance policy (guide 6.4-6.5; not measured plant data).
CREATE TABLE IF NOT EXISTS fact_oee (
    run_id          TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    policy          TEXT NOT NULL,
    machine_id      INT  NOT NULL,
    life_cycles     INT  NOT NULL,
    run_cycles      INT  NOT NULL,
    event           TEXT NOT NULL,
    wasted_cycles   INT  NOT NULL,
    degraded_cycles INT  NOT NULL,
    planned_hours   DOUBLE PRECISION NOT NULL,
    run_hours       DOUBLE PRECISION NOT NULL,
    downtime_hours  DOUBLE PRECISION NOT NULL,
    ideal_units     DOUBLE PRECISION NOT NULL,
    actual_units    DOUBLE PRECISION NOT NULL,
    good_units      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, dataset, policy, machine_id)
);

-- ---- Phase 7: the shift flow (pipeline/flow.py). The replay runs in its own Postgres schema (ops).

-- One pipeline_runs row per shift per dataset; run_id = '<dataset>-test-s<shift>' so a re-run replaces it.
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS shift              INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS last_cycle         INT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS flow_status        TEXT;   -- running | complete | failed
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS flow_run_id        TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS needs_retraining   BOOLEAN;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS retraining_reasons TEXT;

-- Phase 7 migration for databases created before forecast_run_id existed (Phase 6 rows: forecast_run_id = run_id).
-- In the flow an alert can act on a forecast stored by an earlier shift, so the link uses forecast_run_id.
ALTER TABLE fact_alert ADD COLUMN IF NOT EXISTS forecast_run_id TEXT;
UPDATE fact_alert SET forecast_run_id = run_id WHERE forecast_run_id IS NULL;
ALTER TABLE fact_alert ALTER COLUMN forecast_run_id SET NOT NULL;
ALTER TABLE fact_alert DROP CONSTRAINT IF EXISTS fact_alert_run_id_dataset_engine_id_forecast_cycle_fkey;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fact_alert_forecast_fkey'
                   AND conrelid = 'fact_alert'::regclass) THEN
        ALTER TABLE fact_alert ADD CONSTRAINT fact_alert_forecast_fkey FOREIGN KEY (forecast_run_id, dataset, engine_id, forecast_cycle)
            REFERENCES fact_prediction(run_id, dataset, engine_id, cycle);
    END IF;
END $$;

-- A flow work order is created once per engine: the database itself rules out duplicates across shifts.
CREATE UNIQUE INDEX IF NOT EXISTS ux_fact_alert_flow_engine ON fact_alert (dataset, engine_id) WHERE context = 'flow';

-- Fact: model health per run, scored only on engines whose failure has already happened (label available).
CREATE TABLE IF NOT EXISTS fact_model_health (
    run_id                    TEXT PRIMARY KEY,
    dataset                   TEXT NOT NULL,
    shift                     INT  NOT NULL,
    model_id                  TEXT NOT NULL,
    n_labelled                INT  NOT NULL,
    n_scored                  INT  NOT NULL,     -- engines in the rolling window (the 20 most recently labelled)
    rmse                      DOUBLE PRECISION,
    mae                       DOUBLE PRECISION,
    near_failure_mae          DOUBLE PRECISION,
    coverage_90               DOUBLE PRECISION,
    lower_bound_miss_rate     DOUBLE PRECISION,
    cum_rmse                  DOUBLE PRECISION,  -- over every labelled engine so far
    cum_coverage_90           DOUBLE PRECISION,
    cum_lower_bound_miss_rate DOUBLE PRECISION,
    drifted_sensors           TEXT NOT NULL,
    needs_retraining          BOOLEAN NOT NULL,
    retraining_reasons        TEXT NOT NULL
);

-- Fact: KS drift test per run per sensor (per-engine means vs age-matched training engines, Holm-adjusted).
CREATE TABLE IF NOT EXISTS fact_drift (
    run_id      TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    shift       INT  NOT NULL,
    sensor      TEXT NOT NULL,
    n_recent    INT  NOT NULL,
    n_reference INT  NOT NULL,
    ks_stat     DOUBLE PRECISION,
    p_value     DOUBLE PRECISION,
    p_holm      DOUBLE PRECISION,
    tested      BOOLEAN NOT NULL,
    drifted     BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, sensor)
);

-- ---- Phase 7b: the life-matched health rule (pipeline/monitoring2.py)
ALTER TABLE fact_model_health ADD COLUMN IF NOT EXISTS error_ratio DOUBLE PRECISION;   -- squared error / expected for these lives
ALTER TABLE fact_model_health ADD COLUMN IF NOT EXISTS miss_p      DOUBLE PRECISION;   -- binomial p of the lower-bound misses
ALTER TABLE fact_model_health ALTER COLUMN drifted_sensors DROP NOT NULL;
