"""Shared constants for the data pipeline."""
from pathlib import Path

RAW = Path("data/raw")

# Kept after Phase 1 EDA (docs/EDA_FINDINGS.md): constant or two-valued columns are dropped.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
SETTINGS = ["setting1", "setting2"]  # kept for data checks only
KEEP_COLS = SETTINGS + SENSORS

LIMITS_FILE = Path("pipeline/sensor_limits.json")
LIMIT_MARGIN = 0.10  # widen the training min-max range by 10% on each side


def limits_file(dataset: str = "FD001") -> Path:
    """Sensor limits per dataset (FD001 keeps its original file name)."""
    return LIMITS_FILE if dataset == "FD001" else Path(f"pipeline/sensor_limits_{dataset}.json")


# ---- Phase 7: the shift flow (pipeline/flow.py)
FLOW_DATASETS = ("FD001", "FD003")
FLOW_SCHEMA = "ops"                # Postgres schema for the replay, kept apart from the Phase 2 demo loads
CYCLES_PER_SHIFT = 1               # one 8-hour shift = one engine cycle (the Phase 6 plant: 8 h per cycle)
SERVE_INTERVAL_SECONDS = 600       # demo mode: every 10 minutes stands for one 8-hour shift
TASK_RETRIES = 3                   # steps that touch the database or files
TASK_RETRY_DELAY_SECONDS = 10
MC_SEED_BASE = {"FD001": 1_000_000, "FD003": 2_000_000, "FD002": 3_000_000, "FD004": 4_000_000}   # MC-dropout seed = base + shift (deterministic reruns)
