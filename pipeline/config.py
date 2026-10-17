"""Shared constants for the data pipeline."""
from pathlib import Path

RAW = Path("data/raw")

# Kept after Phase 1 EDA (docs/EDA_FINDINGS.md): constant or two-valued columns are dropped.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
SETTINGS = ["setting1", "setting2"]  # kept for data checks only
KEEP_COLS = SETTINGS + SENSORS

LIMITS_FILE = Path("pipeline/sensor_limits.json")
LIMIT_MARGIN = 0.10  # widen the training min-max range by 10% on each side
