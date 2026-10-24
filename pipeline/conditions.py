"""Operating conditions for FD002/FD004 (Phase 7b), as the study guide prescribes for these datasets.

FD002 and FD004 engines fly in six operating conditions, and every sensor's level depends on the condition, so a raw
reading says more about where the engine was flying than about its wear. Each sensor is therefore standardised
within its condition: condition = setting1 rounded to a whole number (on FD002/FD004 training data this gives exactly
six groups, with setting1 varying by at most 0.01 inside a group), and the mean and sd of each sensor per condition
come from the TRAINING-split engines only (features/condition_scaler_<dataset>.json).
"""
from pathlib import Path

import pandas as pd

from features.build import load_json, save_json, split_file, split_engines
from pipeline.check_raw_data import load
from pipeline.config import RAW, SENSORS

CONDITION_DATASETS = ("FD002", "FD004")


def condition(df: pd.DataFrame) -> pd.Series:
    return df.setting1.round().astype(int)


def stats_file(ds: str) -> Path:
    return Path(f"features/condition_scaler_{ds}.json")


def ensure_split_and_stats(ds: str) -> dict:
    """Engine split (the Phase 3 function and seed) and per-condition sensor statistics from training engines."""
    raw = load(RAW / f"train_{ds}.txt")
    if not split_file(ds).exists():
        save_json(split_engines(raw.unit.unique()), split_file(ds))
    split = load_json(split_file(ds))
    if not stats_file(ds).exists():
        tr = raw[raw.unit.isin(split["train"])]
        stats = {str(c): {s: [float(g[s].mean()), float(g[s].std())] for s in SENSORS} for c, g in tr.groupby(condition(tr))}
        if len(stats) != 6:
            raise ValueError(f"{ds}: expected 6 operating conditions, found {sorted(stats)}")
        save_json(stats, stats_file(ds))
    return split


def normalise(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Each sensor standardised within its operating condition (unknown condition -> error, never silently)."""
    out = df.copy()
    cond = condition(df).astype(str)
    unknown = set(cond.unique()) - set(stats)
    if unknown:
        raise ValueError(f"readings in unknown operating conditions {sorted(unknown)}")
    for s in SENSORS:
        mean = cond.map(lambda c: stats[c][s][0])
        sd = cond.map(lambda c: stats[c][s][1])
        out[s] = (df[s] - mean) / sd.where(sd > 0, 1.0)
    return out


def normalised_train(ds: str) -> pd.DataFrame:
    return normalise(load(RAW / f"train_{ds}.txt"), load_json(stats_file(ds)))
