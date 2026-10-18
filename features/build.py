"""Phase 3 feature engineering: pure functions on a readings table (no database access here).

Input: one row per engine per cycle with columns `unit`, `cycle` and the sensors in pipeline.config.SENSORS.
Every feature is causal: a row's features use only that engine's readings up to and including that cycle,
so the same code works on a full history, on a stream of new batches, and on the test set.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pipeline.config import SENSORS

RUL_CAP = 125                 # sensors are flat until ~100-150 cycles before failure (docs/EDA_FINDINGS.md)
WINDOWS = (5, 10, 20)         # rolling windows, in cycles
EMA_SPAN = 10                 # exponential moving average span
FFT_WINDOW = 30               # cycles used for the low-frequency energy feature
FFT_MIN = 8                   # fewer readings than this -> feature is 0

SCALER_FILE = Path("features/scaler.json")
SPLIT_FILE = Path("features/split.json")


def scaler_file(dataset: str = "FD001") -> Path:
    """FD001 keeps its Phase 3 file; other datasets (Phase 5b) get their own, fit on their own training engines."""
    return SCALER_FILE if dataset == "FD001" else Path(f"features/scaler_{dataset}.json")


def split_file(dataset: str = "FD001") -> Path:
    return SPLIT_FILE if dataset == "FD001" else Path(f"features/split_{dataset}.json")


# --- 3.1 target --------------------------------------------------------------------------------------
def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Remaining useful life for run-to-failure engines: last cycle minus current cycle, capped."""
    last = df.groupby("unit")["cycle"].transform("max")
    return df.assign(RUL=(last - df["cycle"]).clip(upper=cap))


# --- 3.2 split by engine -----------------------------------------------------------------------------
def split_engines(units, val_share: float = 0.2, seed: int = 42) -> dict[str, list[int]]:
    """Assign whole engines to train or validation, never individual rows."""
    units = np.asarray(sorted(set(int(u) for u in units)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    tr, va = next(splitter.split(units, groups=units))
    return {"train": units[tr].tolist(), "val": units[va].tolist()}


# --- 3.3 scaling (fit on training engines only) ------------------------------------------------------
def fit_scaler(df: pd.DataFrame, train_units, cols=SENSORS) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each sensor, from training engines only."""
    rows = df[df["unit"].isin(train_units)]
    return {c: {"mean": float(rows[c].mean()), "std": float(rows[c].std())} for c in cols}


def apply_scaler(df: pd.DataFrame, scaler: dict, cols=SENSORS) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (out[c] - scaler[c]["mean"]) / scaler[c]["std"]
    return out


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- 3.4 rolling features ----------------------------------------------------------------------------
def _rolling_slope(cycle: pd.Series, y: pd.Series, groups: pd.Series, w: int) -> pd.Series:
    """Least-squares slope of y against cycle over the last w readings, from rolling sums (fast, exact)."""
    frame = pd.DataFrame({"g": groups, "x": cycle, "y": y, "xy": cycle * y, "xx": cycle * cycle})
    r = frame.groupby("g")[["x", "y", "xy", "xx"]].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
    n = frame.groupby("g")["y"].rolling(w, min_periods=1).count().reset_index(level=0, drop=True)
    denom = n * r["xx"] - r["x"] ** 2
    slope = (n * r["xy"] - r["x"] * r["y"]) / denom.where(denom != 0)
    return slope.fillna(0.0)


def add_rolling(df: pd.DataFrame, cols=SENSORS, windows=WINDOWS) -> pd.DataFrame:
    """Mean (level), standard deviation (noise) and slope (speed of change) over the last w cycles."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")
    new = {}
    for w in windows:
        means = g[cols].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        stds = g[cols].rolling(w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
        for c in cols:
            new[f"{c}_mean{w}"] = means[c]
            new[f"{c}_std{w}"] = stds[c]
            new[f"{c}_slope{w}"] = _rolling_slope(df["cycle"], df[c], df["unit"], w)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# --- 3.5 signal-processing features ------------------------------------------------------------------
def _low_freq_energy(values: np.ndarray, window: int = FFT_WINDOW, min_len: int = FFT_MIN) -> np.ndarray:
    """For each position, strength of the slowest FFT components (bins 1-3) over the last `window` values.

    Bin k is a cycle with period len/k readings, so until an engine has `window` readings the bins cover
    slightly faster cycles. Readings before `min_len` get 0. Dividing by the length keeps values comparable.
    """
    out = np.zeros(len(values))
    for t in range(len(values)):
        seg = values[max(0, t - window + 1): t + 1]
        if len(seg) >= min_len:
            spectrum = np.abs(np.fft.rfft(seg - seg.mean()))
            out[t] = spectrum[1:4].sum() / len(seg)  # divide by length so short and full windows compare
    return out


def add_signal(df: pd.DataFrame, cols=SENSORS) -> pd.DataFrame:
    """Exponential moving average (smoothed level) and low-frequency FFT energy (slow drift)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    new = {}
    for c in cols:
        new[f"{c}_ema"] = df.groupby("unit")[c].transform(lambda s: s.ewm(span=EMA_SPAN, adjust=False).mean())
        lf = np.empty(len(df))
        for _, idx in df.groupby("unit").indices.items():
            lf[idx] = _low_freq_energy(df[c].to_numpy()[idx])
        new[f"{c}_lowfreq"] = lf
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# --- everything together -----------------------------------------------------------------------------
def feature_columns(cols=SENSORS, windows=WINDOWS) -> list[str]:
    names = list(cols)
    for w in windows:
        for c in cols:
            names += [f"{c}_mean{w}", f"{c}_std{w}", f"{c}_slope{w}"]
    for c in cols:
        names += [f"{c}_ema", f"{c}_lowfreq"]
    return names


def build_features(readings: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Scale the sensors, then add rolling and signal features. Returns unit, cycle and feature columns."""
    df = readings[["unit", "cycle"] + SENSORS].sort_values(["unit", "cycle"]).reset_index(drop=True)
    df = apply_scaler(df, scaler)
    df = add_signal(add_rolling(df))
    return df[["unit", "cycle"] + feature_columns()]
