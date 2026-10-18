"""Phase 5: sequence windows for the deep-learning models.

A window is the last W cycles of the 14 scaled sensors, ending at (and including) the current cycle, so it
uses only readings that exist at prediction time. Training gets one window per cycle of every engine; the
test set gets one window per engine, ending at its last observed cycle.

Left padding: when an engine has fewer than W readings so far, the window is filled on the left with copies
of that engine's FIRST reading. This uses only past data (the first reading is always in the past) and stays
inside the sensors' real range, unlike zero-padding, which after scaling would mean "an average engine".
"""
import numpy as np
import pandas as pd

from features.build import RUL_CAP, add_rul, apply_scaler, load_json, scaler_file, split_file
from pipeline.check_raw_data import load
from pipeline.config import RAW, SENSORS


def scaled_readings(split: str, dataset: str = "FD001") -> pd.DataFrame:
    """Raw NASA readings for 'train' or 'test', sensors scaled with the Phase 3 scaler, sorted by engine/cycle."""
    df = load(RAW / f"{split}_{dataset}.txt")
    if split == "train":
        df = add_rul(df)
        df["RUL_true"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df = apply_scaler(df, load_json(scaler_file(dataset)))
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def make_windows(df: pd.DataFrame, length: int, cols=SENSORS, last_only: bool = False):
    """Build causal windows. Returns (X, index): X has shape (n, length, len(cols)), float32.

    `index` has one row per window: unit and cycle of the window's LAST reading (plus RUL columns when present).
    With last_only=True, each engine contributes only its final window (the test-set convention).
    """
    xs, keys = [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        v = g[cols].to_numpy(np.float32)
        if not np.array_equal(g["cycle"].to_numpy(), np.arange(1, len(g) + 1)):
            raise ValueError(f"engine {unit}: cycles must run 1, 2, 3 ... without gaps")
        padded = np.concatenate([np.repeat(v[:1], length - 1, axis=0), v])  # length-1 copies of the first reading
        ends = [len(g) - 1] if last_only else range(len(g))
        for t in ends:  # window ending at reading t covers padded[t : t + length]
            xs.append(padded[t: t + length])
        keys.append(g.iloc[list(ends)])
    index = pd.concat(keys, ignore_index=True)
    keep = [c for c in ("unit", "cycle", "RUL", "RUL_true") if c in index.columns]
    return np.stack(xs).astype(np.float32), index[keep].reset_index(drop=True)


def datasets(length: int, dataset: str = "FD001"):
    """Train / validation / all-100 windows plus the test windows (last cycle per engine, NASA truth attached)."""
    split = load_json(split_file(dataset))
    tr = scaled_readings("train", dataset)
    X_all, i_all = make_windows(tr, length)
    in_train = i_all.unit.isin(split["train"]).to_numpy()
    in_val = i_all.unit.isin(split["val"]).to_numpy()
    te = scaled_readings("test", dataset)
    X_te, i_te = make_windows(te, length, last_only=True)
    truth = pd.read_csv(RAW / f"RUL_{dataset}.txt", header=None, names=["RUL_true"])
    truth["unit"] = range(1, len(truth) + 1)  # line i of the RUL file belongs to engine i
    i_te = i_te.merge(truth, on="unit", validate="one_to_one")
    i_te["RUL"] = i_te.RUL_true.clip(upper=RUL_CAP)
    return {"train": (X_all[in_train], i_all[in_train].reset_index(drop=True)),
            "val": (X_all[in_val], i_all[in_val].reset_index(drop=True)),
            "all": (X_all, i_all),
            "test": (X_te, i_te)}
