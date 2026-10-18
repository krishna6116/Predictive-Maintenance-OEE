"""Modeling datasets for Phase 4+.

Features are built with the exact Phase 3 code (features.build.build_features) and the saved scaler.
They are computed from the raw NASA files, because the official evaluation needs every test engine's full
observed history, while the pipeline streams test data in shifts. Phase 3's verification showed that
features from the files and from the database are identical (max difference 1.9e-13).

Results are cached in data/processed/ (git-ignored) and rebuilt automatically when the inputs change.
"""
import hashlib
from pathlib import Path

import pandas as pd

from features.build import RUL_CAP, add_rul, build_features, load_json, scaler_file, split_file
from pipeline.check_raw_data import load
from pipeline.config import RAW

CACHE = Path("data/processed")


def _fingerprint(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    # Any change to the feature code, the sensor list or the file loader invalidates the cache
    for code in ("features/build.py", "pipeline/config.py", "pipeline/check_raw_data.py", "models/data.py"):
        h.update(Path(code).read_bytes())
    return h.hexdigest()[:16]


def _cached(name: str, inputs: list[Path], make) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{_fingerprint(*inputs)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    df = make()
    df.to_pickle(path)
    return df


def train_frame(dataset: str = "FD001") -> pd.DataFrame:
    """Every training reading: unit, cycle, 168 features, RUL (capped), RUL_true (uncapped), split."""
    src, scaler, splits = RAW / f"train_{dataset}.txt", scaler_file(dataset), split_file(dataset)

    def make():
        raw = add_rul(load(src))
        raw["RUL_true"] = raw.groupby("unit")["cycle"].transform("max") - raw["cycle"]
        feats = build_features(raw, load_json(scaler)).merge(raw[["unit", "cycle", "RUL", "RUL_true"]],
                                                                   on=["unit", "cycle"], validate="one_to_one")
        split = load_json(splits)
        feats["split"] = feats.unit.map({**{u: "train" for u in split["train"]}, **{u: "val" for u in split["val"]}})
        return feats

    return _cached(f"train_{dataset}", [src, scaler, splits], make)


def test_frame(dataset: str = "FD001") -> pd.DataFrame:
    """One row per test engine at its LAST observed cycle, with the true RUL from NASA's RUL file.

    RUL_true is NASA's value; RUL is the same value capped at 125 (the convention used for scoring).
    """
    src, rul_file, scaler = RAW / f"test_{dataset}.txt", RAW / f"RUL_{dataset}.txt", scaler_file(dataset)

    def make():
        raw = load(src)
        feats = build_features(raw, load_json(scaler))
        last = feats.loc[feats.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
        truth = pd.read_csv(rul_file, header=None, names=["RUL_true"])
        truth["unit"] = range(1, len(truth) + 1)  # line i of the RUL file belongs to engine i
        last = last.merge(truth, on="unit", validate="one_to_one")
        last["RUL"] = last["RUL_true"].clip(upper=RUL_CAP)
        return last

    return _cached(f"test_{dataset}", [src, rul_file, scaler], make)
