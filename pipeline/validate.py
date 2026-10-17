"""Batch validation: split each batch into good rows and quarantined rows with a reason.

Rules, checked in this order (a row keeps the first reason it fails):
  1. missing value      - any kept column is empty or not a number
  2. duplicate cycle    - the same engine and cycle appears twice in the batch (the first is kept)
  3. <col> out of range - a reading falls outside the training range widened by LIMIT_MARGIN
  4. cycle gap          - an engine skips cycles: its cycle is more than one above the previous one
                          received (the previous row in the batch, or the last cycle received in an
                          earlier run, whether it was loaded or quarantined)
"""
import json

import numpy as np
import pandas as pd

from pipeline.config import KEEP_COLS, LIMIT_MARGIN, LIMITS_FILE


def compute_limits(train: pd.DataFrame, cols=KEEP_COLS, margin: float = LIMIT_MARGIN) -> dict:
    """Allowed range per column: the training min-max, widened by `margin` of the range on each side."""
    limits = {}
    for c in cols:
        lo, hi = float(train[c].min()), float(train[c].max())
        pad = (hi - lo) * margin
        limits[c] = [lo - pad, hi + pad]
    return limits


def save_limits(limits: dict) -> None:
    LIMITS_FILE.write_text(json.dumps(limits, indent=2), encoding="utf-8")


def load_limits() -> dict:
    return json.loads(LIMITS_FILE.read_text(encoding="utf-8"))


def validate(batch: pd.DataFrame, limits: dict, last_cycle: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (good, bad). `bad` has a `reason` column. `last_cycle` maps engine -> last loaded cycle."""
    batch = batch.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)
    reason = pd.Series([None] * len(batch), index=batch.index, dtype=object)

    def flag(mask: pd.Series, why: str) -> None:
        reason[mask & reason.isna()] = why

    cols = [c for c in limits if c in batch.columns]
    flag(batch[cols].apply(pd.to_numeric, errors="coerce").isna().any(axis=1), "missing value")
    flag(batch.duplicated(["unit", "cycle"], keep="first"), "duplicate cycle")
    for c in cols:
        lo, hi = limits[c]
        flag(~batch[c].between(lo, hi), f"{c} out of range")

    # Cycle continuity. Every received cycle counts (even one quarantined for another reason), so one
    # bad reading does not make the next good one look like a gap. Only duplicates are ignored.
    last_cycle = last_cycle or {}
    seen = ~batch.duplicated(["unit", "cycle"], keep="first")
    prev = batch[seen].groupby("unit")["cycle"].shift(1)
    prev = prev.fillna(batch.loc[seen, "unit"].map(last_cycle))
    gap = pd.Series(False, index=batch.index)
    gap[seen] = (batch.loc[seen, "cycle"] - prev) > 1
    flag(gap, "cycle gap")

    bad = batch[reason.notna()].assign(reason=reason[reason.notna()])
    good = batch[reason.isna()]
    return good.reset_index(drop=True), bad.reset_index(drop=True)


def inject_faults(batch: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Add one duplicate row, one missing value and one wild reading, to demonstrate quarantine."""
    rng = np.random.default_rng(seed)
    out = batch.copy()
    i, j, k = rng.choice(len(out), size=3, replace=False)
    out = pd.concat([out, out.iloc[[i]]], ignore_index=True)          # duplicate
    out.loc[j, "s11"] = np.nan                                         # missing value
    out.loc[k, "s4"] = out.loc[k, "s4"] * 3                            # sensor spike
    return out
