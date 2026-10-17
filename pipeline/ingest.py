"""Simulated live ingestion: replay a C-MAPSS file in batches of cycles, like readings arriving each shift."""
import pandas as pd

from pipeline.check_raw_data import load
from pipeline.config import RAW


def load_file(dataset: str, split: str) -> pd.DataFrame:
    """Read data/raw/{split}_{dataset}.txt, e.g. load_file("FD001", "test")."""
    return load(RAW / f"{split}_{dataset}.txt")


def next_batch(df: pd.DataFrame, start_cycle: int, n_cycles: int) -> pd.DataFrame:
    """Every engine's readings for cycles start_cycle .. start_cycle + n_cycles - 1.

    Engines that stopped before start_cycle simply contribute no rows, as a machine that is
    no longer reporting would.
    """
    mask = (df.cycle >= start_cycle) & (df.cycle < start_cycle + n_cycles)
    return df[mask].reset_index(drop=True)
