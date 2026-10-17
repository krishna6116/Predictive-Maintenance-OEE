"""Integrity check for the raw NASA C-MAPSS files in data/raw/.

Run:  python -m pipeline.check_raw_data
Exits with code 1 if any check fails.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAW = Path("data/raw")
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
# (train engines, test engines) per subset, from NASA's readme.txt. The readme swaps FD004's
# counts (it says 248 train / 249 test); the files hold 249 train and 248 test engines, and
# RUL_FD004.txt has 248 values, so the files are complete and the readme has the typo.
EXPECTED = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}


def field_counts(path: Path) -> set[int]:
    """Number of space-separated values on each non-empty line, read straight from the text."""
    with path.open() as f:
        return {len(line.split()) for line in f if line.strip()}


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def check_series(df: pd.DataFrame, n_engines: int) -> list[str]:
    problems = []
    if sorted(df.unit.unique()) != list(range(1, n_engines + 1)):
        problems.append(f"engine IDs are not exactly 1..{n_engines}")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        problems.append("missing or non-numeric values")
    if df.duplicated(["unit", "cycle"]).any():
        problems.append("duplicate (engine, cycle) rows")
    for unit, g in df.groupby("unit"):
        cycles = g.cycle.to_numpy()
        if cycles[0] != 1 or not (np.diff(cycles) == 1).all():
            problems.append(f"engine {unit}: cycles do not run 1, 2, 3 ... without gaps")
            break
    return problems


def main() -> int:
    failures = 0
    for subset, (n_train, n_test) in EXPECTED.items():
        lines = []
        for split, n in (("train", n_train), ("test", n_test)):
            path = RAW / f"{split}_{subset}.txt"
            if not path.exists():
                print(f"FAIL {path.name}: file missing")
                failures += 1
                continue
            counts = field_counts(path)
            df = load(path)
            problems = check_series(df, n)
            if counts != {26}:
                problems.insert(0, f"lines with {sorted(counts)} values (expected 26)")
            status = "OK  " if not problems else "FAIL"
            failures += bool(problems)
            lines.append(f"{status} {path.name}: {len(df):>6,} rows, {df.unit.nunique()} engines, "
                         f"cycles {df.cycle.min()}-{df.cycle.max()}" + "".join(f"\n       - {p}" for p in problems))

        rul_path = RAW / f"RUL_{subset}.txt"
        if rul_path.exists():
            rul = pd.read_csv(rul_path, header=None, sep=r"\s+")
            ok = rul.shape == (n_test, 1) and (rul[0] > 0).all() and rul[0].dtype.kind == "i"
            failures += not ok
            lines.append(f"{'OK  ' if ok else 'FAIL'} {rul_path.name}: {len(rul)} values "
                         f"(expected {n_test}), range {rul[0].min()}-{rul[0].max()}")
        else:
            failures += 1
            lines.append(f"FAIL {rul_path.name}: file missing")
        print("\n".join(lines))

    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
