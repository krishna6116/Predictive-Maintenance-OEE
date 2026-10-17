"""Independent verification of Phase 3 (run: python -m features.verify).

Recomputes everything from the raw NASA text files with a deliberately different, simple implementation
(explicit loops, two-pass statistics, np.polyfit slopes, a hand-written EMA recursion) and compares it with
what the pipeline stored in PostgreSQL and in features/*.json. Exits with code 1 if any check fails.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sqlalchemy import text

from features.build import EMA_SPAN, FFT_MIN, FFT_WINDOW, RUL_CAP, SCALER_FILE, SPLIT_FILE, WINDOWS, feature_columns
from pipeline.config import RAW, SENSORS
from pipeline.db import get_engine

TOL = 1e-9  # largest acceptable absolute difference between pipeline and independent values
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
results: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    results.append((bool(ok), what))
    print(("PASS " if ok else "FAIL ") + what)


def raw(split: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{split}_FD001.txt", sep=r"\s+", header=None, names=COLS)


def naive_features(df: pd.DataFrame, mean: dict, std: dict) -> pd.DataFrame:
    """One engine at a time, one cycle at a time, straight from the definitions."""
    alpha = 2 / (EMA_SPAN + 1)
    out = []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        cyc = g.cycle.to_numpy(float)
        rec = {"unit": np.full(len(g), unit), "cycle": g.cycle.to_numpy()}
        for s in SENSORS:
            x = (g[s].to_numpy(float) - mean[s]) / std[s]
            rec[s] = x
            for w in WINDOWS:
                m, sd, sl = np.empty(len(x)), np.empty(len(x)), np.empty(len(x))
                for t in range(len(x)):
                    seg, cs = x[max(0, t - w + 1): t + 1], cyc[max(0, t - w + 1): t + 1]
                    m[t] = seg.mean()
                    sd[t] = np.sqrt(((seg - seg.mean()) ** 2).sum() / (len(seg) - 1)) if len(seg) > 1 else 0.0
                    sl[t] = np.polyfit(cs, seg, 1)[0] if len(seg) > 1 else 0.0
                rec[f"{s}_mean{w}"], rec[f"{s}_std{w}"], rec[f"{s}_slope{w}"] = m, sd, sl
            ema = np.empty(len(x))
            for t in range(len(x)):
                ema[t] = x[0] if t == 0 else alpha * x[t] + (1 - alpha) * ema[t - 1]
            rec[f"{s}_ema"] = ema
            lf = np.zeros(len(x))
            for t in range(len(x)):
                seg = x[max(0, t - FFT_WINDOW + 1): t + 1]
                if len(seg) >= FFT_MIN:
                    lf[t] = np.abs(np.fft.fft(seg - seg.mean()))[1:4].sum() / len(seg)  # full FFT, not rfft
            rec[f"{s}_lowfreq"] = lf
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    engine = get_engine()
    train_raw, test_raw = raw("train"), raw("test")

    print("\n== Split (features/split.json)")
    split = json.loads(SPLIT_FILE.read_text())
    units = np.arange(1, 101)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(units, groups=units))
    check(sorted(split["train"]) == sorted(units[tr].tolist()) and sorted(split["val"]) == sorted(units[va].tolist()),
          "split reproduces exactly from seed 42")
    check(set(split["train"]).isdisjoint(split["val"]) and sorted(split["train"] + split["val"]) == list(range(1, 101)),
          "80 + 20 engines, no overlap, every engine 1-100 assigned once")

    print("\n== Scaler (features/scaler.json), recomputed from the raw training file")
    scaler = json.loads(SCALER_FILE.read_text())
    rows = train_raw[train_raw.unit.isin(split["train"])]
    check(len(rows) == 16561, f"scaler inputs: {len(rows):,} readings from training engines only")
    mean = {s: rows[s].to_numpy().mean() for s in SENSORS}
    std = {s: rows[s].to_numpy().std(ddof=1) for s in SENSORS}
    dm = max(abs(mean[s] - scaler[s]["mean"]) for s in SENSORS)
    ds = max(abs(std[s] - scaler[s]["std"]) / std[s] for s in SENSORS)
    check(dm < 1e-9 and ds < 1e-12, f"means and std (sample, ddof=1) match: max mean diff {dm:.1e}, max relative std diff {ds:.1e}")
    leak = train_raw[train_raw.unit.isin(split["val"])]
    check(any(abs(leak[s].mean() - scaler[s]["mean"]) > 1e-6 for s in SENSORS),
          "validation engines are not part of the scaler (their means differ from it)")

    print("\n== RUL target")
    rul = (train_raw.groupby("unit").cycle.transform("max") - train_raw.cycle).clip(upper=RUL_CAP)
    check(rul.min() == 0 and rul.max() == RUL_CAP and (train_raw.groupby("unit").apply(lambda g: rul[g.index].iloc[-1]) == 0).all(),
          "RUL ends at 0 for every engine and never exceeds 125")
    share = (rul == RUL_CAP).mean()
    check(abs(share - 0.39) < 0.005, f"share of training readings at the cap: {share:.2%} (docs say 39%)")

    print("\n== fact_features table")
    with engine.connect() as c:
        stored = pd.read_sql(text("SELECT * FROM fact_features WHERE dataset = 'FD001'"), c)
        src = pd.read_sql(text("SELECT split, engine_id, cycle, run_id FROM fact_sensor_reading WHERE dataset = 'FD001'"), c)
        pk = c.execute(text("""SELECT COUNT(*) FROM information_schema.table_constraints
                               WHERE table_name = 'fact_features' AND constraint_type = 'PRIMARY KEY'""")).scalar()
        types = dict(c.execute(text("""SELECT column_name, data_type FROM information_schema.columns
                                       WHERE table_name = 'fact_features'""")).all())
    fc = feature_columns()
    check(set(fc) <= set(stored.columns) and len(fc) == 168, "all 168 feature columns present")
    check(all(types[c] == "double precision" for c in fc), "every feature column is double precision")
    check(pk == 1, "fact_features has a primary key (dataset, split, engine_id, cycle)")
    check(not stored.duplicated(["split", "engine_id", "cycle"]).any(), "no duplicate engine-cycle rows")
    keys = ["split", "engine_id", "cycle"]
    m = src.merge(stored[keys + ["run_id"]], on=keys, how="outer", suffixes=("_src", "_feat"), indicator=True)
    check((m._merge == "both").all(), f"one feature row per clean reading ({len(stored):,} rows = {len(src):,} readings)")
    check((m.run_id_src == m.run_id_feat).all(), "each feature row carries the run ID of the reading it came from")
    check(np.isfinite(stored[fc].to_numpy()).all(), "no missing or infinite values")

    print("\n== Every stored value vs an independent recomputation from the raw NASA files")
    for split_name, frame in (("train", train_raw), ("test", test_raw)):
        db = stored[stored.split == split_name].rename(columns={"engine_id": "unit"})
        loaded = frame.merge(db[["unit", "cycle"]], on=["unit", "cycle"])
        # Features depend on all earlier cycles, so recompute from cycle 1 up to the last loaded cycle per engine
        last = loaded.groupby("unit").cycle.max()
        hist = frame[frame.cycle <= frame.unit.map(last)]
        ref = naive_features(hist, mean, std).merge(db[["unit", "cycle"]], on=["unit", "cycle"])
        both = ref.merge(db, on=["unit", "cycle"], suffixes=("_ref", "_db"))
        worst = {c: float(np.abs(both[f"{c}_ref"] - both[f"{c}_db"]).max()) for c in fc}
        bad = {c: v for c, v in worst.items() if v > TOL}
        check(len(both) == len(db) and not bad,
              f"{split_name}: {len(both):,} rows x 168 features = {len(both) * 168:,} values, "
              f"max difference {max(worst.values()):.1e}" + (f"; over tolerance: {sorted(bad)[:5]}" if bad else ""))

    print("\n== Numbers quoted in docs/FEATURES.md and the notebook")
    tr_feats = naive_features(train_raw, mean, std).merge(
        train_raw[["unit", "cycle"]].assign(RUL=rul.values), on=["unit", "cycle"])
    tr_feats = tr_feats[tr_feats.unit.isin(split["train"])]
    corr = tr_feats[fc + ["RUL"]].corr()["RUL"].drop("RUL").abs()
    def grp(kind):
        return corr[SENSORS] if kind == "raw" else corr[[f"{s}_{kind}" for s in SENSORS]]
    quoted = {"raw": (0.771, 0.690), "mean5": (0.812, 0.792), "mean10": (0.811, 0.803), "mean20": (0.800, 0.785),
              "ema": (0.811, 0.803), "lowfreq": (0.556, 0.391), "std20": (0.457, 0.201), "slope20": (0.294, 0.184)}
    for kind, (best, med) in quoted.items():
        g = grp(kind)
        check(abs(g.max() - best) < 6e-4 and abs(g.median() - med) < 6e-4,
              f"{kind:8s} best {g.max():.3f} (doc {best}), median {g.median():.3f} (doc {med})")
    check(corr.idxmax() == "s4_mean5", f"top feature is {corr.idxmax()} ({corr.max():.3f})")

    failed = [w for ok, w in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
