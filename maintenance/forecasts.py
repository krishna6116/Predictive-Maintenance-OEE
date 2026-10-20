"""The forecasts Phase 6 decides on: the published LSTM's RUL with its rule-E 90% interval.

Two views, both from the SAME published seed-0 LSTM of each dataset:
  snapshot(ds)  one forecast per test engine at its last observed cycle (the official evaluation point),
                read from the Phase 5 / 5b result files; nothing is re-trained.
  rolling(ds)   a forecast at EVERY observed test cycle, which the predictive policy needs as a machine ages.
                Phase 5 saved only last-cycle forecasts, so `python -m maintenance.forecasts` rebuilds the
                published model (training is bit-for-bit deterministic on the producing OS) and proves it is
                the same model: its last-cycle predictions and MC passes must equal the published ones EXACTLY,
                otherwise nothing is written. Output: maintenance/forecasts_rolling_<ds>.csv.

Interval (rule E, models/calibration_rule.json): MC mean +/- k x MC sd, lower edge clipped at 0, with
k = k_near when the MC mean is <= 50 cycles and k_far otherwise, k learned on the dataset's validation engines.
Point predictions and MC passes are clipped to [0, 125] by the model code (RUL cap 125). Upper edges may exceed
125: they are kept as computed, because the decision rule only uses the lower edge.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models.calibration import zone_interval, zone_scales
from pipeline.check_raw_data import load
from pipeline.config import RAW

CSV = {"float_precision": "round_trip"}
MODEL = "LSTM"
SOURCES = {
    "FD001": {"results": "models/results_phase5.json", "preds": "models/predictions_phase5.csv",
              "val": "models/val_predictions_phase5.csv", "draws": "models/mc_draws_test_phase5.csv"},
    "FD003": {"results": "models/results_phase5b.json", "preds": "models/predictions_phase5b.csv",
              "val": "models/val_predictions_phase5b.csv", "draws": "models/mc_draws_test_phase5b.csv"},
}
ROLLING = "maintenance/forecasts_rolling_{ds}.csv"
COLUMNS = ["dataset", "engine_id", "cycle", "rul_point", "rul_mc_mean", "rul_mc_std", "k", "rul_low", "rul_high",
           "model_id", "source"]


def sha12(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def model_id(ds: str) -> str:
    s = json.loads(Path(SOURCES[ds]["results"]).read_text())["models"][MODEL]["settings"]
    return f"{MODEL}-{ds}-seed0-W{s['window']}-h{s['hidden']}-do{s['dropout']}-lr{s['lr']:g}"


def interval_scales(ds: str) -> tuple[float, float]:
    """Rule-E (k_near, k_far), re-learned from the dataset's saved validation MC predictions."""
    v = pd.read_csv(SOURCES[ds]["val"], **CSV)
    return zone_scales(v.RUL, v["MC mean"], v["MC std"], v.cycle, v.RUL_true)


def with_interval(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    kn, kf = interval_scales(ds)
    lo, hi = zone_interval(df.rul_mc_mean, df.rul_mc_std, kn, kf)
    return df.assign(k=np.where(df.rul_mc_mean <= 50, kn, kf), rul_low=lo, rul_high=hi)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Mark every forecast valid or invalid, with the reason. Invalid forecasts are never repaired or dropped:
    the alert step sends them to manual inspection instead of the scheduler."""
    num = df[["rul_point", "rul_mc_mean", "rul_mc_std", "rul_low", "rul_high"]].to_numpy(float)
    reason = np.full(len(df), "", dtype=object)
    checks = [
        (~np.isfinite(num).all(axis=1), "non-finite value"),
        (df.rul_mc_std.to_numpy(float) < 0, "negative MC sd"),
        (df.rul_low.to_numpy(float) < 0, "negative lower bound"),
        (df.rul_low.to_numpy(float) > df.rul_high.to_numpy(float), "lower bound above upper bound"),
        (df.duplicated(["dataset", "engine_id", "cycle"], keep=False).to_numpy(), "duplicate forecast"),
    ]
    for bad, why in checks:
        bad = np.asarray(bad, bool)
        reason[bad & (reason == "")] = why
    return df.assign(valid=reason == "", invalid_reason=reason)


def snapshot(ds: str) -> pd.DataFrame:
    """One validated forecast per test engine at its last observed cycle, plus the truth for later scoring."""
    src = SOURCES[ds]
    p = pd.read_csv(src["preds"], **CSV)
    last = load(RAW / f"test_{ds}.txt").groupby("unit").cycle.max()
    df = pd.DataFrame({"dataset": ds, "engine_id": p.unit.astype(int), "cycle": p.unit.map(last).astype(int),
                       "rul_point": p[MODEL].astype(float), "rul_mc_mean": p["MC mean"].astype(float),
                       "rul_mc_std": p["MC std"].astype(float)})
    df = with_interval(df, ds).assign(model_id=model_id(ds), source=f"{src['preds']}@{sha12(src['preds'])}")
    df = validate(df[COLUMNS])
    return df.assign(true_rul=p.RUL_true.astype(float).to_numpy(), true_rul_capped=p.RUL.astype(float).to_numpy())


def rolling(ds: str) -> pd.DataFrame:
    df = pd.read_csv(ROLLING.format(ds=ds), **CSV)
    return validate(df[COLUMNS])


def build_rolling(ds: str) -> pd.DataFrame:
    """Rebuild the published seed-0 LSTM and forecast every observed test cycle (see module docstring)."""
    import torch
    from models.deep import mc_dropout, predict, train_fixed
    from models.sequences import datasets, make_windows, scaled_readings

    src = SOURCES[ds]
    r = json.loads(Path(src["results"]).read_text())
    m, env, passes = r["models"][MODEL], r["environment"], r["mc_dropout"]["passes"]
    if env["os"] != sys.platform or env["torch"] != torch.__version__:
        raise SystemExit(f"{ds} was published on {env['os']} / torch {env['torch']}; this is {sys.platform} / "
                         f"torch {torch.__version__}. The rebuild is only exact there, so it is not attempted.")
    cfg = m["settings"]
    d = datasets(cfg["window"], ds)
    X_all, i_all = d["all"]
    X_last, _ = d["test"]
    model, _ = train_fixed(cfg, 0, X_all, i_all.RUL.to_numpy(float), m["epochs_per_seed"][0])

    # Identity proof: same model as published, value for value
    pub = pd.read_csv(src["preds"], **CSV)
    draws = pd.read_csv(src["draws"], **CSV)
    same_point = np.array_equal(predict(model, X_last), pub[MODEL].to_numpy(float))
    same_mc = np.array_equal(mc_dropout(model, X_last, passes, seed=0).T,
                             draws[[f"pass_{k}" for k in range(passes)]].to_numpy(float))
    if not (same_point and same_mc):
        raise SystemExit(f"{ds}: rebuilt model differs from the published one (point {same_point}, MC {same_mc})")

    X, idx = make_windows(scaled_readings("test", ds), cfg["window"])
    mc = mc_dropout(model, X, passes, seed=0)  # a fresh set of draws over all windows, same model
    out = pd.DataFrame({"dataset": ds, "engine_id": idx.unit.astype(int), "cycle": idx.cycle.astype(int),
                        "rul_point": predict(model, X), "rul_mc_mean": mc.mean(0), "rul_mc_std": mc.std(0, ddof=1)})
    out = with_interval(out, ds).assign(model_id=model_id(ds), source=f"{src['preds']}@{sha12(src['preds'])}")
    last = out.groupby("engine_id").tail(1)
    if not np.array_equal(last.rul_point.to_numpy(), pub[MODEL].to_numpy(float)):
        raise SystemExit(f"{ds}: rolling last-cycle point forecasts differ from the published ones")
    return out[COLUMNS]


def main() -> int:
    for ds in SOURCES:
        out = build_rolling(ds)
        out.to_csv(ROLLING.format(ds=ds), index=False)
        bad = (~validate(out).valid).sum()
        print(f"{ds}: {len(out):,} forecasts for {out.engine_id.nunique()} engines; published model reproduced exactly; "
              f"invalid {bad}; wrote {ROLLING.format(ds=ds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
