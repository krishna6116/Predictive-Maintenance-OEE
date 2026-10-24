"""Phase 7b: the deployed LSTM for the fresh confirmation fleets FD002 and FD004 (frozen recipe, nothing chosen here).

Run:  python -m models.run_fresh FD002 FD004       (about 45 minutes per dataset on one CPU thread)

Frozen from FD001 (pipeline/phase7b_criteria.json): the LSTM settings found by the Phase 5 search, seed 0, the engine
split function and seed, rule E for the intervals. Dataset-specific, learned the same way as in Phases 5/5b: sensors
standardised per operating condition on training engines (pipeline/conditions.py), the epoch count by early stopping
on the dataset's validation engines, the refit on all training engines for that count, rule E's scale factors on
validation. Also writes out-of-fold predictions for the training-split engines (models/oof.py), which the
monitoring rule needs. Test RMSE is reported next to the fleet-average and age-only baselines for context; the
monitoring confirmation, not this number, is what Phase 7b tests.
Outputs: models/results_<ds>.json, predictions_<ds>.csv, val_predictions_<ds>.csv, mc_draws_test_<ds>.csv,
oof_predictions_<ds>.csv.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.calibration import zone_interval, zone_scales
from models.deep import THREADS, mc_dropout, predict, train_early_stopping, train_fixed
from models.metrics import rmse
from models.oof import oof
from models.sequences import datasets
from pipeline.conditions import ensure_split_and_stats

FROZEN = json.loads(Path("models/results_phase5.json").read_text())["models"]["LSTM"]["settings"]
PASSES = 100


def run(ds: str) -> dict:
    t0 = time.perf_counter()
    split = ensure_split_and_stats(ds)
    d = datasets(FROZEN["window"], ds)
    (Xtr, itr), (Xva, iva), (Xall, iall), (Xte, ite) = d["train"], d["val"], d["all"], d["test"]
    m80, info = train_early_stopping(FROZEN, 0, Xtr, itr.RUL.to_numpy(float), Xva, iva.RUL.to_numpy(float))
    model, secs = train_fixed(FROZEN, 0, Xall, iall.RUL.to_numpy(float), info["best_epoch"])
    vpt, tpt = predict(m80, Xva), predict(model, Xte)
    vd, td = mc_dropout(m80, Xva, PASSES, seed=0), mc_dropout(model, Xte, PASSES, seed=0)
    kn, kf = zone_scales(iva.RUL, vd.mean(0), vd.std(0, ddof=1), iva.cycle, iva.RUL_true)
    lo, hi = zone_interval(td.mean(0), td.std(0, ddof=1), kn, kf)
    y = ite.RUL.to_numpy(float)
    full = iall.assign(life=iall.groupby("unit").cycle.transform("max"))
    age = np.polyval(np.polyfit(full.cycle, full.RUL, 2), ite.cycle.to_numpy(float)).clip(0, 125)
    preds = pd.DataFrame({"unit": ite.unit, "RUL_true": ite.RUL_true, "RUL": ite.RUL, "LSTM": tpt,
                          "MC mean": td.mean(0), "MC std": td.std(0, ddof=1), "MC 90% low (rule E)": lo, "MC 90% high (rule E)": hi})
    vpreds = iva[["unit", "cycle", "RUL", "RUL_true"]].assign(LSTM=vpt, **{"MC mean": vd.mean(0), "MC std": vd.std(0, ddof=1)})
    preds.to_csv(f"models/predictions_{ds}.csv", index=False)
    vpreds.to_csv(f"models/val_predictions_{ds}.csv", index=False)
    pd.DataFrame(td.T, columns=[f"pass_{k}" for k in range(PASSES)]).assign(unit=ite.unit.to_numpy()).to_csv(
        f"models/mc_draws_test_{ds}.csv", index=False)
    res = {"dataset": ds, "purpose": "deployed model for the Phase 7b monitoring confirmation (frozen FD001 recipe)",
           "models": {"LSTM": {"settings": FROZEN, "epochs_per_seed": [int(info["best_epoch"])],
                               "val_rmse": float(info["val_rmse"]), "rmse": rmse(y, tpt),
                               "seed_test_predictions": [tpt.tolist()]}},
           "baselines_test_rmse": {"Fleet average": rmse(y, np.full(len(y), float(d["train"][1].RUL.mean()))),
                                   "Age only (quadratic in cycle)": rmse(y, age)},
           "mc_dropout": {"passes": PASSES, "k_near": kn, "k_far": kf,
                          "test_coverage": float(np.mean((lo <= y) & (y <= hi)))},
           "engines": {"train": len(split["train"]), "val": len(split["val"]), "test": int(ite.unit.nunique())},
           "environment": {"os": sys.platform, "platform": platform.platform(), "python": platform.python_version(),
                           "torch": torch.__version__, "threads": THREADS, "deterministic": True}}
    Path(f"models/results_{ds}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    o = oof(ds, f"models/results_{ds}.json")
    o.to_csv(f"models/oof_predictions_{ds}.csv", index=False)
    res["seconds_total"] = time.perf_counter() - t0
    Path(f"models/results_{ds}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{ds}: best epoch {info['best_epoch']}, validation RMSE {info['val_rmse']:.2f}; test RMSE LSTM {res['models']['LSTM']['rmse']:.2f} "
          f"vs {res['baselines_test_rmse']}; interval coverage {res['mc_dropout']['test_coverage']:.0%}; "
          f"{res['seconds_total'] / 60:.0f} min", flush=True)
    return res


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
