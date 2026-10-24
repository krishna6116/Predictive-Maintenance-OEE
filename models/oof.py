"""Out-of-fold (OOF) LSTM predictions for the training-split engines (Phase 7b).

Run:  python -m models.oof FD001 FD003     (about 5 minutes per dataset on one CPU thread)

The Phase 7b drift test compares each sensor with its expected level for an engine of that age and PREDICTED
remaining life. The expected level is learned on training engines, so they need predictions that behave like the
flow's predictions on unseen engines: each training-split engine is predicted by a model that never saw it. The 80
training-split engines are divided into 5 folds (numpy default_rng(0) permutation); for each fold the published
LSTM recipe (settings, seed 0, the validated epoch count) is trained on the other 64 engines and predicts the
held-out 16 at every cycle (dropout off). The 20 validation engines already have such predictions (Phase 5/5b
validation files: the 80-engine model). Writes models/oof_predictions_<dataset>.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models.deep import predict, train_fixed
from models.sequences import datasets

RESULTS = {"FD001": "models/results_phase5.json", "FD003": "models/results_phase5b.json"}
FOLDS = 5
OUT = "models/oof_predictions_{ds}.csv"


def oof(ds: str, results_file: str | None = None) -> pd.DataFrame:
    m = json.loads(Path(results_file or RESULTS[ds]).read_text())["models"]["LSTM"]
    cfg, epochs = m["settings"], m["epochs_per_seed"][0]
    X, idx = datasets(cfg["window"], ds)["train"]
    units = np.array(sorted(idx.unit.unique()))
    perm = np.random.default_rng(0).permutation(units)
    fold_of = {int(u): f for f, part in enumerate(np.array_split(perm, FOLDS)) for u in part}
    fold = idx.unit.map(fold_of).to_numpy()
    pred = np.full(len(idx), np.nan)
    for f in range(FOLDS):
        model, secs = train_fixed(cfg, 0, X[fold != f], idx.RUL.to_numpy(float)[fold != f], epochs)
        pred[fold == f] = predict(model, X[fold == f])
        print(f"{ds} fold {f}: trained on {int((fold != f).sum()):,} windows in {secs:.0f}s, predicted {int((fold == f).sum()):,}", flush=True)
    return idx.assign(fold=fold, oof_point=pred)[["unit", "cycle", "RUL", "RUL_true", "fold", "oof_point"]]


def main(argv=None) -> int:
    for ds in (argv or sys.argv[1:]):
        out = oof(ds)
        out.to_csv(OUT.format(ds=ds), index=False)
        rmse = float(np.sqrt(np.mean((out.oof_point - out.RUL) ** 2)))
        print(f"{ds}: {len(out):,} out-of-fold predictions for {out.unit.nunique()} engines; RMSE vs capped RUL {rmse:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
