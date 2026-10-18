"""Evidence for the cross-OS checks in models/verify_deep.py: which check catches which bug?

Run:  python -m models.tolerance_phase5     (after python -m models.run_deep; about 5 minutes)

On another OS the published predictions cannot be reproduced bit for bit, so the verifier relies on three checks
that do not need bit-identical training:
  1. Window rebuild: every training and test window must equal an independent plain-Python rebuild from the raw
     files. Exact on any OS, so it catches every bug that changes the model's inputs.
  2. Epoch consistency: seed 0 is refit for the epoch count found by the validation search.
  3. Statistical band: one re-trained seed's test RMSE must be within BAND of the published 5-seed mean. This is
     the only check that looks at training itself; it is loose (seed noise), so it is meant for broken training.
This script trains the published setting correctly and with deliberate bugs, scores each on the 20 VALIDATION
engines (the test set is not used for scoring) and records which checks catch each bug. Every bug must be caught
by at least one check. Writes models/tolerance_phase5.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from features.build import SPLIT_FILE, add_rul, load_json
from models.deep import predict, train_fixed
from models.metrics import rmse
from models.sequences import datasets, make_windows
from pipeline.check_raw_data import load
from pipeline.config import RAW, SENSORS

OUT = Path("models/tolerance_phase5.json")


def zero_padded(df, length, cols):
    """The bug version of make_windows: pads short histories with zeros instead of the first reading."""
    X, idx = make_windows(df, length, cols=cols)
    for k, c in enumerate(idx.cycle.to_numpy()):
        if c < length:
            X[k, : length - c] = 0.0
    return X, idx


def main() -> int:
    r = json.loads(Path("models/results_phase5.json").read_text())
    pub = r["published_deep_model"]
    m = r["models"][pub]
    cfg, epochs = m["settings"], m["epochs_per_seed"][0]
    W = cfg["window"]
    d = datasets(W)
    (Xtr, itr), (Xva, iva), (Xall, iall), (Xte, ite) = d["train"], d["val"], d["all"], d["test"]
    ytr, yva, yall = itr.RUL.to_numpy(float), iva.RUL.to_numpy(float), iall.RUL.to_numpy(float)
    split = load_json(SPLIT_FILE)
    tr_mask, va_mask = iall.unit.isin(split["train"]).to_numpy(), iall.unit.isin(split["val"]).to_numpy()

    # Inputs for each deliberate bug: (train windows, validation windows, all-100 windows, test windows)
    from models.sequences import scaled_readings
    raw_tr = add_rul(load(RAW / "train_FD001.txt")).sort_values(["unit", "cycle"])
    raw_te = load(RAW / "test_FD001.txt").sort_values(["unit", "cycle"])
    Xr, _ = make_windows(raw_tr, W, cols=SENSORS)
    Xr_te, _ = make_windows(raw_te, W, cols=SENSORS, last_only=True)
    Xz, _ = zero_padded(scaled_readings("train"), W, SENSORS)
    Xz_te, _ = zero_padded(scaled_readings("test"), W, SENSORS)  # last windows are taken below
    te_last = np.cumsum(scaled_readings("test").groupby("unit").size().to_numpy()) - 1
    s4 = SENSORS.index("s4")
    variants = {
        "correct": (Xtr, Xva, Xall, Xte, epochs),
        "half the validated epochs": (Xtr, Xva, Xall, Xte, max(1, epochs // 2)),
        "sensors not scaled": (Xr[tr_mask], Xr[va_mask], Xr, Xr_te, epochs),
        "windows in reverse time order": (Xtr[:, ::-1].copy(), Xva[:, ::-1].copy(), Xall[:, ::-1].copy(), Xte[:, ::-1].copy(), epochs),
        "one sensor (s4) missing": (np.delete(Xtr, s4, 2), np.delete(Xva, s4, 2), np.delete(Xall, s4, 2), np.delete(Xte, s4, 2), epochs),
        "short histories padded with zeros": (Xz[tr_mask], Xz[va_mask], Xz, Xz_te[te_last], epochs),
    }
    val = {}
    for name, (a, b, c, t, e) in variants.items():
        model, _ = train_fixed(cfg, 0, a, ytr, e)
        val[name] = rmse(yva, predict(model, b))
        print(f"{name:34s} val RMSE {val[name]:7.2f}", flush=True)

    sd, n = m["seed_test_rmse_std"], len(m["seed_test_rmse"])
    band = 4.0 * sd * math.sqrt(1 + 1 / n)
    bugs = {}
    for name, (a, b, c, t, e) in variants.items():
        if name == "correct":
            continue
        shift = val[name] - val["correct"]
        windows_differ = c.shape != Xall.shape or t.shape != Xte.shape or not (np.array_equal(c, Xall) and np.array_equal(t, Xte))
        caught = []
        if windows_differ:
            caught.append("window rebuild")
        if e != epochs:
            caught.append("epoch consistency")
        if abs(shift) > band:
            caught.append("statistical band")
        bugs[name] = {"validation_rmse": val[name], "shift_vs_correct": shift, "caught_by": caught}
    out = {"model": pub, "settings": cfg, "epochs": epochs, "metric": "validation RMSE (20 engines)",
           "band": band, "correct_validation_rmse": val["correct"], "bugs": bugs}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, b in bugs.items():
        print(f"{name:34s} caught by: {', '.join(b['caught_by']) or 'NOTHING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
