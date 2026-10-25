"""Phase 7b pre-registration (part 2): choose the new monitoring rule on training and validation engines only.

Run:  python -m pipeline.calibrate_monitoring2              (FD001 + FD003: chooses the rule, writes monitoring_rule_v2.json)
      python -m pipeline.calibrate_monitoring2 FD002 FD004  (later: re-learns ONLY the dataset-specific values with the
                                                             frozen decision parameters; no choice is made)
Reads no test file.

Calibration fleets must reproduce Phase 7's failure, or they cannot judge a fix. Fleets of the 20 validation engines
alone did not: even the old Phase 7 test raised no false alarm on them. So the fleets are CROSS-FITTED: the 80
training-split engines are split into halves A and B (numpy default_rng(5)); the expected-level model, the drift
reference and the expected-error model are fitted on A; the fleet is B + the 20 validation engines (60 engines, all
with predictions from models that never saw them: out-of-fold for B, the 80-engine model for validation); then the
halves swap. Each fleet engine is cut off at a random 25-95% of its life (mimicking NASA's test files) and starts on
shift 1 (synchronized) or at a random offset 0-150 (staggered); 15 replicate fleets per half and mode. The old Phase 7
test is run on the same fleets as evidence that they reproduce the problem.

Decision parameters (frozen afterwards for FD002/FD004):
  alpha, K-of-M persistence, alpha_strong: every combination's false-alarm rate on the clean fleets; among those
  within the drift budget in EVERY dataset x mode x half, the one that catches the most fleet-wide persistent +1
  training-sd steps within 10 shifts (then ramps to +2 sd within 50, then the stricter setting).
  error-ratio limit: bootstrap windows of 20 fleet engines (one random test-like reading each), plus windows in
  order of engine life (shortest first, as labels arrive), within budget.
  miss alpha: bootstrap windows of the 20 validation engines (rule-E bounds, k learned leaving the engine out).
The flow uses the same model classes fitted on all 80 training-split engines.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, ks_2samp

from features.build import split_file
from models.calibration import test_like, zone_interval, zone_scales
from pipeline import monitoring as M1
from pipeline import monitoring2 as M2
from pipeline.check_raw_data import load
from pipeline.config import RAW, SENSORS

CSV = {"float_precision": "round_trip"}
VAL = {"FD001": "models/val_predictions_phase5.csv", "FD003": "models/val_predictions_phase5b.csv"}
BUDGET = {"drift": 0.02, "error": 0.015, "miss": 0.015}
ALPHAS = [0.05, 0.01, 0.005, 0.001]
KM = [(1, 1), (2, 3), (3, 5), (4, 6)]
STRONG = [1e-4, 1e-6, 1e-8, 1e-10]
RATIOS = [round(1.2 + 0.1 * i, 1) for i in range(29)]            # 1.2 .. 4.0
MISS_ALPHAS = [0.05, 0.01, 0.005, 0.001]
REPLICATES, POWER_REPLICATES, MAX_OFFSET, CUT = 15, 3, 150, (0.25, 0.95)
ONSETS, DETECT_WITHIN, RAMP_LEN = (25, 50, 75, 100, 125, 150), 10, 50
N_BOOT = 5000
HALF_SEED = 5


# ------------------------------------------------------------------ data and per-dataset models
def load_dataset(ds: str, val_file: str, oof_file: str, raw_train: pd.DataFrame | None = None) -> dict:
    split = json.loads(Path(split_file(ds)).read_text())
    raw = load(RAW / f"train_{ds}.txt") if raw_train is None else raw_train
    oof = pd.read_csv(oof_file, **CSV)[["unit", "cycle", "oof_point"]].rename(columns={"oof_point": "rhat"})
    tr = raw[raw.unit.isin(split["train"])].merge(oof, on=["unit", "cycle"], validate="one_to_one")
    v = pd.read_csv(val_file, **CSV)
    val = raw[raw.unit.isin(split["val"])].merge(v[["unit", "cycle", "LSTM"]].rename(columns={"LSTM": "rhat"}),
                                                  on=["unit", "cycle"], validate="one_to_one")
    parts = []                    # rule-E bounds for each validation engine, k learned on the other validation engines
    for u in sorted(v.unit.unique()):
        o, me = v[v.unit != u], v[v.unit == u]
        kn, kf = zone_scales(o.RUL, o["MC mean"], o["MC std"], o.cycle, o.RUL_true)
        lo, hi = zone_interval(me["MC mean"], me["MC std"], kn, kf)
        parts.append(me.assign(low=lo, high=hi))
    vb = pd.concat(parts)
    vtl = vb[test_like(vb.cycle, vb.RUL_true)]
    return {"split": split, "tr": tr, "val": val, "vtl": vtl, "p0": float(np.mean(vtl.RUL_true < vtl.low)),
            "sd_raw": tr[SENSORS].std()}


VARIANT = "v2"                     # set by main: "v2" (pre-registered) or "v2.1" (post-hoc spread fix)


def fit_models(train: pd.DataFrame, variant: str | None = None) -> dict:
    """Expected level, drift reference and expected error, fitted on the given training engines (with rhat)."""
    lev = M2.ExpectedLevel(spread=M2.SPREAD[variant or VARIANT]).fit(train)
    ref, n_ref = M2.reference_windows(lev.z(train))
    life = train.groupby("unit").cycle.transform("max")
    tl = test_like(train.cycle, life - train.cycle)
    err = M2.ExpectedError().fit(life[tl], (train.rhat[tl] - (life[tl] - train.cycle[tl]).clip(upper=125)) ** 2)
    return {"level": lev, "ref": ref, "n_ref": n_ref, "error": err}


def halves(d: dict) -> list:
    units = np.random.default_rng(HALF_SEED).permutation(sorted(d["split"]["train"]))
    return [(set(units[:len(units) // 2].tolist()), set(units[len(units) // 2:].tolist())),
            (set(units[len(units) // 2:].tolist()), set(units[:len(units) // 2].tolist()))]


def fleet_arrays(d: dict, models: dict, fleet: pd.DataFrame, ref_raw: pd.DataFrame) -> dict:
    """Per fleet engine (cycles 1..life): z, z-change of +1 training sd, raw readings (for the old test)."""
    f = fleet.sort_values(["unit", "cycle"]).reset_index(drop=True)
    z = models["level"].z(f)
    X = M2._inputs(f.rhat, f.cycle)
    scale = np.column_stack([d["sd_raw"][s] / models["level"].sd(s, X) for s in SENSORS])
    out = {}
    for u, g in z.groupby("unit", sort=True):
        if not np.array_equal(g.cycle.to_numpy(), np.arange(1, len(g) + 1)):
            raise ValueError(f"engine {u}: cycles must run 1..life")
        ix = g.index.to_numpy()
        out[int(u)] = {"z": g[SENSORS].to_numpy(float), "scale": scale[ix], "raw": f.loc[ix, SENSORS].to_numpy(float), "life": len(g)}
    # old-test reference: per reference engine, prefix sums of raw readings by cycle
    refp = {}
    for u, g in ref_raw.sort_values(["unit", "cycle"]).groupby("unit"):
        v = g[SENSORS].to_numpy(float)
        refp[int(u)] = np.vstack([np.zeros((1, len(SENSORS))), np.cumsum(v, axis=0)])
    return {"engines": out, "ref_prefix": refp}


# ------------------------------------------------------------------ fleets and drift statistics
def streams(engines: dict, mode: str, b: int) -> dict:
    rng = np.random.default_rng(1000 * b + (1 if mode == "staggered" else 0))
    eng = {}
    for u in sorted(engines):
        life = engines[u]["life"]
        t0 = max(M2.WINDOW_SHIFTS, int(round(life * rng.uniform(*CUT))))
        off = int(rng.integers(0, MAX_OFFSET + 1)) if mode == "staggered" else 0
        eng[u] = (off, t0)
    return eng


def window_rows(engines, eng, k, key="z", sensor=None, shift_fn=None):
    """Per-engine 3-reading means at shift k (engines with a full window); returns (rows, cycles covered)."""
    rows, cyc = [], []
    for u, (off, t0) in eng.items():
        c = k - off
        if M2.WINDOW_SHIFTS <= c <= t0:
            zz = engines[u][key][c - M2.WINDOW_SHIFTS:c]
            cyc.append(c)
            if shift_fn is None:
                rows.append(np.round(zz.mean(0), M2.DECIMALS))
            else:
                add = np.array([shift_fn(k - M2.WINDOW_SHIFTS + 1 + i) for i in range(M2.WINDOW_SHIFTS)])
                rows.append(np.round((zz[:, sensor] + add * engines[u]["scale"][c - M2.WINDOW_SHIFTS:c, sensor]).mean(), M2.DECIMALS))
    return rows, cyc


def clean_pvalues(fa: dict, models: dict, eng: dict) -> tuple[np.ndarray, np.ndarray]:
    """New test: unadjusted p-values (runs x 14), NaN rows when too few engines. Old Phase 7 test: flag per run."""
    E = fa["engines"]
    last = max(off + t0 for off, t0 in eng.values())
    P = np.full((last, len(SENSORS)), np.nan)
    old = np.full(last, np.nan)
    for k in range(1, last + 1):
        rows, cyc = window_rows(E, eng, k)
        if len(rows) < M2.MIN_ENGINES:
            continue
        R = np.vstack(rows)
        P[k - 1] = [M2.ks_vs_reference(R[:, j], *models["ref"][s], models["n_ref"])[1] for j, s in enumerate(SENSORS)]
        raw_rows, _ = window_rows(E, eng, k, key="raw")
        lo, hi = max(1, min(cyc) - M2.WINDOW_SHIFTS + 1), max(cyc)        # the cycle numbers the window covers
        ref = [(pre[min(hi, len(pre) - 1)] - pre[min(lo - 1, len(pre) - 1)]) / (min(hi, len(pre) - 1) - min(lo - 1, len(pre) - 1))
               for pre in fa["ref_prefix"].values() if len(pre) - 1 >= lo]
        if len(ref) >= M1.MIN_ENGINES:
            Rr, Rf = np.round(np.vstack(raw_rows), M1.MEAN_DECIMALS), np.round(np.vstack(ref), M1.MEAN_DECIMALS)
            p = [ks_2samp(Rr[:, j], Rf[:, j], method="asymp").pvalue for j in range(len(SENSORS))]
            old[k - 1] = float((M1.holm(p) < 0.01).any())
    return P, old


def adjusted(P: np.ndarray) -> np.ndarray:
    A = np.full_like(P, np.nan)
    for i in np.flatnonzero(~np.isnan(P).any(axis=1)):
        A[i] = M2.holm(P[i])
    return A


def decide(A: np.ndarray, a, k, mw, strong):
    """Vectorised monitoring2.drift_decision for every run: (tested per run, flagged per run and sensor).
    Untested runs count as 'no event' in the K-of-M memory, exactly as in drift_decision."""
    tested = ~np.isnan(A).any(axis=1)
    ev = np.where(tested[:, None], A < a, False)
    cs = np.vstack([np.zeros((1, ev.shape[1])), np.cumsum(ev, axis=0)])
    n = np.arange(1, len(ev) + 1)
    persistent = (cs[n] - cs[np.maximum(n - mw, 0)]) >= k
    sudden = np.where(tested[:, None], A < strong, False) & ~persistent
    return tested, (persistent | sudden) & tested[:, None]


def injected_power(pools, settings) -> dict:
    kinds = {"step +1 sd within 10": (lambda r, k0: 1.0 if r >= k0 else 0.0, DETECT_WITHIN),
             "ramp to +2 sd within 50": (lambda r, k0: 2.0 * min(1.0, (r - k0 + 1) / RAMP_LEN) if r >= k0 else 0.0, RAMP_LEN)}
    hits = {g: {kind: [] for kind in kinds} for g in settings}
    for pool in pools:
        E, models = pool["arrays"]["engines"], pool["models"]
        for eng, P0 in pool["clean"]["staggered"][:POWER_REPLICATES]:
            for j, s in enumerate(SENSORS):
                for k0 in ONSETS:
                    if k0 > len(P0):
                        continue
                    for kind, (fn, within) in kinds.items():
                        end = min(len(P0), k0 - 1 + within)
                        P = P0.copy()
                        for k in range(k0, end + 1):
                            if not np.isnan(P[k - 1]).any():
                                rows, _ = window_rows(E, eng, k, sensor=j, shift_fn=lambda r: fn(r, k0))
                                P[k - 1, j] = M2.ks_vs_reference(np.array(rows), *models["ref"][s], models["n_ref"])[1]
                        A, win = adjusted(P), slice(k0 - 1, end)
                        for g in settings:
                            t, f = decide(A, *g)
                            if t[win].any():
                                hits[g][kind].append(bool(f[win, j].any()))
    return {g: {**{kind: float(np.mean(v)) if v else 0.0 for kind, v in h.items()},
                **{f"n {kind}": len(v) for kind, v in h.items()}} for g, h in hits.items()}


# ------------------------------------------------------------------ calibration
def build_pools(data: dict) -> list:
    pools = []
    for ds, d in data.items():
        for h, (A, B) in enumerate(halves(d)):
            models = fit_models(d["tr"][d["tr"].unit.isin(A)])
            fleet = pd.concat([d["tr"][d["tr"].unit.isin(B)], d["val"]])
            arrays = fleet_arrays(d, models, fleet, d["tr"][d["tr"].unit.isin(A)])
            clean, old = {}, {}
            for mode in ("synchronized", "staggered"):
                clean[mode], flags_old = [], []
                for b in range(REPLICATES):
                    eng = streams(arrays["engines"], mode, b)
                    P, o = clean_pvalues(arrays, models, eng)
                    clean[mode].append((eng, P))
                    flags_old.append(o)
                o = np.concatenate(flags_old)
                old[mode] = float(np.nanmean(o)) if np.isfinite(o).any() else None
            pools.append({"ds": ds, "half": h, "models": models, "arrays": arrays, "clean": clean, "old_test_false_alarms": old,
                          "fleet_engines": len(arrays["engines"])})
            print(f"   {ds} half {h}: fleet of {len(arrays['engines'])} engines; old Phase 7 test false alarms {old}", flush=True)
    return pools


def calibrate_drift(pools) -> dict:
    grid = list(product(ALPHAS, KM, STRONG))
    fa = {}
    for (a, (k, mw), st) in grid:
        rates = {}
        for pool in pools:
            for mode, reps in pool["clean"].items():
                tested = flagged = 0
                for _, P in reps:
                    t, f = decide(adjusted(P), a, k, mw, st)
                    tested += int(t.sum())
                    flagged += int(f.any(axis=1).sum())
                rates[f"{pool['ds']} half {pool['half']} {mode}"] = flagged / max(tested, 1)
        fa[(a, k, mw, st)] = rates
    ok = [g for g, r in fa.items() if max(r.values()) <= BUDGET["drift"]]
    if not ok:
        raise SystemExit("no drift setting meets the budget: " + str({g: max(r.values()) for g, r in fa.items()}))
    power = injected_power(pools, ok)
    best = max(ok, key=lambda g: (power[g]["step +1 sd within 10"], power[g]["ramp to +2 sd within 50"], -g[0], g[1], -g[3]))
    label = lambda g: f"alpha {g[0]} K{g[1]}/M{g[2]} strong {g[3]:g}"
    return {"alpha": best[0], "k": best[1], "m": best[2], "alpha_strong": best[3], "window_shifts": M2.WINDOW_SHIFTS,
            "min_engines": M2.MIN_ENGINES, "false_alarm_chosen": fa[best], "candidates_within_budget": len(ok),
            "candidates": len(grid), "power_chosen": power[best],
            "old_phase7_test_false_alarms_on_same_fleets": {f"{p['ds']} half {p['half']}": p["old_test_false_alarms"] for p in pools},
            "false_alarm_table": {label(g): max(r.values()) for g, r in fa.items()},
            "power_table": {label(g): power[g] for g in ok},
            "fleets": f"cross-fitted: models on half A of the training engines, fleet = half B + validation "
                      f"({pools[0]['fleet_engines']} engines); {REPLICATES} replicates per half and mode; cut-off "
                      f"{CUT[0]:.0%}-{CUT[1]:.0%} of life; staggered offsets 0-{MAX_OFFSET}; injections in "
                      f"{POWER_REPLICATES} staggered replicates per half"}


def calibrate_health(data: dict, pools) -> dict:
    rng = np.random.default_rng(0)
    ratios, order_ratios, mp = {}, {}, {}
    for pool in pools:
        d, models = data[pool["ds"]], pool["models"]
        fleet_units = sorted(pool["arrays"]["engines"])
        f = pd.concat([d["tr"], d["val"]])
        f = f[f.unit.isin(fleet_units)].copy()
        f["life"] = f.groupby("unit").cycle.transform("max")
        f["true"] = f.life - f.cycle
        tl = f[test_like(f.cycle, f.true)]
        by = [g for _, g in tl.groupby("unit")]
        r_ = np.empty(N_BOOT)
        for b in range(N_BOOT):
            w = pd.concat([g.iloc[[rng.integers(len(g))]] for g in (by[i] for i in rng.integers(len(by), size=M2.HEALTH_WINDOW))])
            r_[b] = ((w.rhat - w.true.clip(upper=125)) ** 2).sum() / models["error"](w.life).sum()
        key = f"{pool['ds']} half {pool['half']}"
        ratios[key] = r_
        # windows in order of life (labels arrive shortest-lived first), one random test-like reading per engine
        lives = sorted(by, key=lambda g: (g.life.iloc[0], g.unit.iloc[0]))
        oc = []
        for b in range(200):
            pick = pd.concat([g.iloc[[rng.integers(len(g))]] for g in lives])
            err = ((pick.rhat - pick.true.clip(upper=125)) ** 2).to_numpy()
            exp = np.asarray(models["error"](pick.life))
            oc += [err[i:i + M2.HEALTH_WINDOW].sum() / exp[i:i + M2.HEALTH_WINDOW].sum()
                   for i in range(0, len(pick) - M2.HEALTH_WINDOW + 1)]
        order_ratios[key] = np.array(oc)
    for ds, d in data.items():                        # miss rate: validation engines only (they have rule-E bounds)
        by = [g for _, g in d["vtl"].groupby("unit")]
        p_ = np.empty(N_BOOT)
        for b in range(N_BOOT):
            w = pd.concat([g.iloc[[rng.integers(len(g))]] for g in (by[i] for i in rng.integers(len(by), size=M2.HEALTH_WINDOW))])
            p_[b] = binom.sf(int((w.RUL_true < w.low).sum()) - 1, M2.HEALTH_WINDOW, d["p0"])
        mp[ds] = p_
    fa_r = {r: {**{k: float(np.mean(v > r)) for k, v in ratios.items()},
                **{f"{k} (life order)": float(np.mean(v > r)) for k, v in order_ratios.items()}} for r in RATIOS}
    fa_m = {a: {ds: float(np.mean(v < a)) for ds, v in mp.items()} for a in MISS_ALPHAS}
    lim = next(r for r in RATIOS if max(fa_r[r].values()) <= BUDGET["error"])
    ma = max(a for a in MISS_ALPHAS if max(fa_m[a].values()) <= BUDGET["miss"])
    return {"error_ratio_limit": lim, "miss_alpha": ma, "min_labelled_engines": M2.HEALTH_WINDOW,
            "false_alarm_error_chosen": fa_r[lim], "false_alarm_miss_chosen": fa_m[ma],
            "error_ratio_median_in_life_order": {k: float(np.median(v)) for k, v in order_ratios.items()}}


def dataset_entry(d: dict) -> dict:
    return {"p0": d["p0"], "training_engines": int(d["tr"].unit.nunique()), "validation_engines": int(d["val"].unit.nunique())}


def main(argv=None) -> int:
    global VARIANT
    args = list(argv if argv is not None else sys.argv[1:])
    if "--variant" in args:
        i = args.index("--variant")
        VARIANT = args[i + 1]
        del args[i:i + 2]
    out_file = M2.RULE_FILES[VARIANT]
    if args:                                           # fresh datasets: frozen decision parameters, own values
        from pipeline.conditions import normalised_train
        rule = M2.load_rule(VARIANT)
        for ds in args:
            d = load_dataset(ds, f"models/val_predictions_{ds}.csv", f"models/oof_predictions_{ds}.csv", normalised_train(ds))
            rule["datasets"][ds] = dataset_entry(d)
            print(f"{ds}: p0 {d['p0']:.1%}")
        out_file.write_text(json.dumps(rule, indent=2), encoding="utf-8")
        return 0
    data = {ds: load_dataset(ds, VAL[ds], f"models/oof_predictions_{ds}.csv") for ds in VAL}
    pools = build_pools(data)
    d, h = calibrate_drift(pools), calibrate_health(data, pools)
    fa_total = 1 - (1 - max(d["false_alarm_chosen"].values())) * (1 - max(h["false_alarm_error_chosen"].values())) * \
        (1 - max(h["false_alarm_miss_chosen"].values()))
    note = ("chosen on training and validation engines of FD001 and FD003 only; no test file was read. Decision parameters "
            "are frozen for FD002/FD004; only the dataset-specific values are re-learned.")
    if VARIANT != "v2":
        note = ("POST HOC (v2.1): the spread model was changed after the pre-registered v2 rule failed on FD004 test data "
                "(negative spread fits). Chosen with the same validation-only procedure, but every test set had already been "
                "seen, so no test result with this rule is a confirmation. " + note)
    rule = {"variant": VARIANT, "pre_registered": note,
            "budgets": BUDGET, "drift": d, "health": h, "datasets": {ds: dataset_entry(x) for ds, x in data.items()},
            "combined_false_alarm_upper_estimate": fa_total}
    out_file.write_text(json.dumps(rule, indent=2, default=float), encoding="utf-8")
    print(f"drift: alpha {d['alpha']}, K{d['k']}/M{d['m']}, strong {d['alpha_strong']:g}; {d['candidates_within_budget']} of "
          f"{d['candidates']} settings within budget; new-test false alarms {d['false_alarm_chosen']}")
    print(f"   old Phase 7 test on the same fleets: {d['old_phase7_test_false_alarms_on_same_fleets']}")
    print(f"   power: {d['power_chosen']}")
    print(f"health: error-ratio limit {h['error_ratio_limit']} (FA {h['false_alarm_error_chosen']}), miss alpha {h['miss_alpha']} "
          f"(FA {h['false_alarm_miss_chosen']}); life-order median ratio {h['error_ratio_median_in_life_order']}")
    print(f"combined false-alarm estimate {fa_total:.2%}; wrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
