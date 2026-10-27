"""Phase 7b fault suite (pre-registered protocol in pipeline/phase7b_criteria.json).

Run:  python -m pipeline.fault_suite FD002 FD004          (after the staggered clean replay of those datasets)

A fault is fleet-wide (every engine reporting from the onset shift), in one sensor at a time, injected into the
staggered replay at onsets 25, 50, 75, 100, 125, 150. Running the whole flow for each of the 84 injections per fault
type would take days, so each injection MIRRORS the flow's steps from the onset on, starting from the clean replay:
  - each engine's history gets the fault ONCE (Mirror.prepare), so every step sees the same faulty readings;
  - the validator's range check decides what is quarantined; from an engine's first quarantined cycle on, it gets no
    new forecast (a gap in its history), so its wear estimate is its newest earlier valid forecast, aged;
  - for loaded readings whose 40-cycle window contains faulty readings, the deployed model RE-PREDICTS from the faulty
    inputs (fixed batches of 100, as in the flow), so the model can partly absorb a fault; other windows keep the
    clean replay's forecast;
  - the Phase 7b drift test and its K-of-M / sudden rule run with the history of earlier runs.
A detection is a flag whose reason names the injected sensor. Self-check: re-predicting clean windows gives exactly the
clean replay's forecasts. check_equivalence() compares the mirror with the real flow on an injected replay.
Faults (sd = the sensor's training sd, within its operating condition for FD002/FD004; pipeline.flow.sensor_sd):
  step +1 sd (C3: flagged within 10 shifts), ramp 0 -> +2 sd over 50 shifts (C4: flagged before it ends);
  reported without pass/fail: stuck (frozen at the engine's last pre-onset reading), noise x3 (Gaussian noise of
  2*sqrt(2) x the sensor's normal residual spread added, so total noise is about 3x), dropout (10% of readings missing:
  quarantined as unreadable, invisible to the drift test by design).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features.build import load_json
from pipeline import flow as F
from pipeline import monitoring2 as M2
from pipeline.conditions import CONDITION_DATASETS, normalise, stats_file
from pipeline.config import SENSORS
from pipeline.validate import load_limits

ONSETS = (25, 50, 75, 100, 125, 150)
CSV = {"float_precision": "round_trip"}
OUT = Path("results/phase7b")


class Fault:
    def __init__(self, kind: str, sensor: str, onset: int):
        self.kind, self.sensor, self.onset = kind, sensor, onset

    def amount(self, shift: np.ndarray) -> np.ndarray:
        on = shift >= self.onset
        if self.kind == "step":
            return np.where(on, 1.0, 0.0)
        if self.kind == "ramp":
            return np.where(on, 2.0 * np.minimum(1.0, (shift - self.onset + 1) / 50.0), 0.0)
        return np.zeros(len(shift))


class Mirror:
    def __init__(self, ds: str, clean_dir: Path, fleet: str = "staggered", variant: str = "v2"):
        self.ds, self.fleet, self.variant = ds, fleet, variant
        self.off = F.offsets(ds, fleet)
        t = F.test_file(ds)
        self.raw = {int(u): g.sort_values("cycle").reset_index(drop=True) for u, g in t.groupby("unit")}
        self.limits = load_limits(ds)
        p = pd.read_csv(clean_dir / "phase7_predictions.csv", **CSV)
        p = p[p.dataset == ds]
        self.pred = {int(u): g.set_index("cycle") for u, g in p.groupby("engine_id")}
        d = pd.read_csv(clean_dir / "phase7_drift.csv", **CSV)
        d = d[d.dataset == ds]
        self.hist = {int(s): ({r.sensor: r.p_holm for r in g.itertuples()} if g.tested.all() else {}) for s, g in d.groupby("shift")}
        self.models = F.v2_models(ds, variant)
        self.rule = M2.load_rule(variant)["drift"]
        self.window = json.loads(Path(F.SOURCES[ds]["results"]).read_text())["models"]["LSTM"]["settings"]["window"]
        self._model = None
        grid = M2._inputs(np.repeat(np.linspace(0, 125, 26), 15), np.tile(np.linspace(1, 300, 15), 26))
        self.noise_sd = {}                                        # normal residual spread, in training-sd units
        for s in SENSORS:
            spread = float(np.median(self.models["level"].sd(s, grid)))
            unit_sd = 1.0 if ds in CONDITION_DATASETS else float(F.training_engines(ds)[s].std())
            self.noise_sd[s] = spread / unit_sd

    def model(self):
        if self._model is None:
            self._model = F.model_for(self.ds)[0]
        return self._model

    def prepare(self, fault: Fault) -> None:
        """Apply the fault once to every engine's whole history; record each engine's first quarantined cycle."""
        s, rng = fault.sensor, np.random.default_rng(SENSORS.index(fault.sensor) * 1000 + fault.onset)
        lo, hi = self.limits[s]
        self.inj, self.first_q, self.first_fault = {}, {}, {}
        for u, g in self.raw.items():
            g = g.copy()
            g["shift"] = g.cycle + self.off[u]
            hit = g["shift"].to_numpy() >= fault.onset
            x = g[s].to_numpy(float).copy()
            unread = np.zeros(len(g), bool)
            sd = F.sensor_sd(self.ds, s, g["setting1"]).to_numpy(float)
            if fault.kind in ("step", "ramp"):
                x = x + fault.amount(g["shift"].to_numpy()) * sd
            elif fault.kind == "noise":
                x = np.where(hit, x + rng.normal(0, 2 * np.sqrt(2) * self.noise_sd[s], len(x)) * sd, x)
            elif fault.kind == "stuck":
                pre = np.flatnonzero(~hit)
                val = x[pre[-1]] if len(pre) else (x[np.flatnonzero(hit)[0]] if hit.any() else 0.0)
                x = np.where(hit, val, x)
            elif fault.kind == "dropout":
                unread = hit & (rng.random(len(x)) < 0.10)
            g[s] = x
            g["unreadable"] = unread
            g["quarantined"] = unread | (hit & ((x < lo) | (x > hi)))
            self.inj[u] = g
            q = np.flatnonzero(g.quarantined.to_numpy())
            self.first_q[u] = int(g.cycle.iloc[q[0]]) if len(q) else 10 ** 9
            f = np.flatnonzero(hit)
            self.first_fault[u] = int(g.cycle.iloc[f[0]]) if len(f) else 10 ** 9

    def forecast(self, u: int, c: int, cache: dict):
        """The point forecast the flow would hold for engine u at cycle c (None if invalid)."""
        if (u, c) in cache:
            return cache[(u, c)]
        if c >= self.first_q[u]:
            v = None                                              # gap in the engine's history
        elif c < self.first_fault[u]:
            pr = self.pred[u].loc[c] if c in self.pred[u].index else None
            v = float(pr.rul_point) if pr is not None and bool(pr.valid) else None
        else:
            v = self.repredict(u, [c])[c]
        cache[(u, c)] = v
        return v

    def repredict(self, u: int, cycles: list) -> dict:
        from models.deep import predict
        H = F.scale(self.ds, self.inj[u].iloc[:max(cycles)])[SENSORS].to_numpy(np.float32)
        X = np.stack([np.concatenate([np.repeat(H[:1], max(0, self.window - c), axis=0), H[max(0, c - self.window):c]])
                      for c in cycles]).astype(np.float32)
        pts = F._fixed_batches(lambda x: predict(self.model(), x), X)
        return {c: float(v) for c, v in zip(cycles, pts)}

    def warm(self, shifts, cache) -> None:
        """Re-predict, in fixed batches, every faulty window these shifts will need (much faster than one by one)."""
        for u, g in self.inj.items():
            need = [int(s - self.off[u]) for s in shifts if 1 <= s - self.off[u] <= len(g)]
            need = [c for c in need if self.first_fault[u] <= c < self.first_q[u] and (u, c) not in cache]
            if need:
                cache.update({(u, c): v for c, v in self.repredict(u, need).items()})

    def drift_at(self, k: int, cache: dict) -> dict:
        rows = []
        for u, g in self.inj.items():
            c = np.array([s - self.off[u] for s in range(max(1, k - M2.WINDOW_SHIFTS + 1), k + 1)])
            c = c[(c >= 1) & (c <= len(g))]
            if len(c):
                rows.append(g.iloc[c - 1])
        if not rows:
            return {}
        rows = pd.concat(rows, ignore_index=True)
        rows = rows[~rows.unreadable]
        rh = []
        for r in rows.itertuples():
            u, c = int(r.unit), int(r.cycle)
            fc = None if r.quarantined else self.forecast(u, c, cache)
            back = c
            while fc is None and back > 1:                        # newest earlier valid forecast, aged
                back -= 1
                f0 = self.forecast(u, back, cache)
                fc = None if f0 is None else max(0.0, f0 - (c - back))
            rh.append(fc)
        rows = rows.assign(rhat=rh).dropna(subset=["rhat"])
        if self.ds in CONDITION_DATASETS:
            rows = normalise(rows, load_json(stats_file(self.ds)))
        d = M2.drift(self.models["level"].z(rows), self.models["ref"], self.models["n_ref"])
        return dict(zip(d.sensor, d.p_holm)) if d.tested.all() else {}

    def run(self, fault: Fault, within: int) -> dict:
        self.prepare(fault)
        cache = {}
        self.warm(range(max(1, fault.onset - M2.WINDOW_SHIFTS), fault.onset + within), cache)
        hist = [self.hist.get(k, {}) for k in range(max(1, fault.onset - self.rule["m"] + 1), fault.onset)]
        tested, out = False, []
        for k in range(fault.onset, fault.onset + within):
            now = self.drift_at(k, cache)
            tested |= bool(now)
            hist.append(now)
            per, sud = M2.drift_decision(hist[-self.rule["m"]:], self.rule)
            out.append({"shift": k, "p_holm": now.get(fault.sensor), "flag": fault.sensor in per + sud})
            if fault.sensor in per + sud:
                return {"detected": True, "delay": k - fault.onset + 1, "tested": True,
                        "how": "persistent" if fault.sensor in per else "sudden", "trace": out}
        return {"detected": False, "delay": None, "tested": tested, "how": None, "trace": out}


def run_suite(ds: str, clean_dir: Path, kinds=("step", "ramp", "stuck", "noise", "dropout")) -> dict:
    m = Mirror(ds, clean_dir)
    out = {}
    for kind in kinds:
        within = 50 if kind == "ramp" else 10
        res = []
        for s in SENSORS:
            for k0 in ONSETS:
                r = m.run(Fault(kind, s, k0), within)
                res.append({"sensor": s, "onset": k0, **{k: v for k, v in r.items() if k != "trace"}})
        t = pd.DataFrame(res)
        tt = t[t.tested]
        out[kind] = {"injections": len(t), "testable": int(len(tt)), "detected": int(tt.detected.sum()),
                     "rate": float(tt.detected.mean()) if len(tt) else None,
                     "median_delay": float(tt[tt.detected].delay.median()) if tt.detected.any() else None,
                     "by_sensor": {k: float(v) for k, v in tt.groupby("sensor").detected.mean().items()}, "rows": res}
        print(f"{ds} {kind}: {out[kind]['detected']}/{out[kind]['testable']} detected "
              f"({(out[kind]['rate'] or 0):.0%}), median delay {out[kind]['median_delay']}", flush=True)
    return out


def self_check(ds: str, clean_dir: Path) -> float:
    """With no fault, re-predicted windows must equal the clean replay's forecasts exactly."""
    m = Mirror(ds, clean_dir)
    m.prepare(Fault("step", SENSORS[0], 10 ** 9))                 # never starts
    rng = np.random.default_rng(0)
    worst = 0.0
    for u in rng.choice(sorted(m.raw), size=min(10, len(m.raw)), replace=False):
        u = int(u)
        cyc = sorted(rng.choice(np.arange(1, len(m.raw[u]) + 1), size=min(20, len(m.raw[u])), replace=False).tolist())
        got = m.repredict(u, cyc)
        worst = max(worst, max(abs(got[c] - float(m.pred[u].loc[c].rul_point)) for c in cyc))
    return worst


def check_equivalence(ds: str = "FD001", sensor: str = "s11", onset: int = 60, within: int = 10) -> dict:
    """Run the REAL flow (published model, staggered, Phase 7b rule) with a +1 sd step from `onset` in a throwaway
    schema, and require the mirror to give the same adjusted p-value for that sensor and the same flag every shift."""
    import uuid
    from prefect.testing.utilities import prefect_test_harness
    from sqlalchemy import text
    from pipeline.db import get_engine
    schema, engine = f"faulteq_{uuid.uuid4().hex[:6]}", get_engine()
    inject = {"dataset": ds, "sensor": sensor, "shifts": list(range(onset, onset + within)),
              "sd_by_shift": {str(k): 1.0 for k in range(onset, onset + within)}}
    try:
        with prefect_test_harness():
            for k in range(1, onset + within):
                F.shift_pipeline(shift=k, predictor="model", db_schema=schema, datasets=[ds], fleet="staggered",
                                 monitor="v2", inject=inject if k >= onset else None)
        with engine.begin() as c:
            flow = pd.read_sql(text(f'SELECT shift, p_holm, drifted FROM "{schema}".fact_drift WHERE sensor = :s '
                                    'AND shift >= :o ORDER BY shift'), c, params={"s": sensor, "o": onset})
    finally:
        with engine.begin() as c:
            c.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    m = Mirror(ds, OUT / "posthoc_staggered")
    r = m.run(Fault("step", sensor, onset), within)
    tr = pd.DataFrame(r["trace"])
    n = len(tr)
    same_flag = flow.drifted.tolist()[:n] == tr.flag.tolist()
    gap = float(np.nanmax(np.abs(flow.p_holm.to_numpy(float)[:n] - tr.p_holm.to_numpy(float)))) if n else 0.0
    return {"dataset": ds, "sensor": sensor, "onset": onset, "shifts_compared": n, "same_flags": same_flag,
            "max_p_gap": gap, "mirror_detected": r["detected"], "flow_flags": flow.drifted.tolist()[:n]}


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["--equivalence"]:
        eq = check_equivalence()
        print(eq)
        path = OUT / "fault_suite.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        res["mirror_equivalence"] = eq
        path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        return 0 if eq["same_flags"] and eq["max_p_gap"] < 1e-9 else 1
    path = OUT / "fault_suite.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for ds in args:
        clean = OUT / ("fresh_staggered" if ds in CONDITION_DATASETS else "posthoc_staggered")
        diff = self_check(ds, clean)
        print(f"{ds}: mirror re-predictions vs clean replay, max diff {diff:.1e}", flush=True)
        res[ds] = {"mirror_self_check_max_diff": diff, **run_suite(ds, clean)}
        path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
