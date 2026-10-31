"""Phase 8 Step 1: freeze the rows the dashboards show into small CSVs with a manifest (dashboards/phase8_spec.json).

Run:  python -m dashboards.export_snapshots          (needs the local PostgreSQL database)
      python -m dashboards.export_snapshots --check  (rebuild in memory and compare with the committed files)

Database tables (public.*, ops, ops7b_*) are read in a read-only transaction; committed results files fill in what the
database does not hold (true RUL, model metrics, per-run drift). Output is byte-for-byte deterministic: fixed column
order, rows sorted by key, floats written round-trip exact, LF line endings, no timestamps. The manifest stores each
file's rows, sha256, key and run IDs, the sha256 of every source file, and one data_checksum over all snapshot files.
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

OUT = Path("dashboards/snapshots")
RUN6 = json.loads(Path("maintenance/results_phase6.json").read_text())["run_id"]
SCENARIOS = {"phase7_v1": "ops", "phase7b_v2_synchronized": "ops7b_fresh_synchronized",
             "phase7b_v2_staggered": "ops7b_fresh_staggered"}
SOURCES = ["maintenance/snapshot_decisions_phase6.csv", "maintenance/results_phase6.json", "maintenance/work_orders_phase6.csv",
           "maintenance/machines_phase6.csv", "maintenance/forecasts_rolling_FD001.csv",
           "maintenance/forecasts_rolling_FD003.csv", "models/results_phase4.json", "models/results_phase5.json",
           "models/results_phase5b.json", "models/predictions_phase4.csv", "models/predictions_phase5.csv",
           "models/predictions_phase5b.csv", "results/phase7/phase7_drift.csv", "results/phase7/phase7_health.csv",
           "results/phase7/results_phase7.json", "results/phase7b/fresh_synchronized/phase7_health.csv",
           "results/phase7b/fresh_staggered/phase7_health.csv", "results/phase7b/checks.json"]
# model display name -> (key in the results files, column in the predictions files)
MODELS = {"Fleet average": "Fleet average", "Age only": "Age only (cycle count)", "Ridge (168 features)": "Ridge, all 168 features",
          "XGBoost (168 features)": "XGBoost, all 168 features", "LSTM": "LSTM", "CNN": "CNN"}
KEYS = {"fleet.csv": ["dataset", "engine_id"], "models.csv": ["dataset", "model"], "plan.csv": ["dataset", "engine_id"],
        "work_orders.csv": ["dataset", "engine_id", "decision_cycle"], "oee_machines.csv": ["dataset", "policy", "machine_id"],
        "oee_policies.csv": ["scope", "policy"], "monitoring_runs.csv": ["scenario", "dataset", "shift"]}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def read_db(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), conn)


def fleet(conn) -> pd.DataFrame:
    """The published forecasts the Phase 6 plan acted on, plus the database's rolling forecast at the same cycle.
    The rolling forecasts used a fresh set of MC-dropout passes (maintenance/forecasts.py), so their MC mean and bounds
    differ from the published ones by Monte Carlo noise; point forecasts must be identical."""
    d = pd.read_csv("maintenance/snapshot_decisions_phase6.csv", float_precision="round_trip")
    out = d[["dataset", "engine_id", "cycle", "model_id", "rul_point", "rul_mc_mean", "rul_mc_std", "k", "rul_low", "rul_high",
             "true_rul_capped", "flagged", "risk", "outcome", "run_id"]].copy()
    p = read_db(conn, f"""SELECT DISTINCT ON (dataset, engine_id) dataset, engine_id, cycle, rul_point AS db_rul_point,
                                 rul_mc_mean AS rolling_mc_mean, rul_low AS rolling_rul_low, rul_high AS rolling_rul_high
                          FROM public.fact_prediction WHERE run_id = '{RUN6}' ORDER BY dataset, engine_id, cycle DESC""")
    out = out.merge(p, on=["dataset", "engine_id", "cycle"], how="left", validate="one_to_one")
    if not (out.db_rul_point == out.rul_point).all():
        raise RuntimeError("published point forecasts differ from public.fact_prediction at the last cycle")
    a = read_db(conn, f"SELECT dataset, engine_id, decision_cycle, rul_low_used FROM public.fact_alert "
                      f"WHERE run_id = '{RUN6}' AND context = 'snapshot'")
    chk = out[out.flagged].merge(a, on=["dataset", "engine_id"], how="outer", indicator=True)
    if not ((chk._merge == "both").all() and (chk.decision_cycle == chk.cycle).all() and (chk.rul_low_used == chk.rul_low).all()):
        raise RuntimeError("flagged engines differ from the snapshot alerts in public.fact_alert")
    out["inside_interval"] = (out.true_rul_capped >= out.rul_low) & (out.true_rul_capped <= out.rul_high)
    return out.drop(columns="db_rul_point")


def models() -> pd.DataFrame:
    res = {"FD001": {**json.loads(Path("models/results_phase4.json").read_text())["models"],
                     **json.loads(Path("models/results_phase5.json").read_text())["models"]},
           "FD003": json.loads(Path("models/results_phase5b.json").read_text())["models"]}
    preds = {"FD001": pd.read_csv("models/predictions_phase4.csv", float_precision="round_trip"),
             "FD003": pd.read_csv("models/predictions_phase5b.csv", float_precision="round_trip")}
    published = json.loads(Path("models/results_phase5b.json").read_text())["verdict"]["published_winner"]
    rows = []
    for ds, r in res.items():
        pr = preds[ds]
        near = pr.RUL <= 50
        for name, key in MODELS.items():
            nf = r[key].get("near_failure_mae")
            if nf is None:  # Phase 4 kept near-failure MAE in its report only; same rule as models/report_phase4.py
                nf = float(np.mean(np.abs(pr.RUL[near] - pr[key][near])))
            rows.append({"dataset": ds, "model": name, "published": name == published, "rmse": r[key]["rmse"],
                         "near_failure_mae": nf, "engines": len(pr), "near_failure_engines": int(near.sum())})
    return pd.DataFrame(rows)


def plan(conn) -> pd.DataFrame:
    return read_db(conn, f"""SELECT dataset, engine_id, decision_cycle, rul_low_used, risk, reason, scheduled_day, service_cycle,
                                    status, run_id FROM public.fact_alert WHERE run_id = '{RUN6}' AND context = 'snapshot'""")


def work_orders(conn) -> pd.DataFrame:
    return read_db(conn, f"""SELECT dataset, engine_id, decision_cycle, forecast_cycle, rul_low_used, risk, scheduled_day,
                                    service_cycle, status, run_id FROM public.fact_alert WHERE run_id = '{RUN6}' AND context <> 'snapshot'""")


def oee_machines(conn) -> pd.DataFrame:
    return read_db(conn, f"SELECT * FROM public.fact_oee WHERE run_id = '{RUN6}'")


def oee_policies(m: pd.DataFrame) -> pd.DataFrame:
    from maintenance.simulate import oee
    r6 = json.loads(Path("maintenance/results_phase6.json").read_text())
    t = m.rename(columns={"machine_id": "machine", "life_cycles": "life"})
    rows = []
    for scope, g in [("Pooled", t), *t.groupby("dataset")]:
        for pol, gp in g.groupby("policy"):
            rows.append({"scope": scope, "policy": pol, "kind": "policy", **oee(gp)})
    for gap, v in r6["pooled_gaps"].items():
        rows.append({"scope": "Pooled", "policy": gap, "kind": "gap", "oee_gap": v["diff"], "ci_low": v["ci"][0], "ci_high": v["ci"][1]})
    return pd.DataFrame(rows)


def monitoring(conn) -> pd.DataFrame:
    cols = "run_id, dataset, shift, n_labelled, n_scored, rmse, coverage_90, drifted_sensors, needs_retraining, retraining_reasons"
    parts = []
    for sc, schema in SCENARIOS.items():
        h = read_db(conn, f'SELECT {cols} FROM "{schema}".fact_model_health')
        h.insert(0, "scenario", sc)
        parts.append(h)
    out = pd.concat(parts, ignore_index=True)
    d = pd.read_csv("results/phase7/phase7_drift.csv")
    agg = d.groupby("run_id").agg(drift_tested=("tested", "any"), any_drifted=("drifted", "any")).reset_index()
    agg["any_drifted"] &= agg.drift_tested  # a sensor counts only in a run that was tested
    out = out.merge(agg.assign(scenario="phase7_v1"), on=["scenario", "run_id"], how="left", validate="one_to_one")
    for c in ("drift_tested", "any_drifted"):  # Phase 7b runs: not part of this column (spec), stays empty
        out[c] = out[c].astype("boolean")
    return out


def build(conn) -> dict[str, pd.DataFrame]:
    m = oee_machines(conn)
    frames = {"fleet.csv": fleet(conn), "models.csv": models(), "plan.csv": plan(conn), "work_orders.csv": work_orders(conn),
              "oee_machines.csv": m, "oee_policies.csv": oee_policies(m), "monitoring_runs.csv": monitoring(conn)}
    for name, df in frames.items():
        df = df.sort_values(KEYS[name], kind="mergesort").reset_index(drop=True)
        if df.duplicated(KEYS[name]).any():
            raise RuntimeError(f"{name}: key {KEYS[name]} is not unique")
        frames[name] = df
    return frames


def manifest(blobs: dict[str, bytes], frames: dict[str, pd.DataFrame]) -> dict:
    files = {}
    for name, b in blobs.items():
        ids = sorted(frames[name].run_id.dropna().unique()) if "run_id" in frames[name] else []
        files[name] = {"rows": len(frames[name]), "sha256": sha256(b), "key": KEYS[name], "run_ids_count": len(ids),
                       "run_ids": ids if len(ids) <= 5 else None, "run_ids_sha256": sha256("\n".join(ids).encode()) if ids else None}
    data = "".join(f"{n}:{files[n]['sha256']}\n" for n in sorted(files))
    return {"spec": "dashboards/phase8_spec.json", "builder": "python -m dashboards.export_snapshots",
            "phase6_run_id": RUN6, "files": files,
            "sources": {s: sha256(Path(s).read_bytes().replace(b"\r\n", b"\n")) for s in SOURCES},
            "data_checksum": sha256(data.encode())}


def export() -> dict[str, bytes]:
    from pipeline.db import get_engine
    with get_engine().connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        frames = build(conn)
    blobs = {n: to_bytes(df) for n, df in frames.items()}
    blobs["manifest.json"] = (json.dumps(manifest(blobs, frames), indent=2) + "\n").encode()
    return blobs


def main() -> int:
    blobs = export()
    if "--check" in sys.argv:
        bad = [n for n, b in blobs.items() if not (OUT / n).exists() or (OUT / n).read_bytes() != b]
        print("snapshots identical to the committed files" if not bad else f"DIFFERENT: {bad}")
        return 1 if bad else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, b in blobs.items():
        (OUT / n).write_bytes(b)
    man = json.loads(blobs["manifest.json"])
    total = sum(len(b) for b in blobs.values())
    for n, f in man["files"].items():
        print(f"{n:22s} {f['rows']:6d} rows  {f['sha256'][:12]}")
    print(f"total {total / 1e6:.2f} MB; data checksum {man['data_checksum'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
