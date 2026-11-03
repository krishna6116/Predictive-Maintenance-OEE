"""Phase 9: every number the write-up publishes, read from the results files into named facts with unit and source.

Run:  python -m writeup.facts            (prints every fact with its source)
The renderer (writeup/report_phase9.py) takes numbers only from here; the verifier re-reads the files independently.
"""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every results file the write-up may draw on (the verifier's allowed values are the numeric leaves of these files)
SOURCES = [
    "models/results_phase4.json",
    "models/results_phase5.json",
    "models/results_phase5b.json",
    "models/results_FD002.json",
    "models/results_FD004.json",
    "maintenance/results_phase6.json",
    "results/phase7/results_phase7.json",
    "results/phase7b/checks.json",
    "dashboards/snapshots/facts.json",
    "dashboards/phase8_results.json",
]


@dataclass(frozen=True)
class Fact:
    value: object
    unit: str
    source: str  # file and key path, or 'derived: <formula>'


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def facts() -> dict[str, Fact]:
    p4, p5, p5b = _load(SOURCES[0]), _load(SOURCES[1]), _load(SOURCES[2])
    p6, p7, p7b, snap = _load(SOURCES[5]), _load(SOURCES[6]), _load(SOURCES[7]), _load(SOURCES[8])
    p8 = _load(SOURCES[9])
    F: dict[str, Fact] = {}

    def put(name, value, unit, rel, path):
        F[name] = Fact(value, unit, f"{rel}: {path}")

    # Remaining-life models (cycles); FD003 is the fresh confirmation set, FD001 the development set
    m5b = p5b["models"]
    for key, name in (("LSTM", "lstm"), ("CNN", "cnn"), ("XGBoost, all 168 features", "xgb"),
                      ("Ridge, all 168 features", "ridge"), ("Age only (cycle count)", "age"), ("Fleet average", "fleet")):
        put(f"rmse_{name}_fd003", m5b[key]["rmse"], "cycles", SOURCES[2], f"models.{key}.rmse")
    put("rmse_lstm_fd001", p5["models"]["LSTM"]["rmse"], "cycles", SOURCES[1], "models.LSTM.rmse")
    put("rmse_xgb_fd001", p4["models"]["XGBoost, all 168 features"]["rmse"], "cycles", SOURCES[0],
        "models.XGBoost, all 168 features.rmse")
    for key, name in (("LSTM", "lstm"), ("XGBoost, all 168 features", "xgb")):
        pooled = p5b["pooled_fd001_fd003"][key]
        put(f"pooled_rmse_{name}", pooled["rmse"], "cycles", SOURCES[2], f"pooled_fd001_fd003.{key}.rmse")
        put(f"pooled_nf_mae_{name}", pooled["near_failure_mae"], "cycles", SOURCES[2], f"pooled_fd001_fd003.{key}.near_failure_mae")
    h1 = p5b["tests"]["H1 LSTM vs XGBoost, RMSE, FD001+FD003"]
    put("h1_diff", h1["diff"], "cycles", SOURCES[2], "tests.H1.diff")
    put("h1_ci_lo", h1["ci"][0], "cycles", SOURCES[2], "tests.H1.ci[0]")
    put("h1_ci_hi", h1["ci"][1], "cycles", SOURCES[2], "tests.H1.ci[1]")
    put("h1_p_holm", h1["p_holm"], "p-value", SOURCES[2], "tests.H1.p_holm")
    put("tests_total", len(p5b["tests"]), "count", SOURCES[2], "tests (number of pre-registered tests)")
    put("tests_significant", sum(t["significant"] for t in p5b["tests"].values()), "count", SOURCES[2],
        "tests.*.significant (count)")
    put("published_model", p5b["verdict"]["published_winner"], "name", SOURCES[2], "verdict.published_winner")
    mc = p5b["mc_dropout"]
    put("coverage_fd003", mc["FD003"]["test_coverage"], "share", SOURCES[2], "mc_dropout.FD003.test_coverage")
    put("coverage_fd003_near", mc["FD003"]["test_coverage_near_failure"], "share", SOURCES[2],
        "mc_dropout.FD003.test_coverage_near_failure")
    put("coverage_fd001_post_hoc", mc["FD001_post_hoc"]["test_coverage"], "share", SOURCES[2], "mc_dropout.FD001_post_hoc.test_coverage")
    put("interval_nominal", snap["interval_nominal"]["value"], "share", SOURCES[8], "interval_nominal.value")
    put("mc_passes", mc["passes"], "count", SOURCES[2], "mc_dropout.passes")
    put("window", p5b["frozen_from_fd001"]["deep_settings"]["LSTM"]["window"], "cycles", SOURCES[2],
        "frozen_from_fd001.deep_settings.LSTM.window")

    # Decisions and the simulated plant (Phase 6, 200 machines pooled)
    st = p6["settings"]
    put("alert_threshold", st["alert_threshold"], "cycles", SOURCES[5], "settings.alert_threshold")
    put("crews_per_day", st["crews_per_day"], "crews", SOURCES[5], "settings.crews_per_day")
    put("horizon_days", st["horizon_days"], "days", SOURCES[5], "settings.horizon_days")
    put("hours_per_cycle", p6["plant"]["hours_per_cycle"], "hours (simulated)", SOURCES[5], "plant.hours_per_cycle")
    for pol, name in (("Predictive", "pred"), ("Fixed interval", "fixed"), ("Run to failure", "rtf")):
        r = p6["pooled"][pol]
        put(f"oee_{name}", r["oee"], "share (simulated)", SOURCES[5], f"pooled.{pol}.oee")
        put(f"failures_{name}", r["unplanned_failures"], "count (simulated)", SOURCES[5], f"pooled.{pol}.unplanned_failures")
        put(f"services_{name}", r["planned_services"], "count (simulated)", SOURCES[5], f"pooled.{pol}.planned_services")
        put(f"downtime_{name}", r["downtime_hours"], "hours (simulated)", SOURCES[5], f"pooled.{pol}.downtime_hours")
        put(f"wasted_{name}", r["wasted_cycles"], "cycles (simulated)", SOURCES[5], f"pooled.{pol}.wasted_cycles")
        put("machines", r["machines"], "count (simulated)", SOURCES[5], f"pooled.{pol}.machines")
    for gap, name in (("Predictive - Fixed interval", "fixed"), ("Predictive - Run to failure", "rtf")):
        g = p6["pooled_gaps"][gap]
        put(f"gap_{name}", g["diff"], "share (simulated)", SOURCES[5], f"pooled_gaps.{gap}.diff")
        put(f"gap_{name}_lo", g["ci"][0], "share (simulated)", SOURCES[5], f"pooled_gaps.{gap}.ci[0]")
        put(f"gap_{name}_hi", g["ci"][1], "share (simulated)", SOURCES[5], f"pooled_gaps.{gap}.ci[1]")
    # derived (each formula is pre-registered in writeup/phase9_spec.json number_rules.derived)
    F["failures_avoided_vs_fixed"] = Fact(F["failures_fixed"].value - F["failures_pred"].value, "count (simulated)",
                                          "derived: failures_fixed - failures_pred")
    F["failures_avoided_vs_rtf"] = Fact(F["failures_rtf"].value - F["failures_pred"].value, "count (simulated)",
                                        "derived: failures_rtf - failures_pred")
    F["wasted_ratio"] = Fact(F["wasted_fixed"].value / F["wasted_pred"].value, "ratio (simulated)",
                             "derived: wasted_fixed / wasted_pred")
    snap1 = p6["datasets"]["FD001"]["snapshot"]
    put("fd001_flagged", snap1["flagged"], "engines", SOURCES[5], "datasets.FD001.snapshot.flagged")
    put("fd001_scheduled", snap1["scheduled"], "engines", SOURCES[5], "datasets.FD001.snapshot.scheduled")

    # Monitoring (Phase 7 and 7b) and its two reported misses
    put("p7_runs", p7["runs"], "runs", SOURCES[6], "runs")
    put("p7_shifts", p7["shifts"], "shifts", SOURCES[6], "shifts")
    put("p7_false_alarm", p7["drift_false_alarm_rate_test"], "share", SOURCES[6], "drift_false_alarm_rate_test")
    put("p7_target", snap["p7_drift_target"]["value"], "share", SOURCES[8], "p7_drift_target.value")
    put("c1_limit", snap["c1_limit"]["value"], "share", SOURCES[8], "c1_limit.value")
    for ds in ("FD002", "FD004"):
        for fleet in ("synchronized", "staggered"):
            put(f"c1_{ds.lower()}_{fleet}", p7b["C1"][f"{ds} {fleet}"]["rate"], "share", SOURCES[7], f"C1.{ds} {fleet}.rate")
    put("rmse_lstm_fd002", _load(SOURCES[3])["models"]["LSTM"]["rmse"], "cycles", SOURCES[3], "models.LSTM.rmse")
    put("rmse_lstm_fd004", _load(SOURCES[4])["models"]["LSTM"]["rmse"], "cycles", SOURCES[4], "models.LSTM.rmse")

    # Dashboards (Phase 8)
    put("p8_score", p8["score"], "checks", SOURCES[9], "score")
    put("p8_out_of", p8["out_of"], "checks", SOURCES[9], "out_of")
    return F


def main() -> None:
    for name, f in facts().items():
        print(f"{name:28} {f.value!s:24} {f.unit:20} {f.source}")


if __name__ == "__main__":
    main()
