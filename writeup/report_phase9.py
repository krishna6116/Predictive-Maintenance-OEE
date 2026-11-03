"""Phase 9: render the write-up (README.md and docs/*) from writeup.facts; never hand-type a number.

Run:  python -m writeup.report_phase9            (writes every file; re-running gives byte-identical output)
      python -m writeup.report_phase9 --check    (exit 1 if any committed file differs from what would be written)
"""
import subprocess
import sys
from pathlib import Path

from features.build import RUL_CAP
from models.calibration import NEAR
from writeup.facts import ROOT, facts

APP = "https://predictive-maintenance-oee.streamlit.app"
REPO = "https://github.com/krishna6116/Predictive-Maintenance-OEE"
LINKEDIN = "https://www.linkedin.com/in/krishnakumar-balasundaram-3a92b6422"
GITHUB = "https://github.com/Krishna6116"
TAG = "v1.0.0"


# ---- formatting: every number in the write-up goes through one of these
def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def pts(x: float, d: int = 2, sign: bool = True) -> str:
    """A difference of two shares, in percentage points."""
    return f"{100 * x:+.{d}f}" if sign else f"{100 * abs(x):.{d}f}"


def cyc(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def n(x) -> str:
    return f"{int(x):,}"


def render_all() -> dict[str, str]:
    F = {k: f.value for k, f in facts().items()}
    return {"README.md": readme(F), "docs/RESULTS.md": results(F), "docs/CASE_STUDY.md": case_study(F),
            "docs/ARCHITECTURE.md": architecture(F), "docs/DECISIONS.md": decisions(F), "docs/GLOSSARY.md": glossary(),
            "docs/RELEASE_NOTES.md": release_notes(F), "docs/README.md": docs_index(), **notebooks(F)}


def notebook_summaries(F) -> dict[str, tuple[str, str, str]]:
    """notebook -> (question, answer with numbers from the facts, doc link)."""
    return {
        "01_eda": ("How do the engines wear out, and which sensors carry the signal?",
                   f"Most sensors stay flat for most of a life and drift only near failure, so remaining life is capped at "
                   f"{RUL_CAP} cycles for training and sensors with no signal are dropped.", "../docs/EDA_FINDINGS.md"),
        "03_features": ("Can features built only from past readings predict remaining life without leaking the future?",
                        f"Yes: on these features XGBoost reaches {cyc(F['rmse_xgb_fd001'])} cycles RMSE on FD001 (Phase 4), "
                        "with every split made by engine.", "../docs/FEATURES.md"),
        "04_baselines": ("How well do simple models predict remaining life?",
                         f"XGBoost reaches {cyc(F['rmse_xgb_fd001'])} cycles RMSE (95% interval {F['rmse_xgb_fd001_lo']:.1f}-"
                         f"{F['rmse_xgb_fd001_hi']:.1f}) against {F['rmse_age_fd001']:.1f} for an age-only rule and "
                         f"{F['rmse_fleet_fd001']:.1f} for the fleet average.", "../docs/MODEL_METRICS.md"),
        "05_deep": ("Do sequence models (LSTM, CNN) beat XGBoost?",
                    f"On FD001 the LSTM reaches {cyc(F['rmse_lstm_fd001'])} cycles RMSE against {cyc(F['rmse_xgb_fd001'])}, "
                    "but the paired, Holm-corrected gap is not significant, so XGBoost stayed published until Phase 5b.",
                    "../docs/MODEL_METRICS_PHASE5.md"),
        "05b_fd003": ("Does the LSTM's advantage hold on a fresh dataset?",
                      f"Yes: on FD003 it reaches {cyc(F['rmse_lstm_fd003'])} against {cyc(F['rmse_xgb_fd003'])}; pooled, it is "
                      f"better by {cyc(abs(F['h1_diff']))} cycles RMSE (Holm p {F['h1_p_holm']:.3f}), so it became the "
                      f"published model; its interval covers {pct(F['coverage_fd003'], 0)} of FD003 engines.",
                      "../docs/MODEL_METRICS_PHASE5B.md"),
        "06_maintenance": ("Which maintenance policy gives the highest OEE on a simulated plant?",
                           f"Predictive: simulated OEE {pct(F['oee_pred'])} vs {pct(F['oee_fixed'])} fixed interval vs "
                           f"{pct(F['oee_rtf'])} run to failure, with {n(F['failures_pred'])} / {n(F['failures_fixed'])} / "
                           f"{n(F['failures_rtf'])} simulated unplanned failures.", "../docs/MAINTENANCE_OEE.md"),
        "07_monitoring": ("Can the pipeline run itself every shift and say when to stop trusting the model?",
                          f"It ran {n(F['p7_runs'])} runs over {n(F['p7_shifts'])} shifts with retries and resume, but its "
                          f"drift test raised false alarms on {pct(F['p7_false_alarm'])} of clean runs against a "
                          f"{pct(F['p7_target'], 0)} target (not met, reported).", "../docs/MONITORING.md"),
        "07b_monitoring": ("Does a wear-adjusted drift rule fix that on data nothing has seen?",
                           f"On FD002 yes ({pct(F['c1_fd002_synchronized'])} / {pct(F['c1_fd002_staggered'])} clean runs "
                           f"flagged); on FD004 no ({pct(F['c1_fd004_synchronized'])} / {pct(F['c1_fd004_staggered'])} "
                           f"against a {pct(F['c1_limit'], 0)} limit), reported as a failure.", "../docs/MONITORING_7B.md"),
    }


def notebooks(F) -> dict[str, str]:
    """Each notebook: a tagged summary cell first and the 'Why we did these tasks' cell last. Markdown only, so no
    output changes and nothing is re-executed; applying it twice gives the same file."""
    import json

    def text(c) -> str:
        return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]

    out = {}
    for stem, (q, a, doc) in notebook_summaries(F).items():
        path = f"notebooks/{stem}.ipynb"
        # plain JSON, not nbformat: nbformat assigns random ids to cells without one, which breaks byte-identical output
        nb = json.loads((ROOT / path).read_text(encoding="utf-8"))
        cells = [c for c in nb["cells"] if "phase9-summary" not in c.get("metadata", {}).get("tags", [])]
        why = [c for c in cells if c["cell_type"] == "markdown" and "Why we did these tasks" in text(c)]
        if len(why) != 1:
            raise SystemExit(f"{path}: expected one 'Why we did these tasks' cell, found {len(why)}")
        lines = [f"**Question:** {q}  \n", f"**Answer:** {a}  \n", f"**Read more:** [{Path(doc).name}]({doc})"]
        summary = {"cell_type": "markdown", "metadata": {"tags": ["phase9-summary"]}, "source": lines}
        if any("id" in c for c in cells):
            summary = {"cell_type": "markdown", "id": "phase9-summary", "metadata": summary["metadata"], "source": lines}
        nb["cells"] = [summary] + [c for c in cells if c is not why[0]] + [why[0]]
        out[path] = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    return out


GENERATED = "<!-- Generated by `python -m writeup.report_phase9` from the results files; do not edit by hand. -->"


def misses(F) -> str:
    return (f"Phase 7's drift test raised false alarms on {pct(F['p7_false_alarm'])} of clean test runs against a "
            f"{pct(F['p7_target'], 0)} target (not met). Phase 7b's new rule failed its pre-registered criterion C1 on "
            f"FD004: {pct(F['c1_fd004_synchronized'])} and {pct(F['c1_fd004_staggered'])} of clean runs flagged "
            f"(synchronized and staggered fleets) against a {pct(F['c1_limit'], 0)} limit. Both are reported, not re-tuned.")


def limitations(F) -> list[str]:
    return [
        "**Simulated plant.** The OEE, downtime and failure figures come from a simulated plant driven by real NASA "
        "engine lives and real forecasts; they are not measured production data.",
        f"**Monitoring failed on FD004.** The wear-adjusted drift rule flagged {pct(F['c1_fd004_synchronized'])} and "
        f"{pct(F['c1_fd004_staggered'])} of clean FD004 runs against a {pct(F['c1_limit'], 0)} limit; the likely cause "
        "is a mix of two fault modes, which is not proven.",
        "**Unseen data needed.** A fault-mode-aware drift reference would need data no model has seen, such as NASA's "
        "N-CMAPSS, to confirm; all four C-MAPSS test sets are already used.",
        "**Censored labels.** Once maintenance acts, engines are serviced before they fail, so failure labels are "
        "censored and labels-only monitoring underestimates misses; a deployment needs inspection labels or a control group.",
        "**No cost data.** Impact is shown in counts and hours, not money: no real cost figures were available.",
        "**Simulation data.** C-MAPSS itself is simulated engine data; field data would add sensor noise, gaps and "
        "maintenance records this project could not test.",
    ]


def readme(F) -> str:
    L = [
        "# Predictive Maintenance & OEE", "",
        "Predicts when jet engines will need maintenance from their sensor data, then plans crew work so engines are "
        "serviced before they fail.", "",
        f"[![CI]({REPO}/actions/workflows/ci.yml/badge.svg)]({REPO}/actions/workflows/ci.yml) "
        "[![Python](https://img.shields.io/badge/python-3.14-blue)](https://www.python.org/) "
        f"[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) "
        f"[![Live app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]({APP})", "",
        "**TL;DR**", "",
        "- **Problem:** unplanned equipment failures stop production; fixed service schedules throw away useful life.",
        "- **Built end to end:** data pipeline into PostgreSQL, remaining-life forecasts with uncertainty, a crew "
        "scheduler, a shift-by-shift monitoring flow, and dashboards (Streamlit and Power BI), every result checked by "
        "independent code in CI.",
        f"- **Forecast:** the published {F['published_model']} model's error on unseen NASA engines (FD003) is "
        f"{cyc(F['rmse_lstm_fd003'])} cycles (RMSE) against {cyc(F['rmse_xgb_fd003'])} for XGBoost.",
        f"- **Simulated plant, 200 machines:** OEE {pct(F['oee_pred'])} with predictive maintenance vs "
        f"{pct(F['oee_fixed'])} fixed interval vs {pct(F['oee_rtf'])} run to failure; unplanned failures "
        f"{n(F['failures_pred'])} / {n(F['failures_fixed'])} / {n(F['failures_rtf'])} (simulated).",
        f"- **Live app:** {APP} (free hosting; if it shows a sleep page, click the wake-up button and wait about a minute)", "",
        "**Tech stack:** Python · PostgreSQL · pandas · scikit-learn · XGBoost · PyTorch (LSTM, 1D-CNN) · OR-Tools "
        "CP-SAT · Prefect · Streamlit · Power BI · pytest · GitHub Actions", "",
        "![The live app's Home page: headline results and a guide to each page](docs/figures/phase9/app_home.png)", "",
        "## Results at a glance", "",
        "| What | Result | Compared with |", "| --- | --- | --- |",
        f"| Remaining-life error, unseen FD003 engines (RMSE, cycles; lower is better) | **{F['published_model']} "
        f"{cyc(F['rmse_lstm_fd003'])}** | XGBoost {cyc(F['rmse_xgb_fd003'])}, age-only rule {cyc(F['rmse_age_fd003'])} |",
        f"| Error near failure (mean absolute error on engines near failure, see [glossary](docs/GLOSSARY.md); FD001 + FD003) | "
        f"**{cyc(F['pooled_nf_mae_lstm'])} cycles** | XGBoost {cyc(F['pooled_nf_mae_xgb'])} cycles |",
        f"| Published {pct(F['interval_nominal'], 0)} interval: share of FD003 engines whose true remaining life falls "
        f"inside | **{pct(F['coverage_fd003'], 0)}** | {pct(F['coverage_fd003_near'], 0)} of engines near failure |",
        f"| Simulated OEE: predictive / fixed interval / run to failure | **{pct(F['oee_pred'])}** / {pct(F['oee_fixed'])} "
        f"/ {pct(F['oee_rtf'])} | predictive minus fixed {pts(F['gap_fixed'])} percentage points (95% interval "
        f"{pts(F['gap_fixed_lo'])} to {pts(F['gap_fixed_hi'])}, simulated) |",
        f"| Simulated unplanned failures: predictive / fixed interval / run to failure | **{n(F['failures_pred'])}** / "
        f"{n(F['failures_fixed'])} / {n(F['failures_rtf'])} | every simulated machine fails under run to failure |", "",
        f"Every number on this page is generated from the results files and re-checked by independent code on every "
        f"push. Full results with intervals and sources: [docs/RESULTS.md](docs/RESULTS.md).", "",
        "## Business impact (simulated)", "",
        "What the maintenance policies mean in operations terms, on the same 200 simulated machines (counts and hours "
        "only; no real cost data was available):", "",
        "| Simulated outcome | Predictive | Fixed interval | Run to failure |", "| --- | --- | --- | --- |",
        f"| Unplanned failures (simulated) | **{n(F['failures_pred'])}** | {n(F['failures_fixed'])} | {n(F['failures_rtf'])} |",
        f"| Planned services (simulated) | {n(F['services_pred'])} | {n(F['services_fixed'])} | {n(F['services_rtf'])} |",
        f"| Downtime, hours (simulated) | **{n(F['downtime_pred'])}** | {n(F['downtime_fixed'])} | {n(F['downtime_rtf'])} |",
        f"| Useful life thrown away at early services, cycles (simulated) | **{n(F['wasted_pred'])}** | "
        f"{n(F['wasted_fixed'])} | {n(F['wasted_rtf'])} |", "",
        f"In the simulated plant, predictive maintenance avoids {n(F['failures_avoided_vs_fixed'])} unplanned failures "
        f"compared with a fixed service interval and {n(F['failures_avoided_vs_rtf'])} compared with run to failure, "
        f"while the fixed interval throws away {F['wasted_ratio']:.1f} times as much useful engine life. The crew plan "
        f"respects {F['crews_per_day']} crews a day: on FD001 it books {n(F['fd001_scheduled'])} of the "
        f"{n(F['fd001_flagged'])} flagged engines into the {F['horizon_days']}-day horizon.", "",
        "## How it works", "",
        "```mermaid",
        "flowchart LR",
        "    A[NASA C-MAPSS<br/>sensor readings] --> B[Ingest and validate<br/>bad rows quarantined]",
        "    B --> C[(PostgreSQL<br/>star schema)]",
        "    C --> D[Causal features<br/>and sensor windows]",
        "    D --> E[LSTM remaining-life forecast<br/>with 90% interval]",
        "    E --> F[Alert when the lower bound<br/>reaches 20 cycles]",
        "    F --> G[OR-Tools CP-SAT<br/>crew schedule]",
        "    G --> H[Simulated plant<br/>and OEE]",
        "    E --> I[Model health, drift test,<br/>retraining flag]",
        "    C --> J[Power BI report<br/>read-only login]",
        "    H --> K[Snapshots] --> L[Streamlit app<br/>Community Cloud]",
        "    I --> K",
        "    P{Prefect flow: one run per 8-hour shift} -.-> B",
        "    P -.-> E",
        "    P -.-> G",
        "    P -.-> I",
        "```", "",
        "Architecture, star schema and technology choices: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).", "",
        "## What makes this trustworthy", "",
        "- **Decided before seeing results:** model choices, statistical tests, monitoring rules and every phase's "
        "success criteria were committed before the test data was scored ([decision log](docs/DECISIONS.md)).",
        "- **Validation only:** models were chosen on validation engines; the test set only confirms.",
        "- **Fair comparisons:** every model is compared with simple baselines on the same engines, with paired "
        "bootstrap intervals and Holm correction for multiple tests.",
        "- **Confirmed on fresh data:** the switch from XGBoost to the LSTM was tested on a dataset (FD003) that nothing "
        "had seen, with every setting frozen.",
        "- **Independent verifiers in CI:** each phase has a separate script that recomputes its published numbers "
        "with different code; CI runs them on every push.",
        "- **Misses reported, not re-tuned:** " + misses(F), "",
        "## Skills demonstrated", "",
        "| Job function | What this project shows | Proof |", "| --- | --- | --- |",
        "| Data engineering | PostgreSQL star schema, validation with a quarantine table, idempotent run-ID loads | "
        "[pipeline/schema.sql](pipeline/schema.sql) · [pipeline/validate.py](pipeline/validate.py) · [pipeline/load.py](pipeline/load.py) |",
        "| Machine learning | XGBoost and Ridge baselines, PyTorch LSTM and 1D-CNN, Monte Carlo dropout uncertainty | "
        "[models/baselines.py](models/baselines.py) · [models/deep.py](models/deep.py) |",
        "| Statistics | Engine-level bootstrap, paired tests, Holm correction, KS drift tests | "
        "[models/metrics.py](models/metrics.py) · [models/run_fd003.py](models/run_fd003.py) · [pipeline/monitoring.py](pipeline/monitoring.py) |",
        "| Operations research | Crew scheduling with OR-Tools CP-SAT, optimality proved by a second method | "
        "[maintenance/schedule.py](maintenance/schedule.py) |",
        "| MLOps | Prefect shift flow with retries and resume, drift and retraining rules, CI verification | "
        "[pipeline/flow.py](pipeline/flow.py) · [pipeline/monitoring2.py](pipeline/monitoring2.py) · [.github/workflows/ci.yml](.github/workflows/ci.yml) |",
        "| BI and dashboards | Streamlit app on frozen snapshots, Power BI on the star schema, both checked against "
        "independent numbers | [app/streamlit_app.py](app/streamlit_app.py) · [docs/DASHBOARDS.md](docs/DASHBOARDS.md) |",
        "| Operations and quality | OEE (availability x performance x quality) and maintenance-policy comparison on a "
        "simulated plant | [maintenance/simulate.py](maintenance/simulate.py) · [docs/MAINTENANCE_OEE.md](docs/MAINTENANCE_OEE.md) |", "",
        "## Progress", "",
        "| Phase | Status | Read more |", "| --- | --- | --- |",
        *[f"| {name} | Done | {more} |" for name, more in PHASES], "",
        "## Repository map", "",
        "```text",
        *REPO_MAP,
        "```", "",
        "## Run locally", "",
        "Python 3.14 (the version CI uses). `requirements-lock.txt` pins every package to the versions behind the "
        "published results; the first line installs the smaller CPU-only PyTorch build.", "",
        "Windows (PowerShell):", "",
        "```powershell",
        *RUN_WINDOWS,
        "```", "",
        "macOS / Linux (bash):", "",
        "```bash",
        *RUN_BASH,
        "```", "",
        "The full pipeline (PostgreSQL, Prefect, dashboards) is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).", "",
        "## Data", "",
        "NASA C-MAPSS Turbofan Engine Degradation Simulation (A. Saxena and K. Goebel, NASA Prognostics Center of "
        "Excellence, 2008): four datasets, FD001-FD004, each with training engines run to failure and test engines that "
        "stop before failure. One command downloads them and checks every file against the SHA-256 checksums in "
        "`data/checksums.sha256`, so results are always reproduced on byte-identical data. Data files are not committed.", "",
        "```bash",
        "python -m pipeline.download_data",
        "```", "",
        "## Continuous verification", "",
        "Every push runs these GitHub Actions jobs ([ci.yml](.github/workflows/ci.yml)):", "",
        *CI_JOBS, "",
        "## Limitations and what I would do next", "",
        *[f"- {x}" for x in limitations(F)], "",
        "## How this was built", "",
        "I built this phase by phase from a study guide, with Claude (Anthropic's AI assistant) as my coding assistant. "
        "I chose the project, the phase plan and the success criteria for each phase, approved every pre-registration "
        "and publication decision, built the Power BI report, deployed the app and produced the manual evidence. "
        "Claude wrote most of the code, tests and documentation under my direction. Every published claim is re-checked "
        "by independent verifier code in CI, so it does not depend on trusting either of us.", "",
        "## Author", "",
        f"Krishnakumar Balasundaram · MS Business Analytics, UMass Amherst (Isenberg) · [LinkedIn]({LINKEDIN}) · "
        f"[GitHub]({GITHUB})", "",
        "## License", "",
        "MIT, see [LICENSE](LICENSE).", "",
    ]
    return "\n".join(L)


PHASES = [
    ("0. Setup: structure, tests, CI", "[CI workflow](.github/workflows/ci.yml)"),
    ("1. Exploratory data analysis", "[EDA findings](docs/EDA_FINDINGS.md) · [notebook](notebooks/01_eda.ipynb)"),
    ("2. Data pipeline: PostgreSQL star schema, validation and quarantine, idempotent loads", "[Pipeline](docs/PIPELINE.md)"),
    ("3. Feature engineering: engine-level split, causal features, incremental builds",
     "[Features](docs/FEATURES.md) · [notebook](notebooks/03_features.ipynb)"),
    ("4. Baseline models: fleet average, age-only, Ridge, XGBoost", "[Model metrics](docs/MODEL_METRICS.md) · [notebook](notebooks/04_baselines.ipynb)"),
    ("5. Deep learning: LSTM and 1D-CNN, Monte Carlo dropout intervals",
     "[Deep models](docs/MODEL_METRICS_PHASE5.md) · [notebook](notebooks/05_deep.ipynb)"),
    ("5b. Confirmation on a fresh dataset (FD003) with pre-registered tests",
     "[Confirmation](docs/MODEL_METRICS_PHASE5B.md) · [notebook](notebooks/05b_fd003.ipynb)"),
    ("6. Alerts, CP-SAT crew schedule, simulated plant and OEE, three-policy comparison",
     "[Maintenance and OEE](docs/MAINTENANCE_OEE.md) · [notebook](notebooks/06_maintenance.ipynb)"),
    ("7. Prefect shift flow, model health, drift and retraining rules", "[Monitoring](docs/MONITORING.md) · [notebook](notebooks/07_monitoring.ipynb)"),
    ("7b. Monitoring re-tested on unseen FD002/FD004 (one criterion failed on FD004, reported)",
     "[Phase 7b](docs/MONITORING_7B.md) · [notebook](notebooks/07b_monitoring.ipynb)"),
    ("8. Dashboards: deployed Streamlit app and Power BI report", f"[Dashboards](docs/DASHBOARDS.md) · [live app]({APP})"),
    ("9. Write-up: README, case study, results, architecture, decisions, release", "[Docs index](docs/README.md) · [case study](docs/CASE_STUDY.md)"),
]

REPO_MAP = [
    "app/            Streamlit app (reads frozen snapshots only)",
    "dashboards/     snapshots, Phase 8 verifier, Power BI theme and read-only login",
    "data/           NASA files (downloaded, not committed) and their checksums",
    "docs/           write-up, phase docs, figures (start at docs/README.md)",
    "features/       causal feature engineering and the engine-level split",
    "maintenance/    alerts, CP-SAT crew schedule, simulated plant and OEE",
    "models/         baselines, LSTM/CNN, calibration, results files",
    "notebooks/      one notebook per phase, built by scripts in the same folder",
    "pipeline/       ingest, validation, PostgreSQL loads, Prefect flow, monitoring",
    "results/        Phase 7 and 7b monitoring results",
    "scripts/        screenshot capture and the optional keep-awake visit",
    "tests/          pytest suite",
    "writeup/        Phase 9: facts, renderer, verifier and scorecard",
    ".github/        CI workflows",
]

RUN_WINDOWS = [
    "python -m venv .venv",
    r".\.venv\Scripts\Activate.ps1",
    "pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu",
    "pip install -r requirements-lock.txt",
    "pytest -q",
]

RUN_BASH = [
    "python -m venv .venv",
    "source .venv/bin/activate",
    "pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu",
    "pip install -r requirements-lock.txt",
    "pytest -q",
]

CI_JOBS = [
    "- **test**: installs the pinned packages and runs the test suite.",
    "- **verify**: downloads the pinned NASA data, runs the pipeline into a fresh PostgreSQL database, builds the "
    "features and runs the Phase 3, 4, 5, 5b and 6 verifiers plus the Phase 8 database checks.",
    "- **phase7**: replays shifts of the Prefect flow in a fresh database and runs the Phase 7 and 7b verifiers.",
    "- **dashboards**: installs only the app's pinned requirements on Python 3.12, 3.13 and 3.14 and runs the Phase 8 verifier.",
    "- **writeup**: re-renders the write-up, checks it is unchanged and runs the Phase 9 verifier (external links are "
    "checked in a non-blocking step).",
    "- **writeup-repro**: runs the macOS/Linux commands above, exactly as written, on a fresh machine.",
    "- **keep-awake** (separate workflow, every 6 hours): one visit to the public app so the free host does not put it to sleep.",
]


def results(F) -> str:
    rows = [
        ("Published model error, FD003 (fresh test set)", f"{F['published_model']} RMSE {cyc(F['rmse_lstm_fd003'])} cycles",
         f"XGBoost {cyc(F['rmse_xgb_fd003'])}, CNN {cyc(F['rmse_cnn_fd003'])}, Ridge {cyc(F['rmse_ridge_fd003'])}, "
         f"age-only {cyc(F['rmse_age_fd003'])}, fleet average {cyc(F['rmse_fleet_fd003'])}",
         "models/results_phase5b.json", "[MODEL_METRICS_PHASE5B.md](MODEL_METRICS_PHASE5B.md)"),
        ("Error on the development test set, FD001", f"LSTM RMSE {cyc(F['rmse_lstm_fd001'])} cycles",
         f"XGBoost {cyc(F['rmse_xgb_fd001'])} (not significant on FD001 alone)", "models/results_phase5.json, results_phase4.json",
         "[MODEL_METRICS_PHASE5.md](MODEL_METRICS_PHASE5.md)"),
        ("LSTM vs XGBoost, pooled FD001 + FD003 (primary pre-registered test)",
         f"LSTM better by {cyc(abs(F['h1_diff']))} cycles RMSE (95% interval {cyc(abs(F['h1_ci_hi']))} to "
         f"{cyc(abs(F['h1_ci_lo']))}, Holm p {F['h1_p_holm']:.3f})",
         f"pooled RMSE {cyc(F['pooled_rmse_lstm'])} vs {cyc(F['pooled_rmse_xgb'])}", "models/results_phase5b.json (tests.H1)",
         "[MODEL_METRICS_PHASE5B.md](MODEL_METRICS_PHASE5B.md)"),
        ("Error near failure, pooled (engines near failure, see [glossary](GLOSSARY.md))", f"LSTM {cyc(F['pooled_nf_mae_lstm'])} cycles mean absolute error",
         f"XGBoost {cyc(F['pooled_nf_mae_xgb'])}", "models/results_phase5b.json", "[MODEL_METRICS_PHASE5B.md](MODEL_METRICS_PHASE5B.md)"),
        (f"Published {pct(F['interval_nominal'], 0)} interval coverage", f"FD003 {pct(F['coverage_fd003'], 0)} "
         f"(confirmation); {pct(F['coverage_fd003_near'], 0)} near failure",
         f"FD001 {pct(F['coverage_fd001_post_hoc'], 0)} (post hoc: FD001 had been seen)", "models/results_phase5b.json (mc_dropout)",
         "[MODEL_METRICS_PHASE5B.md](MODEL_METRICS_PHASE5B.md)"),
        ("Simulated OEE, 200 machines pooled", f"predictive {pct(F['oee_pred'])}",
         f"fixed interval {pct(F['oee_fixed'])}, run to failure {pct(F['oee_rtf'])}; predictive minus fixed "
         f"{pts(F['gap_fixed'])} percentage points (95% interval {pts(F['gap_fixed_lo'])} to {pts(F['gap_fixed_hi'])}, simulated)",
         "maintenance/results_phase6.json", "[MAINTENANCE_OEE.md](MAINTENANCE_OEE.md)"),
        ("Simulated unplanned failures", f"predictive {n(F['failures_pred'])}",
         f"fixed interval {n(F['failures_fixed'])}, run to failure {n(F['failures_rtf'])} (simulated)",
         "maintenance/results_phase6.json", "[MAINTENANCE_OEE.md](MAINTENANCE_OEE.md)"),
        ("Simulated downtime", f"predictive {n(F['downtime_pred'])} hours",
         f"fixed interval {n(F['downtime_fixed'])} hours, run to failure {n(F['downtime_rtf'])} hours (simulated)",
         "maintenance/results_phase6.json", "[MAINTENANCE_OEE.md](MAINTENANCE_OEE.md)"),
        ("Simulated useful life thrown away at early services", f"predictive {n(F['wasted_pred'])} cycles",
         f"fixed interval {n(F['wasted_fixed'])} cycles, {F['wasted_ratio']:.1f} times as much (simulated)",
         "maintenance/results_phase6.json", "[MAINTENANCE_OEE.md](MAINTENANCE_OEE.md)"),
        ("Phase 7 drift false alarms on clean test runs (miss)", f"{pct(F['p7_false_alarm'])} of runs",
         f"target {pct(F['p7_target'], 0)}: not met", "results/phase7/results_phase7.json", "[MONITORING.md](MONITORING.md)"),
        ("Phase 7b criterion C1, clean-run flags on unseen data (FD004 miss)",
         f"FD004 {pct(F['c1_fd004_synchronized'])} / {pct(F['c1_fd004_staggered'])}: failed",
         f"FD002 {pct(F['c1_fd002_synchronized'])} / {pct(F['c1_fd002_staggered'])}: passed; limit {pct(F['c1_limit'], 0)}",
         "results/phase7b/checks.json", "[MONITORING_7B.md](MONITORING_7B.md)"),
        ("Remaining-life error on the fresh monitoring fleets", f"LSTM RMSE FD002 {cyc(F['rmse_lstm_fd002'])}, "
         f"FD004 {cyc(F['rmse_lstm_fd004'])} cycles", "same frozen recipe as FD001", "models/results_FD002.json, results_FD004.json",
         "[MONITORING_7B.md](MONITORING_7B.md)"),
    ]
    L = ["# Results", "", GENERATED, "",
         "Every headline result in one place: the value, what it is compared with, the file it is read from and the doc "
         "that explains it. Intervals are 95% bootstrap intervals over whole engines. OEE, downtime and failure figures "
         "come from a simulated plant.", "",
         "| Result | Value | Compared with | Source file | Explained in |", "| --- | --- | --- | --- | --- |",
         *[f"| {a} | {b} | {c} | `{d}` | {e} |" for a, b, c, d, e in rows], "",
         "## Reading the misses", "", misses(F), "",
         "## Technical highlights by phase", "",
         f"- **Baselines (Phase 4, FD001):** XGBoost reaches {cyc(F['rmse_xgb_fd001'])} RMSE (95% interval "
         f"{F['rmse_xgb_fd001_lo']:.1f}-{F['rmse_xgb_fd001_hi']:.1f}), against a fleet average ({F['rmse_fleet_fd001']:.1f}) "
         f"and an age-only rule ({F['rmse_age_fd001']:.1f}). Near-failure errors are in [MODEL_METRICS.md](MODEL_METRICS.md).", "",
         f"- **Deep models (Phase 5, FD001):** the LSTM reaches {cyc(F['rmse_lstm_fd001'])} RMSE (95% interval "
         f"{F['rmse_lstm_fd001_lo']:.1f}-{F['rmse_lstm_fd001_hi']:.1f}), better than XGBoost on paper but not significantly, "
         f"so XGBoost stayed published. Raw MC-dropout intervals covered only {pct(F['p5_raw_val_coverage'], 0)} of validation "
         f"truths at a nominal {pct(F['interval_nominal'], 0)}; one scale factor learned on validation brought test coverage "
         f"to {pct(F['p5_calibrated_test_coverage'], 0)}.", "",
         f"- **Confirmation (Phase 5b, FD003):** the LSTM reaches {cyc(F['rmse_lstm_fd003'])} RMSE against XGBoost's "
         f"{cyc(F['rmse_xgb_fd003'])}. Pooled over FD001 and FD003 its advantage is {cyc(abs(F['h1_diff']))} cycles RMSE "
         f"(95% interval {cyc(abs(F['h1_ci_hi']))}-{cyc(abs(F['h1_ci_lo']))}, Holm p {F['h1_p_holm']:.3f}) and "
         f"{cyc(F['pooled_nf_mae_lstm'])} vs {cyc(F['pooled_nf_mae_xgb'])} cycles near failure, so the LSTM is now the "
         f"published model. The two-factor interval rule covers {pct(F['coverage_fd003'], 0)} of FD003 test engines.", "",
         f"- **Maintenance (Phase 6, simulated plant):** the predictive policy reaches {pct(F['oee_pred'])} simulated OEE "
         f"with {n(F['failures_pred'])} unplanned failures, against {pct(F['oee_rtf'])} for run to failure "
         f"({n(F['failures_rtf'])} failures) and {pct(F['oee_fixed'])} for a fixed service interval "
         f"({n(F['failures_fixed'])} failures, and {F['wasted_ratio']:.1f} times as much useful life thrown away).", "",
         f"- **Monitoring (Phase 7):** drift is a KS test per sensor on per-engine means, Holm-corrected, alpha "
         f"{F['p7_alpha']:g}; retraining triggers when rolling RMSE exceeds {F['p7_rmse_multiplier']:.2f} x validation RMSE. "
         f"The drift test fired on {pct(F['p7_false_alarm'])} of clean test runs against a {pct(F['p7_target'], 0)} target.", "",
         f"- **Monitoring re-tested (Phase 7b):** clean-run flags {pct(F['c1_fd002_synchronized'])} / "
         f"{pct(F['c1_fd002_staggered'])} on FD002 and {pct(F['c1_fd004_synchronized'])} / {pct(F['c1_fd004_staggered'])} on "
         f"FD004 (synchronized / staggered fleets), against a {pct(F['c1_limit'], 0)} limit: passed on FD002, failed on FD004.", "",
         "Back to the [README](../README.md) · [case study](CASE_STUDY.md) · [docs index](README.md)", ""]
    return "\n".join(L)


def case_study(F) -> str:
    L = ["# Case study: predicting engine maintenance before failures stop production", "", GENERATED, "",
         "## Situation", "",
         "In maintenance-heavy operations (manufacturing lines, fleets, hospital equipment), an unplanned failure stops "
         "production and a fixed service calendar replaces parts that still had life left. This project tests, on NASA's "
         "turbofan engine data and a simulated plant, whether forecasting each machine's remaining life lets a planner do "
         "better than both. My background is mechanical engineering with production and quality-control experience, where "
         "this trade-off between downtime and wasted parts is a daily decision.", "",
         "## Task", "",
         "Build the whole chain an operations team would need: clean data, a forecast it can trust (with uncertainty), a "
         "maintenance plan that respects crew limits, monitoring that says when to stop trusting the forecast, and "
         "dashboards a manager can read, with every claim checked by independent code.", "",
         "## Action", "",
         "1. **Data (Phases 1-3):** loaded NASA's sensor readings into a PostgreSQL star schema with validation and a "
         "quarantine for bad rows; built features only from past readings; split by engine, never by row.",
         f"2. **Forecasts (Phases 4-5b):** compared simple baselines, XGBoost and deep models (LSTM, CNN) on the same "
         f"engines; chose on validation data only; then confirmed the switch to the {F['published_model']} on a fresh "
         "dataset with tests written down in advance.",
         f"3. **Decisions (Phase 6):** flag an engine when the lower edge of its {pct(F['interval_nominal'], 0)} interval "
         f"reaches {F['alert_threshold']} cycles, schedule it with an optimizer under {F['crews_per_day']} crews a day, and "
         "replay three maintenance policies on a simulated plant.",
         "4. **Operations (Phases 7-7b):** a Prefect flow runs every shift with retries and monitoring rules fixed before "
         "testing; re-tested on two datasets nothing had seen.",
         "5. **Dashboards and write-up (Phases 8-9):** a public Streamlit app and a Power BI report on the same frozen "
         "numbers, and this write-up generated from the results files.", "",
         "## Result", "",
         f"- Forecast error on unseen engines: {F['published_model']} {cyc(F['rmse_lstm_fd003'])} cycles RMSE vs "
         f"{cyc(F['rmse_xgb_fd003'])} for XGBoost; near failure {cyc(F['pooled_nf_mae_lstm'])} vs "
         f"{cyc(F['pooled_nf_mae_xgb'])} cycles mean absolute error (FD001 + FD003).",
         f"- Simulated plant of 200 machines: OEE {pct(F['oee_pred'])} with predictive maintenance vs "
         f"{pct(F['oee_fixed'])} fixed interval and {pct(F['oee_rtf'])} run to failure; {n(F['failures_pred'])} "
         f"unplanned failures vs {n(F['failures_fixed'])} and {n(F['failures_rtf'])} (simulated).",
         f"- In the same simulation, the fixed interval threw away {F['wasted_ratio']:.1f} times as much useful engine life "
         f"as predictive maintenance ({n(F['wasted_fixed'])} vs {n(F['wasted_pred'])} cycles, simulated).",
         "- **What did not work:** " + misses(F), "",
         "## Limitations", "",
         *[f"- {x}" for x in limitations(F)], "",
         "## What I learned", "",
         "- Writing the success criteria down before seeing results turned disappointing results into clear, honest ones, "
         "and once caught my own wrong first explanation.",
         "- A forecast is only useful with its uncertainty: alerting on the interval's lower edge, not the point "
         "forecast, is what keeps late failures rare.",
         "- Checks that run without the real environment miss real problems: the dashboards' dark-mode bug only showed "
         "on a phone.", "",
         f"More detail: [results](RESULTS.md) · [decisions](DECISIONS.md) · [architecture](ARCHITECTURE.md) · "
         f"[live app]({APP})", ""]
    return "\n".join(L)


def architecture(F) -> str:
    L = ["# Architecture", "", GENERATED, "",
         "## System", "",
         "```mermaid",
         "flowchart LR",
         "    subgraph Local machine",
         "      PG[(PostgreSQL 17<br/>star schema)]",
         "      PF[Prefect flow<br/>one run per shift]",
         "      PBI[Power BI Desktop<br/>read-only login]",
         "    end",
         "    subgraph GitHub",
         "      CI[GitHub Actions<br/>tests + verifiers]",
         "      SNAP[Frozen snapshots<br/>dashboards/snapshots]",
         "    end",
         "    subgraph Streamlit Community Cloud",
         "      APP[Streamlit app]",
         "    end",
         "    NASA[NASA C-MAPSS files] --> PF --> PG",
         "    PG --> PBI",
         "    PG --> SNAP --> APP",
         "    SNAP --> CI",
         "```", "",
         "## Star schema (PostgreSQL)", "",
         "| Table | Grain (one row per) | Key |", "| --- | --- | --- |",
         "| `pipeline_runs` | load or shift run | `run_id` |",
         "| `dim_engine` | engine in a dataset and split | `dataset, split, engine_id` |",
         "| `fact_sensor_reading` | engine and cycle | `dataset, split, engine_id, cycle` |",
         "| `quarantine` | rejected input row, with its reason | `id` |",
         "| `fact_features` | engine and cycle (causal features) | `dataset, split, engine_id, cycle` |",
         "| `fact_prediction` | forecast for an engine at a cycle in a run | `run_id, dataset, engine_id, cycle` |",
         "| `fact_alert` | maintenance decision for an engine in a run | `run_id, dataset, context, engine_id` |",
         "| `fact_oee` | simulated machine under one policy | `run_id, dataset, policy, machine_id` |",
         "| `fact_model_health` | monitoring run (shift) | `run_id` |",
         "| `fact_drift` | sensor drift test in a run | `run_id, sensor` |", "",
         "Definitions: [pipeline/schema.sql](../pipeline/schema.sql). Loads are idempotent: each run has a fixed run ID, so "
         "re-running it replaces its rows instead of duplicating them.", "",
         "## Data flow per shift", "",
         "1. Ingest the readings that arrived in the shift; validate them; quarantine bad rows with a reason.",
         "2. Load valid rows into `fact_sensor_reading` and build features for the new cycles.",
         f"3. Predict remaining life with the published model and its {pct(F['interval_nominal'], 0)} interval "
         f"({F['mc_passes']} Monte Carlo dropout passes).",
         f"4. Flag engines whose interval's lower bound is at or below {F['alert_threshold']} cycles and schedule them "
         f"with CP-SAT under {F['crews_per_day']} crews a day over a {F['horizon_days']}-day horizon.",
         "5. Score the model on engines whose failure is already known, test every sensor for drift, and set a retraining "
         "flag when the pre-registered rule fires.", "",
         "## What runs where", "",
         "| Component | Where | Notes |", "| --- | --- | --- |",
         "| PostgreSQL, Prefect, model training | local machine | `.env` holds the database login (never committed) |",
         "| Power BI report | local machine | read-only login `powerbi_ro`; the .pbip project is not committed |",
         "| Streamlit app | Streamlit Community Cloud | reads only the committed snapshots; no database or model weights |",
         "| Tests and verifiers | GitHub Actions | fresh PostgreSQL service container; pinned NASA data |", "",
         "## Why these technologies", "",
         "| Choice | Over | Why |", "| --- | --- | --- |",
         "| PostgreSQL star schema | flat CSV files | keys and foreign keys catch duplicates and orphans; Power BI reads it directly |",
         "| XGBoost and Ridge baselines | deep models only | a deep model must beat strong simple models on the same engines to earn its place |",
         "| PyTorch LSTM | transformer models | sequences are short; the LSTM is small, fast on a CPU and reproducible with one thread |",
         "| Monte Carlo dropout | ensembles | uncertainty from one trained model; calibrated on validation engines |",
         "| OR-Tools CP-SAT | greedy rules | proves the plan optimal under crew limits; a second method re-checks it |",
         "| Prefect | cron scripts | retries, fixed run IDs and resume after failure come built in |",
         "| Streamlit on frozen snapshots | a live database connection | free public hosting, no credentials in the cloud, numbers checkable by checksum |",
         "| Power BI | only Streamlit | the BI tool most hiring teams use, on the same star schema |", "",
         f"Settings that shape the system: remaining life is capped at {RUL_CAP} cycles (`features/build.py`), the LSTM "
         f"reads {F['window']}-cycle sensor windows, and every split is by engine.", "",
         "Back to the [README](../README.md) · [docs index](README.md)", ""]
    return "\n".join(L)


def decisions(F) -> str:
    rows = [
        ("2026-10-17", f"Cap remaining life at {RUL_CAP} cycles", "sensors are flat for most of a life and drift only near failure (EDA)",
         "no cap (asks the model to predict the unpredictable)", "8223226"),
        ("2026-10-17", "Split by engine, never by row", "rows from one engine are near-copies; a row split leaks the answer",
         "random row split", "8223226"),
        ("2026-10-18", "Choose models on validation only, test once", "validation near-failure error decided XGBoost over Ridge",
         "choosing on the test set", "b280d0b"),
        ("2026-10-18", f"Switch the published model from XGBoost to the {F['published_model']}",
         f"pre-registered pooled test: LSTM better by {cyc(abs(F['h1_diff']))} cycles RMSE, Holm p {F['h1_p_holm']:.3f}; "
         f"{F['tests_significant']} of {F['tests_total']} tests significant",
         "switching on FD001 alone (gap not significant there)", "e00eaac, da9eb3d"),
        ("2026-10-18", "Interval rule E: two scale factors switched on predicted remaining life",
         "chosen on FD001 validation before FD003 was touched", "one scale factor (82% FD003 coverage)", "db8bf1b"),
        ("2026-10-20", f"Alert on the interval's lower bound at {F['alert_threshold']} cycles",
         "acting on the pessimistic edge keeps late failures rare", "alert on the point forecast", "7e6761e"),
        ("2026-10-20", "Threshold 30 NOT adopted", "it looked better on the test replay, so adopting it would be tuning to the test set",
         "threshold 30", "7e6761e"),
        ("2026-10-23", "Retraining when rolling RMSE exceeds 1.45x validation RMSE",
         "1.3x raised too many false alarms on validation windows", "the guide's 30% rule", "cce5187"),
        ("2026-10-24", "Phase 7b criteria and rule v2 pre-registered", "Phase 7's drift test missed its target on clean data",
         "re-tuning Phase 7's rule on the test replay", "ce5a16e, b7b0cb2"),
        ("2026-10-24", "Rule v2.1 (log-scale spread) labelled post hoc", "fixed a real defect found after the results; FD004 still failed",
         "presenting it as pre-registered", "9cd577c"),
        ("2026-10-31", "Dashboards read frozen snapshots with a checksum", "the public app needs no database or credentials and is checkable",
         "a live database connection from the cloud", "8fe0ad9"),
        ("2026-11-03", "Write-up generated from results files and pre-registered", "no number can drift from its source",
         "hand-written docs", "b0a5b95"),
    ]
    L = ["# Decision log", "", GENERATED, "",
         "Each major decision with its date, the evidence, the alternative rejected, and the commit that holds it.", "",
         "| Date | Decision | Evidence | Rejected | Commit |", "| --- | --- | --- | --- | --- |",
         *[f"| {d} | {a} | {e} | {r} | `{c}` |" for d, a, e, r, c in rows], "",
         "Back to the [README](../README.md) · [docs index](README.md)", ""]
    return "\n".join(L)


def glossary() -> str:
    terms = [
        ("RUL (remaining useful life)", "How many more cycles an engine will run before it fails. Capped in training, "
         "because early in life it cannot be predicted from sensors."),
        ("OEE (overall equipment effectiveness)", "Availability x performance x quality: the share of planned time that "
         "produced good output. Here it comes from a simulated plant."),
        ("RMSE", "Root mean squared error: the typical size of a forecast's error, in cycles; large misses count more."),
        ("PHM08 score", "The scoring rule from NASA's 2008 prognostics challenge; it punishes late forecasts (predicting "
         "more life than is left) more than early ones."),
        ("MC dropout", "Monte Carlo dropout: running the network many times with random parts switched off to get a "
         "spread of forecasts, which gives an uncertainty interval."),
        ("Holm correction", "A way to adjust p-values when several tests are run together, so that running many tests does "
         "not make a chance result look significant."),
        ("Near failure", f"An engine with {NEAR} or fewer cycles of remaining life (`models/calibration.py`); errors there matter most, because that is when maintenance must be booked."),
        ("Bootstrap interval", "A range for a result found by re-sampling whole engines many times; here 95% intervals."),
        ("KS test", "Kolmogorov-Smirnov test: checks whether two sets of numbers come from the same distribution; used "
         "to detect sensor drift."),
        ("CP-SAT", "Google OR-Tools' constraint solver; it finds the best maintenance plan under crew limits and proves no "
         "better plan exists."),
        ("Pre-registration", "Writing the rules and success criteria down, and committing them, before seeing the results "
         "they will judge."),
        ("Post hoc", "Decided after seeing results; reported as such and never presented as pre-registered."),
        ("Simulated", "Produced by a model of a plant, not measured; every OEE, downtime or failure figure here is simulated."),
    ]
    L = ["# Glossary", "", GENERATED, "", *[f"- **{t}:** {d}" for t, d in terms], "",
         "Back to the [README](../README.md) · [docs index](README.md)", ""]
    return "\n".join(L)


def release_notes(F) -> str:
    L = [f"# Predictive Maintenance & OEE {TAG}", "", GENERATED, "",
         "First complete release: data pipeline, remaining-life forecasts with uncertainty, crew scheduling, a simulated "
         "plant, shift-by-shift monitoring, dashboards and the write-up, every number re-checked by independent code in CI.", "",
         "## Headline results", "",
         f"- {F['published_model']} remaining-life error on unseen NASA engines (FD003): {cyc(F['rmse_lstm_fd003'])} cycles RMSE "
         f"vs {cyc(F['rmse_xgb_fd003'])} for XGBoost.",
         f"- Simulated OEE, 200 machines: {pct(F['oee_pred'])} predictive vs {pct(F['oee_fixed'])} fixed interval vs "
         f"{pct(F['oee_rtf'])} run to failure; simulated unplanned failures {n(F['failures_pred'])} / "
         f"{n(F['failures_fixed'])} / {n(F['failures_rtf'])}.",
         "- Misses, reported as measured: " + misses(F), "",
         "## Links", "",
         f"- Live app: {APP}",
         f"- Case study: {REPO}/blob/main/docs/CASE_STUDY.md",
         f"- All results: {REPO}/blob/main/docs/RESULTS.md",
         f"- Docs index: {REPO}/blob/main/docs/README.md", ""]
    return "\n".join(L)


DOC_DESCRIPTIONS = {
    "docs/ARCHITECTURE.md": "System diagram, star schema, data flow per shift and why each technology was chosen.",
    "docs/CASE_STUDY.md": "One-page story of the project for a non-technical reader: situation, task, action, result.",
    "docs/DASHBOARDS.md": "Phase 8: the Streamlit app and Power BI report, their checks and the Phase 8 scorecard.",
    "docs/DECISIONS.md": "Decision log: each major choice with its evidence, the rejected alternative and its commit.",
    "docs/EDA_FINDINGS.md": "Phase 1: what the NASA sensor data looks like and which sensors were kept.",
    "docs/FEATURES.md": "Phase 3: the causal features, the engine-level split and their independent check.",
    "docs/GLOSSARY.md": "Plain-language definitions of the technical terms used across the project.",
    "docs/LEARNING_LOG.md": "Dated diary of every working session: what was done, found and learned.",
    "docs/MAINTENANCE_OEE.md": "Phase 6: alerts, the CP-SAT crew schedule and the simulated plant's policy comparison.",
    "docs/MODEL_METRICS.md": "Phase 4: baseline models on FD001 with intervals, comparisons and ablations.",
    "docs/MODEL_METRICS_PHASE5.md": "Phase 5: LSTM and CNN models and their Monte Carlo dropout intervals on FD001.",
    "docs/MODEL_METRICS_PHASE5B.md": "Phase 5b: the pre-registered confirmation on FD003 that made the LSTM the published model.",
    "docs/MONITORING.md": "Phase 7: the Prefect shift flow, model health and drift rules, and the missed false-alarm target.",
    "docs/MONITORING_7B.md": "Phase 7b: the re-tested monitoring rule on unseen FD002 and FD004, including the FD004 failure.",
    "docs/PHASE8_CHECKLIST.md": "Phase 8 manual evidence checklist (deployed checksum, phone screenshots, Power BI report).",
    "docs/PHASE9_CHECKLIST.md": "Phase 9 manual release checklist (redeploy, tag, repository settings, social preview).",
    "docs/PIPELINE.md": "Phase 2: ingest, validation, quarantine and idempotent loads into PostgreSQL.",
    "docs/RELEASE_NOTES.md": "Release notes for v1.0.0, generated from the results files.",
    "docs/RESULTS.md": "Every headline result with its interval, comparison, source file and explaining doc.",
}
NOTEBOOK_DESCRIPTIONS = {
    "01_eda": "Phase 1 exploratory analysis of the FD001 sensor data",
    "03_features": "Phase 3 feature engineering and its checks",
    "04_baselines": "Phase 4 baseline models, intervals and ablations",
    "05_deep": "Phase 5 LSTM and CNN models with Monte Carlo dropout intervals",
    "05b_fd003": "Phase 5b confirmation on the fresh FD003 dataset",
    "06_maintenance": "Phase 6 alerts, crew schedule and simulated OEE policies",
    "07_monitoring": "Phase 7 shift flow and monitoring results",
    "07b_monitoring": "Phase 7b monitoring re-tested on unseen data",
}


def tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return sorted(set(out.split()))


def docs_index() -> str:
    docs = sorted({p for p in tracked("docs") if p.count("/") == 1} | set(DOC_DESCRIPTIONS))
    docs = [p for p in docs if p != "docs/README.md"]
    missing = [p for p in docs if p not in DOC_DESCRIPTIONS]
    if missing:
        raise SystemExit(f"docs index: add a description for {missing}")
    L = ["# Documentation index", "", GENERATED, "",
         "Start with the [case study](CASE_STUDY.md) (5 minutes, non-technical) or the [results](RESULTS.md); the phase "
         "docs go deeper.", "", "## Documents", ""]
    L += [f"- [{Path(p).name}]({Path(p).name}): {DOC_DESCRIPTIONS[p]}" for p in docs]
    L += ["", "## Notebooks", "",
          "Each notebook is built by the `build_*.py` script beside it, so its findings come from the results files.", ""]
    for p in tracked("notebooks"):
        stem = Path(p).stem.removeprefix("build_")
        if stem not in NOTEBOOK_DESCRIPTIONS:
            raise SystemExit(f"docs index: add a notebook description for {p}")
        what = NOTEBOOK_DESCRIPTIONS[stem]
        text = f"{what} (notebook with outputs)." if p.endswith(".ipynb") else f"Script that builds the notebook: {what.lower()}."
        L.append(f"- [{Path(p).name}](../{p}): {text}")
    L += ["", "Back to the [README](../README.md)", ""]
    return "\n".join(L)


def main() -> int:
    files = render_all()
    if "--check" in sys.argv:
        stale = [p for p, text in files.items()
                 if not (ROOT / p).exists() or (ROOT / p).read_text(encoding="utf-8").replace("\r\n", "\n") != text]
        print("all generated files current" if not stale else f"stale: {stale}")
        return 1 if stale else 0
    for p, text in files.items():
        (ROOT / p).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
