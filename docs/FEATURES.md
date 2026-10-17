# Features (Phase 3)

Code: `features/build.py` (pure functions), `features/store.py` (database read/write), `features/run_features.py` (command line). Tests: `tests/test_features.py`. Walkthrough: `notebooks/03_features.ipynb`.

## Target

Remaining useful life (RUL) = last cycle - current cycle, **capped at 125**. Sensors are flat until about the last 100-150 cycles (`docs/EDA_FINDINGS.md`), so early life cannot be predicted. 39.4% of training readings sit at the cap.

## Split and scaling

- **Split by engine:** 80 training and 20 validation engines (`GroupShuffleSplit`, seed 42), saved in `features/split.json`. No engine appears in both. Median lifetime: 199 cycles (training), 194 (validation).
- **Scaling:** each of the 14 sensors is standardised to mean 0 and standard deviation 1, using **training engines only** (16,561 readings). Saved in `features/scaler.json`.

## Feature set: 14 sensors x 12 = 168 features

| Feature | Meaning |
| --- | --- |
| `sX` | scaled reading |
| `sX_mean{5,10,20}` | rolling mean over the last 5, 10, 20 cycles |
| `sX_std{5,10,20}` | rolling standard deviation |
| `sX_slope{5,10,20}` | least-squares slope against cycle |
| `sX_ema` | exponential moving average, span 10 |
| `sX_lowfreq` | FFT energy in frequency bins 1-3 over the last 30 cycles, divided by the window length (0 until 8 readings exist). Until an engine has 30 readings, the window is shorter, so the bins cover slightly faster cycles |

Every feature is **causal**: it uses only the engine's own readings up to the current cycle.

## How much signal each group carries

Absolute correlation with capped RUL, training engines:

| Group | Best | Median |
| --- | --- | --- |
| raw (scaled) | 0.771 | 0.690 |
| mean5 | 0.812 | 0.792 |
| mean10 | 0.811 | 0.803 |
| mean20 | 0.800 | 0.785 |
| ema | 0.811 | 0.803 |
| lowfreq (FFT) | 0.556 | 0.391 |
| std20 | 0.457 | 0.201 |
| slope20 | 0.294 | 0.184 |

- Smoothing helps: the median sensor rises from 0.69 (raw) to about 0.80 (10-cycle mean or EMA). Top single feature: `s4_mean5` (0.812).
- Slopes and spreads are weak on their own. They may still help models in combination; Phase 4 tests this.
- **Known limitation:** slopes are unreliable in an engine's first ~10 cycles (a line through 2-5 noisy points swings widely). Early cycles have RUL at the cap anyway, so the effect on training is small.
- FFT energy is weaker than smoothing. C-MAPSS gives one reading per cycle, not a high-frequency signal, so this is expected. Phase 4 measures whether removing it changes accuracy.

## Production use

`python -m features.run_features --run-id <run>` builds features only for the readings one pipeline run delivered, using each engine's full earlier history. On the real database, incremental features for test cycles 21-25 matched a full rebuild exactly (maximum difference 0.0), and rerunning the batch replaced its 500 rows without duplicating them.

`fact_features` (created on first write) holds one row per (dataset, split, engine, cycle) with the delivering `run_id` and the 168 features. A **primary key** on (dataset, split, engine_id, cycle) makes the database itself reject duplicates; a test insert of a duplicate row fails with `UniqueViolation`.

## Commands

```powershell
python -m features.run_features --fit            # split + scaler from FD001 training data
python -m features.run_features --build train    # all training features (about 2 minutes)
python -m features.run_features --build test
python -m features.run_features --run-id <run>   # incremental, one batch
```

## Independent verification

`python -m features.verify` recomputes everything from the raw NASA text files with a separate, deliberately simple implementation (explicit loops, two-pass statistics, `np.polyfit` slopes, a hand-written EMA, full FFT) and compares it with the database and the saved JSON files. Result on 2026-10-17: **25/25 checks passed**.

| Check | Result |
| --- | --- |
| Split reproduces from seed 42; 80 + 20 engines; no overlap | Pass |
| Scaler: 16,561 training-engine readings; means and sample std (ddof=1) match exactly; validation engines excluded | Pass |
| RUL ends at 0 for every engine, never above 125; 39.41% at the cap | Pass |
| fact_features: 168 double-precision columns, primary key, no duplicates, one row per clean reading (23,131), run IDs match their readings, no missing or infinite values | Pass |
| Train: 3,466,008 stored values vs recomputation | Max difference 1.9e-13 |
| Test: 420,000 stored values vs recomputation | Max difference 6.9e-15 |
| Every correlation quoted in this document and the notebook | Matches to 3 decimals |

Differences around 1e-13 are floating-point rounding (the pipeline uses rolling sums, the check uses direct formulas); the tolerance is 1e-9.

Tests: 27 pass, including 3 database tests for the feature store (rebuild is idempotent and keyed; incremental equals full rebuild in PostgreSQL; unknown run IDs are rejected).
