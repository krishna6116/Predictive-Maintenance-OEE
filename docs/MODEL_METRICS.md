# Model metrics

All results: NASA C-MAPSS FD001, 100 official test engines, each scored at its last observed cycle. Produced by `python -m models.baselines`; checked independently by `python -m models.verify_baselines` (12/12 checks pass). Walkthrough: `notebooks/04_baselines.ipynb`.

**Protocol:** settings chosen on the 20 validation engines only; each model refit on all 100 training engines; test set scored once. True RUL is capped at 125 (the training convention); scores against NASA's uncapped values are in `models/results_phase4.json`. Intervals are 95% bootstrap intervals over test engines (2,000 resamples).

## Phase 4: baselines

| Model | Inputs | Validation RMSE | Test RMSE (95% interval) | MAE | PHM08 | Late predictions |
| --- | --- | --- | --- | --- | --- | --- |
| Fleet average | none | 41.72 | 41.94 (37.4-46.3) | 34.83 | 33,354 | 50% |
| Ridge regression | 14 raw sensors | 19.90 | 20.59 (17.8-23.3) | 16.37 | 1,224 | 58% |
| Ridge regression | 168 features | 16.27 | 16.28 (14.0-18.4) | 12.96 | 465 | 51% |
| **XGBoost (published baseline)** | **168 features** | **13.20** | **15.38 (12.4-18.2)** | **11.02** | **588** | **54%** |
| XGBoost | 154, no FFT | 13.59 | 15.87 (12.8-19.0) | 11.30 | 558 | 59% |
| XGBoost | 140, no EMA or FFT | 13.63 | 15.68 (12.8-18.6) | 11.36 | 523 | 59% |
| XGBoost | 144, without s9 and s14 | 13.90 | 15.51 (12.5-18.4) | 11.03 | 561 | 58% |
| XGBoost | 14 raw sensors | 16.82 | 17.36 (14.0-20.5) | 12.21 | 1,038 | 55% |

XGBoost settings: depth 5 (chosen on validation from 3, 5, 7: validation RMSE 13.61, 13.27, 13.45), learning rate 0.05, early stopping on validation. Across 5 random seeds, test RMSE for the main model is 15.17 +/- 0.20. Ridge's penalty barely matters: validation RMSE on raw sensors is 19.90 for every alpha from 0 to 100.

## Which differences are real (paired bootstrap, same 100 engines)

| A vs B | RMSE gap A - B (95% interval) | Verdict |
| --- | --- | --- |
| Ridge, 168 features vs Ridge, raw sensors | -4.31 (-6.65 to -2.04) | Features clearly help (PHM08 too) |
| XGBoost vs Ridge, raw sensors | -5.21 (-8.64 to -1.93) | XGBoost better |
| XGBoost vs Ridge, 168 features | -0.90 (-2.81 to +0.95) | No clear difference overall (Ridge has the better PHM08, also not significant) |
| XGBoost vs XGBoost without FFT | -0.50 (-1.79 to +0.57) | No clear difference |
| XGBoost vs XGBoost without EMA and FFT | -0.31 (-1.61 to +0.85) | No clear difference |
| XGBoost vs XGBoost without s9 and s14 | -0.13 (-1.44 to +1.07) | No clear difference |
| XGBoost vs XGBoost on raw sensors | -1.98 (-4.52 to +0.39) | Not significant on RMSE; XGBoost better on PHM08 (-450, interval -1,019 to -50) |

## Near failure: why XGBoost is the published baseline

Over all engines, XGBoost and Ridge (same features) tie. Near failure they do not:

| Engines with <= 50 cycles left (33) | MAE | Mean signed error (+ = late) | Late share |
| --- | --- | --- | --- |
| XGBoost, 168 features | 4.8 | +1.8 | 48% |
| Ridge, 168 features | 13.3 | +9.3 | 70% |

The MAE gap is -8.5 cycles (95% interval -10.9 to -6.3). Because the 50-cycle cut-off was chosen after looking at the data, it was re-tested at 25, 40, 60 and 75 cycles: XGBoost is clearly better at each (gaps of 5.9 to 8.5 cycles, every interval excluding 0). At 100 cycles, with mostly healthy engines included, the gap is no longer significant (-2.2, interval -4.6 to +0.2).

XGBoost's error by true RUL: 3.6 cycles MAE for 0-25 left, 6.5 for 26-50, 24.0 for 51-75, 15.5 for 76-100, 10.0 for 101-125.

## Conclusions

1. Feature engineering is the largest real gain (-4.3 RMSE for the same linear model).
2. FFT, EMA and s9/s14 have no measurable effect; they are kept and the result is stated.
3. Validation RMSE (13.2) is lower than test RMSE (15.4) because validation scores every cycle, many at the easy 125 cap, while the test scores one point per engine.
4. **Bar for Phase 5:** beat XGBoost's 15.38 RMSE and 588 PHM08 with a paired interval that excludes 0, and stay at least as accurate near failure.

Top XGBoost features by gain: s4_ema, s4_mean10, s15_mean10, s4_mean5, s11_mean10, s11_mean5, s2_mean10, s15_mean5, s3_ema, s9_mean5.
