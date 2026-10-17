# EDA findings: FD001

Source: `notebooks/01_eda.ipynb` (run on 2026-10-17). Figures are in `docs/figures/`.

## The data

- 100 training engines (20,631 rows) and 100 test engines (13,096 rows), one row per engine per cycle, no missing values.
- Training engines last **128 to 362 cycles** (median 199, mean 206.3, standard deviation 46.3). Lifetimes vary almost 3x, so a single fleet-average prediction will be far off for many engines.

## Inputs dropped, and why

| Columns | Reason |
| --- | --- |
| s1, s5, s10, s16, s18, s19, setting3 | One value across all 20,631 rows; no information |
| s6 | Only two distinct values; effectively constant |
| setting1, setting2 | FD001 has one operating condition; their small variation is noise. Kept only for data checks |

**14 sensors kept:** s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.

## How the sensors behave

- Most kept sensors are **flat for much of an engine's life and drift in roughly the last 100 to 150 cycles**. This supports capping remaining useful life at 125 cycles in Phase 3: early in life there is no wear signal to predict from.
- Sensors that **rise** toward failure: s2, s3, s4, s8, s11, s13, s15, s17. Sensors that **fall**: s7, s12, s20, s21.
- **s9 and s14 are engine-specific:** near failure they rise for some engines and fall for others. This explains their weak correlation with remaining life (-0.39 and -0.31). They are also near-duplicates of each other (correlation 0.963).
- **s17 is stepped:** it takes whole-number values (390 to 399), so rolling averages will help more for it than for the smooth sensors.

## Link to remaining life (correlation with uncapped RUL, training data)

| Sensor | Correlation | | Sensor | Correlation |
| --- | --- | --- | --- | --- |
| s11 | -0.696 | | s2 | -0.606 |
| s4 | -0.679 | | s17 | -0.606 |
| s12 | +0.672 | | s3 | -0.585 |
| s7 | +0.657 | | s8 | -0.564 |
| s15 | -0.643 | | s13 | -0.563 |
| s21 | +0.636 | | s9 | -0.390 |
| s20 | +0.629 | | s14 | -0.307 |

## Test set

- Test engines are observed for **31 to 303 cycles** (median 133.5), so some engines give the model only a short history. Sliding windows of about 30 cycles (Phase 5) fit every test engine.
- True remaining life ranges from **7 to 145 cycles** (median 86). **11% of test engines have true RUL above 125**, so the cap convention for the test labels must be stated with the results.

## Decisions for the next phases

1. Model on the 14 kept sensors; drop the rest.
2. Cap the training target at 125 cycles, and report test results with the cap convention stated.
3. Keep s9 and s14 for now, but check in Phase 4 (feature importance and an ablation) whether dropping one or both changes accuracy.
4. Use rolling-window features to smooth noise, especially for the stepped s17.
5. Split training and validation by engine, never by row.
