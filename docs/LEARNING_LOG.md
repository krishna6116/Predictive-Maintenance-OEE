# Learning log

## 2026-10-17: Phase 0, setup
- Created the project structure, requirements, pytest config and GitHub Actions CI.
- Python 3.14 is used (already installed), not 3.12 as in the study guide.

## 2026-10-17: Data download
- Downloaded NASA C-MAPSS (FD001-FD004) into data/raw/ (git-ignored).
- FD001 check: train 20,631 rows and test 13,096 rows, 26 columns, 100 engines each, no missing values. Test RUL ranges 7-145 cycles.

## 2026-10-17: Raw data integrity check
- Added `pipeline/check_raw_data.py`: checks every subset for 26 values per line, numeric/finite values, engine IDs 1..N, no duplicate cycles, cycles 1, 2, 3 ... with no gaps, and one positive integer RUL per test engine.
- All files are byte-identical to NASA's archive and pass every check.
- Found an error in NASA's readme: FD004 has 249 train / 248 test engines (readme says 248 / 249). The RUL file has 248 values, which confirms the files are right.
