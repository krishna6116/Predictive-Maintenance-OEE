# Learning log

## 2026-10-17: Phase 0, setup
- Created the project structure, requirements, pytest config and GitHub Actions CI.
- Python 3.14 is used (already installed), not 3.12 as in the study guide.

## 2026-10-17: Data download
- Downloaded NASA C-MAPSS (FD001-FD004) into data/raw/ (git-ignored).
- FD001 check: train 20,631 rows and test 13,096 rows, 26 columns, 100 engines each, no missing values. Test RUL ranges 7-145 cycles.
