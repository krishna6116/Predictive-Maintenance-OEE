# Predictive Maintenance & OEE

Predicts turbofan engine failures from sensor data, schedules maintenance under crew limits, and shows the effect on OEE.

> Work in progress. Phase 0 (setup) complete.

## Data

NASA C-MAPSS Turbofan Engine Degradation Simulation (A. Saxena and K. Goebel, NASA Prognostics Center of Excellence, 2008). Download the zip from NASA's PCoE data repository mirror:

```powershell
curl.exe -L -o cmapss.zip "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
```

It contains `CMAPSSData.zip`; extract the `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` files into `data/raw/` (not committed). FD001: 100 training engines (20,631 rows), 100 test engines (13,096 rows), 26 columns.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```
