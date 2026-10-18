"""Phase 5b: interval rule E, per-dataset files and the stratified bootstrap."""
import numpy as np

from features.build import SCALER_FILE, SPLIT_FILE, scaler_file, split_file
from models.calibration import NEAR, zone_interval, zone_scales
from models.calibration import test_like as is_test_like  # an imported name starting with test_ would be collected
from models.metrics import rmse
from models.run_fd003 import stratified_paired_bootstrap


def test_fd001_keeps_its_phase3_files_and_other_datasets_get_their_own():
    assert scaler_file() == SCALER_FILE and split_file("FD001") == SPLIT_FILE
    assert scaler_file("FD003").name == "scaler_FD003.json" and split_file("FD003").name == "split_FD003.json"


def test_test_like_uses_fd001_definition():
    assert is_test_like([30, 31, 31], [100, 145, 146]).tolist() == [False, True, False]


def test_zone_scales_hit_nominal_coverage_in_each_zone():
    rng = np.random.default_rng(0)
    n = 4000
    mean = np.r_[rng.uniform(30, NEAR, n), rng.uniform(NEAR + 1, 125, n)]
    std = np.full(2 * n, 2.0)
    noise = np.r_[rng.normal(0, 3 * 2.0, n), rng.normal(0, 1 * 2.0, n)]  # near zone is three times noisier; RUL stays > 0
    y = mean + noise
    k_near, k_far = zone_scales(y, mean, std, np.full(2 * n, 50), np.full(2 * n, 100))
    assert k_near > 2.5 * k_far  # the noisier zone gets the wider factor
    lo, hi = zone_interval(mean, std, k_near, k_far)
    covered = (y >= lo) & (y <= hi)
    assert abs(covered[:n].mean() - 0.9) < 0.02 and abs(covered[n:].mean() - 0.9) < 0.02


def test_zone_switches_on_predicted_not_true_rul_and_clips_at_zero():
    lo, hi = zone_interval([10.0, 80.0], [5.0, 5.0], k_near=3.0, k_far=1.0)
    assert lo.tolist() == [0.0, 75.0] and hi.tolist() == [25.0, 85.0]


def test_stratified_bootstrap_keeps_each_dataset_size_and_detects_a_real_gap():
    rng = np.random.default_rng(1)
    y1, y2 = rng.uniform(0, 125, 100), rng.uniform(0, 125, 100)
    good = lambda y: y + rng.normal(0, 5, len(y))
    bad = lambda y: y + rng.normal(0, 15, len(y))
    r = stratified_paired_bootstrap([(y1, good(y1), bad(y1)), (y2, good(y2), bad(y2))], metric=rmse, n=500)
    assert r["diff"] < 0 and r["ci"][1] < 0 and r["p"] <= 0.01
    same = stratified_paired_bootstrap([(y1, y1 + 1, y1 + 1)], metric=rmse, n=200)
    assert same["diff"] == 0 and same["ci"] == (0.0, 0.0)
