"""Phase 4 metric tests: known values worked out by hand."""
import math

import numpy as np
import pytest

from models.metrics import bootstrap_ci, mae, paired_bootstrap, phm_score, rmse


def test_rmse_and_mae_known_values():
    assert rmse([0, 0, 0, 0], [3, -3, 3, -3]) == 3.0
    assert mae([10, 20], [12, 16]) == 3.0
    assert rmse([5, 5], [5, 5]) == 0.0


def test_phm_matches_the_formula_by_hand():
    # 10 cycles late:  e^(10/10) - 1 = e - 1;  10 cycles early: e^(10/13) - 1
    assert phm_score([50], [60]) == pytest.approx(math.e - 1)
    assert phm_score([50], [40]) == pytest.approx(math.exp(10 / 13) - 1)
    assert phm_score([50, 50], [60, 40]) == pytest.approx(math.e - 1 + math.exp(10 / 13) - 1)


def test_phm_punishes_late_more_than_early():
    assert phm_score([50], [60]) > phm_score([50], [40])
    assert phm_score([30], [30]) == 0.0


def test_bootstrap_interval_contains_the_estimate_and_is_repeatable():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 125, 100)
    p = y + rng.normal(0, 15, 100)
    lo, hi = bootstrap_ci(y, p)
    assert lo < rmse(y, p) < hi
    assert (lo, hi) == bootstrap_ci(y, p)  # fixed seed -> same answer every run


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 125, 100)
    good, bad = y + rng.normal(0, 5, 100), y + rng.normal(0, 30, 100)
    out = paired_bootstrap(y, good, bad)
    assert out["diff"] < 0 and out["ci"][1] < 0 and out["a_wins"] > 0.99
