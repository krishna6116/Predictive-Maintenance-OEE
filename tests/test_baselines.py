"""Reproducibility settings of the Phase 4 models (fast; no data files needed)."""
import numpy as np
from xgboost import XGBRegressor

from models.baselines import SEEDS, XGB_FIXED


def test_xgboost_runs_on_one_fixed_thread():
    # Thread count changes XGBoost's trees, so it must not depend on the machine's core count
    assert XGB_FIXED["n_jobs"] == 1


def test_xgboost_same_seed_same_predictions():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(500, 8)), rng.normal(size=500)
    preds = [XGBRegressor(n_estimators=30, max_depth=3, random_state=SEEDS[0], **XGB_FIXED).fit(x, y).predict(x)
             for _ in range(2)]
    assert np.array_equal(preds[0], preds[1])


def test_xgboost_seed_matters():
    # Subsampling is random, so a different seed must give a different model (otherwise "5 seeds" means nothing)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(500, 8)), rng.normal(size=500)
    a, b = (XGBRegressor(n_estimators=30, max_depth=3, random_state=s, **XGB_FIXED).fit(x, y).predict(x)
            for s in SEEDS[:2])
    assert not np.array_equal(a, b)
