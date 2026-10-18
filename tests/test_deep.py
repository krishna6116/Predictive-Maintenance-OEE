"""Phase 5 model tests: determinism, MC dropout, shapes (tiny synthetic data, a few seconds)."""
import numpy as np
import pytest
import torch

from features.build import RUL_CAP
from models.deep import THREADS, build, mc_dropout, predict, train_early_stopping, train_fixed


def toy(n=300, window=10, features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, features)).astype(np.float32)
    y = np.clip(60 + 20 * X[:, -1, 0] + rng.normal(0, 2, n), 0, RUL_CAP)
    return X, y


CFG = {"arch": "LSTM", "hidden": 8, "dropout": 0.2, "lr": 3e-3}


@pytest.mark.parametrize("arch", ["LSTM", "CNN"])
def test_output_is_one_number_per_window(arch):
    m = build(arch, n_features=3, hidden=8, dropout=0.2, window=10)
    assert m(torch.zeros(5, 10, 3)).shape == (5,)


@pytest.mark.parametrize("arch", ["LSTM", "CNN"])
def test_same_seed_gives_identical_models(arch):
    X, y = toy()
    cfg = {**CFG, "arch": arch}
    a, _ = train_fixed(cfg, 0, X, y, epochs=2)
    b, _ = train_fixed(cfg, 0, X, y, epochs=2)
    assert np.array_equal(predict(a, X), predict(b, X))
    assert torch.get_num_threads() == THREADS == 1


def test_different_seed_gives_a_different_model():
    X, y = toy()
    a, _ = train_fixed(CFG, 0, X, y, epochs=2)
    b, _ = train_fixed(CFG, 1, X, y, epochs=2)
    assert not np.array_equal(predict(a, X), predict(b, X))


def test_early_stopping_returns_the_best_epoch():
    X, y = toy()
    m, info = train_early_stopping(CFG, 0, X[:200], y[:200], X[200:], y[200:], max_epochs=6, patience=2)
    assert info["val_rmse"] == pytest.approx(min(info["curve"]))
    assert info["best_epoch"] == int(np.argmin(info["curve"])) + 1
    # the returned model IS the best epoch's model
    assert np.sqrt(np.mean((predict(m, X[200:]) - y[200:]) ** 2)) == pytest.approx(info["val_rmse"])


def test_mc_dropout_passes_vary_point_prediction_does_not():
    X, y = toy()
    m, _ = train_fixed(CFG, 0, X, y, epochs=2)
    draws = mc_dropout(m, X, passes=20, seed=0)
    assert draws.shape == (20, len(X))
    assert draws.std(axis=0).mean() > 0                       # dropout on: passes differ
    assert np.array_equal(predict(m, X), predict(m, X))       # dropout off: identical
    assert not m.training                                     # model left in eval mode
    assert np.array_equal(draws, mc_dropout(m, X, passes=20, seed=0))  # MC draws are reproducible too
    assert draws.min() >= 0 and draws.max() <= RUL_CAP
