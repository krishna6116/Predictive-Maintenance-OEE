"""Scoring for remaining-useful-life predictions (Phase 4).

d = predicted RUL - true RUL. d > 0 is a LATE prediction (the engine fails sooner than predicted, the
dangerous case); d < 0 is EARLY (serviced too soon, wasting some life).
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in cycles."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(np.abs(d)))


def phm_score(y_true, y_pred) -> float:
    """NASA PHM08 scoring function (Saxena et al., 2008). Lower is better; late errors cost more."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def bootstrap_ci(y_true, y_pred, metric=rmse, n: int = 2000, seed: int = 0, level: float = 0.95):
    """Confidence interval for a metric by resampling engines with replacement."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    stats = np.array([metric(y_true[i], y_pred[i]) for i in idx])
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric=rmse, n: int = 2000, seed: int = 0) -> dict:
    """How often model A beats model B when the same engines are resampled.

    Returns the observed difference (A - B, negative = A better), its 95% interval, and the share of
    resamples in which A had the lower error. An interval that excludes 0 means the gap is not luck.
    """
    y, a, b = (np.asarray(v, float) for v in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = np.array([metric(y[i], a[i]) - metric(y[i], b[i]) for i in idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": metric(y, a) - metric(y, b), "ci": (float(lo), float(hi)), "a_wins": float((diffs < 0).mean())}
