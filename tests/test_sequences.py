"""Phase 5 window tests: shape, causality, padding, no mixing between engines (no data files needed)."""
import numpy as np
import pandas as pd
import pytest

from models.sequences import make_windows

COLS = ["a", "b"]


def engines(lengths, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"unit": u, "cycle": c, "a": rng.normal(), "b": 1000 * u + c, "RUL": n - c}
            for u, n in lengths.items() for c in range(1, n + 1)]
    return pd.DataFrame(rows)


def test_one_window_per_cycle_with_the_right_shape():
    X, idx = make_windows(engines({1: 12, 2: 7}), length=5, cols=COLS)
    assert X.shape == (19, 5, 2) and X.dtype == np.float32
    assert list(idx.cycle) == list(range(1, 13)) + list(range(1, 8))


def test_window_ends_at_the_current_cycle():
    df = engines({1: 12})
    X, idx = make_windows(df, length=5, cols=COLS)
    t = 9  # window for cycle 9 is cycles 5..9, in order
    assert list(X[idx.cycle.tolist().index(t), :, 1]) == [1000 + c for c in range(5, 10)]


def test_short_history_is_left_padded_with_the_first_reading():
    X, idx = make_windows(engines({1: 12}), length=5, cols=COLS)
    first = X[0]                                   # cycle 1: four copies of cycle 1, then cycle 1
    assert (first[:, 1] == 1001).all()
    third = X[2]                                   # cycle 3: [1, 1, 1, 2, 3]
    assert list(third[:, 1]) == [1001, 1001, 1001, 1002, 1003]


def test_future_readings_never_enter_a_window():
    df = engines({1: 15})
    X, _ = make_windows(df, length=6, cols=COLS)
    changed = df.copy()
    changed.loc[changed.cycle > 8, COLS] += 999.0   # rewrite the future
    X2, _ = make_windows(changed, length=6, cols=COLS)
    assert np.array_equal(X[:8], X2[:8])            # windows ending at cycles 1-8 are untouched
    assert not np.array_equal(X[8:], X2[8:])


def test_windows_never_mix_engines():
    X, idx = make_windows(engines({1: 4, 2: 4}), length=6, cols=COLS)
    for w, unit in zip(X, idx.unit):
        assert (w[:, 1] // 1000 == unit).all()      # every reading in the window belongs to its own engine


def test_last_only_gives_each_engines_final_window():
    df = engines({1: 9, 2: 3})
    full, idx = make_windows(df, length=4, cols=COLS)
    last, lidx = make_windows(df, length=4, cols=COLS, last_only=True)
    assert list(lidx.cycle) == [9, 3]
    assert np.array_equal(last[0], full[8]) and np.array_equal(last[1], full[11])


def test_gaps_in_cycles_are_rejected():
    df = engines({1: 6})
    with pytest.raises(ValueError):
        make_windows(df[df.cycle != 3], length=4, cols=COLS)
