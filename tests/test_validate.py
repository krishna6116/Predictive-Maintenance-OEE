import numpy as np
import pandas as pd

from pipeline.ingest import next_batch
from pipeline.validate import compute_limits, inject_faults, validate

LIMITS = {"s2": [600.0, 700.0], "s4": [1300.0, 1500.0]}


def make(rows):
    return pd.DataFrame(rows, columns=["unit", "cycle", "s2", "s4"])


def test_clean_batch_passes():
    good, bad = validate(make([[1, 1, 642.0, 1400.0], [1, 2, 642.5, 1401.0]]), LIMITS)
    assert len(good) == 2 and bad.empty


def test_duplicate_keeps_first_and_quarantines_second():
    good, bad = validate(make([[1, 1, 642.0, 1400.0], [1, 1, 642.0, 1400.0]]), LIMITS)
    assert len(good) == 1 and bad.reason.tolist() == ["duplicate cycle"]


def test_missing_value_is_quarantined():
    good, bad = validate(make([[1, 1, np.nan, 1400.0]]), LIMITS)
    assert good.empty and bad.reason.tolist() == ["missing value"]


def test_out_of_range_names_the_sensor():
    good, bad = validate(make([[1, 1, 642.0, 9999.0]]), LIMITS)
    assert good.empty and bad.reason.tolist() == ["s4 out of range"]


def test_gap_inside_batch():
    good, bad = validate(make([[1, 1, 642.0, 1400.0], [1, 3, 642.0, 1400.0]]), LIMITS)
    assert good.cycle.tolist() == [1] and bad.reason.tolist() == ["cycle gap"]


def test_gap_against_last_received_cycle():
    _, bad = validate(make([[1, 8, 642.0, 1400.0]]), LIMITS, last_cycle={1: 5})
    assert bad.reason.tolist() == ["cycle gap"]
    good, _ = validate(make([[1, 6, 642.0, 1400.0]]), LIMITS, last_cycle={1: 5})
    assert good.cycle.tolist() == [6]


def test_one_bad_reading_does_not_cascade_into_gaps():
    batch = make([[1, 1, 642.0, 1400.0], [1, 2, np.nan, 1400.0], [1, 3, 642.0, 1400.0]])
    good, bad = validate(batch, LIMITS)
    assert good.cycle.tolist() == [1, 3] and bad.reason.tolist() == ["missing value"]


def test_engines_are_checked_independently():
    batch = make([[1, 1, 642.0, 1400.0], [2, 1, 642.0, 1400.0], [1, 2, 642.0, 1400.0], [2, 2, 642.0, 1400.0]])
    good, bad = validate(batch, LIMITS)
    assert len(good) == 4 and bad.empty


def test_limits_widen_training_range():
    limits = compute_limits(make([[1, 1, 640.0, 1400.0], [1, 2, 650.0, 1420.0]]), cols=["s2"], margin=0.1)
    assert limits["s2"] == [639.0, 651.0]


def test_inject_faults_produces_three_quarantined_rows():
    batch = make([[u, c, 642.0, 1400.0] for u in (1, 2) for c in range(1, 6)])
    limits = {"s2": [600.0, 700.0], "s4": [1300.0, 1500.0], "s11": [40.0, 55.0]}
    good, bad = validate(inject_faults(batch.assign(s11=47.5)), limits)
    assert len(bad) == 3 and len(good) == len(batch) - 2
    assert sorted(bad.reason) == ["duplicate cycle", "missing value", "s4 out of range"]


def test_next_batch_window():
    df = make([[u, c, 642.0, 1400.0] for u in (1, 2) for c in range(1, 11)])
    batch = next_batch(df, 4, 3)
    assert sorted(batch.cycle.unique()) == [4, 5, 6] and batch.unit.nunique() == 2
