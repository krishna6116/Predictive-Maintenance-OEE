"""Alert rule (guide 6.2): flag an engine when the LOWER edge of its 90% interval is at or below the threshold.

Using the pessimistic edge builds in a safety margin against late predictions (the costly kind, see PHM08).
Invalid forecasts are never scheduled on their numbers: they are flagged "invalid forecast" for manual inspection.
"""
import numpy as np
import pandas as pd

from maintenance.config import ALERT_THRESHOLD, RISK_BINS, RISK_LABELS


def flag_engines(pred: pd.DataFrame, threshold: float = ALERT_THRESHOLD) -> pd.DataFrame:
    """pred has engine_id, rul_point, rul_low, rul_high and, from maintenance.forecasts.validate, valid and
    invalid_reason. Returns the flagged engines, most urgent first, each with a risk level and a reason."""
    valid = pred["valid"].to_numpy(bool) if "valid" in pred else np.ones(len(pred), bool)
    ok = pred[valid & (pred.rul_low.to_numpy(float) <= threshold)].copy()
    # guide bins (-1, 5, 10, threshold]; with a threshold below 10 (sensitivity runs) the upper bins shrink
    bins = [b for b in RISK_BINS[:-1] if b < threshold] + [threshold]
    ok["risk"] = pd.cut(ok.rul_low, bins=bins, labels=RISK_LABELS[: len(bins) - 1]).astype(str)
    ok["reason"] = [f"lower 90% bound {lo:.1f} <= {threshold:g} cycles (point forecast {pt:.1f}, interval "
                    f"{lo:.1f}-{hi:.1f})" for lo, pt, hi in zip(ok.rul_low, ok.rul_point, ok.rul_high)]
    bad = pred[~valid].copy()
    bad["risk"] = "invalid forecast"
    bad["reason"] = "manual inspection: " + bad.get("invalid_reason", pd.Series("invalid", index=bad.index)).astype(str)
    out = pd.concat([ok.sort_values(["rul_low", "engine_id"]), bad.sort_values("engine_id")])
    return out.reset_index(drop=True)
