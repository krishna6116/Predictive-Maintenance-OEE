"""Shared helpers for the Phase 8 Streamlit app: snapshot loading (checked against the manifest), palette, formats.

The app reads ONLY dashboards/snapshots/ (no database, no model weights, no network). Every file is checked against
its sha256 in manifest.json before use, so a stale or corrupt snapshot shows an error instead of wrong numbers.
PDM_SNAPSHOT_DIR overrides the folder (used by the tests for missing/corrupt snapshots).
"""
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/krishna6116/Predictive-Maintenance-OEE"
DOCS = f"{REPO}/blob/main/docs"

# Page titles are the questions in dashboards/phase8_spec.json (checked by the tests)
TITLES = {
    "fleet": "Which engines are closest to failure, and how sure is the model?",
    "plan": "Which engines should the crews service in the next 7 days?",
    "oee": "Which maintenance policy gives the highest OEE? (Simulated plant)",
    "monitoring": "Can we trust the model's forecasts this shift?",
    "about": "How were these numbers made, and can I check them?",
}

# Okabe-Ito colour-blind-safe palette; colour is always paired with a second channel (shape, dash, label or axis)
OKABE_ITO = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}


class SnapshotError(Exception):
    pass


def snapshot_dir() -> Path:
    return Path(os.environ.get("PDM_SNAPSHOT_DIR", ROOT / "dashboards" / "snapshots"))


@st.cache_data(show_spinner=False)
def _read(folder: str, name: str):
    d = Path(folder)
    if not (d / "manifest.json").exists():
        raise SnapshotError(f"manifest.json is missing from {d}")
    try:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        entry = man["files"][name]
    except (ValueError, KeyError) as e:
        raise SnapshotError(f"manifest.json is unreadable or does not list {name} ({type(e).__name__})")
    p = d / name
    if not p.exists():
        raise SnapshotError(f"snapshot file {name} is missing")
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != entry["sha256"]:
        raise SnapshotError(f"{name} does not match its sha256 in manifest.json (corrupt or stale snapshot)")
    try:
        if name.endswith(".json"):
            return json.loads(b)
        df = pd.read_csv(io.BytesIO(b), float_precision="round_trip")
    except Exception as e:
        raise SnapshotError(f"{name} could not be parsed ({type(e).__name__})")
    if entry["rows"] is not None and len(df) != entry["rows"]:
        raise SnapshotError(f"{name} has {len(df)} rows; manifest.json says {entry['rows']}")
    return df


def load(name: str):
    """A snapshot (DataFrame, or dict for JSON); on any problem show a clear message and stop the page."""
    try:
        return _read(str(snapshot_dir()), name)
    except SnapshotError as e:
        st.error(f"Data unavailable: {e}. Rebuild the snapshots with `python -m dashboards.export_snapshots` "
                 "or restore them from the repository.")
        st.stop()


def manifest() -> dict:
    try:
        return json.loads((snapshot_dir() / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        st.error(f"Data unavailable: manifest.json could not be read ({type(e).__name__}).")
        st.stop()


def fact(name: str):
    return load("facts.json")[name]["value"]


def define(term: str) -> str:
    return fact("definitions")[term]


def header(page: str, intro: str) -> None:
    st.title(TITLES[page])
    st.caption(intro)


def simulated_notice() -> None:
    st.warning(f"**Simulated.** {fact('simulation_notice')}. OEE = Availability x Performance x Quality "
               "from a simulated plant driven by the real NASA engine lives and the real forecasts.")


def has_rows(df: pd.DataFrame, what: str) -> bool:
    if len(df):
        return True
    st.info(f"No {what} match the current filter. Choose at least one option above.")
    return False


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def points(x: float, d: int = 2) -> str:
    return f"{100 * x:+.{d}f} pts"
