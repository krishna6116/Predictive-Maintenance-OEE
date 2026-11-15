"""v1.1: the page entries copied into the Phase 8 spec stay identical to their pre-registered v1.1 definitions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase8_spec_pages_equal_their_v11_definitions():
    v11 = json.loads((ROOT / "dashboards" / "v11_spec.json").read_text(encoding="utf-8"))["phase8_spec_pages"]
    p8 = {p["id"]: p for p in json.loads((ROOT / "dashboards" / "phase8_spec.json").read_text(encoding="utf-8"))["pages"]}
    for pid, entry in v11.items():
        assert pid in p8, f"page '{pid}' is defined in v11_spec.json but missing from phase8_spec.json"
        assert p8[pid] == entry, f"page '{pid}' differs between phase8_spec.json and v11_spec.json"
    added = {pid for pid, p in p8.items() if "v1.1 amendment" in p.get("added", "")}
    assert added == set(v11), f"phase8_spec.json v1.1 pages {sorted(added)} != v11_spec.json {sorted(v11)}"
