"""Phase 8: the app checks must be able to fail (each detector catches a planted problem)."""
import pytest

pytest.importorskip("streamlit")

from dashboards import app_checks as A

GOOD = {"layer": [
    {"mark": "point", "encoding": {"x": {"field": "e", "type": "ordinal", "title": "Engine"},
                                   "y": {"field": "r", "type": "quantitative", "title": "RUL (cycles)"},
                                   "color": {"field": "s", "scale": {"range": ["#D55E00", "#0072B2"]}},
                                   "shape": {"field": "s"}}}]}


def with_(**enc):
    spec = {"layer": [{"mark": "point", "encoding": {**GOOD["layer"][0]["encoding"], **enc}}]}
    return {k: v for k, v in spec.items()}


def test_good_chart_passes():
    assert A.chart_problems(GOOD) == []


def test_non_palette_colour_is_caught():
    spec = with_(color={"field": "s", "scale": {"range": ["#FF0000", "#0072B2"]}})
    assert any("Okabe-Ito" in p for p in A.chart_problems(spec))


def test_colour_without_second_channel_is_caught():
    enc = dict(GOOD["layer"][0]["encoding"])
    enc.pop("shape")
    assert any("second channel" in p for p in A.chart_problems({"layer": [{"mark": "point", "encoding": enc}]}))


def test_text_layer_counts_as_second_channel():
    enc = dict(GOOD["layer"][0]["encoding"])
    enc.pop("shape")
    spec = {"layer": [{"mark": "bar", "encoding": enc}, {"mark": "text", "encoding": {"text": {"field": "s"}}}]}
    assert A.chart_problems(spec) == []


def test_axis_without_unit_or_title_is_caught():
    assert any("names no unit" in p for p in A.chart_problems(with_(y={"field": "r", "type": "quantitative", "title": "RUL"})))
    assert any("no title" in p for p in A.chart_problems(with_(y={"field": "r", "type": "quantitative"})))


def test_texts_collects_metric_help(tmp_path):
    at = A.run_page("about")
    t = A.texts(at)
    assert "Snapshot data checksum" in t and "sha256" in t
