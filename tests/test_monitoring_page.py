"""Monitoring page: switching scenario keeps a usable Datasets filter (Chrome review 2026-11-15)."""
from dashboards.app_checks import run_page


def test_switching_rule_keeps_datasets_selected():
    at = run_page("monitoring")
    radio = at.radio(key="scenario")
    assert at.multiselect(key="datasets").value == ["FD001", "FD003"]
    for scenario, want in ((radio.options[1], ["FD002", "FD004"]), (radio.options[2], ["FD002", "FD004"]),
                           (radio.options[0], ["FD001", "FD003"])):
        at.radio(key="scenario").set_value(scenario).run()
        assert at.multiselect(key="datasets").value == want, scenario
        assert not at.exception and not any("No runs match" in i.value for i in at.info)


def test_emptied_filter_still_shows_the_message():
    at = run_page("monitoring")
    at.multiselect(key="datasets").set_value([]).run()
    assert any("No runs match" in i.value for i in at.info)
