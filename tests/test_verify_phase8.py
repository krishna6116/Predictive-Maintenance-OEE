"""Phase 8 verifier: passes on the committed snapshots and catches every planted bug."""
import pytest

from dashboards import verify_phase8 as V


def test_committed_snapshots_pass_the_data_checks():
    card = V.Card()
    V.check_manifest(V.SNAP, card)
    V.check_cells(V.truth(), V.SNAP, card)
    V.check_facts(V.truth(), V.SNAP, card)
    assert card.res["D1.2"][0] and card.res["D1.3"][0], card.res


@pytest.fixture(scope="module")
def planted():
    return {name: (ok, caught) for name, ok, caught in V.planted_bugs()}


@pytest.mark.parametrize("bug", list(V.PLANTED))
def test_planted_bug_is_caught(planted, bug):
    ok, caught = planted[bug]
    assert ok, f"{bug}: {caught}"


def test_at_least_five_planted_bugs():
    assert len(V.PLANTED) >= 5
