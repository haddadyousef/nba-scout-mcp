import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config

pytestmark = pytest.mark.skipif(
    not config.PLAYERS_PATH.exists(), reason="artifacts not built — run scripts/export_artifacts.py first"
)

ENVELOPE_KEYS = {"answer", "summary", "confidence", "confidence_reasons", "caveats", "provenance"}


def _assert_envelope(result: dict):
    assert ENVELOPE_KEYS <= result.keys()
    assert result["confidence"] in ("high", "medium", "low", "abstain")
    assert set(result["provenance"].keys()) == {"prediction_type", "model_version", "data_through"}
    assert result["provenance"]["prediction_type"] in ("out_of_fold", "holdout", "new_input")


@pytest.fixture(scope="module", autouse=True)
def fresh_engine():
    from scout_mcp import server

    server._engine = None
    yield
    server._engine = None


def test_search_players_envelope():
    from scout_mcp.server import search_players

    _assert_envelope(search_players("Trae Young"))


def test_get_skill_profile_envelope():
    from scout_mcp.server import get_skill_profile

    result = get_skill_profile("Trae Young")
    _assert_envelope(result)
    assert set(result["answer"]["skills"].keys()) == set(config.SKILLS)


def test_get_skill_profile_ambiguous_name_returns_candidates_not_guess():
    from scout_mcp.server import get_skill_profile

    result = get_skill_profile("Qqnonexistentplayer")
    _assert_envelope(result)
    assert result["confidence"] == "abstain"


def test_compare_prospects_envelope():
    from scout_mcp.server import compare_prospects

    result = compare_prospects("Trae Young", "Jalen Brunson", ["Playmaking"])
    _assert_envelope(result)


def test_explain_tier_envelope():
    from scout_mcp.server import explain_tier

    result = explain_tier("Trae Young", "Playmaking")
    _assert_envelope(result)


def test_draft_class_board_2018_envelope_and_holdout_type():
    from scout_mcp.server import draft_class_board

    result = draft_class_board(2018, "Playmaking", 20)
    _assert_envelope(result)
    assert result["provenance"]["prediction_type"] == "holdout"
    names = [r["player"] for r in result["answer"]["board"]]
    assert "Jalen Brunson" in names


def test_project_custom_prospect_rejects_gibberish_without_crashing():
    from scout_mcp.server import project_custom_prospect

    result = project_custom_prospect({"PTS": -5, "Games": 0}, "Nonsense Stat Line")
    _assert_envelope(result)
    assert result["confidence"] in ("abstain", "low")


def test_model_card_resource_has_known_limitations():
    from scout_mcp.server import model_card_resource

    card = model_card_resource()
    assert "known_limitations" in card
    assert set(card["accuracy_by_skill"].keys()) == set(config.SKILLS)
