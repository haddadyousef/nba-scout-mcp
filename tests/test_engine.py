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


@pytest.fixture(scope="module")
def engine():
    from scout_mcp.engine import ScoutEngine

    return ScoutEngine()


def test_search_exact_match(engine):
    matches = engine.search("Trae Young")
    assert matches[0]["player"] == "Trae Young"
    assert matches[0]["in_2018_holdout"] is True


def test_search_misspelled_name_still_resolves(engine):
    matches = engine.search("Tray Yung")
    assert any(m["player"] == "Trae Young" for m in matches)


def test_search_unknown_player_returns_no_crash(engine):
    matches = engine.search("Zzyzx Qqplonk")
    assert isinstance(matches, list)


def test_skill_profile_has_all_four_skills(engine):
    profile = engine.skill_profile("Trae Young")
    assert set(profile["skills"].keys()) == set(config.SKILLS)
    for info in profile["skills"].values():
        assert info["prediction_type"] == "holdout"
        assert info["confidence"] in ("high", "medium", "low", "abstain")


def test_compare_prospects_picks_an_edge_or_abstains(engine):
    result = engine.compare("Trae Young", "Jalen Brunson", ["Playmaking"])
    entry = result["by_skill"]["Playmaking"]
    assert "comparable" in entry


def test_draft_board_2018_includes_brunson(engine):
    board = engine.draft_board(2018, "Playmaking", top_n=60)
    players_on_board = [r["player"] for r in board["board"]]
    assert "Jalen Brunson" in players_on_board


def test_draft_board_other_year_raises(engine):
    with pytest.raises(ValueError):
        engine.draft_board(2015, None, 15)


def test_explain_tier_returns_top_contributions(engine):
    result = engine.explain("Trae Young", "Playmaking")
    assert len(result["top_contributions"]) == 5
    for c in result["top_contributions"]:
        assert c["feature"]
        assert c["raw_value"] is not None


def test_project_custom_prospect_abstains_on_sparse_input(engine):
    from scout_mcp.trust import is_out_of_distribution

    pred = engine.score_new_input({"PTS": 999, "Games": 1}, "Shooter")
    ood, _reasons = is_out_of_distribution(
        pred["dist_to_nearest_cluster"], engine.ood_threshold(), pred["n_missing"]
    )
    assert ood is True
