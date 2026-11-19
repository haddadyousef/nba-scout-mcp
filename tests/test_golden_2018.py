"""Checks evals/golden_2018.json's claims against a live engine, so the
eval file can't silently go stale."""

import json
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
def golden():
    with open(ROOT / "evals" / "golden_2018.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def engine():
    from scout_mcp.engine import ScoutEngine

    return ScoutEngine()


def test_undervalued_playmakers_still_rank_ahead_of_their_pick(engine, golden):
    board = engine.draft_board(2018, "Playmaking", top_n=60)
    by_player = {r["player"]: r for r in board["board"]}
    for player in golden["draft_class_board_playmaking_2018"]["expect_on_board"]:
        assert player in by_player, f"{player} missing from the 2018 Playmaking board"
        row = by_player[player]
        assert row["draft_pick"] is not None and row["model_rank"] < row["draft_pick"]


def test_trae_young_still_abstains_on_playmaking(engine, golden):
    profile = engine.skill_profile("Trae Young")
    info = profile["skills"]["Playmaking"]
    assert info["confidence"] == golden["trae_young"]["observed"]["confidence"]
    assert info["predicted_tier"] is None


def test_nonsense_stat_line_still_abstains(engine, golden):
    from scout_mcp import trust

    stats = golden["out_of_distribution_check"]["input"]
    pred = engine.score_new_input(stats, "Shooter")
    confidence, _reasons, _caveats = trust.confidence_for(
        pred["probs"], "Shooter", pred["dist_to_nearest_cluster"], engine.ood_threshold(), pred["n_missing"],
    )
    assert confidence == golden["out_of_distribution_check"]["expect_confidence"]
