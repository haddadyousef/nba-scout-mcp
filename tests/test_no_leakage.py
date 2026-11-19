"""The most important correctness rule in the project (brief Section 0,
hard rule 2): no training-set player is ever served an in-sample
prediction, and the 2018 holdout stays genuinely held out.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config

pytestmark = pytest.mark.skipif(
    not config.PLAYERS_PATH.exists(), reason="artifacts not built — run scripts/export_artifacts.py first"
)


@pytest.fixture(scope="module")
def players():
    return pd.read_parquet(config.PLAYERS_PATH)


@pytest.fixture(scope="module")
def oof():
    return pd.read_parquet(config.OOF_PREDICTIONS_PATH)


def test_training_and_holdout_are_disjoint(players):
    overlap = players.loc[players["in_training_set"] & players["in_2018_holdout"]]
    assert overlap.empty, f"players in both training and 2018 holdout: {overlap['player'].tolist()}"


def test_every_training_player_has_out_of_fold_predictions(players, oof):
    training_players = set(players.loc[players["in_training_set"], "player"])
    for skill in config.SKILLS:
        scored = set(oof.loc[oof["skill"] == skill, "player"])
        scored_oof = set(
            oof.loc[(oof["skill"] == skill) & (oof["prediction_type"] == "out_of_fold"), "player"]
        )
        covered = training_players & scored
        assert covered, f"no training players scored for {skill}"
        assert covered <= scored_oof, (
            f"{skill}: some training players were scored with a non-out_of_fold prediction_type"
        )


def test_no_training_player_is_labeled_holdout_or_new_input(players, oof):
    training_players = set(players.loc[players["in_training_set"], "player"])
    bad = oof.loc[
        oof["player"].isin(training_players) & (oof["prediction_type"] != "out_of_fold")
    ]
    assert bad.empty, f"training players with a non-OOF label: {bad[['player', 'skill', 'prediction_type']].to_dict('records')}"


def test_every_holdout_player_is_labeled_holdout(players, oof):
    holdout_players = set(players.loc[players["in_2018_holdout"], "player"])
    for skill in config.SKILLS:
        rows = oof.loc[(oof["skill"] == skill) & (oof["player"].isin(holdout_players))]
        assert not rows.empty
        assert (rows["prediction_type"] == "holdout").all()
