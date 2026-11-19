"""Thresholds, paths, and version string for the scout engine and server.

These are the trust-layer knobs described in the project brief, Section
5.1: tune here, not inline in `trust.py` or `engine.py`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

COLLEGE_STATS_CSV = DATA_DIR / "college_stats_final.csv"
NBA_RATINGS_CSV = DATA_DIR / "nba_player_ratings_25_99_newv2.csv"
DRAFT_2018_CSV = DATA_DIR / "draft_2018_board.csv"

PREPROCESS_PATH = ARTIFACTS_DIR / "preprocess.joblib"
SKILL_MODELS_DIR = ARTIFACTS_DIR / "skill_models"
OOF_PREDICTIONS_PATH = ARTIFACTS_DIR / "oof_predictions.parquet"
PLAYERS_PATH = ARTIFACTS_DIR / "players.parquet"
PLAYER_FEATURES_PATH = ARTIFACTS_DIR / "player_features.parquet"

RANDOM_STATE = 42
N_FOLDS = 5
N_ARCHETYPE_CLUSTERS = 5
N_PREDICTIVE_CLUSTERS = 5

SKILLS = ["Shooter", "Rebounding", "Playmaking", "Defense"]
SKILL_DISPLAY_NAMES = {
    "Shooter": "Shooting",
    "Rebounding": "Rebounding",
    "Playmaking": "Playmaking",
    "Defense": "Defense",
}
TIER_NAMES = ["Bottom 25%", "Middle 50%", "Top 25%"]

# Skills the published research flags as weakest (brief Section 3.4).
WEAK_SKILLS = {"Defense"}

# --- Confidence rules (brief Section 5.1) ---
HIGH_CONFIDENCE_PROB = 0.70
MEDIUM_CONFIDENCE_PROB = 0.50
CLOSE_CLASS_MARGIN = 0.10

# --- Abstain / out-of-distribution rules ---
OOD_CENTROID_PERCENTILE = 97.5
MAX_MISSING_INPUTS = 2

# Fuzzy name matching
FUZZY_MATCH_SCORE_CUTOFF = 60


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except OSError:
        return "unknown"
    if out.returncode == 0:
        return out.stdout.strip()
    return "unknown"


MODEL_VERSION = f"svi-2026.1+{_git_sha()}"
DATA_THROUGH = "college 1985-2019, NBA outcomes 1996-2025"

RESEARCH_REPO = "https://github.com/haddadyousef/NBA-Draft-prediction"
RESEARCH_REPO_COMMIT = "b0db46844007a939bfc9d644f613b8815e6f10a4"
