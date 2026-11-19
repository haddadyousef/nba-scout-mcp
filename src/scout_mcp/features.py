"""Feature engineering shared by artifact export and live inference.

The per-40 stats, the engineered ratios, and the category feature lists
below are reproduced from the research pipeline
(`vendor/predict_success.py`, `add_advanced_features` /
`engineer_features` / `category_features`, from
https://github.com/haddadyousef/NBA-Draft-prediction @
b0db46844007a939bfc9d644f613b8815e6f10a4) so that a player is scored with
exactly the inputs the published models were trained on. This module is
imported by both `scripts/export_artifacts.py` (bulk, at export time) and
`scout_mcp.engine` (one row at a time, at query time) so the two paths
can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Base + engineered features per skill (vendor/predict_success.py:55-137).
CATEGORY_FEATURES: dict[str, list[str]] = {
    "Rebounding": [
        "TRB", "TRB_per_40", "TRB%",
        "Games", "Minutes_Per_Game",
        "DWS", "WS", "WS/40",
        "Rebound_per_Minute", "Rebound_Efficiency", "Impact_Rebounding",
    ],
    "Playmaking": [
        "AST", "AST_per_40", "AST%",
        "TOV", "TOV%", "USG%",
        "OWS", "WS", "WS/40",
        "AST_to_TOV", "Playmaking_Impact", "Ball_Control",
    ],
    "Defense": [
        "STL", "STL_per_40", "STL%",
        "BLK", "BLK_per_40", "BLK%",
        "DWS", "WS", "WS/40",
        "Stocks", "Stocks_per_40", "Defense_Impact",
        "STL_BLK_Interaction", "STL_BLK_Impact",
    ],
    "Shooter": [
        "FG", "FGA", "FG%",
        "3P", "3PA", "3P%",
        "FT", "FTA", "FT%",
        "TS%", "eFG%", "OWS",
        "Three_Point_Rate", "Shot_Selection", "Pure_Shooting",
    ],
}

# Raw stats used to fit the predictive (per-skill-input) clustering
# (vendor/predict_success.py:981-984, `cluster_base_feats`).
CLUSTER_BASE_FEATS = [
    "DWS", "AST", "BLK", "3P%", "TRB", "TOV",
    "FT%", "USG%", "STL", "FTA", "WS",
]

# Archetype clustering adds PPG (vendor/predict_success.py:404-417); PPG
# isn't a raw column in college_stats_final.csv, so we derive it (PTS /
# Games) rather than silently dropping it to 11 features. This is the only
# feature we add that the published script doesn't already compute inline.
ARCHETYPE_FEATS = CLUSTER_BASE_FEATS + ["PPG"]

RAW_STATS_REQUIRED = [
    "Games", "Minutes_Per_Game", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PTS", "TS%",
    "eFG%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS",
    "WS", "WS/40",
]


def add_per_40_features(df: pd.DataFrame) -> pd.DataFrame:
    """vendor/predict_success.py:139-146, verbatim logic."""
    df = df.copy()
    if "Minutes_Per_Game" in df.columns:
        minutes_factor = 40 / df["Minutes_Per_Game"].replace(0, 40)
        for stat in ["BLK", "STL", "TRB", "AST"]:
            if stat in df.columns:
                df[f"{stat}_per_40"] = df[stat] * minutes_factor
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """vendor/predict_success.py:104-131, verbatim logic."""
    df = df.copy()

    df["Three_Point_Rate"] = df["3PA"] / (df["FGA"] + 1e-6)
    df["Shot_Selection"] = df["eFG%"] / (df["USG%"] + 1e-6)
    df["Pure_Shooting"] = (df["3P%"] + df["FT%"]) / 2

    df["Rebound_per_Minute"] = df["TRB"] / (df["Minutes_Per_Game"] + 1e-6)
    df["Rebound_Efficiency"] = df["TRB%"] * df["Minutes_Per_Game"] / 40
    df["Impact_Rebounding"] = df["TRB_per_40"] * df["WS/40"]

    df["AST_to_TOV"] = df["AST"] / (df["TOV"] + 1e-6)
    df["Playmaking_Impact"] = df["AST%"] / (df["TOV%"] + 1e-6)
    df["Ball_Control"] = (df["AST_per_40"] * df["WS/40"]) / (df["TOV"] + 1e-6)

    df["Stocks"] = df["STL"] + df["BLK"]
    df["Stocks_per_40"] = df["STL_per_40"] + df["BLK_per_40"]
    df["Defense_Impact"] = (df["STL%"] + df["BLK%"]) * df["DWS"]

    df["STL_BLK_Interaction"] = df["STL_per_40"] * df["BLK_per_40"]
    df["STL_BLK_Impact"] = df["STL%"] * df["BLK%"]

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering chain for one or many raw college-stat rows."""
    df = df.copy()
    for col in RAW_STATS_REQUIRED:
        if col not in df.columns:
            df[col] = 0.0
    df[RAW_STATS_REQUIRED] = df[RAW_STATS_REQUIRED].fillna(0.0)
    df = add_per_40_features(df)
    df = engineer_features(df)
    games = df["Games"].replace(0, np.nan)
    df["PPG"] = (df["PTS"] / games).fillna(0.0)
    return df


def conference_dummies(df: pd.DataFrame, known_columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode LastConf. At inference, align to the training-time
    column set (`known_columns`) so an unseen conference doesn't change
    the model's input width."""
    df = df.copy()
    conf = df["LastConf"] if "LastConf" in df.columns else pd.Series([""] * len(df), index=df.index)
    dummies = pd.get_dummies(conf.fillna(""), prefix="Conf")

    if known_columns is not None:
        for col in known_columns:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[known_columns]

    return pd.concat([df, dummies], axis=1)
