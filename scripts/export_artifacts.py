"""Build every artifact `scout_mcp.engine.ScoutEngine` needs to serve the
Shopping by Skill model, without training or scraping at request time.

This re-implements the published pipeline's feature engineering and model
family (vendor/predict_success.py), but fixes two things the published
pipeline does not do (see README.md's model card for the full rationale,
and Section 12 answers recorded in the project chat):

1. The 2018 draft class is excluded from training, so the case-study demo
   is a genuine forecast rather than an in-sample replay (the published
   pipeline has no season column and trains on all 1,268 merged players,
   2018 class included).
2. Out-of-fold predictions cover every training-set player, aggregated
   across all 5 folds, instead of only the single best-accuracy fold per
   model that the published pipeline keeps.

Run with:  uv run python scripts/export_artifacts.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from scout_mcp.features import (
    ARCHETYPE_FEATS,
    CATEGORY_FEATURES,
    CLUSTER_BASE_FEATS,
    add_derived_features,
    conference_dummies,
)

RANDOM_STATE = config.RANDOM_STATE
N_FOLDS = config.N_FOLDS
TIER_NAMES = config.TIER_NAMES


def log(msg: str) -> None:
    print(f"[export_artifacts] {msg}", flush=True)


def load_and_merge() -> tuple[pd.DataFrame, list[str]]:
    """vendor/predict_success.py:1132-1177 (`prepare_data`), reproduced."""
    nba_data = pd.read_csv(config.NBA_RATINGS_CSV)
    college_stats = pd.read_csv(config.COLLEGE_STATS_CSV)
    college_stats["Player"] = college_stats["Player"].astype(str).str.strip()

    college_stats = add_derived_features(college_stats)
    college_stats = conference_dummies(college_stats)
    conf_cols = [c for c in college_stats.columns if c.startswith("Conf_")]

    merged = pd.merge(college_stats, nba_data, on="Player", how="inner")
    merged = merged.fillna(0)

    for skill in config.SKILLS:
        merged[f"{skill}_Category"] = pd.qcut(
            merged[f"{skill}_Score"],
            q=[0, 0.25, 0.75, 1],
            labels=["Bottom 25%", "Middle 50%", "Top 25%"],
            duplicates="drop",
        ).cat.codes

    log(f"Merged dataset: {len(merged)} players, {len(conf_cols)} conference columns.")
    return merged, conf_cols


def mark_2018_holdout(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(config.DRAFT_2018_CSV)
    holdout_names = set(board["player"].str.lower())
    is_holdout = merged["Player"].str.lower().isin(holdout_names)

    training = merged.loc[~is_holdout].reset_index(drop=True)
    holdout = merged.loc[is_holdout].reset_index(drop=True)
    holdout = holdout.merge(
        board.rename(columns={"player": "Player_lower_join"}),
        left_on=merged["Player"].str.lower()[is_holdout].values,
        right_on="Player_lower_join",
        how="left",
    ).drop(columns=["Player_lower_join"])

    log(
        f"2018 holdout: {len(board)} drafted, {len(holdout)} matched to college stats "
        f"and excluded from training. Training population: {len(training)}."
    )
    return training, holdout


def fit_archetype_clustering(merged: pd.DataFrame) -> dict:
    """Descriptive archetypes, fit on the full population (draft-class
    members included. This is unsupervised and purely descriptive, so it
    doesn't leak into any model's accuracy). vendor/predict_success.py:
    396-546, `add_clustering_features`."""
    valid = [f for f in ARCHETYPE_FEATS if f in merged.columns]
    X = merged[valid].fillna(0)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=min(5, len(valid)), random_state=RANDOM_STATE).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    kmeans = KMeans(n_clusters=config.N_ARCHETYPE_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    descriptions: dict[int, str] = {}
    examples: dict[int, list[str]] = {}
    for c in range(config.N_ARCHETYPE_CLUSTERS):
        mask = labels == c
        cluster_means = X[mask].mean()
        distinctive = []
        for feat in valid:
            overall_mean, std = X[feat].mean(), X[feat].std()
            if overall_mean != 0 and std > 0:
                z = (cluster_means[feat] - overall_mean) / std
                distinctive.append((feat, z))
        distinctive.sort(key=lambda t: abs(t[1]), reverse=True)
        words = [f"high {f}" if z > 1 else f"low {f}" for f, z in distinctive[:3] if abs(z) > 1]
        descriptions[c] = "Players with " + ", ".join(words) if words else "Balanced players"
        examples[c] = merged.loc[mask, "Player"].head(5).tolist()

    log("Archetypes: " + "; ".join(f"{c}={d}" for c, d in descriptions.items()))
    return {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "features": valid,
        "descriptions": descriptions,
        "examples": examples,
        "labels_by_index": labels,
        "pca_coords": X_pca,
    }


def fit_predictive_clustering(training: pd.DataFrame) -> dict:
    """Fit once on the training population and persist. This is the brief's
    Section 3.3 fix for the published pipeline's per-fold, never-saved
    clustering (vendor/predict_success.py:548-558)."""
    valid = [f for f in CLUSTER_BASE_FEATS if f in training.columns]
    X = training[valid].fillna(0)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=min(5, len(valid)), random_state=RANDOM_STATE).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    kmeans = KMeans(n_clusters=config.N_PREDICTIVE_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_pca)

    dists = np.stack(
        [np.linalg.norm(X_pca - kmeans.cluster_centers_[i], axis=1) for i in range(kmeans.n_clusters)],
        axis=1,
    )
    min_dist = dists.min(axis=1)
    ood_threshold = float(np.percentile(min_dist, config.OOD_CENTROID_PERCENTILE))
    log(f"Predictive clustering fit on {len(training)} rows. OOD distance threshold (p{config.OOD_CENTROID_PERCENTILE}) = {ood_threshold:.3f}")

    return {"scaler": scaler, "pca": pca, "kmeans": kmeans, "features": valid, "ood_threshold": ood_threshold}


def add_predictive_cluster_cols(df: pd.DataFrame, cluster_model: dict) -> pd.DataFrame:
    df = df.copy()
    valid = cluster_model["features"]
    X = df[valid].fillna(0)
    X_scaled = cluster_model["scaler"].transform(X)
    X_pca = cluster_model["pca"].transform(X_scaled)
    kmeans = cluster_model["kmeans"]
    labels = kmeans.predict(X_pca)
    for i in range(kmeans.n_clusters):
        df[f"cluster_{i}"] = (labels == i).astype(int)
        df[f"dist_to_cluster_{i}"] = np.linalg.norm(X_pca - kmeans.cluster_centers_[i], axis=1)
    dist_cols = [f"dist_to_cluster_{i}" for i in range(kmeans.n_clusters)]
    df["dist_to_nearest_cluster"] = df[dist_cols].min(axis=1)
    return df


def class_weight_dict_for(y: pd.Series) -> dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    d = dict(zip(classes, weights))
    if 2 in d:
        d[2] *= 1.5
    if 0 in d:
        d[0] *= 1.25
    return d


def make_base_models(class_weight: dict[int, float]) -> dict:
    return {
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, random_state=RANDOM_STATE
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.1, max_depth=8, subsample=0.9,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=8, class_weight=class_weight, random_state=RANDOM_STATE
        ),
    }


def fit_one(name: str, model, X_train, y_train, sample_weight=None):
    if name in ("hist_gb", "xgb") and sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def run_oof_cv(X: pd.DataFrame, y: pd.Series, skill: str) -> tuple[dict, dict]:
    """5-fold stratified CV with SMOTE inside each training fold
    (vendor/predict_success.py:994-1079). Unlike the published pipeline,
    every fold's held-out predictions are kept (not just the single
    best-accuracy fold), so every row gets exactly one real out-of-fold
    prediction per model type."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_names = ["hist_gb", "xgb", "rf", "voting"]
    oof_proba = {m: np.zeros((len(X), 3)) for m in model_names}
    fold_acc: dict[str, list[float]] = {m: [] for m in model_names}
    classes_ = np.array([0, 1, 2])

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cw = class_weight_dict_for(y_train)

        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        sw_res = pd.Series(y_res).map(cw).to_numpy()

        fitted = {}
        for name, model in make_base_models(cw).items():
            fit_one(name, model, X_res, y_res, sample_weight=sw_res)
            proba = model.predict_proba(X_test)
            proba_full = np.zeros((len(X_test), 3))
            for j, c in enumerate(model.classes_):
                proba_full[:, c] = proba[:, j]
            oof_proba[name][test_idx] = proba_full
            acc = accuracy_score(y_test, classes_[proba_full.argmax(axis=1)])
            fold_acc[name].append(acc)
            fitted[name] = model

        voting = VotingClassifier(
            estimators=[(n, m) for n, m in fitted.items()], voting="soft"
        )
        voting.fit(X_res, y_res)
        proba = voting.predict_proba(X_test)
        proba_full = np.zeros((len(X_test), 3))
        for j, c in enumerate(voting.classes_):
            proba_full[:, c] = proba[:, j]
        oof_proba["voting"][test_idx] = proba_full
        fold_acc["voting"].append(accuracy_score(y_test, classes_[proba_full.argmax(axis=1)]))

        log(
            f"  {skill} fold {fold_idx}/{N_FOLDS}: "
            + ", ".join(f"{m}={fold_acc[m][-1]:.3f}" for m in model_names)
        )

    mean_acc = {m: float(np.mean(v)) for m, v in fold_acc.items()}
    return oof_proba, {"fold_accuracies": fold_acc, "mean_accuracy": mean_acc}


def train_final_model(best_name: str, X: pd.DataFrame, y: pd.Series, member_accs: dict):
    cw = class_weight_dict_for(y)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    sw_res = pd.Series(y_res).map(cw).to_numpy()

    base_models = make_base_models(cw)
    if best_name != "voting":
        model = base_models[best_name]
        fit_one(best_name, model, X_res, y_res, sample_weight=sw_res)
        return model, "self"

    fitted = {}
    for name, m in base_models.items():
        fit_one(name, m, X_res, y_res, sample_weight=sw_res)
        fitted[name] = m
    voting = VotingClassifier(estimators=[(n, m) for n, m in fitted.items()], voting="soft")
    voting.fit(X_res, y_res)
    strongest = max(member_accs, key=member_accs.get)
    return voting, f"voting_member:{strongest}"


def export_skill(skill: str, training: pd.DataFrame, holdout: pd.DataFrame, cluster_model: dict):
    log(f"=== {skill} ===")
    base_features = CATEGORY_FEATURES[skill]
    cluster_cols = [f"cluster_{i}" for i in range(cluster_model["kmeans"].n_clusters)]
    dist_cols = [f"dist_to_cluster_{i}" for i in range(cluster_model["kmeans"].n_clusters)]
    full_features = base_features + cluster_cols + dist_cols

    scaler = StandardScaler().fit(training[base_features])
    X_base = pd.DataFrame(
        scaler.transform(training[base_features]), columns=base_features, index=training.index
    )
    X_full = pd.concat([X_base, training[cluster_cols + dist_cols]], axis=1)[full_features]
    y = training[f"{skill}_Category"].astype(int)

    oof_proba, cv_stats = run_oof_cv(X_full, y, skill)
    best_name = max(cv_stats["mean_accuracy"], key=cv_stats["mean_accuracy"].get)
    member_accs = {m: cv_stats["mean_accuracy"][m] for m in ("hist_gb", "xgb", "rf")}
    log(
        f"{skill}: mean CV accuracy by model = "
        + ", ".join(f"{m}={a:.3f}" for m, a in cv_stats["mean_accuracy"].items())
        + f" -> serving model = {best_name}"
    )

    final_model, shap_source = train_final_model(best_name, X_full, y, member_accs)

    holdout_X_base = pd.DataFrame(
        scaler.transform(holdout[base_features]), columns=base_features, index=holdout.index
    )
    holdout_X_full = pd.concat([holdout_X_base, holdout[cluster_cols + dist_cols]], axis=1)[full_features]
    holdout_proba = final_model.predict_proba(holdout_X_full)
    holdout_proba_full = np.zeros((len(holdout), 3))
    for j, c in enumerate(final_model.classes_):
        holdout_proba_full[:, c] = holdout_proba[:, j]

    model_record = {
        "model": final_model,
        "model_type": best_name,
        "shap_source": shap_source,
        "scaler": scaler,
        "base_features": base_features,
        "full_features": full_features,
        "mean_cv_accuracy": cv_stats["mean_accuracy"],
        "fold_accuracies": cv_stats["fold_accuracies"],
        "served_accuracy": cv_stats["mean_accuracy"][best_name],
        "is_weak_skill": skill in config.WEAK_SKILLS,
    }

    oof_rows = pd.DataFrame(
        oof_proba[best_name], columns=["prob_bottom25", "prob_middle50", "prob_top25"]
    )
    oof_rows["player"] = training["Player"].values
    oof_rows["skill"] = skill
    oof_rows["true_tier"] = y.values
    oof_rows["pred_tier"] = oof_rows[["prob_bottom25", "prob_middle50", "prob_top25"]].to_numpy().argmax(axis=1)
    oof_rows["prediction_type"] = "out_of_fold"
    oof_rows["dist_to_nearest_cluster"] = training["dist_to_nearest_cluster"].values

    holdout_rows = pd.DataFrame(
        holdout_proba_full, columns=["prob_bottom25", "prob_middle50", "prob_top25"]
    )
    holdout_rows["player"] = holdout["Player"].values
    holdout_rows["skill"] = skill
    holdout_rows["true_tier"] = holdout[f"{skill}_Category"].astype(int).values
    holdout_rows["pred_tier"] = holdout_rows[["prob_bottom25", "prob_middle50", "prob_top25"]].to_numpy().argmax(axis=1)
    holdout_rows["prediction_type"] = "holdout"
    holdout_rows["dist_to_nearest_cluster"] = holdout["dist_to_nearest_cluster"].values

    return model_record, pd.concat([oof_rows, holdout_rows], ignore_index=True)


def build_players_table(merged: pd.DataFrame, training: pd.DataFrame, holdout: pd.DataFrame, archetype: dict) -> pd.DataFrame:
    board = pd.read_csv(config.DRAFT_2018_CSV)
    board["player_lower"] = board["player"].str.lower()

    rows = merged[["Player", "LastConf", "YearsPlayed", "Games", "Minutes_Per_Game"]].copy()
    rows["player_lower"] = rows["Player"].str.lower()
    rows["archetype_cluster"] = archetype["labels_by_index"]
    rows["archetype_description"] = rows["archetype_cluster"].map(archetype["descriptions"])
    pca_coords = archetype["pca_coords"]
    for i in range(pca_coords.shape[1]):
        rows[f"archetype_pca_{i}"] = pca_coords[:, i]
    rows["in_training_set"] = rows["Player"].isin(training["Player"])
    rows["in_2018_holdout"] = rows["Player"].isin(holdout["Player"])

    rows = rows.merge(
        board[["player_lower", "pick", "round", "team", "college"]],
        on="player_lower", how="left",
    )
    rows = rows.rename(
        columns={
            "Player": "player",
            "LastConf": "conference",
            "YearsPlayed": "years_played",
            "Games": "games",
            "Minutes_Per_Game": "minutes_per_game",
            "pick": "draft_pick",
            "round": "draft_round",
            "team": "draft_team",
            "college": "draft_college",
        }
    ).drop(columns=["player_lower"])
    return rows


def main():
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    config.SKILL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    merged, conf_cols = load_and_merge()
    training, holdout = mark_2018_holdout(merged)

    archetype = fit_archetype_clustering(merged)
    predictive_cluster = fit_predictive_clustering(training)

    training = add_predictive_cluster_cols(training, predictive_cluster)
    holdout = add_predictive_cluster_cols(holdout, predictive_cluster)

    model_records = {}
    oof_frames = []
    for skill in config.SKILLS:
        record, oof_df = export_skill(skill, training, holdout, predictive_cluster)
        model_records[skill] = record
        oof_frames.append(oof_df)
        joblib.dump(record, config.SKILL_MODELS_DIR / f"{skill}.joblib")

    oof_all = pd.concat(oof_frames, ignore_index=True)
    oof_all.to_parquet(config.OOF_PREDICTIONS_PATH, index=False)

    players = build_players_table(merged, training, holdout, archetype)
    players.to_parquet(config.PLAYERS_PATH, index=False)

    all_base_features = sorted({f for feats in CATEGORY_FEATURES.values() for f in feats})
    cluster_cols = [f"cluster_{i}" for i in range(predictive_cluster["kmeans"].n_clusters)]
    dist_cols = [f"dist_to_cluster_{i}" for i in range(predictive_cluster["kmeans"].n_clusters)]
    feat_cols = ["Player"] + all_base_features + cluster_cols + dist_cols + ["dist_to_nearest_cluster"]
    player_features = pd.concat([training[feat_cols], holdout[feat_cols]], ignore_index=True)
    player_features = player_features.rename(columns={"Player": "player"})
    player_features.to_parquet(config.PLAYER_FEATURES_PATH, index=False)

    preprocess = {
        "archetype": {
            k: v for k, v in archetype.items() if k not in ("labels_by_index", "pca_coords")
        },
        "predictive_cluster": predictive_cluster,
        "conf_columns": conf_cols,
        "model_version": config.MODEL_VERSION,
        "data_through": config.DATA_THROUGH,
        "exported_at": pd.Timestamp.utcnow().isoformat(),
    }
    joblib.dump(preprocess, config.PREPROCESS_PATH)

    summary = {
        skill: {
            "served_model": rec["model_type"],
            "served_accuracy": rec["served_accuracy"],
            "mean_cv_accuracy_by_model": rec["mean_cv_accuracy"],
        }
        for skill, rec in model_records.items()
    }
    (config.ARTIFACTS_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Done in {time.time() - t0:.1f}s. Summary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
