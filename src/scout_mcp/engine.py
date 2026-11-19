"""ScoutEngine: loads the artifacts `scripts/export_artifacts.py` built and
answers every tool's question. No training, scraping, or network calls
happen here, only inference against saved models (brief Section 6:
"server.py is a thin adapter. All ML lives in engine.py").
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
import shap

import config
from scout_mcp import trust
from scout_mcp.features import add_derived_features, conference_dummies
from scout_mcp.names import NameResolver

logger = logging.getLogger("scout_mcp.engine")


class UnknownPlayerError(Exception):
    def __init__(self, query: str, candidates: list[str]):
        self.query = query
        self.candidates = candidates
        super().__init__(f"'{query}' did not resolve to one player. Candidates: {candidates}")


class UnknownSkillError(Exception):
    pass


class ScoutEngine:
    def __init__(self, artifacts_dir=None):
        artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
        self.preprocess = joblib.load(artifacts_dir / "preprocess.joblib")
        self.players = pd.read_parquet(artifacts_dir / "players.parquet")
        self.oof = pd.read_parquet(artifacts_dir / "oof_predictions.parquet")
        self.skill_models = {
            skill: joblib.load(artifacts_dir / "skill_models" / f"{skill}.joblib")
            for skill in config.SKILLS
        }
        self.player_features = pd.read_parquet(artifacts_dir / "player_features.parquet").set_index(
            "player", drop=False
        )
        self.resolver = NameResolver(self.players)
        self._shap_cache: dict[str, tuple] = {}

    # --- name resolution ----------------------------------------------

    def resolve(self, query: str) -> str:
        canonical, candidates = self.resolver.resolve_one(query)
        if canonical is None:
            raise UnknownPlayerError(query, [c.player for c in candidates])
        return canonical

    def search(self, query: str, limit: int = 5) -> list[dict]:
        matches = self.resolver.search(query, limit=limit)
        out = []
        for m in matches:
            row = self.players.loc[self.players["player"] == m.player].iloc[0]
            out.append(
                {
                    "player": m.player,
                    "match_score": round(m.score, 1),
                    "conference": row["conference"] or None,
                    "draft_college": row["draft_college"] if pd.notna(row["draft_college"]) else None,
                    "in_training_set": bool(row["in_training_set"]),
                    "in_2018_holdout": bool(row["in_2018_holdout"]),
                }
            )
        return out

    # --- per-skill scoring ----------------------------------------------

    def _player_row(self, player: str) -> pd.Series:
        rows = self.players.loc[self.players["player"] == player]
        if rows.empty:
            raise UnknownPlayerError(player, [])
        return rows.iloc[0]

    def _skill_record(self, skill: str) -> dict:
        if skill not in self.skill_models:
            raise UnknownSkillError(f"Unknown skill '{skill}'. Valid skills: {config.SKILLS}")
        return self.skill_models[skill]

    def skill_prediction_for_known_player(self, player: str, skill: str) -> dict:
        """Look up the player's precomputed out-of-fold / holdout prediction.
        Never scores a training-set player with their own trained model
        (brief hard rule 2)."""
        record = self._skill_record(skill)
        rows = self.oof.loc[(self.oof["player"] == player) & (self.oof["skill"] == skill)]
        if rows.empty:
            return None
        row = rows.iloc[0]
        probs = [row["prob_bottom25"], row["prob_middle50"], row["prob_top25"]]
        return {
            "probs": probs,
            "pred_tier": int(row["pred_tier"]),
            "true_tier": int(row["true_tier"]) if pd.notna(row["true_tier"]) else None,
            "prediction_type": row["prediction_type"],
            "dist_to_nearest_cluster": float(row["dist_to_nearest_cluster"]),
            "model_type": record["model_type"],
        }

    def score_new_input(self, stats: dict, skill: str) -> dict:
        """Run a stat line that isn't in the database through the saved
        scaler, predictive clustering, and model (brief Section 5.2,
        `project_custom_prospect`)."""
        record = self._skill_record(skill)
        df = pd.DataFrame([stats])
        n_missing = sum(1 for col in ("Games", "Minutes_Per_Game") if col not in stats or stats.get(col) in (None, 0))
        df = add_derived_features(df)
        df = conference_dummies(df, known_columns=self.preprocess["conf_columns"])

        pc = self.preprocess["predictive_cluster"]
        valid = pc["features"]
        X_cluster_raw = df.reindex(columns=valid, fill_value=0.0)
        X_scaled = pc["scaler"].transform(X_cluster_raw)
        X_pca = pc["pca"].transform(X_scaled)
        kmeans = pc["kmeans"]
        dists = np.stack(
            [np.linalg.norm(X_pca - kmeans.cluster_centers_[i], axis=1) for i in range(kmeans.n_clusters)],
            axis=1,
        )
        dist_to_nearest = float(dists.min())
        cluster_cols = {f"cluster_{i}": int(np.argmin(dists[0]) == i) for i in range(kmeans.n_clusters)}
        dist_cols = {f"dist_to_cluster_{i}": float(dists[0, i]) for i in range(kmeans.n_clusters)}

        base_features = record["base_features"]
        X_base = pd.DataFrame(
            record["scaler"].transform(df.reindex(columns=base_features, fill_value=0.0)),
            columns=base_features,
        )
        for k, v in {**cluster_cols, **dist_cols}.items():
            X_base[k] = v
        X_full = X_base[record["full_features"]]

        proba = record["model"].predict_proba(X_full)[0]
        full_proba = [0.0, 0.0, 0.0]
        for j, c in enumerate(record["model"].classes_):
            full_proba[c] = float(proba[j])

        return {
            "probs": full_proba,
            "pred_tier": int(np.argmax(full_proba)),
            "true_tier": None,
            "prediction_type": "new_input",
            "dist_to_nearest_cluster": dist_to_nearest,
            "model_type": record["model_type"],
            "n_missing": n_missing,
        }

    def archetype_for(self, player: str) -> dict:
        row = self._player_row(player)
        cluster = int(row["archetype_cluster"])
        return {
            "cluster": cluster,
            "description": self.preprocess["archetype"]["descriptions"][cluster],
            "examples": self.preprocess["archetype"]["examples"].get(cluster, []),
        }

    def ood_threshold(self) -> float:
        return self.preprocess["predictive_cluster"]["ood_threshold"]

    # --- comps -----------------------------------------------------------

    def _archetype_pca_cols(self) -> list[str]:
        return [c for c in self.players.columns if c.startswith("archetype_pca_")]

    def nearest_comps(self, player: str, k: int = 5, skill: str | None = None) -> list[dict]:
        """Nearest neighbors in archetype PCA space (brief: `find_comps`,
        and the 3-comp block inside `explain_tier`)."""
        pca_cols = self._archetype_pca_cols()
        row = self._player_row(player)
        target = row[pca_cols].to_numpy(dtype=float)
        others = self.players.loc[self.players["player"] != player]
        dists = np.linalg.norm(others[pca_cols].to_numpy(dtype=float) - target, axis=1)
        order = np.argsort(dists)[:k]
        nearest = others.iloc[order]
        nearest_dists = dists[order]

        comps = []
        for (_, r), d in zip(nearest.iterrows(), nearest_dists):
            comp = {
                "player": r["player"],
                "conference": r["conference"] or None,
                "archetype_description": r["archetype_description"],
                "pca_distance": round(float(d), 3),
                "in_2018_holdout": bool(r["in_2018_holdout"]),
            }
            if skill is not None:
                known = self.skill_prediction_for_known_player(r["player"], skill)
                comp["actual_nba_tier"] = (
                    config.TIER_NAMES[known["true_tier"]]
                    if known and known["true_tier"] is not None
                    else None
                )
            comps.append(comp)
        return comps

    # --- explain -----------------------------------------------------------

    def _explained_model(self, skill: str):
        record = self._skill_record(skill)
        model = record["model"]
        if record["shap_source"] == "self":
            return model
        member_name = record["shap_source"].split(":", 1)[1]
        return next(m for n, m in model.estimators_ if n == member_name)

    def _tree_explainer(self, skill: str) -> shap.TreeExplainer:
        if skill in self._shap_cache:
            return self._shap_cache[skill]
        explainer = shap.TreeExplainer(self._explained_model(skill))
        self._shap_cache[skill] = explainer
        return explainer

    def explain(self, player: str, skill: str) -> dict:
        record = self._skill_record(skill)
        if player not in self.player_features.index:
            raise UnknownPlayerError(player, [])
        feat_row = self.player_features.loc[player]
        base_features = record["base_features"]
        full_features = record["full_features"]

        X_base = pd.DataFrame(
            record["scaler"].transform(feat_row[base_features].to_frame().T),
            columns=base_features,
        )
        for col in full_features:
            if col not in X_base.columns:
                X_base[col] = feat_row[col]
        X_full = X_base[full_features]

        known = self.skill_prediction_for_known_player(player, skill)
        pred_tier = known["pred_tier"] if known else int(np.argmax(record["model"].predict_proba(X_full)[0]))

        explainer = self._tree_explainer(skill)
        shap_values = explainer.shap_values(X_full)
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_tier][0]
        elif shap_values.ndim == 3:
            class_shap = shap_values[0, :, pred_tier]
        else:
            class_shap = shap_values[0]

        training_players = set(self.players.loc[self.players["in_training_set"], "player"])
        train_pop = self.player_features.loc[self.player_features.index.isin(training_players)]

        contributions = []
        order = np.argsort(-np.abs(class_shap))[:5]
        for idx in order:
            feat = full_features[idx]
            raw_value = float(feat_row[feat]) if feat in feat_row.index else None
            percentile = None
            if feat in train_pop.columns and raw_value is not None:
                percentile = float((train_pop[feat] < raw_value).mean() * 100)
            contributions.append(
                {
                    "feature": feat,
                    "shap_contribution": round(float(class_shap[idx]), 4),
                    "raw_value": raw_value,
                    "training_percentile": round(percentile, 1) if percentile is not None else None,
                }
            )

        archetype = self.archetype_for(player)
        comps = self.nearest_comps(player, k=3, skill=skill)

        caveats = []
        if record["shap_source"] != "self":
            caveats.append(
                f"Served model for {skill} is a voting ensemble; SHAP explains its strongest "
                f"member ({record['shap_source'].split(':', 1)[1]}), not the ensemble vote itself."
            )

        return {
            "known": known,
            "pred_tier": pred_tier,
            "top_contributions": contributions,
            "archetype": archetype,
            "dist_to_nearest_cluster": float(feat_row["dist_to_nearest_cluster"]),
            "nearest_comps": comps,
            "caveats": caveats,
        }

    # --- high-level tool methods -----------------------------------------

    def _prediction_for(self, player: str, skill: str) -> dict:
        known = self.skill_prediction_for_known_player(player, skill)
        if known is not None:
            return known
        if player in self.player_features.index:
            # In players.parquet / player_features.parquet but not scored
            # for this skill (e.g. missing a target score) — treat as a
            # fresh input through the full scoring path.
            feat_row = self.player_features.loc[player]
            stats = feat_row.to_dict()
            result = self.score_new_input(stats, skill)
            result["prediction_type"] = "new_input"
            return result
        raise UnknownPlayerError(player, [])

    def skill_profile(self, player: str) -> dict:
        canonical = self.resolve(player)
        archetype = self.archetype_for(canonical)
        skills_out = {}
        for skill in config.SKILLS:
            pred = self._prediction_for(canonical, skill)
            confidence, reasons, caveats = trust.confidence_for(
                pred["probs"], skill, pred["dist_to_nearest_cluster"], self.ood_threshold(),
                pred.get("n_missing", 0),
            )
            skills_out[skill] = {
                "predicted_tier": None if confidence == "abstain" else config.TIER_NAMES[pred["pred_tier"]],
                "probabilities": dict(zip(config.TIER_NAMES, pred["probs"])),
                "actual_tier": config.TIER_NAMES[pred["true_tier"]] if pred["true_tier"] is not None else None,
                "confidence": confidence,
                "confidence_reasons": reasons,
                "caveats": caveats,
                "prediction_type": pred["prediction_type"],
            }
        return {"player": canonical, "archetype": archetype, "skills": skills_out}

    def compare(self, player_a: str, player_b: str, skills: list[str] | None = None) -> dict:
        a = self.resolve(player_a)
        b = self.resolve(player_b)
        skills = skills or config.SKILLS
        out = {}
        for skill in skills:
            pred_a = self._prediction_for(a, skill)
            pred_b = self._prediction_for(b, skill)
            conf_a, _reasons_a, _ = trust.confidence_for(
                pred_a["probs"], skill, pred_a["dist_to_nearest_cluster"], self.ood_threshold(), pred_a.get("n_missing", 0)
            )
            conf_b, _reasons_b, _ = trust.confidence_for(
                pred_b["probs"], skill, pred_b["dist_to_nearest_cluster"], self.ood_threshold(), pred_b.get("n_missing", 0)
            )
            if conf_a == "abstain" or conf_b == "abstain":
                out[skill] = {"comparable": False, "reason": "can't compare: one side abstained"}
                continue
            top25_a, top25_b = pred_a["probs"][2], pred_b["probs"][2]
            edge_player = a if top25_a >= top25_b else b
            out[skill] = {
                "comparable": True,
                "edge": edge_player,
                "margin_top25_prob": round(abs(top25_a - top25_b), 3),
                "player_a": {"player": a, "predicted_tier": config.TIER_NAMES[pred_a["pred_tier"]], "confidence": conf_a},
                "player_b": {"player": b, "predicted_tier": config.TIER_NAMES[pred_b["pred_tier"]], "confidence": conf_b},
            }
        return {"player_a": a, "player_b": b, "by_skill": out}

    def draft_board(self, year: int, skill: str | None, top_n: int) -> dict:
        if year != 2018:
            raise ValueError("Only the 2018 draft class has a built holdout board right now.")
        board = self.players.loc[self.players["in_2018_holdout"]].copy()
        rows = []
        for _, r in board.iterrows():
            player = r["player"]
            if skill:
                pred = self._prediction_for(player, skill)
                confidence, _reasons, _caveats = trust.confidence_for(
                    pred["probs"], skill, pred["dist_to_nearest_cluster"], self.ood_threshold(), pred.get("n_missing", 0)
                )
                score = None if confidence == "abstain" else pred["probs"][2]
            else:
                preds = [self._prediction_for(player, s) for s in config.SKILLS]
                confidences = [
                    trust.confidence_for(
                        p["probs"], s, p["dist_to_nearest_cluster"], self.ood_threshold(), p.get("n_missing", 0)
                    )[0]
                    for p, s in zip(preds, config.SKILLS)
                ]
                non_abstain = [p["probs"][2] for p, c in zip(preds, confidences) if c != "abstain"]
                confidence = "abstain" if not non_abstain else ("low" if "abstain" in confidences else "medium")
                score = float(np.mean(non_abstain)) if non_abstain else None
            rows.append(
                {
                    "player": player,
                    "draft_pick": int(r["draft_pick"]) if pd.notna(r["draft_pick"]) else None,
                    "draft_round": int(r["draft_round"]) if pd.notna(r["draft_round"]) else None,
                    "draft_team": r["draft_team"] if pd.notna(r["draft_team"]) else None,
                    "model_score_top25_prob": round(score, 3) if score is not None else None,
                    "confidence": confidence,
                }
            )
        rows.sort(key=lambda d: (d["model_score_top25_prob"] is not None, d["model_score_top25_prob"] or 0), reverse=True)
        for i, row in enumerate(rows[:top_n], 1):
            row["model_rank"] = i
        return {"year": year, "skill": skill, "board": rows[:top_n]}
