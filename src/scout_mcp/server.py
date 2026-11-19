"""The MCP server: a thin adapter over `scout_mcp.engine.ScoutEngine`. No
ML logic lives here. Every tool resolves names, calls the engine, and
packages the result into the response envelope (project brief Section 5.1).
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

import config
from scout_mcp import trust
from scout_mcp.engine import ScoutEngine, UnknownPlayerError, UnknownSkillError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scout_mcp.server")

mcp = MCPServer(
    name="nba-scout-mcp",
    title="NBA Scout",
    instructions=(
        "Consult a published NBA draft-scouting model (Shopping by Skill, MIT Sloan SSAC 2026) "
        "the way you'd consult a human scout: skill-by-skill tier predictions, archetypes, "
        "comparable players, the reasons behind a prediction, and an honest confidence/abstain "
        "signal. The model only ever sees pre-draft college box-score stats: no measurements, "
        "no film, no injury history. Start with search_players to resolve a name, then "
        "get_skill_profile for an overview, compare_prospects to put two players side by side, "
        "and explain_tier when you need the reasoning behind one skill's grade."
    ),
    version=config.MODEL_VERSION,
)

_engine: ScoutEngine | None = None


def get_engine() -> ScoutEngine:
    global _engine
    if _engine is None:
        _engine = ScoutEngine()
    return _engine


def envelope(answer: dict, summary: str, confidence: str, confidence_reasons: list[str],
             caveats: list[str], prediction_type: str) -> dict:
    return {
        "answer": answer,
        "summary": summary,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "caveats": caveats,
        "provenance": {
            "prediction_type": prediction_type,
            "model_version": config.MODEL_VERSION,
            "data_through": config.DATA_THROUGH,
        },
    }


def error_envelope(message: str, candidates: list[str] | None = None) -> dict:
    return {
        "answer": {"error": message, "candidates": candidates or []},
        "summary": message,
        "confidence": "abstain",
        "confidence_reasons": ["could not resolve the request"],
        "caveats": [],
        "provenance": {
            "prediction_type": "new_input",
            "model_version": config.MODEL_VERSION,
            "data_through": config.DATA_THROUGH,
        },
    }


@mcp.tool()
def search_players(query: str, limit: int = 5) -> dict:
    """Resolve a (possibly misspelled) player name to one or more canonical
    names. Use this FIRST for any player you don't have an exact, verified
    name for. Every other tool here accepts a canonical name and will
    return candidates instead of guessing if the name is ambiguous or not
    found. The college pipeline runs 1985-2019 and NBA outcomes 1996-2025,
    so very recent or very old players may not be covered; this tool tells
    you whether a player is in the model's training set, in the 2018 draft
    holdout, or neither.
    """
    try:
        matches = get_engine().search(query, limit=limit)
    except Exception as e:  # never a stack trace to the agent
        logger.exception("search_players failed")
        return error_envelope(f"search_players failed: {e}")

    if not matches:
        return envelope(
            {"matches": []}, f"No player matched '{query}'.", "abstain",
            ["no fuzzy match above the score cutoff"], [], "new_input",
        )
    summary = f"Found {len(matches)} match(es) for '{query}'; best match: {matches[0]['player']}."
    return envelope({"matches": matches}, summary, "high", ["exact or high-confidence name match"], [], "new_input")


@mcp.tool()
def get_skill_profile(player: str) -> dict:
    """Full skill-by-skill scouting profile for one player: predicted tier
    (Bottom 25% / Middle 50% / Top 25% of all NBA players at that skill),
    class probabilities, confidence, the actual NBA outcome when known, and
    the player's archetype. Tiers are relative to the full 1996-2025 NBA
    population at that skill, not to the player's draft class. Defense is
    the model's weakest skill, so expect lower confidence there. Uses college
    box-score stats only.
    """
    try:
        canonical = get_engine().resolve(player)
    except UnknownPlayerError as e:
        return error_envelope(f"'{player}' is ambiguous or not found.", e.candidates)
    except Exception as e:
        logger.exception("get_skill_profile failed")
        return error_envelope(f"get_skill_profile failed: {e}")

    profile = get_engine().skill_profile(canonical)
    confidences = [s["confidence"] for s in profile["skills"].values()]
    overall_confidence = "abstain" if "abstain" in confidences else (
        "low" if "low" in confidences else ("medium" if "medium" in confidences else "high")
    )
    best_skill = max(
        (s for s in config.SKILLS if profile["skills"][s]["predicted_tier"]),
        key=lambda s: profile["skills"][s]["probabilities"]["Top 25%"],
        default=None,
    )
    summary = f"{canonical} ({profile['archetype']['description']})."
    if best_skill:
        summary += f" Strongest projected skill: {best_skill} ({profile['skills'][best_skill]['predicted_tier']})."
    caveats = sorted({c for s in profile["skills"].values() for c in s["caveats"]})
    return envelope(profile, summary, overall_confidence, ["see per-skill confidence_reasons in answer"], caveats, "out_of_fold")


@mcp.tool()
def compare_prospects(player_a: str, player_b: str, skills: list[str] | None = None) -> dict:
    """Side-by-side comparison of two players, skill by skill, with the
    edge (which player, and by how much modeled probability of reaching
    Top 25%) for each skill. If either player abstains on a skill (out of
    distribution, or too little data), that skill is marked
    not-comparable instead of picking a winner. `skills` defaults to all
    four (Shooter, Rebounding, Playmaking, Defense).
    """
    try:
        result = get_engine().compare(player_a, player_b, skills)
    except UnknownPlayerError as e:
        return error_envelope(f"'{e.query}' is ambiguous or not found.", e.candidates)
    except UnknownSkillError as e:
        return error_envelope(str(e))
    except Exception as e:
        logger.exception("compare_prospects failed")
        return error_envelope(f"compare_prospects failed: {e}")

    comparable = [s for s, v in result["by_skill"].items() if v.get("comparable")]
    edges = [f"{s}: {v['edge']}" for s, v in result["by_skill"].items() if v.get("comparable")]
    summary = f"{result['player_a']} vs {result['player_b']}: " + (
        "; ".join(edges) if edges else "no comparable skills (abstained on all)"
    )
    confidence = "medium" if comparable else "abstain"
    return envelope(result, summary, confidence, [f"{len(comparable)}/{len(result['by_skill'])} skills comparable"], [], "out_of_fold")


@mcp.tool()
def explain_tier(player: str, skill: str) -> dict:
    """Why the model predicts what it predicts: the top 5 feature
    contributions (SHAP) for this player and skill, each feature's raw
    value and training-set percentile, the player's archetype and distance
    to its centroid, and the 3 nearest comparable players in archetype
    space with their actual NBA tiers. This is the model's reasoning, not
    just its answer. Use it when an agent needs to justify a scouting
    claim. `skill` is one of Shooter, Rebounding, Playmaking, Defense.
    """
    try:
        canonical = get_engine().resolve(player)
        result = get_engine().explain(canonical, skill)
    except UnknownPlayerError as e:
        return error_envelope(f"'{player}' is ambiguous or not found.", e.candidates)
    except UnknownSkillError as e:
        return error_envelope(str(e))
    except Exception as e:
        logger.exception("explain_tier failed")
        return error_envelope(f"explain_tier failed: {e}")

    known = result["known"]
    tier_name = config.TIER_NAMES[result["pred_tier"]]
    top_feat = result["top_contributions"][0]["feature"] if result["top_contributions"] else "n/a"
    summary = (
        f"{canonical} projects to {tier_name} in {skill}, driven most by {top_feat}. "
        f"Archetype: {result['archetype']['description']}."
    )
    confidence = "abstain" if known is None else "medium"
    return envelope(result, summary, confidence, [], result["caveats"], known["prediction_type"] if known else "new_input")


@mcp.tool()
def draft_class_board(year: int = 2018, skill: str | None = None, top_n: int = 15) -> dict:
    """Rank a draft class by modeled P(Top 25%) for one skill, or by the
    average across all four skills. Only the 2018 class has a genuine
    holdout board right now (the model never trained on these players'
    college stats). This is the tool for the "who was undervalued"
    story: compare `model_rank` against `draft_pick` to find players the
    model liked more than NBA teams did at the time.
    """
    try:
        result = get_engine().draft_board(year, skill, top_n)
    except ValueError as e:
        return error_envelope(str(e))
    except UnknownSkillError as e:
        return error_envelope(str(e))
    except Exception as e:
        logger.exception("draft_class_board failed")
        return error_envelope(f"draft_class_board failed: {e}")

    undervalued = [
        r for r in result["board"]
        if r["draft_pick"] and r["model_rank"] < r["draft_pick"] - 10
    ]
    summary = f"{year} board ({skill or 'combined'}), top {len(result['board'])}."
    if undervalued:
        names = ", ".join(f"{r['player']} (pick {r['draft_pick']}, model rank {r['model_rank']})" for r in undervalued[:3])
        summary += f" Model-undervalued-by-the-room: {names}."
    return envelope(result, summary, "high", ["genuine 2018 holdout: never in the training set"], [], "holdout")


@mcp.tool()
def find_comps(player: str, k: int = 5) -> dict:
    """Nearest neighbors to a player in archetype space (PCA over 12
    standardized college stats), with their actual NBA outcomes when
    known. Good for "who does this remind you of" questions.
    """
    try:
        canonical = get_engine().resolve(player)
        comps = get_engine().nearest_comps(canonical, k=k)
    except UnknownPlayerError as e:
        return error_envelope(f"'{player}' is ambiguous or not found.", e.candidates)
    except Exception as e:
        logger.exception("find_comps failed")
        return error_envelope(f"find_comps failed: {e}")

    summary = f"Nearest comps to {canonical}: " + ", ".join(c["player"] for c in comps)
    return envelope({"player": canonical, "comps": comps}, summary, "medium", ["archetype-space distance, not an outcome prediction"], [], "out_of_fold")


@mcp.tool()
def project_custom_prospect(stats: dict[str, float], name: str = "Custom prospect") -> dict:
    """Score a college stat line that isn't in the database, e.g. a
    current prospect you found elsewhere. Provide college box-score
    totals/percentages using the same field names as the model (Games,
    Minutes_Per_Game, FG, FGA, FG%, 3P, 3PA, 3P%, FT, FTA, FT%, TRB, AST,
    STL, BLK, TOV, PTS, TS%, eFG%, TRB%, AST%, STL%, BLK%, TOV%, USG%, OWS,
    DWS, WS, WS/40, LastConf). Missing fields are treated as 0 and count
    toward the abstain rule, so this will usually abstain or run low
    confidence on a sparse stat line. That's the trust layer working as
    intended, not a bug.
    """
    try:
        out = {}
        for skill in config.SKILLS:
            pred = get_engine().score_new_input(stats, skill)
            confidence, reasons, caveats = trust.confidence_for(
                pred["probs"], skill, pred["dist_to_nearest_cluster"], get_engine().ood_threshold(), pred["n_missing"],
            )
            out[skill] = {
                "predicted_tier": None if confidence == "abstain" else config.TIER_NAMES[pred["pred_tier"]],
                "probabilities": dict(zip(config.TIER_NAMES, pred["probs"])),
                "confidence": confidence,
                "confidence_reasons": reasons,
                "caveats": caveats,
            }
    except Exception as e:
        logger.exception("project_custom_prospect failed")
        return error_envelope(f"project_custom_prospect failed: {e}")

    confidences = [v["confidence"] for v in out.values()]
    overall = "abstain" if "abstain" in confidences else ("low" if "low" in confidences else "medium")
    summary = f"{name}: " + ", ".join(f"{s}={v['predicted_tier'] or 'abstain'}" for s, v in out.items())
    return envelope({"name": name, "skills": out}, summary, overall, [], [], "new_input")


@mcp.tool()
def get_model_card() -> dict:
    """The model card: data coverage, features, per-skill accuracy, and
    known limitations. Also available as the `scout://model-card` resource.
    """
    return _model_card()


def _model_card() -> dict:
    engine = get_engine()
    accuracy = {
        skill: {
            "served_model": rec["model_type"],
            "served_cv_accuracy": round(rec["served_accuracy"], 3),
            "mean_cv_accuracy_by_model": {k: round(v, 3) for k, v in rec["mean_cv_accuracy"].items()},
            "weak_skill": rec["is_weak_skill"],
        }
        for skill, rec in engine.skill_models.items()
    }
    return {
        "model_version": config.MODEL_VERSION,
        "data_through": config.DATA_THROUGH,
        "research_repo": config.RESEARCH_REPO,
        "research_repo_commit": config.RESEARCH_REPO_COMMIT,
        "accuracy_by_skill": accuracy,
        "n_training_players": int(engine.players["in_training_set"].sum()),
        "n_2018_holdout_players": int(engine.players["in_2018_holdout"].sum()),
        "ood_distance_threshold": engine.ood_threshold(),
        "known_limitations": [
            "College box-score stats only: no measurements, no film, no injury history.",
            "Defense is the weakest skill; treat Defense predictions with extra caution.",
            "Tiers are relative to the full 1996-2025 NBA outcome population, not to a draft class.",
            "The 2018 holdout only covers players who played NCAA basketball before the draft.",
        ],
    }


@mcp.resource("scout://model-card")
def model_card_resource() -> dict:
    """Model card: data, features, per-skill accuracy, and known limitations."""
    return _model_card()


@mcp.resource("scout://archetypes")
def archetypes_resource() -> dict:
    """The 5 player archetypes (PCA + K-means over 12 standardized college
    stats) with descriptions and example players."""
    engine = get_engine()
    archetype = engine.preprocess["archetype"]
    return {
        str(cluster): {"description": desc, "examples": archetype["examples"].get(cluster, [])}
        for cluster, desc in archetype["descriptions"].items()
    }


@mcp.prompt()
def scouting_report(player: str) -> str:
    """Template: call get_skill_profile, then explain_tier on the
    strongest and weakest skills, then write a short report that quotes
    the confidence and caveats."""
    return (
        f"Write a short scouting report on {player} using the nba-scout-mcp tools. Steps:\n"
        f"1. Call search_players if you don't have {player}'s exact canonical name.\n"
        f"2. Call get_skill_profile({player!r}) to get the full skill-by-skill picture.\n"
        "3. Call explain_tier on the player's strongest skill and on their weakest skill "
        "(by Top 25% probability) to get the reasoning behind each.\n"
        "4. Write 3-5 sentences: lead with the strongest skill and the archetype, note the "
        "weakest skill and why (quote the SHAP-driving feature), and explicitly state the "
        "confidence level and any caveats for each skill you mention. Never state a tier "
        "without its confidence."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
