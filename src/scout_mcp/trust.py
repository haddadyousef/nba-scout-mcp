"""The trust layer: confidence rules and the out-of-distribution abstain
check (project brief Section 5.1). Implemented exactly as specified there;
tune thresholds in `config.py`, not here.
"""

from __future__ import annotations

import logging

import config
from scout_mcp.schemas import Confidence

logger = logging.getLogger("scout_mcp.trust")


def is_out_of_distribution(dist_to_nearest_cluster: float, ood_threshold: float, n_missing: int) -> tuple[bool, list[str]]:
    reasons = []
    ood = False
    if dist_to_nearest_cluster > ood_threshold:
        ood = True
        reasons.append(
            f"distance to nearest archetype centroid ({dist_to_nearest_cluster:.2f}) exceeds the "
            f"{config.OOD_CENTROID_PERCENTILE}th percentile of training distances ({ood_threshold:.2f})"
        )
    if n_missing > config.MAX_MISSING_INPUTS:
        ood = True
        reasons.append(f"{n_missing} required input stats were missing or imputed")
    return ood, reasons


def confidence_for(
    probs: list[float],
    skill: str,
    dist_to_nearest_cluster: float,
    ood_threshold: float,
    n_missing: int = 0,
) -> tuple[Confidence, list[str], list[str]]:
    """Returns (confidence, confidence_reasons, caveats)."""
    ood, ood_reasons = is_out_of_distribution(dist_to_nearest_cluster, ood_threshold, n_missing)
    caveats = []
    if skill in config.WEAK_SKILLS:
        caveats.append(
            f"{skill} is the model's weakest skill (65-75% tier accuracy does not hold for it; see model card)."
        )

    if ood:
        logger.info("abstain: skill=%s reasons=%s", skill, ood_reasons)
        return "abstain", ood_reasons, caveats

    sorted_probs = sorted(probs, reverse=True)
    top_prob = sorted_probs[0]
    second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob

    reasons = [f"top-class probability {top_prob:.2f}", "inside training distribution"]

    if top_prob >= config.HIGH_CONFIDENCE_PROB and skill not in config.WEAK_SKILLS:
        return "high", reasons, caveats

    if top_prob < config.MEDIUM_CONFIDENCE_PROB:
        reasons = [f"top-class probability {top_prob:.2f} (below {config.MEDIUM_CONFIDENCE_PROB})"]
        return "low", reasons, caveats

    if margin < config.CLOSE_CLASS_MARGIN:
        reasons = [f"top two classes within {margin:.2f} of each other"]
        return "low", reasons, caveats

    if skill in config.WEAK_SKILLS:
        reasons.append(f"{skill} has known weak accuracy")
        return "medium", reasons, caveats

    return "medium", reasons, caveats
