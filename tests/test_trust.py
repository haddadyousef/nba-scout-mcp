import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from scout_mcp import trust


def test_high_confidence_requires_high_prob_and_in_distribution():
    conf, _reasons, _caveats = trust.confidence_for(
        [0.05, 0.10, 0.85], "Rebounding", dist_to_nearest_cluster=1.0, ood_threshold=3.7
    )
    assert conf == "high"


def test_low_confidence_prob_below_floor():
    conf, _reasons, _caveats = trust.confidence_for(
        [0.40, 0.35, 0.25], "Rebounding", dist_to_nearest_cluster=1.0, ood_threshold=3.7
    )
    assert conf == "low"


def test_low_confidence_close_margin():
    conf, _reasons, _caveats = trust.confidence_for(
        [0.30, 0.34, 0.36], "Rebounding", dist_to_nearest_cluster=1.0, ood_threshold=3.7
    )
    assert conf == "low"


def test_abstain_when_out_of_distribution():
    conf, reasons, _caveats = trust.confidence_for(
        [0.05, 0.10, 0.85], "Rebounding", dist_to_nearest_cluster=10.0, ood_threshold=3.7
    )
    assert conf == "abstain"
    assert any("centroid" in r for r in reasons)


def test_abstain_when_too_many_missing_inputs():
    conf, reasons, _caveats = trust.confidence_for(
        [0.05, 0.10, 0.85], "Rebounding", dist_to_nearest_cluster=1.0, ood_threshold=3.7,
        n_missing=config.MAX_MISSING_INPUTS + 1,
    )
    assert conf == "abstain"
    assert any("missing" in r for r in reasons)


def test_weak_skill_never_reports_high():
    conf, _reasons, caveats = trust.confidence_for(
        [0.02, 0.03, 0.95], "Defense", dist_to_nearest_cluster=1.0, ood_threshold=3.7
    )
    assert conf != "high"
    assert any("weakest" in c for c in caveats)
