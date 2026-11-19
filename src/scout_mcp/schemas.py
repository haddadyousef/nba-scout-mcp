"""Pydantic models for every tool's input and the shared response envelope
(project brief Section 5.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "abstain"]
PredictionType = Literal["out_of_fold", "holdout", "new_input"]
Tier = Literal["Bottom 25%", "Middle 50%", "Top 25%"]


class Provenance(BaseModel):
    prediction_type: PredictionType
    model_version: str
    data_through: str


class Envelope(BaseModel):
    """Every tool returns this shape."""

    answer: dict
    summary: str
    confidence: Confidence
    confidence_reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    provenance: Provenance


# --- tool inputs -----------------------------------------------------------


class SearchPlayersInput(BaseModel):
    query: str
    limit: int = 5


class GetSkillProfileInput(BaseModel):
    player: str


class CompareProspectsInput(BaseModel):
    player_a: str
    player_b: str
    skills: list[str] | None = None


class ExplainTierInput(BaseModel):
    player: str
    skill: str


class DraftClassBoardInput(BaseModel):
    year: int = 2018
    skill: str | None = None
    top_n: int = 15


class FindCompsInput(BaseModel):
    player: str
    k: int = 5


class ProjectCustomProspectInput(BaseModel):
    stats: dict[str, float]
    name: str = "Custom prospect"


class ScoutingReportArgs(BaseModel):
    player: str
