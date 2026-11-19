"""Fuzzy player-name resolution (project brief Section 5.2, `search_players`).

Agents misspell names, so every tool that takes a player name resolves it
through here first. An ambiguous or unmatched query returns candidates
instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

import config


@dataclass
class NameMatch:
    player: str
    score: float


class NameResolver:
    def __init__(self, players: pd.DataFrame):
        self.players = players
        self._names = players["player"].tolist()
        self._lower_to_name = {n.lower(): n for n in self._names}

    def exact(self, query: str) -> str | None:
        return self._lower_to_name.get(query.strip().lower())

    def search(self, query: str, limit: int = 5) -> list[NameMatch]:
        exact = self.exact(query)
        if exact:
            return [NameMatch(player=exact, score=100.0)]

        results = process.extract(
            query, self._names, scorer=fuzz.WRatio, limit=limit,
            score_cutoff=config.FUZZY_MATCH_SCORE_CUTOFF,
        )
        return [NameMatch(player=name, score=score) for name, score, _ in results]

    def resolve_one(self, query: str) -> tuple[str | None, list[NameMatch]]:
        """Returns (canonical_name, candidates). canonical_name is None
        when the query is ambiguous or unmatched, in which case the
        caller should return `candidates` to the agent rather than guess."""
        matches = self.search(query, limit=5)
        if not matches:
            return None, []
        if len(matches) == 1 or matches[0].score >= 99.0:
            return matches[0].player, matches
        # Ambiguous: top match isn't a clear winner over the runner-up.
        if matches[0].score - matches[1].score < 5.0:
            return None, matches
        return matches[0].player, matches
