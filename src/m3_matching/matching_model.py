"""P2-06 (Owner: Kiệt) – Ranking Model cho Reference Matching."""
from __future__ import annotations


class MatchingModel:
    def fit(self, X, y): raise NotImplementedError
    def rank(self, X, k: int = 5): raise NotImplementedError

