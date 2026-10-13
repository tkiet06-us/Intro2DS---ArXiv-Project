"""P2-06 (Owner: Kiệt & Bảo) – Preprocessing & Feature Engineering: Levenshtein, Jaccard n-gram, TF-IDF cosine."""
from __future__ import annotations


def preprocess(text: str) -> str:
    raise NotImplementedError


def build_features(bib_entry: dict, candidate: dict) -> list[float]:
    raise NotImplementedError

