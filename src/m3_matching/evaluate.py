"""P4-01 (Owner: Kiệt) – Đánh giá MRR trên tập Test."""
from __future__ import annotations


def mrr(ranked: list[list[str]], truth: list[str]) -> float:
    total = 0.0
    for cands, t in zip(ranked, truth):
        for i, c in enumerate(cands, 1):
            if c == t:
                total += 1 / i
                break
    return total / max(len(truth), 1)

