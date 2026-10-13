"""P2-04 (Owner: Khang) – Dedup 2 cấp: BibTeX Field Union + đổi tên \cite{} trùng; Full-text Dedup giữa các version SAU cleanup."""
from __future__ import annotations


def dedup_bibtex(entries: list[dict]) -> tuple[list[dict], dict]:
    raise NotImplementedError


def dedup_fulltext(versions: list[str]) -> list[str]:
    raise NotImplementedError

