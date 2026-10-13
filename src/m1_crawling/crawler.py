"""P2-01 (Owner: Kiên) – Crawl arXiv source mọi version.

Quy định: export.arxiv.org/src/{id}v{n}; User-Agent rõ; delay >=3s; 1 kết nối/lần.
Nhận diện .tar.gz & .gz đơn; xóa ảnh nhưng GIỮ \begin{figure}; log size_before/after.
PDF/Word-only -> record rỗng. CLI --id_start --id_end + Checkpoint/Resume.
"""
from __future__ import annotations


def crawl_paper(arxiv_id: str, out_dir: str) -> dict:
    raise NotImplementedError


def crawl_range(id_start: str, id_end: str, out_dir: str, resume: bool = True) -> None:
    raise NotImplementedError

