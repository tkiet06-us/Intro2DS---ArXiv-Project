"""P4-02 (Owner: Bảo) – Xuất 7 bảng Parquet phẳng + upload Hugging Face."""
from __future__ import annotations


TABLES = ["papers", "elements", "element_parents", "bibtex_entries",
          "paper_references", "predictions", "sources"]


def export_all(processed_dir: str, out_dir: str) -> None:
    raise NotImplementedError

