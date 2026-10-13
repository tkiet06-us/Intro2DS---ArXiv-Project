"""P1-02 (Owner: Bảo) – Pydantic Schema phẳng, chuẩn Parquet cho 7 bảng HF.

Bảng: papers, elements, element_parents, bibtex_entries,
      paper_references, predictions, sources.
Chốt schema từ Tuần 1 (Interface-Driven Development).
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class Paper(BaseModel):
    arxiv_id: str
    title: Optional[str] = None
    authors: list[str] = []
    submission_date: Optional[str] = None  # ISO 8601
    revised_dates: list[str] = []
    publication_venue: Optional[str] = None
    # TODO: đủ 5 trường metadata ISO theo đề


class Element(BaseModel):
    element_id: str            # chứa arXiv ID
    arxiv_id: str
    kind: str                  # section|sentence|formula|figure|table|...
    content: str = ""
    level: Optional[int] = None


class ElementParent(BaseModel):
    element_id: str
    parent_id: Optional[str] = None
    order: int = 0


class BibtexEntry(BaseModel):
    arxiv_id: str
    citation_key: str
    title: Optional[str] = None
    authors: list[str] = []
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None


class PaperReference(BaseModel):
    arxiv_id: str
    cited_arxiv_id: str


class Prediction(BaseModel):
    arxiv_id: str
    partition: str             # train|valid|test
    citation_key: str
    groundtruth_arxiv_id: Optional[str] = None
    cand_1: Optional[str] = None
    cand_2: Optional[str] = None
    cand_3: Optional[str] = None
    cand_4: Optional[str] = None
    cand_5: Optional[str] = None


class Source(BaseModel):
    arxiv_id: str
    filename: str
    content: str               # nội dung .tex/.bib gốc
