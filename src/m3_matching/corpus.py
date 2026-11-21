"""Đọc mục BibTeX và pool reference của từng bài từ thư mục làm việc của M1/M2 (M3).

Dùng khi CHƯA có bảng ``bibtex``/``references`` dạng Parquet (task P4-02): gán nhãn tự động,
khảo sát luật gán nhãn và gán nhãn tay có thể chạy ngay trên thư mục từng bài.

Nguồn dữ liệu của một bài ``<root>/<yymm-id>/``:
    - ``references.json`` (M1, ``s2_api``): pool ứng viên, chỉ gồm ref có arXiv ID, key dạng
      tên thư mục (``2403-13187``, ``hep-th_9901001``).
    - Mục BibTeX, theo thứ tự ưu tiên:
        1. ``refs.bib`` đã chuẩn hóa và khử trùng của M2 (``run_m2``) nếu có ``processed_root``;
        2. nếu chưa có, trích lại từ nguồn TeX thô ``v1/``, ``v2/``... bằng đúng các hàm của M2
           (``resolve_paper`` -> ``extract_bibtex`` -> ``dedup_bibtex_versions``).

Lưu ý: ``citation_key`` ở đường 2 chỉ trùng với bảng ``bibtex`` cuối cùng nếu parser M2 không
đổi giữa hai lần chạy (README mục 16, "đóng băng parser"). Nhãn tay nên gán trên đường 1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.common.ids import from_folder_name
from src.m2_parsing import bibtex_extractor, dedup, latex_parser
from src.schemas import Bibtex, Reference

logger = logging.getLogger(__name__)

REFERENCES_NAME = "references.json"
PROCESSED_BIB_NAME = "refs.bib"


@dataclass
class PaperRecord:
    """Một bài: các mục BibTeX (sau dedup) và pool reference có arXiv ID."""

    arxiv_id: str
    entries: list[Bibtex] = field(default_factory=list)
    candidates: list[Reference] = field(default_factory=list)
    entry_source: str = "none"  # processed | raw | none


def load_candidates(paper_dir: Path, arxiv_id: str) -> list[Reference]:
    """Đọc pool ứng viên từ ``references.json`` của M1, sắp tăng dần theo ``ref_arxiv_id``.

    Args:
        paper_dir: Thư mục bài, ví dụ ``data/raw/papers_kiet/2410-13025``.
        arxiv_id: ID chuẩn của bài đang trích dẫn.

    Returns:
        Danh sách ``Reference``; rỗng nếu chưa có ``references.json``.
    """
    path = paper_dir / REFERENCES_NAME
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    refs: list[Reference] = []
    for key in sorted(raw):
        info = raw[key] or {}
        refs.append(Reference(
            arxiv_id=arxiv_id,
            ref_arxiv_id=from_folder_name(key),
            ref_title=info.get("ref_title"),
            ref_authors=[str(a) for a in info.get("ref_authors") or []],
            ref_submission_date=info.get("ref_submission_date"),
            ref_s2_id=info.get("ref_s2_id"),
        ))
    return refs


def _to_bibtex(arxiv_id: str, entry: dict) -> Bibtex:
    """Đổi dict entry của M2 thành ``Bibtex``, bỏ các khóa ngoài schema."""
    allowed = set(Bibtex.model_fields) - {"arxiv_id"}
    data = {k: v for k, v in entry.items() if k in allowed}
    data["authors"] = [str(a) for a in data.get("authors") or []]
    return Bibtex(arxiv_id=arxiv_id, **data)


def _version_dirs(paper_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in paper_dir.glob("v*"):
        if path.is_dir() and path.name[1:].isdigit():
            found.append((int(path.name[1:]), path))
    return sorted(found)


def extract_entries_from_raw(paper_dir: Path) -> list[dict]:
    """Trích mục BibTeX từ nguồn TeX thô của mọi version rồi khử trùng như M2.

    Args:
        paper_dir: Thư mục bài chứa ``v1/``, ``v2/``...

    Returns:
        Danh sách entry (dict) sau ``dedup_bibtex_versions``; rỗng nếu không version nào có TeX.
    """
    by_version: dict[int, list[dict]] = {}
    for version, vdir in _version_dirs(paper_dir):
        latex = latex_parser.resolve_paper(vdir)
        # Trích trên nội dung đã ghép, TRƯỚC cleanup, giống run_m2.
        by_version[version] = bibtex_extractor.extract_bibtex(latex, vdir) if latex else []
    entries, _ = dedup.dedup_bibtex_versions(by_version)
    return entries


def load_paper(
    paper_dir: Path, processed_root: Path | None = None
) -> PaperRecord:
    """Đọc một bài: pool từ M1, mục BibTeX từ M2 (``refs.bib``) hoặc trích lại từ TeX thô.

    Args:
        paper_dir: Thư mục bài của M1 (``<raw_root>/<yymm-id>``).
        processed_root: Thư mục đầu ra M2 (chứa ``<yymm-id>/refs.bib``); ``None`` thì luôn
            trích lại từ TeX thô.

    Returns:
        ``PaperRecord``; bài lỗi khi trích thì ``entries`` rỗng và lỗi được log.
    """
    arxiv_id = from_folder_name(paper_dir.name)
    record = PaperRecord(arxiv_id=arxiv_id, candidates=load_candidates(paper_dir, arxiv_id))
    processed_bib = processed_root / paper_dir.name / PROCESSED_BIB_NAME if processed_root else None
    try:
        if processed_bib is not None and processed_bib.is_file():
            raw_entries = bibtex_extractor.parse_bib_file(processed_bib)
            record.entry_source = "processed"
        else:
            raw_entries = extract_entries_from_raw(paper_dir)
            record.entry_source = "raw"
        seen: set[str] = set()
        for entry in raw_entries:
            key = str(entry.get("citation_key", "")).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            record.entries.append(_to_bibtex(arxiv_id, entry))
    except Exception:  # một bài lỗi không được làm sập cả dải (README 6.9.8)
        logger.exception("Không đọc được mục BibTeX của %s", arxiv_id)
        record.entries = []
        record.entry_source = "none"
    return record


def iter_paper_dirs(raw_root: Path, ids: set[str] | None = None) -> list[Path]:
    """Liệt kê thư mục bài trong ``raw_root`` (có ``references.json``), lọc theo ``ids`` nếu có.

    Args:
        raw_root: Thư mục chứa các thư mục ``<yymm-id>`` của M1.
        ids: Tập arXiv ID chuẩn cần lấy; ``None`` là lấy hết.

    Returns:
        Danh sách thư mục, sắp theo tên.
    """
    dirs = []
    for path in sorted(raw_root.iterdir()):
        if not path.is_dir() or not (path / REFERENCES_NAME).is_file():
            continue
        if ids is not None and from_folder_name(path.name) not in ids:
            continue
        dirs.append(path)
    return dirs


__all__ = [
    "PaperRecord", "extract_entries_from_raw", "iter_paper_dirs", "load_candidates", "load_paper",
]
