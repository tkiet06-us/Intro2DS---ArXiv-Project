"""Chạy stage M2 (parse) cho một dải bài: Resolver -> Cleanup -> Tree -> BibTeX -> Dedup.

Đặt tại ``src/m2_parsing/run_m2.py``, chạy từ thư mục gốc repo::

    python -m src.m2_parsing.run_m2 --id_start 2506.00788 --id_end 2506.05787

Stage M2 đọc các bài do M1 đã tải về (``data/raw/papers_<name>``). Với mỗi version:
gộp nguồn LaTeX (``resolve_paper``), trích BibTeX trên nội dung đã ghép (TRƯỚC cleanup),
làm sạch + chuẩn hóa toán, rồi dựng cây phân cấp. Sau khi xong mọi version mới khử trùng lặp
BibTeX (Field Union) và full-text, cuối cùng ghi ``hierarchy.json`` + ``refs.bib`` vào thư mục
đầu ra (``data/processed/papers_<name>``).

Ghi chú đo đạc: thời gian/RAM/đĩa do ``Profiler`` của M1 đo và CHỈ phản ánh lần chạy hiện tại;
các chỉ số trạng thái/cấu trúc/dedup được gộp từ checkpoint (kể cả bài đã xong ở lần trước).
RAM đã tính cả tiến trình con (worker xử lý bài lớn), nhưng mẫu RAM có thể bỏ lỡ đỉnh rất ngắn
nên peak chỉ là xấp xỉ. Benchmark chính thức nên chạy ``--no_resume`` trên máy sạch.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field

from src.common.ids import to_folder_name
from src.common.utils_io import write_json
from src.m2_parsing import (
    bibtex_extractor,
    dedup,
    latex_cleanup,
    latex_parser,
    tree_builder,
)
from src.m2_parsing.tree_builder import ParsedNode
from src.schemas import make_element_id

logger = logging.getLogger(__name__)

# Trạng thái của một bài sau khi M2 xử lý.
STATUS_OK = "ok"  # xử lý xong, có ít nhất một version có TeX
STATUS_EMPTY = "empty_record"  # PDF-only / không có .tex: record rỗng hợp lệ
STATUS_FAILED = "failed"  # lỗi khiến không version nào dựng được
STATUS_SKIPPED_LARGE = "skipped_large"  # bài quá lớn, bỏ qua có chủ đích
STATUS_TIMEOUT = "timeout"  # vượt trần thời gian xử lý một bài
# Các trạng thái "đã xử lý": resume bỏ qua, không thử lại.
TERMINAL_STATUSES = frozenset({STATUS_OK, STATUS_EMPTY, STATUS_SKIPPED_LARGE, STATUS_TIMEOUT})

HIERARCHY_NAME = "hierarchy.json"
BIB_NAME = "refs.bib"
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

_VERSION_DIR = re.compile(r"^v(\d+)$", re.IGNORECASE)


class M2Settings(BaseModel):
    """Tham số stage M2; giá trị mặc định khớp ``configs/config.yaml`` mục ``m2``."""

    max_tex_bytes: int = Field(default=5_000_000, ge=0)
    max_tex_bytes_skip: int = Field(default=20_000_000, ge=0)
    max_include_depth: int = Field(default=10, ge=1)
    paper_timeout_s: float = Field(default=120.0, gt=0)
    max_nodes: int = Field(default=200_000, ge=1)
    failures_max: int = Field(default=100, ge=1)
    use_worker: bool = True
    profiler_interval_s: float = Field(default=0.5, gt=0)
    profiler_disk_every: int = Field(default=10, ge=1)

    @classmethod
    def from_config(cls, config_path: Path | str = DEFAULT_CONFIG_PATH) -> Self:
        """Đọc mục ``m2`` của file cấu hình; thiếu khóa nào thì dùng mặc định.

        Args:
            config_path: Đường dẫn ``configs/config.yaml``.

        Returns:
            Cấu hình M2 đã kiểm tra kiểu.
        """
        try:
            with Path(config_path).open(encoding="utf-8") as handle:
                section = (yaml.safe_load(handle) or {}).get("m2") or {}
        except FileNotFoundError:
            logger.warning("Khong thay %s, dung cau hinh M2 mac dinh", config_path)
            return cls()
        known = {key: section[key] for key in cls.model_fields if key in section}
        return cls(**known)


@dataclass
class PaperResult:
    """Kết quả xử lý một bài ở M2, đủ để dựng checkpoint và gộp vào summary."""

    arxiv_id: str
    status: str
    n_versions: int = 0
    n_versions_done: int = 0
    nodes_by_kind: dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    n_bib_before: int = 0
    n_bib_after: int = 0
    n_duplicate_nodes_merged: int = 0
    seconds: float = 0.0
    stage: str | None = None  # stage con gây lỗi: resolver/cleanup/tree/bibtex/dedup
    reason: str | None = None  # lý do lỗi (ngắn gọn)
    degraded: bool = False  # đã chạy chế độ nhẹ do bài lớn
    truncated: bool = False  # cây bị cắt do vượt max_nodes
    partial: bool = False  # còn version lỗi/chưa xử lý

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả sang dict phẳng (JSON-hóa được) cho checkpoint/summary."""
        return asdict(self)


def list_version_dirs(paper_dir: Path) -> list[tuple[int, Path]]:
    """Liệt kê các thư mục version ``v1``, ``v2``... của một bài, sắp theo số version.

    Args:
        paper_dir: Thư mục của một bài (chứa ``metadata.json`` và các thư mục version).

    Returns:
        Danh sách ``(số_version, đường_dẫn)`` đã sắp tăng dần; rỗng nếu không có version nào
        (bài PDF-only: git không giữ thư mục rỗng nên có thể không có ``v1``).
    """
    if not paper_dir.is_dir():
        return []
    versions = [
        (int(match.group(1)), child)
        for child in paper_dir.iterdir()
        if child.is_dir() and (match := _VERSION_DIR.match(child.name))
    ]
    versions.sort(key=lambda item: item[0])
    return versions


def _tree_max_depth(nodes: list[ParsedNode]) -> int:
    """Tính độ sâu lớn nhất của cây (nút gốc sâu 0).

    Duyệt theo thứ tự tiền thứ tự nên cha luôn được tính trước con.
    """
    depths: dict[str, int] = {}
    best = 0
    for node in nodes:
        depth = 0 if node.parent is None else depths.get(node.parent, 0) + 1
        depths[node.local_id] = depth
        best = max(best, depth)
    return best


def process_paper_m2(
    paper_dir: Path, arxiv_id: str, settings: M2Settings, out_dir: Path
) -> PaperResult:
    """Xử lý một bài ở M2 và ghi ``hierarchy.json`` + ``refs.bib`` vào ``out_dir``.

    Hàm không tự đặt trần thời gian (việc đó do lớp worker ở ``run_m2_stage`` lo) và không
    raise vì dữ liệu xấu: mọi lỗi ở từng version được log kèm ``arxiv_id`` rồi ghi vào
    ``PaperResult`` để một bài lỗi không làm sập cả dải.

    Args:
        paper_dir: Thư mục bài do M1 sinh ra (chứa các thư mục version).
        arxiv_id: Mã arXiv chuẩn của bài.
        settings: Cấu hình M2.
        out_dir: Thư mục gốc đầu ra của M2.

    Returns:
        ``PaperResult`` với trạng thái và số liệu của bài.
    """
    started = time.perf_counter()
    result = PaperResult(arxiv_id=arxiv_id, status=STATUS_OK)
    versions = list_version_dirs(Path(paper_dir))
    if not versions:
        result.status = STATUS_EMPTY
        result.reason = "khong co thu muc version"
        result.seconds = time.perf_counter() - started
        return result
    result.n_versions = len(versions)
    version_map = dict(versions)

    texts: dict[int, str] = {}
    for version, version_dir in versions:
        try:
            text = latex_parser.resolve_paper(version_dir)
        except Exception as exc:
            logger.exception("%s v%d: resolver loi", arxiv_id, version)
            result.stage = result.stage or "resolver"
            result.reason = result.reason or f"v{version}: {exc}"
            continue
        if text.strip():
            texts[version] = text

    if not texts:
        result.status = STATUS_FAILED if result.stage else STATUS_EMPTY
        if not result.stage:
            result.reason = "khong co .tex trong moi version"
        result.seconds = time.perf_counter() - started
        return result

    elements: dict[str, str] = {}
    hierarchy: dict[str, dict[str, str]] = {}
    kinds: Counter[str] = Counter()
    bib_by_version: dict[int, list[dict]] = {}

    for version in sorted(texts):
        text = texts[version]
        try:
            bib_by_version[version] = bibtex_extractor.extract_bibtex(text, version_map[version])
        except Exception as exc:
            logger.exception("%s v%d: bibtex loi", arxiv_id, version)
            result.stage = result.stage or "bibtex"
            result.reason = result.reason or f"v{version}: {exc}"
            bib_by_version[version] = []
        try:
            cleaned = latex_cleanup.cleanup_latex(text)
            cleaned = latex_cleanup.normalize_math(cleaned)
            nodes = tree_builder.build_tree(cleaned, version)
        except Exception as exc:
            logger.exception("%s v%d: cleanup/tree loi", arxiv_id, version)
            result.stage = result.stage or "tree"
            result.reason = result.reason or f"v{version}: {exc}"
            continue
        result.n_versions_done += 1
        result.max_depth = max(result.max_depth, _tree_max_depth(nodes))
        for node in nodes:
            element_id = make_element_id(arxiv_id, node.local_id)
            elements[element_id] = node.content
            kinds[str(node.kind)] += 1
            if node.parent is not None:
                parent_id = make_element_id(arxiv_id, node.parent)
                hierarchy.setdefault(str(version), {})[element_id] = parent_id

    all_bib = [entry for version in sorted(bib_by_version) for entry in bib_by_version[version]]
    result.n_bib_before = len(all_bib)
    deduped_bib, key_map = dedup.dedup_bibtex(all_bib)
    result.n_bib_after = len(deduped_bib)

    payload = {"elements": elements, "hierarchy": hierarchy}
    deduped_ft = dedup.dedup_fulltext(payload, key_map or None)
    result.n_duplicate_nodes_merged = max(len(elements) - len(deduped_ft["elements"]), 0)
    result.nodes_by_kind = dict(kinds)

    folder = Path(out_dir) / to_folder_name(arxiv_id)
    write_json(folder / HIERARCHY_NAME, deduped_ft)
    bibtex_extractor.write_bibtex(deduped_bib, folder / BIB_NAME)

    result.partial = result.stage is not None or result.n_versions_done < len(versions)
    if result.n_versions_done == 0:
        result.status = STATUS_FAILED if result.stage else STATUS_EMPTY
    result.seconds = time.perf_counter() - started
    return result


__all__ = [
    "STATUS_EMPTY",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_SKIPPED_LARGE",
    "STATUS_TIMEOUT",
    "TERMINAL_STATUSES",
    "M2Settings",
    "PaperResult",
    "list_version_dirs",
    "process_paper_m2",
]
