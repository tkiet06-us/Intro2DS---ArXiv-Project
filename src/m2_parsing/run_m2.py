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

import argparse
import logging
import multiprocessing
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field

from src.common.ids import from_folder_name, to_folder_name, validate_arxiv_id
from src.common.utils_io import append_jsonl, iter_jsonl, write_json
from src.m1_crawling.profiler import Profiler
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

# Thứ tự kind trong summary cho khớp schema (schemas.Kind).
NODE_KINDS = ("document", "heading", "itemize", "item", "sentence", "formula", "figure")

# Mã thoát của CLI (khớp quy ước main.py: 0 xong, 1 lỗi tham số/dữ liệu, 130 khi Ctrl+C).
EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_INTERRUPTED = 130

# Tên stage dùng làm khóa checkpoint (đổi dải thì không dùng nhầm kết quả cũ).
CHECKPOINT_STAGE = "m2"

_VERSION_DIR = re.compile(r"^v([1-9]\d*)$")


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


def _output_folder(out_dir: Path, arxiv_id: str) -> Path:
    """Thư mục đầu ra của một bài (dùng chung cho lúc ghi và lúc dọn kết quả cũ)."""
    return Path(out_dir) / to_folder_name(arxiv_id)


def _discard_outputs(folder: Path) -> None:
    """Xóa ``hierarchy.json``/``refs.bib`` cũ của bài để lần lỗi không để lại output giả hợp lệ."""
    for name in (HIERARCHY_NAME, BIB_NAME):
        try:
            (folder / name).unlink(missing_ok=True)
        except OSError:
            logger.warning("Khong xoa duoc output cu: %s", Path(folder) / name)


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

    Hàm không tự đặt trần thời gian (việc đó do ``PaperWorker`` ở ``run_stage`` lo) và không
    raise vì dữ liệu xấu: mọi lỗi ở từng version được log kèm ``arxiv_id`` rồi ghi vào
    ``PaperResult`` để một bài lỗi không làm sập cả dải.

    Bài lớn được xử lý hai mức: vượt ``max_tex_bytes`` thì chạy chế độ nhẹ (bỏ ``normalize_math``
    và ``dedup_fulltext``, đặt cờ ``degraded``); vượt ``max_tex_bytes_skip`` thì trả
    ``skipped_large``. Cây vượt ``max_nodes`` bị cắt và đặt cờ ``truncated``.

    Chỉ ghi ``hierarchy.json``/``refs.bib`` khi có ít nhất một version dựng được cây. Bài
    ``failed``/``empty_record``/``skipped_large`` không để lại output cũ (nếu có) trong thư mục.

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
        _discard_outputs(_output_folder(out_dir, arxiv_id))
        result.seconds = time.perf_counter() - started
        return result
    result.n_versions = len(versions)
    version_map = dict(versions)

    texts: dict[int, str] = {}
    for version, version_dir in versions:
        try:
            text = latex_parser.resolve_paper(version_dir, max_depth=settings.max_include_depth)
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
        _discard_outputs(_output_folder(out_dir, arxiv_id))
        result.seconds = time.perf_counter() - started
        return result

    total_bytes = sum(len(text.encode("utf-8")) for text in texts.values())
    if total_bytes > settings.max_tex_bytes_skip:
        result.status = STATUS_SKIPPED_LARGE
        result.reason = f"tong {total_bytes} byte > {settings.max_tex_bytes_skip}"
        _discard_outputs(_output_folder(out_dir, arxiv_id))
        result.seconds = time.perf_counter() - started
        return result
    # Chế độ nhẹ: bỏ normalize_math nặng và dedup_fulltext, giữ BibTeX + cây cơ bản.
    degraded = total_bytes > settings.max_tex_bytes
    result.degraded = degraded

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
            if not degraded:
                cleaned = latex_cleanup.normalize_math(cleaned)
            nodes = tree_builder.build_tree(cleaned, version)
        except Exception as exc:
            logger.exception("%s v%d: cleanup/tree loi", arxiv_id, version)
            result.stage = result.stage or "tree"
            result.reason = result.reason or f"v{version}: {exc}"
            continue
        if len(nodes) > settings.max_nodes:
            # Cắt đuôi an toàn: duyệt tiền thứ tự nên cha luôn nằm trước con trong danh sách.
            nodes = nodes[: settings.max_nodes]
            result.truncated = True
        result.n_versions_done += 1
        result.max_depth = max(result.max_depth, _tree_max_depth(nodes))
        for node in nodes:
            element_id = make_element_id(arxiv_id, node.local_id)
            elements[element_id] = node.content
            kinds[str(node.kind)] += 1
            if node.parent is not None:
                parent_id = make_element_id(arxiv_id, node.parent)
                hierarchy.setdefault(str(version), {})[element_id] = parent_id

    result.partial = result.stage is not None or result.n_versions_done < len(versions)
    if result.n_versions_done == 0:
        # Không version nào dựng được cây: không ghi hierarchy.json/refs.bib rỗng (trông như
        # hợp lệ) và dọn luôn output của lần chạy trước nếu có.
        result.status = STATUS_FAILED if result.stage else STATUS_EMPTY
        if not result.stage:
            result.reason = result.reason or "khong version nao dung duoc cay"
        _discard_outputs(_output_folder(out_dir, arxiv_id))
        result.seconds = time.perf_counter() - started
        return result

    all_bib = [entry for version in sorted(bib_by_version) for entry in bib_by_version[version]]
    result.n_bib_before = len(all_bib)
    deduped_bib, key_map = dedup.dedup_bibtex(all_bib)
    result.n_bib_after = len(deduped_bib)

    payload = {"elements": elements, "hierarchy": hierarchy}
    if degraded:
        output = payload
    else:
        output = dedup.dedup_fulltext(payload, key_map or None)
        result.n_duplicate_nodes_merged = max(len(elements) - len(output["elements"]), 0)
    result.nodes_by_kind = dict(kinds)

    folder = _output_folder(out_dir, arxiv_id)
    write_json(folder / HIERARCHY_NAME, output)
    bibtex_extractor.write_bibtex(deduped_bib, folder / BIB_NAME)

    result.seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# Dải ID, checkpoint/resume và vòng lặp stage
# ---------------------------------------------------------------------------


def id_range_key(arxiv_id: str) -> tuple[str, int]:
    """Khóa so sánh ID theo thứ tự ``(tháng, số thứ tự)`` thay vì so chuỗi thô.

    ID mới ``YYMM.NNNNN`` cho khóa ``("YYMM", NNNNN)``; ID cũ ``archive/YYMMNNN`` cho khóa
    ``("archive", YYMMNNN)``. Vì tiền tố cùng dạng (đều là chuỗi 4 chữ số với ID mới) nên so
    sánh cặp này đúng thứ tự, kể cả khi dải đi qua nhiều tháng.

    Args:
        arxiv_id: Mã arXiv của bài (không kèm hậu tố version).

    Returns:
        Cặp ``(tiền_tố, số_thứ_tự)`` so sánh được bằng toán tử thông thường.

    Raises:
        ValueError: Khi ``arxiv_id`` không đúng dạng chuẩn.
    """
    validate_arxiv_id(arxiv_id)
    if "/" in arxiv_id:
        prefix, _, number = arxiv_id.partition("/")
        return prefix, int(number)
    prefix, _, number = arxiv_id.partition(".")
    return prefix, int(number)


def validate_id_range(id_start: str, id_end: str) -> None:
    """Kiểm tra hai đầu dải ID có cùng dạng chuẩn và đúng thứ tự.

    Không hỗ trợ dải trộn giữa ID cũ (archive/YYMMNNN) và ID mới (YYMM.NNNNN), hoặc giữa hai
    archive khác nhau của ID cũ (vd hep-th và math), vì chúng không thuộc cùng một dãy liên tục.

    Args:
        id_start: ID đầu dải.
        id_end: ID cuối dải.

    Raises:
        ValueError: Khi ID không hợp lệ, hai ID khác dạng, khác archive, hoặc id_start > id_end.
    """
    validate_arxiv_id(id_start)
    validate_arxiv_id(id_end)
    is_old_start = "/" in id_start
    is_old_end = "/" in id_end
    if is_old_start != is_old_end:
        raise ValueError(
            f"--id_start ({id_start}) va --id_end ({id_end}) khong cung dang ID (cu / moi)"
        )
    if is_old_start:
        arch_start = id_start.partition("/")[0]
        arch_end = id_end.partition("/")[0]
        if arch_start != arch_end:
            raise ValueError(
                f"--id_start ({id_start}) va --id_end ({id_end}) khac archive ({arch_start} != {arch_end})"
            )
    if id_range_key(id_start) > id_range_key(id_end):
        raise ValueError(f"--id_start ({id_start}) phai <= --id_end ({id_end})")


def list_papers_in_range(
    papers_root: Path, id_start: str, id_end: str, limit: int | None = None
) -> list[tuple[str, Path]]:
    """Liệt kê bài M1 đã tải nằm trong ``[id_start, id_end]`` (không tự sinh ID mới).

    Duyệt thư mục ``papers_<name>`` do M1 sinh ra nên dải đi qua nhiều tháng vẫn đúng mà không
    cần giả định mỗi tháng có bao nhiêu bài. Mục không đúng dạng tên thư mục ID hoặc không cùng
    dạng ID/archive với dải bị bỏ qua (dữ liệu lạ không làm sập cả dải).

    Args:
        papers_root: Thư mục chứa các thư mục bài (``data/raw/papers_<name>``).
        id_start: ID đầu dải (gồm cả ID này).
        id_end: ID cuối dải (gồm cả ID này).
        limit: Chỉ lấy N bài đầu (chạy thử); None là lấy hết.

    Returns:
        Danh sách ``(arxiv_id, đường_dẫn_thư_mục_bài)`` sắp theo thứ tự ID.

    Raises:
        ValueError: Khi ``id_start`` và ``id_end`` khác dạng ID/archive hoặc ``id_start > id_end``.
    """
    if not papers_root.is_dir():
        return []
    validate_id_range(id_start, id_end)
    low, high = id_range_key(id_start), id_range_key(id_end)
    is_old_range = "/" in id_start
    range_archive = id_start.partition("/")[0] if is_old_range else None
    found: list[tuple[str, Path]] = []
    for child in papers_root.iterdir():
        if not child.is_dir():
            continue
        try:
            arxiv_id = from_folder_name(child.name)
            if ("/" in arxiv_id) != is_old_range:
                continue
            if is_old_range and arxiv_id.partition("/")[0] != range_archive:
                continue
            key = id_range_key(arxiv_id)
        except ValueError:
            logger.warning("Bo qua muc khong dung dang ID trong %s: %s", papers_root, child.name)
            continue
        if low <= key <= high:
            found.append((arxiv_id, child))
    found.sort(key=lambda item: id_range_key(item[0]))
    return found[:limit] if limit else found


@dataclass
class M2Paths:
    """Đường dẫn của một lần chạy M2, suy ra từ ``--name`` để nhiều người không đè dữ liệu nhau."""

    name: str
    papers_in: Path
    papers_out: Path
    reports: Path
    checkpoint: Path
    summary: Path

    @classmethod
    def build(
        cls, name: str, data_root: Path, processed_root: Path, reports: Path
    ) -> M2Paths:
        """Dựng bộ đường dẫn cho ``--name`` từ các thư mục gốc."""
        return cls(
            name=name,
            papers_in=data_root / f"papers_{name}",
            papers_out=processed_root / f"papers_{name}",
            reports=reports,
            checkpoint=reports / f"checkpoint_m2_{name}.jsonl",
            summary=reports / f"summary_stats_m2_{name}.json",
        )


def load_checkpoint(path: Path, id_start: str, id_end: str) -> dict[str, dict]:
    """Đọc checkpoint và trả về bản ghi MỚI NHẤT của từng ``arxiv_id`` thuộc đúng dải.

    Bản ghi của dải khác (khác ``id_start``/``id_end``) bị bỏ qua để đổi dải không dùng nhầm
    kết quả cũ.

    Args:
        path: File checkpoint ``.jsonl`` (không tồn tại thì trả dict rỗng).
        id_start: ID đầu dải của lần chạy hiện tại.
        id_end: ID cuối dải của lần chạy hiện tại.

    Returns:
        ``arxiv_id`` -> bản ghi checkpoint mới nhất.
    """
    state: dict[str, dict] = {}
    for record in iter_jsonl(path):
        if (
            record.get("stage") == CHECKPOINT_STAGE
            and record.get("id_start") == id_start
            and record.get("id_end") == id_end
            and "arxiv_id" in record
        ):
            state[record["arxiv_id"]] = record
    return state


def _resume_skip(record: dict, retry_failed: bool = False) -> bool:
    """True nếu checkpoint cho thấy bài đã "quyết định xong" (không thử lại khi resume).

    Gồm cả ``failed``: mặc định không thử lại để số đo benchmark ổn định. Bật ``retry_failed``
    (cờ ``--retry_failed``) để chỉ chạy lại các bài ``failed`` mà không phải bỏ cả checkpoint.
    """
    status = record.get("status")
    if retry_failed and status == STATUS_FAILED:
        return False
    return status in TERMINAL_STATUSES or status == STATUS_FAILED


def _paper_result(data: dict, arxiv_id: str, fallback_status: str = STATUS_FAILED) -> PaperResult:
    """Dựng ``PaperResult`` từ dict (bỏ qua khóa lạ/thiếu, giữ giá trị mặc định của dataclass)."""
    known = {item.name: data[item.name] for item in fields(PaperResult) if item.name in data}
    known.setdefault("arxiv_id", arxiv_id)
    known.setdefault("status", fallback_status)
    return PaperResult(**known)


def _result_from_record(record: dict, arxiv_id: str) -> PaperResult:
    """Dựng lại ``PaperResult`` từ bản ghi checkpoint (bỏ qua khóa lạ/thiếu)."""
    return _paper_result(record.get("result") or {}, arxiv_id, record.get("status", STATUS_FAILED))


def _checkpoint_record(id_start: str, id_end: str, result: PaperResult) -> dict:
    """Bản ghi checkpoint của một bài (đủ để dựng lại summary khi resume)."""
    return {
        "stage": CHECKPOINT_STAGE,
        "id_start": id_start,
        "id_end": id_end,
        "arxiv_id": result.arxiv_id,
        "status": result.status,
        "result": result.to_dict(),
    }


def _flush_checkpoint(path: Path, records: list[dict]) -> None:
    """Ghi nối các bản ghi checkpoint; lỗi I/O được log và thử lại một lần.

    Không làm mất kết quả đã tính: nếu vẫn không ghi được thì chỉ log lỗi, kết quả còn nguyên
    trong RAM để gộp vào summary.
    """
    for record in records:
        try:
            append_jsonl(path, record)
        except OSError:
            logger.exception("Ghi checkpoint that bai, thu lai mot lan: %s", path)
            try:
                append_jsonl(path, record)
            except OSError:
                logger.error("Van khong ghi duoc checkpoint %s; ket qua van giu trong RAM", path)
                continue


def _worker_loop(conn: Any, settings_dict: dict) -> None:
    """Tiến trình con: nhận từng bài qua ``conn``, trả ``PaperResult`` dạng dict.

    Hàm ở cấp module để ``multiprocessing`` (context ``spawn``) pickle được. Worker tự cài
    logging vì tiến trình spawn không thừa hưởng cấu hình logging của tiến trình cha.
    """
    # Ctrl+C do tiến trình cha điều khiển; worker bỏ qua để không in traceback trùng.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = M2Settings(**settings_dict)
    try:
        while True:
            task = conn.recv()
            if task is None:  # lệnh dừng
                break
            paper_dir, arxiv_id, out_dir = task
            try:
                result = process_paper_m2(Path(paper_dir), arxiv_id, settings, Path(out_dir))
                payload = result.to_dict()
            except Exception as exc:
                # Giữ worker sống: một bài lỗi không được kéo sập tiến trình con.
                logger.exception("Loi bat ngo khi xu ly %s trong worker", arxiv_id)
                payload = {
                    "arxiv_id": arxiv_id,
                    "status": STATUS_FAILED,
                    "stage": "process",
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            conn.send(payload)
    except EOFError:  # cha đóng pipe (bị ngắt/kill): thoát êm
        logger.warning("Worker mat ket noi voi tien trinh cha, dung lai.")
    finally:
        conn.close()


class PaperWorker:
    """Một tiến trình con bền vững xử lý từng bài, bị hủy khi một bài vượt thời gian.

    Dùng 1 worker cho cả dải (không spawn mỗi bài) để chi phí khởi động không làm sai số đo.
    Chỉ khi một bài quá ``paper_timeout_s`` mới ``terminate()`` rồi dựng worker mới cho bài kế.
    """

    def __init__(self, settings: M2Settings, target: Any = _worker_loop) -> None:
        """Khởi tạo và dựng tiến trình worker đầu tiên.

        Args:
            settings: Cấu hình M2 truyền cho worker (dạng dict phẳng, picklable).
            target: Hàm chạy trong worker (để test có thể truyền hàm treo thay thế).
        """
        self._settings = settings
        self._target = target
        self._context = multiprocessing.get_context("spawn")
        self._process: Any = None
        self._conn: Any = None
        self._start()

    def _start(self) -> None:
        """Dựng một tiến trình worker mới cùng pipe giao tiếp."""
        parent_conn, child_conn = self._context.Pipe()
        self._process = self._context.Process(
            target=self._target, args=(child_conn, self._settings.model_dump()), daemon=True
        )
        self._process.start()
        child_conn.close()
        self._conn = parent_conn

    def run(
        self, paper_dir: Path, arxiv_id: str, out_dir: Path, timeout: float
    ) -> dict | None:
        """Gửi một bài cho worker; trả dict kết quả, hoặc ``None`` nếu vượt ``timeout``.

        Worker chết giữa chừng (OOM-kill, crash) được phát hiện qua ``is_alive``/``EOFError``;
        bài gây chết được trả ``failed`` (stage ``worker``) và worker được dựng lại cho bài kế
        tiếp, thay vì để ``BrokenPipeError`` biến mọi bài sau thành ``failed`` vĩnh viễn.
        """
        if self._conn is None or self._process is None or not self._process.is_alive():
            self._kill()
            self._start()
        try:
            self._conn.send((str(paper_dir), arxiv_id, str(out_dir)))
            if self._conn.poll(timeout):
                return self._conn.recv()
        except (EOFError, OSError):
            logger.error("%s: worker chet giua chung", arxiv_id)
            self._kill()
            return {
                "arxiv_id": arxiv_id,
                "status": STATUS_FAILED,
                "stage": "worker",
                "reason": "worker crash",
            }
        logger.error("%s: vuot %.1fs, huy worker", arxiv_id, timeout)
        self._kill()
        return None

    def _kill(self) -> None:
        """Kết thúc tiến trình worker hiện tại (terminate, rồi kill nếu còn sống)."""
        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        if self._conn is not None:
            self._conn.close()
        self._process = None
        self._conn = None

    def close(self) -> None:
        """Dừng worker êm ái; kill nếu không thoát kịp."""
        if self._process is None:
            return
        try:
            if self._conn is not None:
                self._conn.send(None)
                self._process.join(timeout=5)
        except (OSError, ValueError):
            logger.warning("Khong gui duoc lenh dung worker, se kill.")
        if self._process.is_alive():
            self._kill()
        else:
            if self._conn is not None:
                self._conn.close()
            self._process = None
            self._conn = None


def _process_one(
    worker: PaperWorker | None,
    paper_dir: Path,
    arxiv_id: str,
    settings: M2Settings,
    out_dir: Path,
) -> PaperResult:
    """Xử lý một bài với lưới an toàn cuối: lỗi bất ngờ -> ``failed``, không làm sập dải."""
    started = time.perf_counter()
    try:
        if worker is None:
            return process_paper_m2(paper_dir, arxiv_id, settings, out_dir)
        data = worker.run(paper_dir, arxiv_id, out_dir, settings.paper_timeout_s)
        if data is None:
            return PaperResult(
                arxiv_id=arxiv_id,
                status=STATUS_TIMEOUT,
                stage="timeout",
                reason=f"vuot {settings.paper_timeout_s}s",
                seconds=time.perf_counter() - started,
            )
        return _paper_result(data, arxiv_id)
    except Exception as exc:
        logger.exception("Loi bat ngo khi xu ly %s", arxiv_id)
        return PaperResult(
            arxiv_id=arxiv_id,
            status=STATUS_FAILED,
            stage="process",
            reason=f"{type(exc).__name__}: {str(exc)[:300]}",
            seconds=time.perf_counter() - started,
        )


@dataclass
class _RangeContext:
    """Tham số cố định dùng chung cho cả vòng lặp xử lý một dải bài M2."""

    paths: M2Paths
    settings: M2Settings
    id_start: str
    id_end: str
    checkpoint_every: int
    profiler: Profiler | None
    retry_failed: bool = False


def _process_range(
    context: _RangeContext, papers: list[tuple[str, Path]], done: dict[str, dict]
) -> tuple[list[PaperResult], int, bool]:
    """Xử lý lần lượt các bài, ghi checkpoint; trả ``(kết quả, số bài resume, bị ngắt)``.

    Chỉ ghi số đo thời gian cho bài THỰC SỰ chạy ở lần này (bài bỏ qua nhờ checkpoint không được
    tính) để ``avg_seconds_per_paper`` không bị kéo lệch.
    """
    results: list[PaperResult] = []
    resumed = 0
    buffer: list[dict] = []
    interrupted = False
    worker = PaperWorker(context.settings) if context.settings.use_worker else None
    try:
        for index, (arxiv_id, paper_dir) in enumerate(papers, start=1):
            previous = done.get(arxiv_id)
            if previous and _resume_skip(previous, context.retry_failed):
                resumed += 1
                results.append(_result_from_record(previous, arxiv_id))
                logger.info(
                    "[%d/%d] %s -> bo qua (da xong tu truoc)", index, len(papers), arxiv_id
                )
                continue
            result = _process_one(
                worker, paper_dir, arxiv_id, context.settings, context.paths.papers_out
            )
            results.append(result)
            if context.profiler is not None:
                context.profiler.record_paper(arxiv_id, result.seconds)
            buffer.append(_checkpoint_record(context.id_start, context.id_end, result))
            logger.info(
                "[%d/%d] %s -> %s (%.1fs)",
                index, len(papers), arxiv_id, result.status, result.seconds,
            )
            if len(buffer) >= context.checkpoint_every:
                _flush_checkpoint(context.paths.checkpoint, buffer)
                buffer.clear()
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Da dung theo yeu cau; checkpoint da duoc ghi.")
    finally:
        if buffer:
            _flush_checkpoint(context.paths.checkpoint, buffer)
        if worker is not None:
            worker.close()
    return results, resumed, interrupted


def _checkpoint_every(args: argparse.Namespace) -> int:
    """Số bài giữa hai lần ghi checkpoint: ưu tiên ``--checkpoint_every``, rồi ``pipeline`` config."""
    if args.checkpoint_every is not None:
        return max(int(args.checkpoint_every), 1)
    try:
        with Path(args.config).open(encoding="utf-8") as handle:
            value = (yaml.safe_load(handle) or {}).get("pipeline", {}).get("checkpoint_every", 1)
    except FileNotFoundError:
        return 1
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def _ratio(num: float, den: float) -> float | None:
    """Tỉ số làm tròn 4 chữ số, hoặc ``None`` khi mẫu số bằng 0 (khớp cách M1 làm)."""
    return round(num / den, 4) if den else None


def build_summary(
    args: argparse.Namespace,
    settings: M2Settings,
    results: list[PaperResult],
    resumed: int,
    profile: dict,
) -> dict:
    """Gộp số đo ``Profiler`` (chỉ lần chạy này) với số liệu M2 (gộp cả bài đã xong trước đó).

    Bỏ hai khóa ``n_refs_total``/``n_refs_co_arxiv_id_total`` của M1 (không có nghĩa ở M2) và
    ghép khối chỉ số M2. ``success_rate`` là tỉ lệ bài xử lý xong không lỗi (``ok`` hoặc
    ``empty_record``); ``bib_dedup_ratio`` là tỉ lệ mục BibTeX bị gộp bỏ. ``avg_versions_per_paper``
    và ``avg_tree_depth`` chỉ chia cho số bài dựng được cây (``n_papers_with_tree``); bài lỗi
    một phần/bị cắt cây được đếm riêng ở ``n_partial``/``n_truncated``, còn ``n_timeout`` luôn
    có mặt để các thành viên so sánh (timeout phụ thuộc tốc độ máy).

    Args:
        args: Namespace CLI (dùng ``id_start``/``id_end``).
        settings: Cấu hình M2 (dùng ``failures_max``).
        results: Kết quả mọi bài trong dải (kể cả bài bỏ qua nhờ checkpoint).
        resumed: Số bài bỏ qua nhờ checkpoint.
        profile: ``Profiler.summary().model_dump(mode="json")``.

    Returns:
        Dict phẳng JSON-hóa được cho ``summary_stats_m2_<name>.json``.
    """
    base = {
        key: value
        for key, value in profile.items()
        if key not in ("n_refs_total", "n_refs_co_arxiv_id_total")
    }
    statuses = Counter(result.status for result in results)
    n_total = len(results)
    n_ok = statuses.get(STATUS_OK, 0)
    n_empty = statuses.get(STATUS_EMPTY, 0)
    n_failed = statuses.get(STATUS_FAILED, 0)
    n_skipped = statuses.get(STATUS_SKIPPED_LARGE, 0)
    n_timeout = statuses.get(STATUS_TIMEOUT, 0)
    n_partial = sum(1 for result in results if result.partial)
    n_truncated = sum(1 for result in results if result.truncated)
    # Chỉ bài dựng được cây mới có độ sâu/số version có nghĩa; chia cho cả bài
    # empty_record/failed/skipped sẽ kéo các trung bình này xuống.
    with_tree = [result for result in results if result.n_versions_done > 0]
    n_with_tree = len(with_tree)
    kinds: Counter[str] = Counter()
    for result in results:
        kinds.update(result.nodes_by_kind)
    n_bib_before = sum(result.n_bib_before for result in results)
    n_bib_after = sum(result.n_bib_after for result in results)
    failures = [
        {"arxiv_id": result.arxiv_id, "stage": result.stage, "reason": result.reason}
        for result in results
        if result.status == STATUS_FAILED
    ][: settings.failures_max]
    m2_block = {
        "id_start": args.id_start,
        "id_end": args.id_end,
        "n_papers_total": n_total,
        "n_ok": n_ok,
        "n_empty_record": n_empty,
        "n_failed": n_failed,
        "n_skipped_large": n_skipped,
        "n_timeout": n_timeout,
        "n_degraded": sum(1 for result in results if result.degraded),
        "n_partial": n_partial,
        "n_truncated": n_truncated,
        "n_papers_with_tree": n_with_tree,
        "success_rate": _ratio(n_ok + n_empty, n_total),
        "avg_versions_per_paper": _ratio(
            sum(result.n_versions for result in with_tree), n_with_tree
        ),
        "nodes_by_kind": {kind: kinds.get(kind, 0) for kind in NODE_KINDS},
        "avg_tree_depth": _ratio(sum(result.max_depth for result in with_tree), n_with_tree),
        "max_tree_depth": max((result.max_depth for result in results), default=0),
        "n_bib_entries_before_dedup": n_bib_before,
        "n_bib_entries_after_dedup": n_bib_after,
        "bib_dedup_ratio": _ratio(n_bib_before - n_bib_after, n_bib_before),
        "n_duplicate_nodes_merged": sum(result.n_duplicate_nodes_merged for result in results),
        "n_papers_resumed_skipped": resumed,
        "is_resumed_run": resumed > 0,
        "failures": failures,
    }
    return {**base, **m2_block}


def run_stage(args: argparse.Namespace) -> int:
    """Chạy M2 trên dải bài M1 đã tải, có checkpoint/resume, và xuất ``summary_stats_m2``.

    Args:
        args: Namespace từ ``build_parser`` (hoặc do ``main.py`` dựng với các trường tương ứng).

    Returns:
        ``EXIT_OK`` khi xong (kể cả có bài lỗi đã log), ``EXIT_BAD_INPUT`` khi thiếu dữ liệu đầu
        vào, ``EXIT_INTERRUPTED`` khi người dùng Ctrl+C (checkpoint + summary đã được flush).
    """
    settings = M2Settings.from_config(args.config)
    if getattr(args, "no_worker", False):
        settings = settings.model_copy(update={"use_worker": False})
    paths = M2Paths.build(args.name, args.data_root, args.processed_root, args.reports)
    if not paths.papers_in.is_dir():
        logger.error("Khong thay thu muc bai M1: %s", paths.papers_in)
        return EXIT_BAD_INPUT
    paths.papers_out.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    papers = list_papers_in_range(paths.papers_in, args.id_start, args.id_end, args.limit)
    if not papers:
        logger.error(
            "Khong co bai nao trong dai %s..%s tai %s", args.id_start, args.id_end, paths.papers_in
        )
        return EXIT_BAD_INPUT
    if args.no_resume:
        paths.checkpoint.unlink(missing_ok=True)
    done = load_checkpoint(paths.checkpoint, args.id_start, args.id_end)
    logger.info("M2: %d bai trong dai, %d da xong tu truoc", len(papers), len(done))
    with Profiler(
        watch_dir=paths.papers_out,
        interval=settings.profiler_interval_s,
        disk_every=settings.profiler_disk_every,
    ) as profiler:
        context = _RangeContext(
            paths=paths,
            settings=settings,
            id_start=args.id_start,
            id_end=args.id_end,
            checkpoint_every=max(_checkpoint_every(args), 1),
            profiler=profiler,
            retry_failed=getattr(args, "retry_failed", False),
        )
        results, resumed, interrupted = _process_range(context, papers, done)
    # summary() phải gọi SAU khối with để wall_seconds và final_disk_mb đã được chốt ở __exit__.
    profile = profiler.summary().model_dump(mode="json")
    summary = build_summary(args, settings, results, resumed, profile)
    write_json(paths.summary, summary)
    logger.info("M2 xong %d bai (bo qua %d nho checkpoint) -> %s", len(results), resumed, paths.summary)
    return EXIT_INTERRUPTED if interrupted else EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Bộ phân tích tham số M2 (``--id_start`` và ``--id-start`` đều dùng được)."""
    parser = argparse.ArgumentParser(description="Chay stage M2 (parse) cho mot dai bai da tai.")

    def add(name: str, **kwargs: Any) -> None:
        parser.add_argument(*dict.fromkeys([name, name.replace("_", "-")]), **kwargs)

    add("--name", default="default", help="Ten bo du lieu (khop --name cua M1)")
    add("--id_start", required=True, help="ID day du dau dai, VD 2506.00788")
    add("--id_end", required=True, help="ID day du cuoi dai, VD 2506.05787")
    add("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    add("--data_root", type=Path, default=Path("data/raw"), help="Thu muc M1 da tai bai ve")
    add("--processed_root", type=Path, default=Path("data/processed"), help="Thu muc xuat cua M2")
    add("--reports", type=Path, default=Path("reports"))
    add("--limit", type=int, default=None, help="Chi xu ly N bai dau (chay thu)")
    add("--checkpoint_every", type=int, default=None, help="Ghi checkpoint sau moi K bai")
    add("--no_resume", action="store_true", help="Bo checkpoint, chay lai tu dau")
    add(
        "--retry_failed",
        action="store_true",
        help="Khi resume, thu lai cac bai status failed (mac dinh bo qua)",
    )
    add("--no_worker", action="store_true", help="Xu ly bai ngay trong tien trinh chinh (bo timeout)")
    return parser


def _validate(args: argparse.Namespace) -> str | None:
    """Trả thông báo lỗi tham số, hoặc None nếu hợp lệ."""
    try:
        validate_id_range(args.id_start, args.id_end)
    except ValueError as exc:
        return str(exc)
    if args.limit is not None and args.limit <= 0:
        return "--limit phai > 0"
    return None


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI M2 (chạy độc lập). Trả mã thoát (xem ``run_stage``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    problem = _validate(args)
    if problem:
        logger.error("LOI: %s", problem)
        return EXIT_BAD_INPUT
    return run_stage(args)


__all__ = [
    "CHECKPOINT_STAGE",
    "EXIT_BAD_INPUT",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "STATUS_EMPTY",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_SKIPPED_LARGE",
    "STATUS_TIMEOUT",
    "TERMINAL_STATUSES",
    "M2Paths",
    "M2Settings",
    "PaperResult",
    "PaperWorker",
    "build_parser",
    "build_summary",
    "id_range_key",
    "list_papers_in_range",
    "list_version_dirs",
    "load_checkpoint",
    "main",
    "process_paper_m2",
    "run_stage",
    "validate_id_range",
]


if __name__ == "__main__":
    sys.exit(main())
