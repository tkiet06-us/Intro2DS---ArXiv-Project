"""Module Crawling arXiv Multi-format (task P2-01).

Đọc danh sách bài từ discovery (``discovery_25k.jsonl``, đầu ra của P1-00), tải mã nguồn LaTeX
của MỌI version (v1, v2, ...) qua ``export.arxiv.org/src/{id}v{n}``, chỉ giữ lại ``.tex`` và
``.bib``, rồi ghi ra thư mục làm việc của M1/M2 (khớp ``tests/make_mock.py``)::

    <out>/2506-00788/metadata.json
    <out>/2506-00788/v1/main.tex, refs.bib, sections/intro.tex ...
    <out>/2506-00788/v2/...
    <out>/crawl_log.jsonl      # mỗi dòng một version: trạng thái, size_before/size_after...
    <out>/crawl_summary.json   # tổng hợp cho báo cáo M1

Quy định của arXiv mà module tuân thủ (đề bài + kế hoạch dự án):
    - CHỈ dùng ``export.arxiv.org``, tuyệt đối không dùng ``arxiv.org`` (sẽ bị khóa IP).
    - User-Agent mô tả rõ dự án + email liên hệ.
    - Một kết nối tại một thời điểm, giữa hai request cách nhau tối thiểu 3 giây
      (``--delay`` nhỏ hơn 3 bị từ chối).
    - Gặp HTTP 403 thì DỪNG NGAY, không thử lại (tránh bị khóa IP lâu hơn).

Quy tắc dữ liệu:
    - Nhận diện ``.tar.gz``, ``.tar`` và ``.gz`` đơn (một file ``.tex``). Chỉ giải nén ``.tex`` và
      ``.bib``; ảnh và mọi file khác không bao giờ vào corpus. Lệnh ``\\begin{figure}`` và
      ``\\includegraphics`` trong ``.tex`` được giữ NGUYÊN VĂN (module không sửa nội dung file).
    - Version không có TeX (PDF/Word-only, archive không có ``.tex``...) vẫn có thư mục
      ``v{n}/`` RỖNG ("record rỗng"), không bỏ qua.
    - Ghi ``size_before`` (tổng dung lượng mọi file trong archive, đã giải nén) và
      ``size_after`` (dung lượng ``.tex``/``.bib`` giữ lại) cho báo cáo M1.
    - Checkpoint/Resume: sau mỗi version ghi một dòng vào ``crawl_log.jsonl``; chạy lại cùng
      lệnh sẽ bỏ qua version đã xong và thử lại version bị lỗi.

Dùng từ code khác (Runner ``main.py``): ``crawl_range(id_start, id_end, out_dir, resume=True)`` và
``crawl_paper(arxiv_id, out_dir)``, đúng chữ ký stub ban đầu của task (xem hai hàm đó bên dưới).

Cách chạy (từ thư mục gốc repo)::

    py src/m1_crawling/crawler.py --id_start 2506.00788 --id_end 2506.05787 --contact "email"

Thông điệp in ra màn hình cố ý viết KHÔNG DẤU để console Windows (cp1252) không lỗi mã hóa.
"""

import argparse
import gzip
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import time
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import requests

# Chạy "py src/m1_crawling/crawler.py" chỉ thêm src/m1_crawling vào sys.path, nên package `src`
# không import được. Thêm thư mục gốc repo (cùng cách tests/make_mock.py làm).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.ids import to_folder_name  # noqa: E402
from src.common.utils_io import append_jsonl, iter_jsonl, write_json  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

SRC_URL = "https://export.arxiv.org/src/{arxiv_id}v{version}"  # KHÔNG dùng arxiv.org
DEFAULT_DISCOVERY = Path("data/raw/discovery_25k.jsonl")  # đầu ra của P1-00 (đã vá)
DEFAULT_OUT = Path("data/raw/papers")
CONTACT_ENV = "ARXIV_CONTACT"  # biến môi trường thay cho --contact / tham số contact
MIN_DELAY_S = 3.0  # quy định arXiv: tối thiểu 3 giây giữa hai request
USER_AGENT = "DS-Project-HCMUS-arXiv-crawler/0.1 (student project; contact: {contact})"
KEEP_EXTENSIONS = frozenset({".tex", ".bib"})  # ảnh và mọi loại khác không vào corpus
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024  # chặn file tải về quá lớn
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # chặn "zip bomb"
MAX_KEPT_FILE_BYTES = 50 * 1024 * 1024  # một file .tex/.bib lớn hơn mức này bị bỏ qua
RETRY_BACKOFF_S = (10.0, 30.0, 60.0)  # chờ trước mỗi lần thử lại (lỗi mạng, HTTP 5xx, 429)
SINGLE_FILE_NAME = "main.tex"  # tên file khi nguồn là một .gz đơn
LOG_NAME = "crawl_log.jsonl"
SUMMARY_NAME = "crawl_summary.json"

STATUS_OK = "ok"  # có ít nhất một .tex
STATUS_EMPTY = "empty"  # record rỗng: version không có TeX
STATUS_ERROR = "error"  # lỗi tạm thời, sẽ được thử lại khi chạy lại
STATUS_BLOCKED = "blocked"  # arXiv trả 403: dừng toàn bộ
TERMINAL_STATUSES = frozenset({STATUS_OK, STATUS_EMPTY})  # xong, resume bỏ qua

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_BAD_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')  # ký tự Windows không cho phép
_schema_warned = False  # chỉ cảnh báo thiếu src.schemas một lần


# ---------------------------------------------------------------------------
# Lỗi
# ---------------------------------------------------------------------------


class CrawlBlockedError(RuntimeError):
    """arXiv trả HTTP 403: có nguy cơ bị khóa IP, phải dừng toàn bộ chương trình."""


class CrawlAbortedError(RuntimeError):
    """Quá nhiều lỗi liên tiếp: dừng để không dồn dập request vào server đang có vấn đề."""


class FetchError(RuntimeError):
    """Tải một version thất bại sau khi đã thử lại (lỗi tạm thời, version sẽ được thử lại sau)."""


# ---------------------------------------------------------------------------
# Giải nén an toàn, chỉ giữ .tex/.bib
# ---------------------------------------------------------------------------


@dataclass
class Extracted:
    """Kết quả phân tích một nguồn đã tải về (chưa ghi ra đĩa).

    Attributes:
        status: ``ok`` nếu có ít nhất một ``.tex``, ngược lại ``empty``.
        reason: Lý do khi ``empty`` (``pdf_only``, ``no_tex``, ``not_tex``...), ``None`` khi ok.
        files: Đường dẫn tương đối (dấu ``/``) -> nội dung THÔ (bytes) của các file giữ lại.
        size_before: Tổng dung lượng mọi file trong archive (đã giải nén), gồm cả ảnh.
        n_files_before: Số file thường trong archive.
        skipped: Các file ``.tex``/``.bib`` bị bỏ vì không an toàn, quá lớn hoặc trùng tên.
    """

    status: str
    reason: str | None = None
    files: dict[str, bytes] = field(default_factory=dict)
    size_before: int = 0
    n_files_before: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def size_after(self) -> int:
        """Tổng dung lượng các file đã giữ lại."""
        return sum(len(content) for content in self.files.values())


def _safe_relpath(name: str) -> str | None:
    """Chuẩn hóa tên file trong archive thành đường dẫn tương đối an toàn.

    Chặn đường dẫn tuyệt đối, ổ đĩa (``C:``), ``..`` (path traversal) và file rác của macOS
    (``__MACOSX``, ``._x``). Thay ký tự Windows không hợp lệ bằng ``_`` để ghi được trên
    máy Windows.

    Args:
        name: Tên member trong archive.

    Returns:
        Đường dẫn tương đối dùng dấu ``/``, hoặc None nếu phải bỏ file này.
    """
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute():
        return None
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    if parts[0] == "__MACOSX" or any(part.startswith("._") for part in parts):
        return None
    cleaned = [_BAD_FILENAME_CHARS.sub("_", part).rstrip(". ") for part in parts]
    if any(not part for part in cleaned):
        return None
    return "/".join(cleaned)


def _from_single_file(raw: bytes) -> Extracted:
    """Nguồn là MỘT file (``.gz`` đơn đã giải nén, hoặc văn bản thô).

    Args:
        raw: Nội dung file.

    Returns:
        ``ok`` với file ``main.tex`` nếu trông giống LaTeX, ngược lại record rỗng.
    """
    head = raw[:4096]
    stripped = head.lstrip()[:15].lower()
    if stripped.startswith((b"<!doctype", b"<html", b"<?xml")):
        return Extracted(STATUS_EMPTY, "html_response", size_before=len(raw), n_files_before=1)
    if b"\x00" in head or b"\\" not in raw:
        return Extracted(STATUS_EMPTY, "not_tex", size_before=len(raw), n_files_before=1)
    return Extracted(
        STATUS_OK,
        files={SINGLE_FILE_NAME: raw},
        size_before=len(raw),
        n_files_before=1,
    )


def _from_tar(tar: tarfile.TarFile) -> Extracted:
    """Duyệt archive tar, chỉ đọc nội dung ``.tex`` và ``.bib``.

    Args:
        tar: Archive đã mở.

    Returns:
        Kết quả đã lọc. Chỉ file thường được xét: symlink, hardlink, thiết bị bị bỏ qua.
    """
    result = Extracted(STATUS_EMPTY, "no_tex")
    seen: set[str] = set()
    for member in tar:
        if not member.isfile():
            continue
        result.n_files_before += 1
        result.size_before += member.size
        if result.size_before > MAX_UNCOMPRESSED_BYTES:
            return Extracted(STATUS_EMPTY, "too_large", size_before=result.size_before)
        suffix = PurePosixPath(member.name).suffix.lower()
        if suffix not in KEEP_EXTENSIONS:
            continue  # ảnh, .sty, .cls, .pdf... không bao giờ vào corpus
        relpath = _safe_relpath(member.name)
        if relpath is None or member.size > MAX_KEPT_FILE_BYTES or relpath.lower() in seen:
            result.skipped.append(member.name)
            continue
        handle = tar.extractfile(member)
        if handle is None:
            continue
        seen.add(relpath.lower())  # so khớp không phân biệt hoa thường (Windows)
        result.files[relpath] = handle.read()
    if any(PurePosixPath(name).suffix.lower() == ".tex" for name in result.files):
        result.status, result.reason = STATUS_OK, None
    else:
        # Chỉ có .bib mà không có .tex: không phải bài LaTeX dùng được -> record rỗng.
        result.files.clear()
    return result


def _gunzip_limited(data: bytes) -> bytes | None:
    """Giải nén gzip nhưng dừng khi vượt giới hạn (chống bom nén).

    Returns:
        Nội dung đã giải nén, hoặc None nếu vượt ``MAX_UNCOMPRESSED_BYTES``.
    """
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        out = handle.read(MAX_UNCOMPRESSED_BYTES + 1)
    return None if len(out) > MAX_UNCOMPRESSED_BYTES else out


def extract_tex_sources(data: bytes) -> Extracted:
    """Phân tích nguồn arXiv đã tải và lọc ra ``.tex``/``.bib``.

    Nhận diện theo NỘI DUNG chứ không theo header HTTP (arXiv hay gắn ``Content-Encoding`` làm
    thư viện HTTP tự giải nén một lớp gzip): PDF, ``.tar.gz``, ``.tar``, ``.gz`` đơn, văn bản
    thô. Không bao giờ raise vì dữ liệu hỏng: trả record rỗng kèm lý do.

    Args:
        data: Toàn bộ byte của response.

    Returns:
        Kết quả đã lọc, chưa ghi ra đĩa.
    """
    if data[:5] == b"%PDF-":
        return Extracted(STATUS_EMPTY, "pdf_only", size_before=len(data), n_files_before=1)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            return _from_tar(tar)
    except tarfile.ReadError:
        pass  # không phải tar: thử .gz đơn / văn bản thô bên dưới
    except (EOFError, OSError, zlib.error) as exc:
        logger.warning("Archive hong: %s", exc)
        return Extracted(STATUS_EMPTY, "corrupt_archive", size_before=len(data))
    if data[:2] == b"\x1f\x8b":
        try:
            raw = _gunzip_limited(data)
        except (EOFError, OSError, zlib.error) as exc:
            logger.warning("File gz hong: %s", exc)
            return Extracted(STATUS_EMPTY, "corrupt_archive", size_before=len(data))
        if raw is None:
            return Extracted(STATUS_EMPTY, "too_large", size_before=len(data))
        return _from_single_file(raw)
    if data[:2] == b"PK":  # zip/docx: bài nộp dạng Word
        return Extracted(STATUS_EMPTY, "word_or_zip", size_before=len(data), n_files_before=1)
    return _from_single_file(data)


def write_version(folder: Path, version: int, files: dict[str, bytes]) -> None:
    """Ghi một version ra ``folder/v{n}/`` theo kiểu nguyên tử.

    Ghi vào thư mục tạm rồi đổi tên, nên chương trình bị ngắt giữa chừng không để lại
    thư mục ``v{n}`` ghi dở. ``files`` rỗng tạo thư mục rỗng (record rỗng).

    Args:
        folder: Thư mục của bài (``<out>/2506-00788``).
        version: Số version.
        files: Đường dẫn tương đối -> nội dung bytes (tên đã qua ``_safe_relpath``).
    """
    tmp = folder / f".v{version}.tmp"
    final = folder / f"v{version}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for relpath, content in files.items():
        target = tmp / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if final.exists():
        shutil.rmtree(final)
    tmp.replace(final)


# ---------------------------------------------------------------------------
# Tải về lịch sự (delay, thử lại, dừng khi bị chặn)
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Kết quả một lần tải (đã gồm các lần thử lại).

    Attributes:
        status_code: Mã HTTP cuối cùng.
        body: Nội dung (rỗng khi 404/410).
        content_type: Header Content-Type nếu có.
        attempts: Số request đã gửi cho version này.
        seconds: Tổng thời gian tính cả thời gian chờ thử lại.
    """

    status_code: int
    body: bytes
    content_type: str | None
    attempts: int
    seconds: float


class PoliteFetcher:
    """Tải URL tuần tự với delay tối thiểu giữa các request, đúng quy định arXiv."""

    def __init__(
        self,
        user_agent: str,
        delay: float = MIN_DELAY_S,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        backoff: tuple[float, ...] = RETRY_BACKOFF_S,
        timeout: tuple[float, float] = (15.0, 120.0),
    ) -> None:
        """Khởi tạo.

        Args:
            user_agent: Chuỗi User-Agent mô tả dự án và email liên hệ.
            delay: Giây tối thiểu giữa hai request (tính từ lúc request trước KẾT THÚC).
            session: Phiên HTTP (mặc định tạo mới; tests truyền bản giả). Dùng MỘT kết nối.
            sleep: Hàm ngủ (tests thay bằng bản giả).
            clock: Đồng hồ đơn điệu (tests thay bằng bản giả).
            backoff: Số giây chờ trước mỗi lần thử lại.
            timeout: (timeout kết nối, timeout đọc) tính bằng giây.

        Raises:
            ValueError: Khi ``delay`` nhỏ hơn 3 giây (vi phạm quy định arXiv).
        """
        if delay < MIN_DELAY_S:
            raise ValueError(f"delay phai >= {MIN_DELAY_S} giay (quy dinh arXiv), nhan {delay}")
        self.delay = delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._sleep = sleep
        self._clock = clock
        self._backoff = backoff
        self._timeout = timeout
        self._last_end: float | None = None

    def _wait_turn(self) -> None:
        """Ngủ cho đủ ``delay`` giây kể từ khi request trước kết thúc."""
        if self._last_end is not None:
            remaining = self.delay - (self._clock() - self._last_end)
            if remaining > 0:
                self._sleep(remaining)

    def _read_body(self, response: requests.Response) -> bytes:
        """Đọc body theo từng khối, dừng khi vượt ``MAX_DOWNLOAD_BYTES``."""
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=1 << 16):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise FetchError("file tai ve qua lon")
            chunks.append(chunk)
        return b"".join(chunks)

    def get(self, url: str) -> FetchResult:
        """Tải một URL, thử lại khi lỗi tạm thời.

        Args:
            url: URL ``export.arxiv.org/src/...``.

        Returns:
            Kết quả khi nhận HTTP 200 (kèm body) hoặc 404/410 (body rỗng).

        Raises:
            CrawlBlockedError: Khi nhận HTTP 403 (dừng ngay, không thử lại).
            FetchError: Khi hết số lần thử lại mà vẫn lỗi, hoặc gặp mã HTTP bất thường.
        """
        started = self._clock()
        last_error = "khong ro"
        for attempt in range(len(self._backoff) + 1):
            self._wait_turn()
            retry_after: float | None = None
            try:
                response = self.session.get(url, stream=True, timeout=self._timeout)
                try:
                    status = response.status_code
                    if status == 403:
                        raise CrawlBlockedError(f"HTTP 403 khi tai {url}")
                    if status in (404, 410):
                        return FetchResult(status, b"", None, attempt + 1, self._clock() - started)
                    if status == 200:
                        body = self._read_body(response)
                        content_type = response.headers.get("Content-Type")
                        return FetchResult(
                            status, body, content_type, attempt + 1, self._clock() - started
                        )
                    last_error = f"HTTP {status}"
                    if status == 429 or status >= 500:
                        header = response.headers.get("Retry-After", "")
                        retry_after = float(header) if header.isdigit() else None
                    else:
                        raise FetchError(f"HTTP {status} khi tai {url}")
                finally:
                    response.close()
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._last_end = self._clock()
            if attempt < len(self._backoff):
                wait = max(self._backoff[attempt], retry_after or 0.0)
                logger.warning("Loi tam thoi (%s), thu lai sau %.0fs", last_error, wait)
                self._sleep(wait)
        raise FetchError(f"{last_error} sau {len(self._backoff) + 1} lan thu: {url}")


# ---------------------------------------------------------------------------
# Metadata, checkpoint, vòng lặp chính
# ---------------------------------------------------------------------------


def _iso_date(value: str | None) -> str | None:
    """Lấy phần ngày ``YYYY-MM-DD`` của chuỗi ISO 8601, hoặc None nếu không đọc được."""
    if value and _ISO_DATE.match(value):
        return value[:10]
    return None


def build_metadata(row: dict) -> dict:
    """Dựng nội dung ``metadata.json`` (schema ``Paper``) từ một dòng discovery.

    ``submission_date`` là ngày v1; ``revised_dates`` là ngày các version v2..vn. Dòng discovery
    chưa vá ``version_dates`` thì ngày version cuối lấy từ ``latest_version_date``.

    Args:
        row: Một dòng của ``discovery_25k.jsonl``.

    Returns:
        Dict JSON-hóa được, đã qua kiểm tra của ``Paper`` (ID hợp lệ, không có trường lạ) khi
        ``src.schemas`` import được.
    """
    n_versions = int(row.get("n_versions") or 1)
    version_dates = row.get("version_dates") or {}
    revised: list[str] = []
    for number in range(2, n_versions + 1):
        value = version_dates.get(f"v{number}")
        if value is None and number == n_versions:
            value = row.get("latest_version_date")
        date = _iso_date(value)
        if date:
            revised.append(date)
    metadata = {
        "arxiv_id": row["arxiv_id"],
        "title": row.get("title") or None,
        "authors": list(row.get("authors") or []),
        "submission_date": _iso_date(row.get("v1_date")),
        "revised_dates": revised,
        "venue": row.get("venue") or None,
    }
    # Import khi cần: src.schemas kéo theo pyarrow/pydantic. Thiếu chúng thì vẫn crawl được
    # (metadata ghi cùng khóa nhưng không qua kiểm tra schema), tránh chặn việc crawl hàng giờ.
    global _schema_warned
    try:
        from src.schemas import Paper  # noqa: PLC0415
    except ImportError:
        if not _schema_warned:
            logger.warning(
                "Khong import duoc src.schemas (thieu pyarrow/pydantic?): "
                "metadata khong duoc kiem tra schema"
            )
            _schema_warned = True
        return metadata
    return Paper(**metadata).model_dump(mode="json")


def select_papers(
    discovery_path: Path, id_start: str, id_end: str, limit: int | None = None
) -> list[dict]:
    """Chọn các bài discovery nằm trong ``[id_start, id_end]`` (so sánh chuỗi ID, đã sắp xếp).

    ID đầy đủ ``YYMM.NNNNN`` cùng độ rộng nên so sánh chuỗi đúng thứ tự, kể cả dải đi qua nhiều
    tháng (cùng cách ``discovery.py`` lọc).

    Args:
        discovery_path: File jsonl của discovery.
        id_start: ID đầu dải (gồm cả ID này).
        id_end: ID cuối dải (gồm cả ID này).
        limit: Chỉ lấy N bài đầu (chạy thử); None là lấy hết.

    Returns:
        Danh sách dòng discovery, sắp xếp theo ID.
    """
    rows = [
        row
        for row in iter_jsonl(discovery_path)
        if "arxiv_id" in row and id_start <= row["arxiv_id"] <= id_end
    ]
    rows.sort(key=lambda row: row["arxiv_id"])
    return rows[:limit] if limit else rows


def load_log_state(log_path: Path) -> dict[tuple[str, int], dict]:
    """Đọc log và trả về dòng MỚI NHẤT của từng (arxiv_id, version).

    Args:
        log_path: Đường dẫn ``crawl_log.jsonl`` (không tồn tại thì trả dict rỗng).

    Returns:
        (arxiv_id, version) -> bản ghi log cuối cùng.
    """
    state: dict[tuple[str, int], dict] = {}
    for record in iter_jsonl(log_path):
        if "arxiv_id" in record and "version" in record:
            state[(record["arxiv_id"], int(record["version"]))] = record
    return state


def _peak_rss_mb() -> float | None:
    """RAM đỉnh của tiến trình (MB) nếu có ``psutil``; None nếu không đo được."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss  # Windows có peak_wset
    return round(peak / 1024 / 1024, 1)


def _dir_bytes(path: Path) -> int:
    """Tổng dung lượng mọi file dưới ``path``."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def summarize(
    state: dict[tuple[str, int], dict], papers: list[dict], out_dir: Path, elapsed_s: float
) -> dict:
    """Tổng hợp số liệu cho báo cáo M1 từ log (chỉ các bài trong phạm vi đang chạy).

    Args:
        state: Kết quả ``load_log_state``.
        papers: Các bài đang crawl (dòng discovery).
        out_dir: Thư mục đầu ra (để đo dung lượng đĩa).
        elapsed_s: Thời gian của LẦN CHẠY NÀY (chỉ để tham khảo, resume sẽ ngắn hơn tổng).

    Returns:
        Dict tổng hợp: số bài/version theo trạng thái, size_before/size_after, tỉ lệ nén...
    """
    expected = {(row["arxiv_id"], v) for row in papers for v in range(1, _n_versions(row) + 1)}
    records = [state[key] for key in expected if key in state]
    by_status = Counter(record["status"] for record in records)
    reasons = Counter(r.get("reason") for r in records if r["status"] == STATUS_EMPTY)
    size_before = sum(r.get("size_before") or 0 for r in records)
    size_after = sum(r.get("size_after") or 0 for r in records)
    seconds = [r["seconds"] for r in records if r.get("seconds") is not None]
    return {
        "papers": len(papers),
        "versions_expected": len(expected),
        "versions_logged": len(records),
        "versions_not_attempted": len(expected) - len(records),
        "by_status": dict(by_status),
        "empty_reasons": {str(k): v for k, v in reasons.items()},
        "papers_with_no_tex_at_all": sum(
            1
            for row in papers
            if all(
                state.get((row["arxiv_id"], v), {}).get("status") == STATUS_EMPTY
                for v in range(1, _n_versions(row) + 1)
            )
        ),
        "archive_bytes_downloaded": sum(r.get("archive_bytes") or 0 for r in records),
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
        "size_after_over_before": round(size_after / size_before, 4) if size_before else None,
        "seconds_per_version_mean": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "elapsed_this_run_s": round(elapsed_s, 1),
        "peak_rss_mb": _peak_rss_mb(),
        "output_dir_bytes": _dir_bytes(out_dir) if out_dir.exists() else 0,
    }


def _n_versions(row: dict) -> int:
    """Số version của một dòng discovery (tối thiểu 1: bài chỉ nộp một lần dùng v1)."""
    return max(1, int(row.get("n_versions") or 1))


def crawl_version(
    fetcher: PoliteFetcher, row: dict, version: int, folder: Path, log_path: Path
) -> dict:
    """Tải, lọc và ghi MỘT version; luôn ghi một dòng log.

    Args:
        fetcher: Bộ tải lịch sự.
        row: Dòng discovery của bài.
        version: Số version.
        folder: Thư mục của bài.
        log_path: File log checkpoint.

    Returns:
        Bản ghi log vừa ghi (đã có ``status``).

    Raises:
        CrawlBlockedError: Khi arXiv trả HTTP 403 (đã ghi log ``blocked`` trước khi raise).
    """
    arxiv_id = row["arxiv_id"]
    url = SRC_URL.format(arxiv_id=arxiv_id, version=version)
    record: dict = {"arxiv_id": arxiv_id, "version": version, "url": url}
    started = time.monotonic()
    try:
        fetched = fetcher.get(url)
    except CrawlBlockedError as exc:
        record.update(status=STATUS_BLOCKED, reason=str(exc))
        append_jsonl(log_path, record)
        raise
    except FetchError as exc:
        record.update(
            status=STATUS_ERROR, reason=str(exc), seconds=round(time.monotonic() - started, 2)
        )
        append_jsonl(log_path, record)
        return record
    if fetched.status_code in (404, 410):
        extracted = Extracted(STATUS_EMPTY, f"http_{fetched.status_code}")
    else:
        extracted = extract_tex_sources(fetched.body)
    write_version(folder, version, extracted.files)
    record.update(
        status=extracted.status,
        reason=extracted.reason,
        http_status=fetched.status_code,
        content_type=fetched.content_type,
        attempts=fetched.attempts,
        archive_bytes=len(fetched.body),
        size_before=extracted.size_before,
        size_after=extracted.size_after,
        n_files_before=extracted.n_files_before,
        n_files_after=len(extracted.files),
        files=sorted(extracted.files)[:50],
        skipped=extracted.skipped[:20],
        seconds=round(fetched.seconds, 2),
    )
    append_jsonl(log_path, record)
    return record


def _write_metadata_once(folder: Path, row: dict) -> None:
    """Ghi ``metadata.json`` của bài nếu chưa có (không cần mạng, nên làm trước khi tải).

    Args:
        folder: Thư mục của bài.
        row: Dòng discovery của bài.
    """
    target = folder / "metadata.json"
    if target.exists():
        return
    try:
        metadata = build_metadata(row)
    except ValueError as exc:  # pydantic.ValidationError là con của ValueError
        logger.error("metadata %s khong hop le: %s", row["arxiv_id"], exc)
        return
    write_json(target, metadata)


def run(
    papers: list[dict],
    out_dir: Path,
    fetcher: PoliteFetcher,
    *,
    write_metadata: bool = True,
    max_consecutive_errors: int = 10,
    resume: bool = True,
) -> dict:
    """Crawl lần lượt từng bài, từng version; có Checkpoint/Resume.

    Args:
        papers: Dòng discovery của các bài cần crawl (đã sắp xếp).
        out_dir: Thư mục đầu ra.
        fetcher: Bộ tải lịch sự (MỘT kết nối, có delay).
        write_metadata: Ghi ``metadata.json`` cho mỗi bài.
        max_consecutive_errors: Dừng sau chừng này version lỗi LIÊN TIẾP.
        resume: True bỏ qua version đã xong theo log; False tải lại tất cả (ghi đè).

    Returns:
        Dict tổng hợp (``summarize``), có thêm khóa ``exit_code``.

    Raises:
        CrawlBlockedError: Khi arXiv trả HTTP 403.
        CrawlAbortedError: Khi quá nhiều lỗi liên tiếp.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / LOG_NAME
    state = load_log_state(log_path) if resume else {}
    total = sum(_n_versions(row) for row in papers)
    done_before = sum(
        1
        for row in papers
        for v in range(1, _n_versions(row) + 1)
        if state.get((row["arxiv_id"], v), {}).get("status") in TERMINAL_STATUSES
        and (out_dir / to_folder_name(row["arxiv_id"]) / f"v{v}").is_dir()
    )
    logger.info(
        "Can crawl %d bai / %d version; da xong tu truoc: %d", len(papers), total, done_before
    )
    started = time.monotonic()
    fetched_now = 0
    consecutive_errors = 0
    for index, row in enumerate(papers, start=1):
        folder = out_dir / to_folder_name(row["arxiv_id"])
        if write_metadata:
            _write_metadata_once(folder, row)  # ghi TRƯỚC khi tải: ngắt giữa chừng vẫn có metadata
        for version in range(1, _n_versions(row) + 1):
            previous = state.get((row["arxiv_id"], version), {})
            if previous.get("status") in TERMINAL_STATUSES and (folder / f"v{version}").is_dir():
                continue  # resume: version này đã xong
            folder.mkdir(parents=True, exist_ok=True)
            record = crawl_version(fetcher, row, version, folder, log_path)
            state[(row["arxiv_id"], version)] = record
            fetched_now += 1
            consecutive_errors = consecutive_errors + 1 if record["status"] == STATUS_ERROR else 0
            _log_progress(record, index, len(papers), fetched_now, total - done_before, started)
            if consecutive_errors >= max_consecutive_errors:
                raise CrawlAbortedError(
                    f"{consecutive_errors} loi lien tiep, dung de tranh dan dap"
                )
    summary = summarize(state, papers, out_dir, time.monotonic() - started)
    summary["exit_code"] = 1 if summary["by_status"].get(STATUS_ERROR) else 0
    return summary


def _log_progress(
    record: dict, paper_index: int, n_papers: int, done: int, todo: int, started: float
) -> None:
    """In một dòng tiến độ có ước lượng thời gian còn lại (dựa trên tốc độ lần chạy này)."""
    elapsed = time.monotonic() - started
    eta_min = (elapsed / done) * max(todo - done, 0) / 60 if done else math.nan
    logger.info(
        "[bai %d/%d | %d/%d version | ETA %.0f phut] %sv%d %s%s %s -> %s byte",
        paper_index,
        n_papers,
        done,
        todo,
        eta_min,
        record["arxiv_id"],
        record["version"],
        record["status"],
        f" ({record['reason']})" if record.get("reason") else "",
        record.get("size_before", "-"),
        record.get("size_after", "-"),
    )


# ---------------------------------------------------------------------------
# API cho code khác (Runner main.py...): giữ đúng chữ ký stub ban đầu của P2-01
# ---------------------------------------------------------------------------


def _require_contact(contact: str | None) -> str:
    """Lấy email liên hệ cho User-Agent từ tham số hoặc biến môi trường ``ARXIV_CONTACT``.

    Raises:
        ValueError: Khi không có email (arXiv yêu cầu User-Agent có thông tin liên hệ).
    """
    value = contact or os.environ.get(CONTACT_ENV)
    if not value:
        raise ValueError(
            f"can email lien he cho User-Agent: truyen contact=... hoac dat {CONTACT_ENV}"
        )
    return value


def _load_discovery_row(arxiv_id: str, discovery_path: Path) -> dict | None:
    """Tìm dòng discovery của một bài (None nếu không có)."""
    for row in iter_jsonl(discovery_path):
        if row.get("arxiv_id") == arxiv_id:
            return row
    return None


def crawl_paper(
    arxiv_id: str,
    out_dir: str | Path,
    *,
    row: dict | None = None,
    contact: str | None = None,
    discovery_path: str | Path = DEFAULT_DISCOVERY,
    delay: float = MIN_DELAY_S,
    write_metadata: bool = True,
    resume: bool = True,
    fetcher: PoliteFetcher | None = None,
) -> dict:
    """Crawl MỌI version của MỘT bài, ghi vào ``<out_dir>/<yymm-id>/v{n}/``.

    Số version lấy từ discovery (``n_versions``), nên bài phải có trong ``discovery_path`` hoặc
    truyền sẵn ``row``.

    Args:
        arxiv_id: Mã chuẩn, ví dụ ``"2506.00788"`` hoặc ``"hep-th/9901001"``.
        out_dir: Thư mục đầu ra.
        row: Dòng discovery của bài (nếu có sẵn thì khỏi đọc file).
        contact: Email cho User-Agent (mặc định lấy biến môi trường ``ARXIV_CONTACT``).
        discovery_path: File discovery dùng khi không truyền ``row``.
        delay: Giây giữa hai request, tối thiểu 3.
        write_metadata: Ghi ``metadata.json``.
        resume: True bỏ qua version đã xong theo log; False tải lại.
        fetcher: Bộ tải có sẵn (tests truyền bản giả); None là tạo mới.

    Returns:
        Dict tổng hợp (như ``crawl_range``) kèm khóa ``versions``: trạng thái từng version.

    Raises:
        ValueError: Khi ID sai dạng, bài không có trong discovery, hoặc thiếu email liên hệ.
        CrawlBlockedError: Khi arXiv trả HTTP 403.
        CrawlAbortedError: Khi quá nhiều lỗi liên tiếp.
    """
    to_folder_name(arxiv_id)  # ID sai dạng nổ ngay tại đây
    row = row or _load_discovery_row(arxiv_id, Path(discovery_path))
    if row is None:
        raise ValueError(f"{arxiv_id} khong co trong discovery {discovery_path} (can n_versions)")
    fetcher = fetcher or PoliteFetcher(USER_AGENT.format(contact=_require_contact(contact)), delay)
    out_dir = Path(out_dir)
    summary = run([row], out_dir, fetcher, write_metadata=write_metadata, resume=resume)
    state = load_log_state(out_dir / LOG_NAME)
    summary["versions"] = [
        {
            key: state[(arxiv_id, v)].get(key)
            for key in ("version", "status", "reason", "size_before", "size_after")
        }
        for v in range(1, _n_versions(row) + 1)
        if (arxiv_id, v) in state
    ]
    return summary


def crawl_range(
    id_start: str,
    id_end: str,
    out_dir: str | Path,
    resume: bool = True,
    *,
    contact: str | None = None,
    discovery_path: str | Path = DEFAULT_DISCOVERY,
    delay: float = MIN_DELAY_S,
    limit: int | None = None,
    write_metadata: bool = True,
    max_consecutive_errors: int = 10,
    fetcher: PoliteFetcher | None = None,
) -> dict:
    """Crawl mọi bài trong ``[id_start, id_end]``, có Checkpoint/Resume.

    Bốn tham số đầu giữ đúng chữ ký stub ban đầu của P2-01. Khác stub: trả về dict tổng hợp
    (ngoài ra cũng ghi ``<out_dir>/crawl_summary.json``); người gọi có thể bỏ qua giá trị trả về.

    Args:
        id_start: ID đầu dải (gồm cả ID này).
        id_end: ID cuối dải (gồm cả ID này).
        out_dir: Thư mục đầu ra.
        resume: True bỏ qua version đã xong theo log; False tải lại tất cả.
        contact: Email cho User-Agent (mặc định lấy biến môi trường ``ARXIV_CONTACT``).
        discovery_path: File jsonl của discovery (P1-00).
        delay: Giây giữa hai request, tối thiểu 3.
        limit: Chỉ crawl N bài đầu (chạy thử); None là hết.
        write_metadata: Ghi ``metadata.json`` cho mỗi bài.
        max_consecutive_errors: Dừng sau chừng này version lỗi liên tiếp.
        fetcher: Bộ tải có sẵn (tests truyền bản giả); None là tạo mới.

    Returns:
        Dict tổng hợp: số version theo trạng thái, size_before/size_after, ``exit_code``...

    Raises:
        FileNotFoundError: Khi không thấy file discovery.
        ValueError: Khi không có bài nào trong dải, hoặc thiếu email liên hệ.
        CrawlBlockedError: Khi arXiv trả HTTP 403.
        CrawlAbortedError: Khi quá nhiều lỗi liên tiếp.
    """
    discovery_path = Path(discovery_path)
    if not discovery_path.exists():
        raise FileNotFoundError(f"khong thay file discovery: {discovery_path}")
    papers = select_papers(discovery_path, id_start, id_end, limit)
    if not papers:
        raise ValueError(f"khong co bai nao trong dai {id_start} -> {id_end}")
    fetcher = fetcher or PoliteFetcher(USER_AGENT.format(contact=_require_contact(contact)), delay)
    out_dir = Path(out_dir)
    summary = run(
        papers,
        out_dir,
        fetcher,
        write_metadata=write_metadata,
        max_consecutive_errors=max_consecutive_errors,
        resume=resume,
    )
    write_json(out_dir / SUMMARY_NAME, summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ phân tích tham số dòng lệnh (``--id_start`` và ``--id-start`` đều dùng được)."""
    parser = argparse.ArgumentParser(
        description="Crawl nguon LaTeX (.tex/.bib) cua moi version tu export.arxiv.org."
    )

    def add(name: str, **kwargs) -> None:
        parser.add_argument(name, name.replace("_", "-"), **kwargs)

    add("--id_start", required=True, help="ID day du dau dai, VD 2506.00788")
    add("--id_end", required=True, help="ID day du cuoi dai, VD 2506.05787")
    parser.add_argument(
        "--contact",
        default=os.environ.get(CONTACT_ENV),
        help=f"Email lien he, ghi vao User-Agent (hoac dat bien moi truong {CONTACT_ENV})",
    )
    parser.add_argument("--discovery", default=DEFAULT_DISCOVERY, type=Path)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--fresh", action="store_true", help="Bo checkpoint, tai lai tu dau")
    parser.add_argument(
        "--delay", default=MIN_DELAY_S, type=float, help="Giay giua 2 request (>=3)"
    )
    parser.add_argument("--limit", default=None, type=int, help="Chi crawl N bai dau (chay thu)")
    add("--max_consecutive_errors", default=10, type=int)
    add("--no_metadata", action="store_true", help="Khong ghi metadata.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điểm vào của CLI.

    Args:
        argv: Tham số dòng lệnh (None là dùng ``sys.argv``).

    Returns:
        Mã thoát: 0 xong; 1 còn version lỗi (chạy lại để thử lại); 2 sai tham số;
        3 bị arXiv chặn (HTTP 403); 4 dừng vì quá nhiều lỗi liên tiếp.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.delay < MIN_DELAY_S:
        print(f"LOI: --delay phai >= {MIN_DELAY_S} giay (quy dinh arXiv).", file=sys.stderr)
        return 2
    if not args.contact:
        print(
            f"LOI: can --contact (hoac bien moi truong {CONTACT_ENV}) cho User-Agent.",
            file=sys.stderr,
        )
        return 2
    if not args.discovery.exists():
        print(f"LOI: khong thay file discovery: {args.discovery}", file=sys.stderr)
        return 2
    papers = select_papers(args.discovery, args.id_start, args.id_end, args.limit)
    if not papers:
        print("LOI: khong co bai nao trong dai ID da cho.", file=sys.stderr)
        return 2
    fetcher = PoliteFetcher(USER_AGENT.format(contact=args.contact), args.delay)
    started = time.monotonic()
    try:
        summary = run(
            papers,
            args.out,
            fetcher,
            write_metadata=not args.no_metadata,
            max_consecutive_errors=args.max_consecutive_errors,
            resume=not args.fresh,
        )
    except CrawlBlockedError as exc:
        print(f"\nBI CHAN: {exc}. DUNG NGAY. Cho vai gio, kiem tra User-Agent/delay roi chay lai.")
        return 3
    except CrawlAbortedError as exc:
        print(f"\nDUNG: {exc}. Kiem tra mang roi chay lai CUNG LENH de tiep tuc.")
        return 4
    except KeyboardInterrupt:
        print("\nDa dung theo yeu cau. Chay lai CUNG LENH de tiep tuc.")
        return 1
    write_json(args.out / SUMMARY_NAME, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"Tong ket ghi o {args.out / SUMMARY_NAME} (lan chay nay {time.monotonic() - started:.0f}s)"
    )
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())