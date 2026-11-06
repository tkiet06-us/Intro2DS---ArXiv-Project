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
      lệnh sẽ bỏ qua version đã xong và thử lại version bị lỗi. ``html_response`` và
      ``corrupt_archive`` cũng được thử lại (tối đa ``MAX_SOFT_RETRIES`` lần) trước khi chốt rỗng.

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
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

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
DISK_CHECK_EVERY = 500  # cứ N version vừa tải thì đo dung lượng output một lần (rglob tốn kém)
METADATA_ERRORS_NAME = "metadata_errors.json"  # bài không dựng được metadata.json

# [Review #1] Lý do "nghi lỗi tạm thời" (arXiv trả trang HTML lỗi, tải thiếu/hỏng): KHÔNG chốt
# empty ngay mà ghi STATUS_ERROR để resume thử lại, tối đa MAX_SOFT_RETRIES lần rồi mới chốt.
RETRYABLE_REASONS = frozenset({"html_response", "corrupt_archive"})
MAX_SOFT_RETRIES = 3  # tổng số lần được gặp lý do trên; lần thứ 3 vẫn gặp thì chốt empty

# [Review #8] Redirect: chỉ đi theo khi vẫn ở export.arxiv.org (arxiv.org sẽ bị khóa IP).
ALLOWED_HOST = "export.arxiv.org"
MAX_REDIRECTS = 3
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

# [Review #3] Fallback tra n_versions bằng arXiv API (cũng qua PoliteFetcher: delay 3 giây).
API_URL = "https://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
API_BATCH = 100  # số ID mỗi request
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ATOM_ID = re.compile(r"/abs/(?P<id>.+?)v(?P<n>\d+)\s*$")  # <id>http://arxiv.org/abs/2506.00788v3</id>
_FULL_ID = re.compile(r"^(?P<prefix>\d{4})\.(?P<number>\d{4,5})$")

STATUS_OK = "ok"  # có ít nhất một .tex
STATUS_EMPTY = "empty"  # record rỗng: version không có TeX
STATUS_ERROR = "error"  # lỗi tạm thời (mạng, html_response/corrupt_archive chưa hết lượt thử...)
STATUS_BLOCKED = "blocked"  # arXiv trả 403: dừng toàn bộ
# [Review #1] Chỉ ok/empty mới là "xong". html_response/corrupt_archive KHÔNG còn đi vào empty
# ngay (xem RETRYABLE_REASONS), nên resume sẽ thử lại chúng thay vì bỏ qua vĩnh viễn.
TERMINAL_STATUSES = frozenset({STATUS_OK, STATUS_EMPTY})  # xong, resume bỏ qua

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# [Review #5] Nhận diện "trông giống TeX" bằng marker thật, không chỉ có ký tự "\".
_TEX_MARKERS = re.compile(
    rb"\\(documentclass|documentstyle|begin|end|input|include|section|usepackage|def|"
    rb"newcommand|magnification|font|hsize|catcode|bye)\b"
)
_BINARY_MAGIC = (b"%!PS", b"%PDF", b"\x89PNG", b"GIF8", b"\xff\xd8")  # PS, PDF, PNG, GIF, JPEG
_MARKER_SCAN_BYTES = 200_000  # chỉ tìm marker trong 200KB đầu
_BAD_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')  # ký tự Windows không cho phép
_schema_warned = False  # chỉ cảnh báo thiếu src.schemas một lần


# ---------------------------------------------------------------------------
# Lỗi
# ---------------------------------------------------------------------------


class CrawlBlockedError(RuntimeError):
    """arXiv trả HTTP 403: có nguy cơ bị khóa IP, phải dừng toàn bộ chương trình."""


class CrawlAbortedError(RuntimeError):
    """Quá nhiều lỗi liên tiếp: dừng để không dồn dập request vào server đang có vấn đề."""


class DiskLimitError(CrawlAbortedError):
    """Dung lượng thư mục output vượt ``--max_disk_gb``: dừng sớm, chưa mất checkpoint."""


class FetchError(RuntimeError):
    """Tải một version thất bại sau khi đã thử lại (lỗi tạm thời, version sẽ được thử lại sau)."""


class TooLargeError(FetchError):
    """[Review #1] Nguồn vượt ``MAX_DOWNLOAD_BYTES``: thử lại cũng vô ích -> record rỗng ``too_large``.

    Attributes:
        downloaded: Số byte đã nhận được trước khi dừng.
    """

    def __init__(self, message: str, downloaded: int = 0) -> None:
        super().__init__(message)
        self.downloaded = downloaded


class ForbiddenRedirectError(FetchError):
    """[Review #8] Server chuyển hướng sang host khác ``export.arxiv.org``: không đi theo."""


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
    original, renamed = "/".join(parts), "/".join(cleaned)
    if renamed != original:
        # File bị đổi tên trên đĩa nhưng \input{...} trong .tex vẫn trỏ tên gốc: M2 sẽ không
        # resolve được file này. Chỉ cảnh báo để truy vết, không tự sửa nội dung .tex.
        logger.warning(
            "Doi ten file co ky tu khong hop le: %r -> %r (co the lam hong \\input/\\include)",
            original,
            renamed,
        )
    return renamed


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
        # Nghi trang lỗi của server, KHÔNG phải bài thật: crawl_version sẽ thử lại
        # (RETRYABLE_REASONS) thay vì chốt empty ngay.
        return Extracted(STATUS_EMPTY, "html_response", size_before=len(raw), n_files_before=1)
    if head.startswith(b"%PDF-"):  # PDF bọc trong gzip
        return Extracted(STATUS_EMPTY, "pdf_only", size_before=len(raw), n_files_before=1)
    # [Review #5] Trước đây chỉ cần "không có NUL trong 4KB đầu và có dấu \" nên PostScript,
    # file text bất kỳ... đều bị ghi thành main.tex. Giờ: loại theo magic bytes VÀ bắt buộc có
    # ít nhất một marker TeX/LaTeX trong phần đầu file.
    if (
        b"\x00" in head
        or head.startswith(_BINARY_MAGIC)
        or not _TEX_MARKERS.search(raw[:_MARKER_SCAN_BYTES])
    ):
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
        Archive hỏng GIỮA CHỪNG (đã đọc được ít nhất một member) trả ``corrupt_archive``.

    Raises:
        tarfile.ReadError: Khi không đọc nổi member nào (nghĩa là KHÔNG phải tar), để
            ``extract_tex_sources`` thử tiếp nhánh ``.gz`` đơn / văn bản thô.
        EOFError, OSError, zlib.error: Tương tự, khi hỏng ngay từ đầu.
    """
    result = Extracted(STATUS_EMPTY, "no_tex")
    seen: set[str] = set()
    members_read = 0
    # [Review #6] Trước đây ReadError giữa lúc duyệt bị bắt ở ngoài rồi `pass`, code đi tiếp vào
    # nhánh gunzip và cuối cùng trả not_tex (chốt vĩnh viễn). Giờ bắt NGAY trong vòng lặp.
    try:
        for member in tar:
            members_read += 1
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
    except (tarfile.ReadError, EOFError, OSError, zlib.error) as exc:
        if members_read == 0:
            raise  # chưa đọc được gì: không phải tar (hoặc hỏng từ đầu), để caller phân loại
        logger.warning("Tar hong giua chung (da doc %d member): %s", members_read, exc)
        return Extracted(
            STATUS_EMPTY,
            "corrupt_archive",  # thuộc RETRYABLE_REASONS: crawl_version sẽ thử lại
            size_before=result.size_before,
            n_files_before=result.n_files_before,
        )
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
        # Chỉ tới đây khi KHÔNG đọc được member nào (xem _from_tar): không phải tar,
        # thử .gz đơn / văn bản thô bên dưới. Tar hỏng giữa chừng đã trả corrupt_archive.
        pass
    except (EOFError, OSError, zlib.error) as exc:
        logger.warning("Archive hong: %s", exc)
        # corrupt_archive thuộc RETRYABLE_REASONS: crawl_version thử lại, không chốt ngay
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

    def _request(self, url: str) -> requests.Response:
        """Gửi GET, KHÔNG để ``requests`` tự đi theo redirect.

        [Review #8] Nếu ``export.arxiv.org`` chuyển hướng sang ``arxiv.org`` thì tự động đi theo
        sẽ vi phạm quy định (bị khóa IP). Ở đây chỉ đi theo redirect khi đích vẫn là
        ``export.arxiv.org`` (https); mỗi bước nhảy cũng chờ đủ ``delay`` như một request mới.

        Raises:
            ForbiddenRedirectError: Khi đích redirect nằm ngoài ``export.arxiv.org``.
            FetchError: Khi vượt ``MAX_REDIRECTS`` bước nhảy.
        """
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            if hop:
                self._last_end = self._clock()  # request trước vừa kết thúc
                self._wait_turn()
            response = self.session.get(
                current, stream=True, timeout=self._timeout, allow_redirects=False
            )
            if response.status_code not in _REDIRECT_CODES:
                return response
            location = response.headers.get("Location", "")
            response.close()
            target = urljoin(current, location) if location else ""
            parsed = urlparse(target)
            if parsed.hostname != ALLOWED_HOST or parsed.scheme != "https":
                raise ForbiddenRedirectError(
                    f"redirect sang {target or '(thieu Location)'}, chi duoc dung {ALLOWED_HOST}"
                )
            current = target
        raise FetchError(f"qua {MAX_REDIRECTS} lan redirect: {url}")

    @staticmethod
    def _check_complete(response: requests.Response, body: bytes) -> None:
        """[Review #1] So số byte nhận được với ``Content-Length`` để phát hiện tải thiếu.

        Khi server gắn ``Content-Encoding`` (gzip/deflate) thì ``iter_content`` đã giải nén, nên
        ``len(body)`` không so được với Content-Length: lúc đó đối chiếu số byte THÔ đã đọc từ
        socket (``response.raw.tell()``), và bỏ qua phép so nếu không lấy được.

        Raises:
            requests.exceptions.ChunkedEncodingError: Khi nhận ít hơn Content-Length. Đây là
                ``RequestException`` nên vòng thử lại trong ``get`` sẽ tải lại.
        """
        header = str(response.headers.get("Content-Length", ""))
        if not header.isdigit():
            return
        expected = int(header)
        encoding = str(response.headers.get("Content-Encoding", "") or "identity").lower()
        if encoding == "identity":
            received = len(body)
        else:
            tell = getattr(getattr(response, "raw", None), "tell", None)
            if tell is None:
                return
            try:
                received = tell()
            except (OSError, ValueError):
                return
        if received < expected:
            raise requests.exceptions.ChunkedEncodingError(
                f"tai thieu: nhan {received}/{expected} byte"
            )

    def _read_body(self, response: requests.Response) -> bytes:
        """Đọc body theo từng khối, dừng khi vượt ``MAX_DOWNLOAD_BYTES``.

        Raises:
            TooLargeError: Khi vượt giới hạn (không thử lại, chốt record rỗng ``too_large``).
            requests.exceptions.ChunkedEncodingError: Khi tải thiếu so với Content-Length.
        """
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=1 << 16):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise TooLargeError("file tai ve qua lon", downloaded=total)
            chunks.append(chunk)
        body = b"".join(chunks)
        self._check_complete(response, body)
        return body

    def get(self, url: str) -> FetchResult:
        """Tải một URL, thử lại khi lỗi tạm thời.

        Args:
            url: URL ``export.arxiv.org/src/...``.

        Returns:
            Kết quả khi nhận HTTP 200 (kèm body) hoặc 404/410 (body rỗng).

        Raises:
            CrawlBlockedError: Khi nhận HTTP 403 (dừng ngay, không thử lại).
            TooLargeError: Khi body vượt ``MAX_DOWNLOAD_BYTES`` (không thử lại).
            ForbiddenRedirectError: Khi bị chuyển hướng ra ngoài ``export.arxiv.org``.
            FetchError: Khi hết số lần thử lại mà vẫn lỗi, hoặc gặp mã HTTP bất thường.
        """
        started = self._clock()
        last_error = "khong ro"
        for attempt in range(len(self._backoff) + 1):
            self._wait_turn()
            retry_after: float | None = None
            try:
                response = self._request(url)  # không tự theo redirect (xem _request)
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


def _n_versions(row: dict) -> int:
    """Số version của một dòng discovery.

    [Review #3] Trước đây thiếu/None thì âm thầm coi là 1 version: bài có 4 version vẫn chỉ tải
    v1 rồi được đánh dấu hoàn tất. Giờ thiếu hoặc sai là LỖI, để không mất dữ liệu trong im lặng.

    Raises:
        ValueError: Khi ``n_versions`` thiếu, là None, không đọc được thành số, hoặc nhỏ hơn 1.
    """
    raw = row.get("n_versions")
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"{row.get('arxiv_id', '?')}: thieu truong n_versions trong discovery")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{row.get('arxiv_id', '?')}: n_versions khong hop le: {raw!r}") from None
    if value < 1:
        raise ValueError(f"{row.get('arxiv_id', '?')}: n_versions phai >= 1, nhan {value}")
    return value


def _has_n_versions(row: dict) -> bool:
    """True nếu ``_n_versions(row)`` đọc được."""
    try:
        _n_versions(row)
    except ValueError:
        return False
    return True


def _revised_dates(row: dict) -> tuple[list[str], list[int]]:
    """Ngày của v2..vn và danh sách số version KHÔNG có ngày hợp lệ.

    Dòng discovery chưa vá ``version_dates`` thì ngày version cuối lấy từ ``latest_version_date``.

    Returns:
        (revised_dates, missing): ``missing`` rỗng nghĩa là đủ ``n_versions - 1`` ngày.
    """
    n_versions = _n_versions(row)
    version_dates = row.get("version_dates") or {}
    revised: list[str] = []
    missing: list[int] = []
    for number in range(2, n_versions + 1):
        value = version_dates.get(f"v{number}")
        if value is None and number == n_versions:
            value = row.get("latest_version_date")
        date = _iso_date(value)
        if date:
            revised.append(date)
        else:
            missing.append(number)
    return revised, missing


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
    revised, missing = _revised_dates(row)
    if missing:
        # [Review #9] Thiếu ngày của vài version giữa chừng: revised_dates ngắn hơn
        # n_versions - 1 và các ngày còn lại bị dồn lệch vị trí. Schema Paper không cho thêm
        # trường lạ nên không gắn cờ vào metadata.json; cờ đi vào summary
        # (``revised_dates_incomplete``) qua _write_metadata_once.
        logger.warning(
            "%s: thieu ngay cua version %s -> revised_dates co %d/%d phan tu, "
            "cac ngay con lai co the bi lech vi tri",
            row.get("arxiv_id"),
            missing,
            len(revised),
            _n_versions(row) - 1,
        )
    authors = row.get("authors") or []
    if isinstance(authors, (list, tuple)):
        authors = list(authors)
    else:
        # list("John Doe") sẽ ra ['J', 'o', 'h', ...] mà schema Paper vẫn chấp nhận: bọc lại.
        logger.warning("%s: authors khong phai list (%s), boc thanh list 1 phan tu",
                       row.get("arxiv_id"), type(authors).__name__)
        authors = [str(authors)]
    metadata = {
        "arxiv_id": row["arxiv_id"],
        "title": row.get("title") or None,
        "authors": authors,
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


def find_ids_not_in_discovery(
    papers: list[dict], id_start: str, id_end: str
) -> list[str] | None:
    """[Review #3] ID nằm trong dải được giao nhưng discovery không có (``select_papers`` bỏ qua).

    Chỉ tính được khi hai đầu dải cùng tháng (``YYMM``) và cùng độ rộng số thứ tự, vì khi đó
    mọi số ở giữa đều là ID có thể tồn tại. Dải đi qua nhiều tháng thì không biết mỗi tháng có
    bao nhiêu bài. Lưu ý: ID "thiếu" có thể là bài bị bộ lọc của discovery loại hợp lệ, hoặc
    ID không tồn tại trên arXiv; danh sách này chỉ để đo độ phủ, không tự động là lỗi.

    Args:
        papers: Các bài đã chọn bằng ``select_papers`` (chưa cắt ``limit``).
        id_start: ID đầu dải.
        id_end: ID cuối dải.

    Returns:
        Danh sách ID thiếu (sắp xếp), hoặc None nếu dải không tính được.
    """
    start, end = _FULL_ID.match(id_start), _FULL_ID.match(id_end)
    if not start or not end:
        return None
    if start["prefix"] != end["prefix"] or len(start["number"]) != len(end["number"]):
        return None
    width = len(start["number"])
    have = {row["arxiv_id"] for row in papers}
    return [
        f"{start['prefix']}.{number:0{width}d}"
        for number in range(int(start["number"]), int(end["number"]) + 1)
        if f"{start['prefix']}.{number:0{width}d}" not in have
    ]


def add_coverage(
    summary: dict, papers: list[dict], id_start: str, id_end: str, limit: int | None = None
) -> None:
    """[Review #3] Ghi độ phủ so với dải được giao vào ``summary`` và in cảnh báo.

    Thêm các khóa ``ids_not_in_discovery`` (danh sách, hoặc None nếu không tính được),
    ``ids_not_in_discovery_count`` và ``range_size``. Khi dùng ``limit`` thì bỏ qua, vì
    ``papers`` bị cắt chủ ý và mọi ID sau điểm cắt sẽ bị tính là thiếu.
    """
    missing = None if limit else find_ids_not_in_discovery(papers, id_start, id_end)
    summary["ids_not_in_discovery"] = missing
    summary["ids_not_in_discovery_count"] = None if missing is None else len(missing)
    if missing is None:
        logger.info("Khong tinh duoc do phu so voi dai %s -> %s (limit hoac dai nhieu thang)",
                    id_start, id_end)
        return
    if missing:
        logger.warning(
            "%d ID trong dai %s -> %s KHONG co trong discovery (VD: %s)",
            len(missing),
            id_start,
            id_end,
            ", ".join(missing[:5]),
        )


def validate_n_versions(papers: list[dict]) -> None:
    """[Review #3] Kiểm tra MỌI dòng có ``n_versions`` hợp lệ TRƯỚC khi bắt đầu crawl.

    Raises:
        ValueError: Liệt kê (tối đa 10) ID thiếu/sai ``n_versions``.
    """
    bad = [str(row.get("arxiv_id", "?")) for row in papers if not _has_n_versions(row)]
    if bad:
        raise ValueError(
            f"{len(bad)} bai thieu/sai n_versions trong discovery (se chi tai v1 va mat du lieu), "
            f"VD: {bad[:10]}"
        )


def fetch_n_versions_from_api(fetcher: "PoliteFetcher", arxiv_ids: list[str]) -> dict[str, int]:
    """[Review #3] Tra số version mới nhất của các bài qua arXiv API theo lô.

    Đi qua ``fetcher`` nên vẫn tuân thủ delay 3 giây, một kết nối, dừng khi 403. ID entry trong
    Atom feed có dạng ``.../abs/2506.00788v3``: phần ``v3`` chính là số version mới nhất.

    Args:
        fetcher: Bộ tải lịch sự.
        arxiv_ids: ID cần tra.

    Returns:
        ID -> n_versions của những bài tra được (ID không tra được thì vắng mặt).

    Raises:
        CrawlBlockedError: Khi arXiv trả HTTP 403.
    """
    found: dict[str, int] = {}
    for start in range(0, len(arxiv_ids), API_BATCH):
        batch = arxiv_ids[start : start + API_BATCH]
        url = API_URL.format(ids=",".join(batch), n=len(batch))
        try:
            fetched = fetcher.get(url)
            if fetched.status_code != 200:
                continue
            root = ET.fromstring(fetched.body)
        except (FetchError, ET.ParseError) as exc:
            logger.warning("Tra n_versions bang API that bai (lo %d): %s", start // API_BATCH, exc)
            continue
        for entry in root.iterfind(f"{_ATOM_NS}entry"):
            match = _ATOM_ID.search(entry.findtext(f"{_ATOM_NS}id") or "")
            if match:
                found[match["id"]] = int(match["n"])
    return found


def resolve_n_versions(
    papers: list[dict], fetcher: "PoliteFetcher", *, api_fallback: bool = True
) -> list[str]:
    """[Review #3] Đảm bảo mọi bài có ``n_versions``; bài thiếu thì tra API (nếu được phép).

    Cập nhật ``row["n_versions"]`` TRONG BỘ NHỚ (không sửa file discovery).

    Args:
        papers: Dòng discovery sẽ crawl.
        fetcher: Bộ tải lịch sự dùng cho API.
        api_fallback: False thì không tra API, bài thiếu sẽ làm ``validate_n_versions`` báo lỗi.

    Returns:
        Danh sách ID đã được điền ``n_versions`` từ API.

    Raises:
        ValueError: Khi sau khi tra API vẫn còn bài thiếu/sai ``n_versions``.
        CrawlBlockedError: Khi arXiv trả HTTP 403.
    """
    from_api: list[str] = []
    missing = [row for row in papers if not _has_n_versions(row)]
    if missing and api_fallback:
        logger.warning("%d bai thieu n_versions, tra arXiv API", len(missing))
        found = fetch_n_versions_from_api(fetcher, [row["arxiv_id"] for row in missing])
        for row in missing:
            if row["arxiv_id"] in found:
                row["n_versions"] = found[row["arxiv_id"]]
                from_api.append(row["arxiv_id"])
    validate_n_versions(papers)
    return from_api


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
    """RAM ĐỈNH của tiến trình (MB); None nếu không đo được.

    [Review #4] ``info.rss`` chỉ là RAM tại thời điểm gọi chứ không phải đỉnh. Dùng
    ``resource.getrusage`` (Linux/macOS), Windows không có thì dùng ``peak_wset`` của psutil.
    """
    try:
        import resource  # noqa: PLC0415  (không có trên Windows)

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss: KB trên Linux, byte trên macOS
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except ImportError:
        try:
            import psutil  # noqa: PLC0415

            info = psutil.Process().memory_info()
            return round(getattr(info, "peak_wset", info.rss) / 2**20, 1)
        except ImportError:
            return None


def _current_rss_mb() -> float | None:
    """RAM hiện tại (MB) của tiến trình; None nếu không đo được.

    Gọi sau mỗi version và ghi vào log (``rss_mb``) để ``summarize`` tính RAM TRUNG BÌNH
    (đề yêu cầu), vì peak một mình không cho biết mức dùng điển hình.
    """
    try:
        import psutil  # noqa: PLC0415

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except ImportError:
        pass
    try:  # Linux không có psutil: đọc /proc (cột 2 của statm là số trang resident)
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


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
        Thời gian và RAM lấy từ LOG nên không lệch sau resume (khác ``elapsed_this_run_s``):
        ``wall_seconds_*``, ``seconds_per_paper*`` và ``rss_mb_*``. Log cũ chưa có
        ``wall_seconds``/``rss_mb`` thì bị bỏ qua (xem ``versions_with_wall_seconds``).
    """
    expected = {(row["arxiv_id"], v) for row in papers for v in range(1, _n_versions(row) + 1)}
    records = [state[key] for key in expected if key in state]
    by_status = Counter(record["status"] for record in records)
    reasons = Counter(r.get("reason") for r in records if r["status"] == STATUS_EMPTY)
    size_before = sum(r.get("size_before") or 0 for r in records)
    size_after = sum(r.get("size_after") or 0 for r in records)
    seconds = [r["seconds"] for r in records if r.get("seconds") is not None]
    # [Review #5] wall_seconds = fetch + giải nén + ghi đĩa (đã gồm thời gian chờ delay);
    # cộng theo arxiv_id để có thời gian mỗi bài. Bài mới crawl dở thì tổng mới là phần đã xong.
    wall = [r["wall_seconds"] for r in records if r.get("wall_seconds") is not None]
    per_paper: dict[str, float] = {}
    for record in records:
        if record.get("wall_seconds") is not None:
            per_paper[record["arxiv_id"]] = (
                per_paper.get(record["arxiv_id"], 0.0) + record["wall_seconds"]
            )
    rss = [r["rss_mb"] for r in records if r.get("rss_mb") is not None]
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
        "versions_with_wall_seconds": len(wall),
        "wall_seconds_total": round(sum(wall), 1),
        "wall_seconds_per_version_mean": round(sum(wall) / len(wall), 3) if wall else None,
        "seconds_per_paper_mean": (
            round(sum(per_paper.values()) / len(per_paper), 2) if per_paper else None
        ),
        "seconds_per_paper_max": round(max(per_paper.values()), 2) if per_paper else None,
        "seconds_per_paper": {pid: round(sec, 2) for pid, sec in sorted(per_paper.items())},
        "elapsed_this_run_s": round(elapsed_s, 1),  # chỉ lần chạy này, lệch sau resume
        "rss_mb_mean": round(sum(rss) / len(rss), 1) if rss else None,  # trung bình các mẫu/log
        "rss_mb_max_logged": max(rss) if rss else None,
        "peak_rss_mb": _peak_rss_mb(),  # đỉnh của TIẾN TRÌNH lần chạy này
        "output_dir_bytes": _dir_bytes(out_dir) if out_dir.exists() else 0,
    }


def crawl_version(
    fetcher: PoliteFetcher,
    row: dict,
    version: int,
    folder: Path,
    log_path: Path,
    prev_soft_errors: int = 0,
) -> dict:
    """Tải, lọc và ghi MỘT version; luôn ghi một dòng log.

    Args:
        fetcher: Bộ tải lịch sự.
        row: Dòng discovery của bài.
        version: Số version.
        folder: Thư mục của bài.
        log_path: File log checkpoint.
        prev_soft_errors: Số lần version này đã gặp ``RETRYABLE_REASONS`` ở các lần chạy trước
            (lấy từ ``soft_errors`` trong log; ``run`` truyền vào).

    Returns:
        Bản ghi log vừa ghi (đã có ``status``). Gặp ``RETRYABLE_REASONS`` mà chưa hết
        ``MAX_SOFT_RETRIES`` lượt thì ``status`` là ``error`` và KHÔNG tạo thư mục ``v{n}``.

    Raises:
        CrawlBlockedError: Khi arXiv trả HTTP 403 (đã ghi log ``blocked`` trước khi raise).
    """
    arxiv_id = row["arxiv_id"]
    url = SRC_URL.format(arxiv_id=arxiv_id, version=version)
    record: dict = {"arxiv_id": arxiv_id, "version": version, "url": url}
    started = time.monotonic()
    extracted: Extracted | None = None
    try:
        fetched = fetcher.get(url)
    except CrawlBlockedError as exc:
        record.update(status=STATUS_BLOCKED, reason=str(exc))
        append_jsonl(log_path, record)
        raise
    except TooLargeError as exc:
        # [Review #1] Quá lớn thì thử lại cũng vô ích: chốt record rỗng với reason too_large.
        fetched = FetchResult(200, b"", None, 1, time.monotonic() - started)
        extracted = Extracted(STATUS_EMPTY, "too_large", size_before=exc.downloaded)
    except FetchError as exc:
        elapsed = round(time.monotonic() - started, 2)
        record.update(
            status=STATUS_ERROR,
            reason=str(exc),
            soft_errors=prev_soft_errors,  # giữ bộ đếm, lỗi mạng không làm nó về 0
            seconds=elapsed,
            wall_seconds=elapsed,
            rss_mb=_current_rss_mb(),
        )
        append_jsonl(log_path, record)
        return record

    # [Review #2] extract_tex_sources và write_version nằm TRONG try: một version lỗi (đĩa đầy,
    # tên file lạ, MemoryError...) chỉ ghi log "error" kèm tên lỗi rồi đi tiếp. Việc dừng cả
    # chương trình do lỗi liên tiếp đã có max_consecutive_errors lo trong run().
    status: str = STATUS_ERROR
    reason: str | None = None
    soft_errors = prev_soft_errors
    wrote = False
    try:
        if extracted is None:
            if fetched.status_code in (404, 410):
                extracted = Extracted(STATUS_EMPTY, f"http_{fetched.status_code}")
            else:
                extracted = extract_tex_sources(fetched.body)
        retryable = extracted.reason in RETRYABLE_REASONS
        if retryable:
            soft_errors = prev_soft_errors + 1
        if retryable and soft_errors < MAX_SOFT_RETRIES:
            # [Review #1] html_response / corrupt_archive: ghi error, KHÔNG gọi write_version
            # (không tạo v{n}) nên resume sẽ thử lại. Lần thứ MAX_SOFT_RETRIES mà vẫn gặp thì
            # rơi xuống nhánh dưới và chốt empty.
            status, reason = STATUS_ERROR, extracted.reason
        else:
            write_version(folder, version, extracted.files)
            wrote = True
            status, reason = extracted.status, extracted.reason
    except Exception as exc:  # noqa: BLE001  (cố ý bắt rộng: một version hỏng không giết pipeline)
        logger.exception("Loi khi xu ly %sv%d", arxiv_id, version)
        status, reason = STATUS_ERROR, f"{type(exc).__name__}: {str(exc)[:300]}"
    if extracted is None:  # lỗi xảy ra ngay lúc giải nén: không có số liệu phân tích
        extracted = Extracted(STATUS_ERROR)
    kept = extracted.files if wrote else {}
    record.update(
        status=status,
        reason=reason,
        soft_errors=soft_errors,
        http_status=fetched.status_code,
        content_type=fetched.content_type,
        attempts=fetched.attempts,
        archive_bytes=len(fetched.body) or extracted.size_before,
        size_before=extracted.size_before,
        size_after=extracted.size_after if wrote else 0,
        n_files_before=extracted.n_files_before,
        n_files_after=len(kept),
        files=sorted(kept)[:50],
        skipped=extracted.skipped[:20],
        seconds=round(fetched.seconds, 2),  # chỉ phần tải (gồm chờ delay)
        # [Review #5] toàn bộ version: tải + giải nén + ghi đĩa; rss_mb để tính RAM trung bình
        wall_seconds=round(time.monotonic() - started, 2),
        rss_mb=_current_rss_mb(),
    )
    append_jsonl(log_path, record)
    return record


def _write_metadata_once(folder: Path, row: dict) -> dict:
    """Ghi ``metadata.json`` của bài nếu chưa có (không cần mạng, nên làm trước khi tải).

    Args:
        folder: Thư mục của bài.
        row: Dòng discovery của bài.

    Returns:
        Dict các vấn đề phát hiện (rỗng nếu êm): ``error`` (str) khi không dựng được metadata
        hợp lệ, ``revised_dates_missing`` (list[int]) là các version thiếu ngày. ``run`` gom
        chúng vào summary để không chỉ nằm trên console.
    """
    issues: dict = {}
    _, missing = _revised_dates(row)  # [Review #9] kiểm tra cả khi metadata.json đã có
    if missing:
        issues["revised_dates_missing"] = missing
    target = folder / "metadata.json"
    if target.exists():
        return issues
    try:
        metadata = build_metadata(row)
    # [Review #7] pydantic.ValidationError là con của ValueError; thêm TypeError cho dữ liệu sai
    # kiểu (authors không phải list...). Trả lỗi về cho run() ghi vào summary thay vì chỉ log.
    except (ValueError, TypeError) as exc:
        logger.error("metadata %s khong hop le: %s", row["arxiv_id"], exc)
        issues["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        return issues
    folder.mkdir(parents=True, exist_ok=True)
    write_json(target, metadata)
    return issues


def run(
    papers: list[dict],
    out_dir: Path,
    fetcher: PoliteFetcher,
    *,
    write_metadata: bool = True,
    max_consecutive_errors: int = 10,
    resume: bool = True,
    api_fallback: bool = True,
    max_disk_gb: float | None = None,
) -> dict:
    """Crawl lần lượt từng bài, từng version; có Checkpoint/Resume.

    Args:
        papers: Dòng discovery của các bài cần crawl (đã sắp xếp).
        out_dir: Thư mục đầu ra.
        fetcher: Bộ tải lịch sự (MỘT kết nối, có delay).
        write_metadata: Ghi ``metadata.json`` cho mỗi bài.
        max_consecutive_errors: Dừng sau chừng này version lỗi LIÊN TIẾP.
        resume: True bỏ qua version đã xong theo log; False tải lại tất cả (ghi đè).
        api_fallback: True thì bài thiếu ``n_versions`` được tra bằng arXiv API trước khi crawl.
        max_disk_gb: Dừng (``DiskLimitError``) khi thư mục output vượt mức này (GB); None là
            không giới hạn. Đo mỗi ``DISK_CHECK_EVERY`` version vừa tải, nên có thể vượt nhẹ.

    Returns:
        Dict tổng hợp (``summarize``), có thêm khóa ``exit_code``, ``n_versions_from_api``,
        ``metadata_errors`` và ``revised_dates_incomplete``.

    Raises:
        ValueError: Khi còn bài thiếu/sai ``n_versions`` (kiểm tra TRƯỚC khi tải bất kỳ gì).
        CrawlBlockedError: Khi arXiv trả HTTP 403.
        CrawlAbortedError: Khi quá nhiều lỗi liên tiếp (hoặc ``DiskLimitError`` khi đầy đĩa).
    """
    if max_disk_gb is not None and max_disk_gb <= 0:
        raise ValueError(f"max_disk_gb phai > 0, nhan {max_disk_gb}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # [Review #3] Kiểm tra (và nếu được, tự điền từ API) n_versions của TẤT CẢ bài trước khi
    # crawl, để không có bài nào âm thầm chỉ tải v1.
    from_api = resolve_n_versions(papers, fetcher, api_fallback=api_fallback)
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
    metadata_errors: list[dict] = []
    revised_incomplete: list[dict] = []
    for index, row in enumerate(papers, start=1):
        folder = out_dir / to_folder_name(row["arxiv_id"])
        if write_metadata:
            # ghi TRƯỚC khi tải: ngắt giữa chừng vẫn có metadata
            issues = _write_metadata_once(folder, row)
            if "error" in issues:
                metadata_errors.append({"arxiv_id": row["arxiv_id"], "error": issues["error"]})
            if "revised_dates_missing" in issues:
                revised_incomplete.append(
                    {"arxiv_id": row["arxiv_id"], "missing_versions": issues["revised_dates_missing"]}
                )
        for version in range(1, _n_versions(row) + 1):
            previous = state.get((row["arxiv_id"], version), {})
            if previous.get("status") in TERMINAL_STATUSES and (folder / f"v{version}").is_dir():
                continue  # resume: version này đã xong
            try:
                folder.mkdir(parents=True, exist_ok=True)
                record = crawl_version(
                    fetcher,
                    row,
                    version,
                    folder,
                    log_path,
                    # [Review #1] truyền bộ đếm "lỗi mềm" từ log để giới hạn số lần thử lại
                    prev_soft_errors=int(previous.get("soft_errors") or 0),
                )
            except (CrawlBlockedError, CrawlAbortedError):
                raise  # phải dừng cả chương trình
            except Exception as exc:  # noqa: BLE001  [Review #2] lưới an toàn cuối cùng
                logger.exception("Loi bat ngo o %sv%d", row["arxiv_id"], version)
                record = {
                    "arxiv_id": row["arxiv_id"],
                    "version": version,
                    "status": STATUS_ERROR,
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "soft_errors": int(previous.get("soft_errors") or 0),
                }
                try:
                    append_jsonl(log_path, record)
                except OSError:
                    logger.error("Khong ghi duoc log cho %sv%d", row["arxiv_id"], version)
            state[(row["arxiv_id"], version)] = record
            fetched_now += 1
            consecutive_errors = consecutive_errors + 1 if record["status"] == STATUS_ERROR else 0
            _log_progress(record, index, len(papers), fetched_now, total - done_before, started)
            if consecutive_errors >= max_consecutive_errors:
                raise CrawlAbortedError(
                    f"{consecutive_errors} loi lien tiep, dung de tranh dan dap"
                )
            if max_disk_gb is not None and fetched_now % DISK_CHECK_EVERY == 0:
                disk_gb = _dir_bytes(out_dir) / 1024**3
                if disk_gb > max_disk_gb:
                    raise DiskLimitError(
                        f"Dia output {disk_gb:.2f} GB vuot --max_disk_gb {max_disk_gb} GB "
                        f"(da tai {fetched_now} version trong lan chay nay)"
                    )
    summary = summarize(state, papers, out_dir, time.monotonic() - started)
    summary["exit_code"] = 1 if summary["by_status"].get(STATUS_ERROR) else 0
    summary["n_versions_from_api"] = from_api
    # [Review #7, #9] Đưa vào summary (và file riêng) thay vì chỉ log ra console
    summary["metadata_errors"] = metadata_errors
    summary["revised_dates_incomplete"] = revised_incomplete
    if metadata_errors:
        write_json(out_dir / METADATA_ERRORS_NAME, metadata_errors)
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
    api_fallback: bool = True,
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
        api_fallback: True thì thiếu ``n_versions`` trong discovery sẽ tra arXiv API.
        fetcher: Bộ tải có sẵn (tests truyền bản giả); None là tạo mới.

    Returns:
        Dict tổng hợp (như ``crawl_range``) kèm khóa ``versions``: trạng thái từng version.

    Raises:
        ValueError: Khi ID sai dạng, bài không có trong discovery, thiếu ``n_versions`` (mà tra
            API cũng không ra), hoặc thiếu email liên hệ.
        CrawlBlockedError: Khi arXiv trả HTTP 403.
        CrawlAbortedError: Khi quá nhiều lỗi liên tiếp.
    """
    to_folder_name(arxiv_id)  # ID sai dạng nổ ngay tại đây
    row = row or _load_discovery_row(arxiv_id, Path(discovery_path))
    if row is None:
        raise ValueError(f"{arxiv_id} khong co trong discovery {discovery_path} (can n_versions)")
    fetcher = fetcher or PoliteFetcher(USER_AGENT.format(contact=_require_contact(contact)), delay)
    out_dir = Path(out_dir)
    summary = run(
        [row],
        out_dir,
        fetcher,
        write_metadata=write_metadata,
        resume=resume,
        api_fallback=api_fallback,
    )
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
    api_fallback: bool = True,
    max_disk_gb: float | None = None,
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
        api_fallback: True thì thiếu ``n_versions`` trong discovery sẽ tra arXiv API.
        max_disk_gb: Dừng (``DiskLimitError``) khi thư mục output vượt mức này (GB).
        fetcher: Bộ tải có sẵn (tests truyền bản giả); None là tạo mới.

    Returns:
        Dict tổng hợp: số version theo trạng thái, size_before/size_after, ``exit_code``,
        ``ids_not_in_discovery``...

    Raises:
        FileNotFoundError: Khi không thấy file discovery.
        ValueError: Khi không có bài nào trong dải, thiếu email liên hệ, hoặc còn bài thiếu
            ``n_versions``.
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
        api_fallback=api_fallback,
        max_disk_gb=max_disk_gb,
    )
    add_coverage(summary, papers, id_start, id_end, limit)  # [Review #3] độ phủ so với dải
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
    add(
        "--max_disk_gb",
        default=None,
        type=float,
        help="Dung khi thu muc output vuot nguong nay (GB). Mac dinh khong gioi han.",
    )
    add("--no_metadata", action="store_true", help="Khong ghi metadata.json")
    add(
        "--no_api_fallback",
        action="store_true",
        help="Khong tra n_versions bang arXiv API khi discovery thieu (se bao loi)",
    )
    return parser


def _printable(summary: dict, max_items: int = 20) -> dict:
    """Bản summary để IN: list/dict dài hơn ``max_items`` được rút gọn (file thì giữ đủ)."""
    return {
        key: (
            f"<{len(value)} muc, xem day du trong {SUMMARY_NAME}>"
            if isinstance(value, (list, dict)) and len(value) > max_items
            else value
        )
        for key, value in summary.items()
    }


def main(argv: list[str] | None = None) -> int:
    """Điểm vào của CLI.

    Args:
        argv: Tham số dòng lệnh (None là dùng ``sys.argv``).

    Returns:
        Mã thoát: 0 xong; 1 còn version lỗi (chạy lại để thử lại); 2 sai tham số;
        3 bị arXiv chặn (HTTP 403); 4 dừng vì quá nhiều lỗi liên tiếp; 5 đầy đĩa (--max_disk_gb).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.delay < MIN_DELAY_S:
        print(f"LOI: --delay phai >= {MIN_DELAY_S} giay (quy dinh arXiv).", file=sys.stderr)
        return 2
    if args.max_disk_gb is not None and args.max_disk_gb <= 0:
        print("LOI: --max_disk_gb phai > 0.", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 10:
        # select_papers lay N bai dau theo ID; cac bai lien nhau co the toan PDF-only.
        print(
            f"CANH BAO: --limit {args.limit} chi lay {args.limit} bai dau theo ID, co the toan bai "
            "khong co TeX nen smoke test khong dai dien. Nen dung --limit >= 10.",
            file=sys.stderr,
        )
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
            api_fallback=not args.no_api_fallback,
            max_disk_gb=args.max_disk_gb,
        )
    except ValueError as exc:  # [Review #3] thieu n_versions: dung truoc khi tai gi
        print(f"LOI du lieu discovery: {exc}", file=sys.stderr)
        return 2
    except CrawlBlockedError as exc:
        print(f"\nBI CHAN: {exc}. DUNG NGAY. Cho vai gio, kiem tra User-Agent/delay roi chay lai.")
        return 3
    except DiskLimitError as exc:
        print(f"\nDAY DIA: {exc}. Don/chuyen du lieu hoac tang --max_disk_gb roi chay lai CUNG LENH.")
        return 5
    except CrawlAbortedError as exc:
        print(f"\nDUNG: {exc}. Kiem tra mang roi chay lai CUNG LENH de tiep tuc.")
        return 4
    except KeyboardInterrupt:
        print("\nDa dung theo yeu cau. Chay lai CUNG LENH de tiep tuc.")
        return 1
    add_coverage(summary, papers, args.id_start, args.id_end, args.limit)
    write_json(args.out / SUMMARY_NAME, summary)
    print(json.dumps(_printable(summary), indent=2, ensure_ascii=False))
    missing_ids = summary.get("ids_not_in_discovery")
    if missing_ids:
        print(f"CANH BAO: {len(missing_ids)} ID trong dai khong co trong discovery (VD: "
              f"{', '.join(missing_ids[:5])}); day du trong {SUMMARY_NAME}")
    print(
        f"Tong ket ghi o {args.out / SUMMARY_NAME} (lan chay nay {time.monotonic() - started:.0f}s)"
    )
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())