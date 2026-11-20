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
    - Một kết nối tại một thời điểm, hai lần gửi request cách nhau tối thiểu 3 giây
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

Luồng xử lý (sau đợt tối ưu M1; ba file con, ``crawler.py`` import lại tên cũ để không vỡ code):
    1. ``downloader.py``: tải TUẦN TỰ từng version thẳng xuống đĩa vào thư mục spool
       (``<stem>.part`` -> ``<stem>.src`` + ``<stem>.meta.json``). Đứt thì tải tiếp bằng ``Range``,
       kể cả sau khi tắt tiến trình. Version chỉ có PDF thì đóng kết nối ngay, không tải body.
    2. ``extractor.py``: một luồng nền đọc gói theo luồng, ghi ``.tex``/``.bib`` vào thư mục tạm,
       đổi tên nguyên tử thành ``v{n}/``, ghi dòng log rồi xóa gói. Luồng tải không phải chờ.
    3. Khởi động lại: gói còn trong spool mà chưa trích xuất được làm tiếp, KHÔNG tải lại.
    Mục ``crawler`` của ``configs/config.yaml``: ``timeout_read`` (30), ``extract_workers`` (1),
    ``max_spool_gb`` (2).
    Trường log mới (chỉ THÊM, trường cũ giữ nguyên nghĩa): ``wait_seconds``,
    ``transfer_seconds``, ``extract_seconds``, ``attempt_log``, ``body_skipped``.
    ``wall_seconds = seconds + extract_seconds``.

Dùng từ code khác (Runner ``main.py``): ``crawl_range(id_start, id_end, out_dir, resume=True)`` và
``crawl_paper(arxiv_id, out_dir)``, đúng chữ ký stub ban đầu của task (xem hai hàm đó bên dưới).

Cách chạy (từ thư mục gốc repo)::

    py src/m1_crawling/crawler.py --id_start 2506.00788 --id_end 2506.05787 --contact "email"

Thông điệp in ra màn hình cố ý viết KHÔNG DẤU để console Windows (cp1252) không lỗi mã hóa.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Chạy "py src/m1_crawling/crawler.py" chỉ thêm src/m1_crawling vào sys.path, nên package `src`
# không import được. Thêm thư mục gốc repo (cùng cách tests/make_mock.py làm).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.ids import to_folder_name  # noqa: E402
from src.common.utils_io import append_jsonl, iter_jsonl, write_json  # noqa: E402

# PoliteFetcher và các lỗi tải nay nằm ở downloader.py; import lại ở đây để code cũ
# (``crawler.PoliteFetcher``, ``crawler.FetchError``...) không vỡ.
from src.m1_crawling.downloader import (  # noqa: E402, F401
    ALLOWED_HOST,
    DEFAULT_TIMEOUT,
    MAX_REDIRECTS,
    MIN_DELAY_S,
    RETRY_BACKOFF_S,
    CrawlBlockedError,
    FetchError,
    FetchResult,
    ForbiddenRedirectError,
    PoliteFetcher,
    TooLargeError,
    meta_path,
    part_path,
)

# Tương tự, phần trích xuất nay nằm ở extractor.py.
from src.m1_crawling.extractor import (  # noqa: E402, F401
    KEEP_EXTENSIONS,
    SINGLE_FILE_NAME,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    Extracted,
    _rmtree_with_retry,
    extract_from_path,
    extract_tex_sources,
    staging_dir,
    write_version,
)

# Giới hạn kích thước: chỉnh ``crawler.MAX_*`` vẫn có tác dụng (xem limits.limit).
from src.m1_crawling.limits import (  # noqa: E402, F401
    MAX_DOWNLOAD_BYTES,
    MAX_KEPT_FILE_BYTES,
    MAX_UNCOMPRESSED_BYTES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

SRC_URL = "https://export.arxiv.org/src/{arxiv_id}v{version}"  # KHÔNG dùng arxiv.org
DEFAULT_DISCOVERY = Path("data/raw/discovery_25k.jsonl")  # đầu ra của P1-00 (đã vá)
DEFAULT_OUT = Path("data/raw/papers")
CONTACT_ENV = "ARXIV_CONTACT"  # biến môi trường thay cho --contact / tham số contact
USER_AGENT = "DS-Project-HCMUS-arXiv-crawler/0.1 (student project; contact: {contact})"
LOG_NAME = "crawl_log.jsonl"
SUMMARY_NAME = "crawl_summary.json"
DISK_CHECK_EVERY = 500  # cứ N version vừa tải thì đo dung lượng output một lần (rglob tốn kém)
METADATA_ERRORS_NAME = "metadata_errors.json"  # bài không dựng được metadata.json

# [Review #1] Lý do "nghi lỗi tạm thời" (arXiv trả trang HTML lỗi, tải thiếu/hỏng): KHÔNG chốt
# empty ngay mà ghi STATUS_ERROR để resume thử lại, tối đa MAX_SOFT_RETRIES lần rồi mới chốt.
RETRYABLE_REASONS = frozenset({"html_response", "corrupt_archive"})
MAX_SOFT_RETRIES = 3  # tổng số lần được gặp lý do trên; lần thứ 3 vẫn gặp thì chốt empty

SPOOL_DIRNAME = ".spool"  # nơi chứa gói đã tải chờ trích xuất khi không chỉ định spool_dir
DEFAULT_EXTRACT_WORKERS = 1
DEFAULT_MAX_SPOOL_GB = 2.0

# [Review #3] Fallback tra n_versions bằng arXiv API (cũng qua PoliteFetcher: delay 3 giây).
API_URL = "https://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
API_BATCH = 100  # số ID mỗi request
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ATOM_ID = re.compile(r"/abs/(?P<id>.+?)v(?P<n>\d+)\s*$")  # <id>http://arxiv.org/abs/2506.00788v3</id>
_FULL_ID = re.compile(r"^(?P<prefix>\d{4})\.(?P<number>\d{4,5})$")

# STATUS_OK / STATUS_EMPTY / STATUS_ERROR import từ extractor.py (error: lỗi tạm thời như mạng,
# html_response/corrupt_archive chưa hết lượt thử...).
STATUS_BLOCKED = "blocked"  # arXiv trả 403: dừng toàn bộ
# [Review #1] Chỉ ok/empty mới là "xong". html_response/corrupt_archive KHÔNG còn đi vào empty
# ngay (xem RETRYABLE_REASONS), nên resume sẽ thử lại chúng thay vì bỏ qua vĩnh viễn.
TERMINAL_STATUSES = frozenset({STATUS_OK, STATUS_EMPTY})  # xong, resume bỏ qua

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_schema_warned = False  # chỉ cảnh báo thiếu src.schemas một lần


# ---------------------------------------------------------------------------
# Lỗi
# ---------------------------------------------------------------------------


class CrawlAbortedError(RuntimeError):
    """Quá nhiều lỗi liên tiếp: dừng để không dồn dập request vào server đang có vấn đề."""


class DiskLimitError(CrawlAbortedError):
    """Dung lượng thư mục output vượt ``--max_disk_gb``: dừng sớm, chưa mất checkpoint."""


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
    # Phân rã thời gian và lỗi theo lần thử (chỉ có ở log ghi từ bản tối ưu trở đi).
    attempt_errors: Counter[str] = Counter()
    for record in records:
        for attempt in record.get("attempt_log") or []:
            if attempt.get("error"):
                attempt_errors[str(attempt["error"]).split(":", 1)[0]] += 1
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
        # Body PDF KHÔNG tải (đóng kết nối ngay sau header); archive_bytes của các dòng này là
        # Content-Length, tức số byte đã tránh được chứ không phải số byte đã tải.
        "bytes_skipped_pdf": sum(
            r.get("archive_bytes") or 0 for r in records if r.get("body_skipped")
        ),
        "wait_seconds_total": round(sum(r.get("wait_seconds") or 0 for r in records), 1),
        "transfer_seconds_total": round(sum(r.get("transfer_seconds") or 0 for r in records), 1),
        "extract_seconds_total": round(sum(r.get("extract_seconds") or 0 for r in records), 1),
        "retry_attempts_total": sum(max((r.get("attempts") or 1) - 1, 0) for r in records),
        "attempt_errors_by_type": dict(attempt_errors),
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


_LOG_LOCK = threading.Lock()  # mọi dòng crawl_log.jsonl (lỗi tải lẫn kết quả trích xuất) đi qua khóa này
_SPOOL_SUFFIXES = (".meta.json", ".src", ".part")


def _append_log(log_path: Path, record: dict) -> None:
    """Ghi một dòng log; an toàn khi luồng tải và luồng trích xuất cùng ghi."""
    with _LOG_LOCK:
        append_jsonl(log_path, record)


def spool_stem(arxiv_id: str, version: int) -> str:
    """Tên (không đuôi) của một version trong thư mục spool, ví dụ ``2410-13025v1``."""
    return f"{to_folder_name(arxiv_id)}v{version}"


@dataclass
class CrawlerSettings:
    """Các khóa tùy chỉnh của mục ``crawler`` trong ``configs/config.yaml``.

    Attributes:
        timeout_connect: Timeout kết nối (giây).
        timeout_read: Timeout đọc (giây). Kết nối treo quá mức này thì cắt và tải tiếp bằng Range.
        extract_workers: Số luồng trích xuất chạy nền (tải vẫn tuần tự, một kết nối).
        max_spool_gb: Luồng tải chờ khi gói đã tải mà chưa trích xuất vượt mức này.
    """

    timeout_connect: float = DEFAULT_TIMEOUT[0]
    timeout_read: float = DEFAULT_TIMEOUT[1]
    extract_workers: int = DEFAULT_EXTRACT_WORKERS
    max_spool_gb: float = DEFAULT_MAX_SPOOL_GB

    @classmethod
    def from_config(cls, config_path: Path | str | None) -> "CrawlerSettings":
        """Đọc mục ``crawler`` của file cấu hình; thiếu file/khóa/PyYAML thì dùng mặc định."""
        settings = cls()
        if config_path is None:
            return settings
        try:
            import yaml  # noqa: PLC0415  (crawler vẫn chạy được khi thiếu PyYAML)

            with Path(config_path).open(encoding="utf-8") as handle:
                section = (yaml.safe_load(handle) or {}).get("crawler") or {}
        except (ImportError, OSError, ValueError):
            logger.warning("Khong doc duoc %s, dung cau hinh crawler mac dinh", config_path)
            return settings
        for key, cast in (
            ("timeout_connect", float),
            ("timeout_read", float),
            ("extract_workers", int),
            ("max_spool_gb", float),
        ):
            if section.get(key) is not None:
                setattr(settings, key, cast(section[key]))
        return settings

    def make_fetcher(self, contact: str, delay: float = MIN_DELAY_S) -> PoliteFetcher:
        """Tạo bộ tải theo cấu hình (``delay`` < 3 giây vẫn bị từ chối)."""
        return PoliteFetcher(
            USER_AGENT.format(contact=contact),
            delay,
            timeout=(self.timeout_connect, self.timeout_read),
        )


@dataclass
class Spooled:
    """Một version đã qua bước tải, đang chờ trích xuất.

    Attributes:
        row: Dòng discovery của bài.
        version: Số version.
        folder: Thư mục của bài.
        log_path: File log checkpoint.
        meta: Mô tả lần tải (``url``, ``http_status``, ``attempts``, ``seconds``...), cùng nội
            dung với file ``.meta.json`` trong spool.
        src: Gói đã tải; None khi không có body (404/410, PDF bỏ body, quá lớn).
        preset: Kết quả đã biết mà không cần đọc gói (đi kèm ``src=None``).
        prev_soft_errors: Số lần version này đã gặp ``RETRYABLE_REASONS`` ở các lần trước.
    """

    row: dict
    version: int
    folder: Path
    log_path: Path
    meta: dict
    src: Path | None = None
    preset: Extracted | None = None
    prev_soft_errors: int = 0

    @property
    def spool_bytes(self) -> int:
        """Dung lượng gói đang nằm chờ trong spool."""
        return int(self.meta.get("size") or 0) if self.src is not None else 0


def extract_version(job: Spooled) -> dict:
    """Trích xuất và ghi MỘT version đã tải; luôn ghi một dòng log rồi dọn file spool.

    ``extract_from_path`` và ``write_version`` nằm TRONG try: một version lỗi (đĩa đầy, tên file
    lạ...) chỉ ghi log ``error`` kèm tên lỗi rồi đi tiếp. Gặp ``RETRYABLE_REASONS`` mà chưa hết
    ``MAX_SOFT_RETRIES`` lượt thì ``status`` là ``error`` và KHÔNG tạo thư mục ``v{n}``.

    Returns:
        Bản ghi log vừa ghi (đã có ``status``).
    """
    arxiv_id, version, meta = job.row["arxiv_id"], job.version, job.meta
    started = time.monotonic()
    record: dict = {"arxiv_id": arxiv_id, "version": version, "url": meta.get("url")}
    status: str = STATUS_ERROR
    reason: str | None = None
    soft_errors = job.prev_soft_errors
    extracted = job.preset
    tmp: Path | None = None
    wrote = False
    try:
        tmp = staging_dir(job.folder, version)
        if extracted is None:
            extracted = extract_from_path(job.src, tmp)
        retryable = extracted.reason in RETRYABLE_REASONS
        if retryable:
            soft_errors += 1
        if retryable and soft_errors < MAX_SOFT_RETRIES:
            # html_response / corrupt_archive: ghi error, KHÔNG tạo v{n} nên resume sẽ thử lại.
            # Lần thứ MAX_SOFT_RETRIES mà vẫn gặp thì rơi xuống nhánh dưới và chốt empty.
            status, reason = STATUS_ERROR, extracted.reason
        else:
            write_version(job.folder, version, tmp)
            wrote = True
            status, reason = extracted.status, extracted.reason
    except Exception as exc:  # noqa: BLE001  (cố ý bắt rộng: một version hỏng không giết pipeline)
        logger.exception("Loi khi xu ly %sv%d", arxiv_id, version)
        status, reason = STATUS_ERROR, f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if tmp is not None and not wrote:
            _rmtree_with_retry(tmp, ignore_errors=True)
    if extracted is None:  # lỗi xảy ra ngay lúc giải nén: không có số liệu phân tích
        extracted = Extracted(STATUS_ERROR)
    kept = extracted.kept if wrote else {}
    extract_seconds = time.monotonic() - started
    download_seconds = float(meta.get("seconds") or 0.0)
    record.update(
        status=status,
        reason=reason,
        soft_errors=soft_errors,
        http_status=meta.get("http_status"),
        content_type=meta.get("content_type"),
        attempts=meta.get("attempts"),
        archive_bytes=int(meta.get("size") or 0) or extracted.size_before,
        size_before=extracted.size_before,
        size_after=sum(kept.values()),
        n_files_before=extracted.n_files_before,
        n_files_after=len(kept),
        files=sorted(kept)[:50],
        skipped=extracted.skipped[:20],
        seconds=round(download_seconds, 2),  # chỉ phần tải (gồm chờ delay và thử lại)
        wait_seconds=meta.get("wait_seconds"),  # tổng thời gian ngủ: delay + backoff
        transfer_seconds=meta.get("transfer_seconds"),  # tổng thời gian có kết nối mở
        attempt_log=meta.get("attempt_log") or [],
        extract_seconds=round(extract_seconds, 3),  # giải nén + ghi đĩa
        # Toàn bộ version = tải + trích xuất; KHÔNG tính thời gian nằm chờ trong hàng đợi.
        wall_seconds=round(download_seconds + extract_seconds, 2),
        rss_mb=_current_rss_mb(),
    )
    if meta.get("body_skipped"):
        record["body_skipped"] = True
    _append_log(job.log_path, record)
    if job.src is not None:  # log đã ghi xong mới xóa gói: bị ngắt ở giữa thì lần sau trích xuất lại
        job.src.unlink(missing_ok=True)
        meta_path(job.src).unlink(missing_ok=True)
    return record


def _extract_job(job: Spooled) -> dict:
    """Bọc ``extract_version`` cho luồng nền: mọi lỗi còn sót thành bản ghi ``error``."""
    try:
        return extract_version(job)
    except Exception as exc:  # noqa: BLE001  (lưới an toàn: ví dụ không ghi được log)
        logger.exception("Loi bat ngo khi trich xuat %sv%d", job.row["arxiv_id"], job.version)
        return {
            "arxiv_id": job.row["arxiv_id"],
            "version": job.version,
            "status": STATUS_ERROR,
            "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            "soft_errors": job.prev_soft_errors,
        }


class ExtractPipeline:
    """Hàng đợi có giới hạn nối luồng tải (tuần tự) với các luồng trích xuất chạy nền.

    Luồng tải gọi ``submit`` rồi đi tải version kế tiếp ngay; việc giải nén/ghi đĩa diễn ra song
    song. Kết quả được trả về ``on_record`` THEO ĐÚNG THỨ TỰ nộp, nên bộ đếm lỗi liên tiếp giữ
    nguyên nghĩa như khi chạy tuần tự. Dòng ``crawl_log.jsonl`` do chính luồng trích xuất ghi khi
    xong: với 1 luồng (mặc định) thứ tự dòng trùng thứ tự nộp; với nhiều luồng thì có thể đảo
    giữa các version, điều này vô hại vì ``load_log_state`` đọc theo (bài, version).
    """

    def __init__(
        self,
        on_record: Callable[[dict], None],
        workers: int = DEFAULT_EXTRACT_WORKERS,
        max_spool_bytes: float = DEFAULT_MAX_SPOOL_GB * 1024**3,
    ) -> None:
        """Khởi tạo.

        Args:
            on_record: Hàm nhận bản ghi log của từng version đã trích xuất xong (gọi ở luồng tải).
            workers: Số luồng trích xuất (tối thiểu 1).
            max_spool_bytes: ``submit`` sẽ chờ khi tổng dung lượng gói chưa trích xuất vượt mức này.
        """
        self._on_record = on_record
        self._workers = max(int(workers), 1)
        self._max_pending = 2 * self._workers
        self._max_spool_bytes = max_spool_bytes
        self._pending: deque[tuple[Spooled, Future]] = deque()
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="extract"
        )

    def submit(self, job: Spooled) -> None:
        """Đưa một version vào hàng đợi; chờ (backpressure) nếu hàng đợi hoặc spool đang đầy."""
        self._pending.append((job, self._executor.submit(_extract_job, job)))
        self.settle()

    def _over_limit(self) -> bool:
        if len(self._pending) > self._max_pending:
            return True
        return sum(job.spool_bytes for job, _ in self._pending) > self._max_spool_bytes

    def settle(self) -> None:
        """Nhận các kết quả đã xong ở đầu hàng; chỉ CHỜ khi đang vượt giới hạn."""
        while self._pending and (self._pending[0][1].done() or self._over_limit()):
            _, future = self._pending.popleft()
            self._on_record(future.result())

    def flush(self) -> None:
        """Chờ và nhận hết mọi kết quả đang treo (gọi trước khi ghi một dòng log trực tiếp)."""
        while self._pending:
            _, future = self._pending.popleft()
            self._on_record(future.result())

    def close(self) -> None:
        """Chờ các việc đã nộp chạy xong (kể cả khi đang dừng vì lỗi/Ctrl+C) rồi đóng luồng."""
        self._executor.shutdown(wait=True)


def _log_direct(log_path: Path, record: dict, pipeline: "ExtractPipeline | None") -> None:
    """Ghi một dòng log từ luồng tải, SAU khi các version trước đó đã trích xuất và ghi log xong."""
    if pipeline is not None:
        pipeline.flush()
    _append_log(log_path, record)


def crawl_version(
    fetcher: PoliteFetcher,
    row: dict,
    version: int,
    folder: Path,
    log_path: Path,
    prev_soft_errors: int = 0,
    *,
    spool_dir: Path | None = None,
    pipeline: "ExtractPipeline | None" = None,
) -> dict | None:
    """Tải MỘT version xuống spool rồi trích xuất (ngay tại chỗ hoặc qua ``pipeline``).

    Args:
        fetcher: Bộ tải lịch sự.
        row: Dòng discovery của bài.
        version: Số version.
        folder: Thư mục của bài.
        log_path: File log checkpoint.
        prev_soft_errors: Số lần version này đã gặp ``RETRYABLE_REASONS`` ở các lần chạy trước
            (lấy từ ``soft_errors`` trong log; ``run`` truyền vào).
        spool_dir: Nơi chứa gói đã tải (mặc định ``<thư mục cha của folder>/.spool``).
        pipeline: Có thì việc trích xuất được nộp vào hàng đợi chạy nền và hàm trả None (bản ghi
            sẽ tới ``on_record`` của pipeline); không có thì trích xuất ngay và trả bản ghi.

    Returns:
        Bản ghi log vừa ghi, hoặc None nếu version đã được nộp cho ``pipeline``.

    Raises:
        CrawlBlockedError: Khi arXiv trả HTTP 403 (đã ghi log ``blocked`` trước khi raise).
    """
    arxiv_id = row["arxiv_id"]
    url = SRC_URL.format(arxiv_id=arxiv_id, version=version)
    spool_dir = Path(spool_dir) if spool_dir is not None else folder.parent / SPOOL_DIRNAME
    src: Path | None = spool_dir / f"{spool_stem(arxiv_id, version)}.src"
    record: dict = {"arxiv_id": arxiv_id, "version": version, "url": url}
    started = time.monotonic()
    preset: Extracted | None = None
    try:
        fetched = fetcher.download(url, src)
    except CrawlBlockedError as exc:
        record.update(status=STATUS_BLOCKED, reason=str(exc))
        _log_direct(log_path, record, pipeline)
        raise
    except TooLargeError as exc:
        # Quá lớn thì thử lại cũng vô ích: chốt record rỗng với reason too_large.
        meta = {
            "url": url,
            "http_status": 200,
            "content_type": None,
            "attempts": exc.attempts or 1,
            "seconds": time.monotonic() - started,
            "wait_seconds": round(exc.wait_seconds, 2),
            "transfer_seconds": round(exc.transfer_seconds, 2),
            "attempt_log": exc.attempt_log,
        }
        src, preset = None, Extracted(STATUS_EMPTY, "too_large", size_before=exc.downloaded)
    except FetchError as exc:
        elapsed = round(time.monotonic() - started, 2)
        record.update(
            status=STATUS_ERROR,
            reason=str(exc),
            soft_errors=prev_soft_errors,  # giữ bộ đếm, lỗi mạng không làm nó về 0
            attempts=exc.attempts or None,
            seconds=elapsed,
            wait_seconds=round(exc.wait_seconds, 2),
            transfer_seconds=round(exc.transfer_seconds, 2),
            attempt_log=exc.attempt_log,
            wall_seconds=elapsed,
            rss_mb=_current_rss_mb(),
        )
        _log_direct(log_path, record, pipeline)
        return record
    else:
        meta = {"url": url, **fetched.to_meta()}
        if fetched.status_code in (404, 410):
            src, preset = None, Extracted(STATUS_EMPTY, f"http_{fetched.status_code}")
        elif fetched.body_skipped:
            # Version chỉ có PDF: không tải body. size_before lấy từ Content-Length (0 nếu thiếu).
            meta["body_skipped"] = True
            src = None
            preset = Extracted(
                STATUS_EMPTY,
                "pdf_only",
                size_before=fetched.content_length or 0,
                n_files_before=1,
            )
    job = Spooled(row, version, folder, log_path, meta, src, preset, prev_soft_errors)
    if pipeline is None:
        return extract_version(job)
    pipeline.submit(job)
    return None


def _recover_spool(
    spool_dir: Path,
    papers: list[dict],
    state: dict[tuple[str, int], dict],
    out_dir: Path,
    log_path: Path,
) -> list[Spooled]:
    """Quét spool lúc khởi động để KHÔNG tải lại những gì đã có trên đĩa.

    - Có ``.src`` kèm meta mà version chưa xong: trả về để trích xuất thẳng.
    - Có ``.part``: để nguyên, ``PoliteFetcher.download`` sẽ tải tiếp bằng ``Range``.
    - File của version đã xong (mồ côi), ``.src`` thiếu meta, meta lẻ: xóa.
    File không thuộc các bài đang chạy thì không đụng tới.

    Returns:
        Các version sẵn sàng trích xuất, sắp theo tên.
    """
    if not spool_dir.is_dir():
        return []
    wanted = {
        spool_stem(row["arxiv_id"], v): (row, v)
        for row in papers
        for v in range(1, _n_versions(row) + 1)
    }
    jobs: list[Spooled] = []
    for path in sorted(spool_dir.iterdir()):
        suffix = next((s for s in _SPOOL_SUFFIXES if path.name.endswith(s)), None)
        if suffix is None or not path.is_file():
            continue
        stem = path.name[: -len(suffix)]
        if stem not in wanted:
            continue
        row, version = wanted[stem]
        folder = out_dir / to_folder_name(row["arxiv_id"])
        previous = state.get((row["arxiv_id"], version), {})
        finished = (
            previous.get("status") in TERMINAL_STATUSES and (folder / f"v{version}").is_dir()
        )
        src = spool_dir / f"{stem}.src"
        if finished:
            path.unlink(missing_ok=True)
        elif suffix == ".src":
            try:
                meta = json.loads(meta_path(src).read_text(encoding="utf-8"))
                meta["size"] = src.stat().st_size
            except (OSError, ValueError, TypeError, AttributeError):
                logger.warning("Spool %s thieu/hong meta: xoa de tai lai", path.name)
                path.unlink(missing_ok=True)
                continue
            jobs.append(
                Spooled(
                    row,
                    version,
                    folder,
                    log_path,
                    meta,
                    src,
                    prev_soft_errors=int(previous.get("soft_errors") or 0),
                )
            )
        elif suffix == ".meta.json" and not src.exists():
            path.unlink(missing_ok=True)  # meta lẻ (bị ngắt ngay trước khi đổi tên .part -> .src)
    return jobs


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
    spool_dir: Path | None = None,
    extract_workers: int = DEFAULT_EXTRACT_WORKERS,
    max_spool_gb: float = DEFAULT_MAX_SPOOL_GB,
    stop_event: threading.Event | None = None,
) -> dict:
    """Crawl lần lượt từng bài, từng version; có Checkpoint/Resume.

    Tải vẫn TUẦN TỰ (một kết nối, hai lần gửi cách nhau >= 3 giây). Mỗi gói tải xong nằm trong
    ``spool_dir`` và được trích xuất ở luồng nền trong lúc luồng chính chờ/tải version kế tiếp.

    Args:
        papers: Dòng discovery của các bài cần crawl (đã sắp xếp).
        out_dir: Thư mục đầu ra.
        fetcher: Bộ tải lịch sự (MỘT kết nối, có delay).
        write_metadata: Ghi ``metadata.json`` cho mỗi bài.
        max_consecutive_errors: Dừng sau chừng này version lỗi LIÊN TIẾP.
        resume: True bỏ qua version đã xong theo log và dùng lại gói còn trong spool; False tải
            lại tất cả (ghi đè).
        api_fallback: True thì bài thiếu ``n_versions`` được tra bằng arXiv API trước khi crawl.
        max_disk_gb: Dừng (``DiskLimitError``) khi thư mục output CỘNG spool vượt mức này (GB);
            None là không giới hạn. Đo mỗi ``DISK_CHECK_EVERY`` version, nên có thể vượt nhẹ.
        spool_dir: Nơi chứa gói đã tải chờ trích xuất (mặc định ``out_dir/.spool``).
        extract_workers: Số luồng trích xuất chạy nền.
        max_spool_gb: Luồng tải chờ khi gói chưa trích xuất trong spool vượt mức này (GB).
        stop_event: Được đặt thì ngừng tải version mới, trích xuất nốt hàng đợi rồi trả kết quả
            (dùng khi ``run`` chạy trong luồng phụ và người dùng bấm Ctrl+C ở luồng chính).

    Returns:
        Dict tổng hợp (``summarize``), có thêm khóa ``exit_code``, ``n_versions_from_api``,
        ``metadata_errors``, ``revised_dates_incomplete`` và ``stopped``.

    Raises:
        ValueError: Khi còn bài thiếu/sai ``n_versions`` (kiểm tra TRƯỚC khi tải bất kỳ gì).
        CrawlBlockedError: Khi arXiv trả HTTP 403.
        CrawlAbortedError: Khi quá nhiều lỗi liên tiếp (hoặc ``DiskLimitError`` khi đầy đĩa).
    """
    if max_disk_gb is not None and max_disk_gb <= 0:
        raise ValueError(f"max_disk_gb phai > 0, nhan {max_disk_gb}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Kiểm tra (và nếu được, tự điền từ API) n_versions của TẤT CẢ bài trước khi crawl, để
    # không có bài nào âm thầm chỉ tải v1.
    from_api = resolve_n_versions(papers, fetcher, api_fallback=api_fallback)
    log_path = out_dir / LOG_NAME
    spool_dir = Path(spool_dir) if spool_dir is not None else out_dir / SPOOL_DIRNAME
    state = load_log_state(log_path) if resume else {}
    total = sum(_n_versions(row) for row in papers)

    def is_done(row: dict, version: int) -> bool:
        previous = state.get((row["arxiv_id"], version), {})
        folder = out_dir / to_folder_name(row["arxiv_id"])
        return previous.get("status") in TERMINAL_STATUSES and (folder / f"v{version}").is_dir()

    done_before = sum(
        1 for row in papers for v in range(1, _n_versions(row) + 1) if is_done(row, v)
    )
    logger.info(
        "Can crawl %d bai / %d version; da xong tu truoc: %d", len(papers), total, done_before
    )
    started = time.monotonic()
    paper_index = {row["arxiv_id"]: index for index, row in enumerate(papers, start=1)}
    counters = {"handled": 0, "consecutive_errors": 0}

    def disk_gb() -> float:
        used = _dir_bytes(out_dir)
        if spool_dir.is_dir() and out_dir not in spool_dir.parents:
            used += _dir_bytes(spool_dir)
        return used / 1024**3

    def handle(record: dict) -> None:
        """Nhận kết quả cuối của một version (từ luồng tải hoặc từ hàng đợi trích xuất)."""
        state[(record["arxiv_id"], int(record["version"]))] = record
        counters["handled"] += 1
        counters["consecutive_errors"] = (
            counters["consecutive_errors"] + 1 if record["status"] == STATUS_ERROR else 0
        )
        _log_progress(
            record,
            paper_index.get(record["arxiv_id"], 0),
            len(papers),
            counters["handled"],
            total - done_before,
            started,
        )
        if counters["consecutive_errors"] >= max_consecutive_errors:
            raise CrawlAbortedError(
                f"{counters['consecutive_errors']} loi lien tiep, dung de tranh dan dap"
            )
        if max_disk_gb is not None and counters["handled"] % DISK_CHECK_EVERY == 0:
            used = disk_gb()
            if used > max_disk_gb:
                raise DiskLimitError(
                    f"Dia output {used:.2f} GB vuot --max_disk_gb {max_disk_gb} GB "
                    f"(da xu ly {counters['handled']} version trong lan chay nay)"
                )

    def crawl_one(row: dict, version: int) -> None:
        """Tải một version (nếu chưa xong) và chuyển kết quả cho ``handle``/hàng đợi."""
        if is_done(row, version) or (row["arxiv_id"], version) in in_flight:
            return  # resume: version này đã xong, hoặc gói của nó vừa được lấy lại từ spool
        folder = out_dir / to_folder_name(row["arxiv_id"])
        previous = state.get((row["arxiv_id"], version), {})
        try:
            folder.mkdir(parents=True, exist_ok=True)
            record = crawl_version(
                fetcher,
                row,
                version,
                folder,
                log_path,
                # truyền bộ đếm "lỗi mềm" từ log để giới hạn số lần thử lại
                prev_soft_errors=int(previous.get("soft_errors") or 0),
                spool_dir=spool_dir,
                pipeline=pipeline,
            )
        except (CrawlBlockedError, CrawlAbortedError):
            raise  # phải dừng cả chương trình
        except Exception as exc:  # noqa: BLE001  lưới an toàn cuối cùng
            logger.exception("Loi bat ngo o %sv%d", row["arxiv_id"], version)
            record = {
                "arxiv_id": row["arxiv_id"],
                "version": version,
                "status": STATUS_ERROR,
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                "soft_errors": int(previous.get("soft_errors") or 0),
            }
            try:
                _log_direct(log_path, record, pipeline)
            except OSError:
                logger.error("Khong ghi duoc log cho %sv%d", row["arxiv_id"], version)
        if record is not None:
            handle(record)
        pipeline.settle()

    metadata_errors: list[dict] = []
    revised_incomplete: list[dict] = []
    in_flight: set[tuple[str, int]] = set()
    stopped = False
    pipeline = ExtractPipeline(handle, extract_workers, max_spool_gb * 1024**3)
    try:
        if resume:
            # Gói đã tải xong từ lần chạy trước nhưng chưa kịp trích xuất: làm tiếp, KHÔNG tải lại.
            for job in _recover_spool(spool_dir, papers, state, out_dir, log_path):
                logger.info("Lay lai tu spool: %sv%d", job.row["arxiv_id"], job.version)
                in_flight.add((job.row["arxiv_id"], job.version))
                pipeline.submit(job)
        elif spool_dir.is_dir():
            for path in spool_dir.iterdir():
                if path.is_file() and path.name.endswith(_SPOOL_SUFFIXES):
                    path.unlink(missing_ok=True)
        for row in papers:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if write_metadata:
                # ghi TRƯỚC khi tải: ngắt giữa chừng vẫn có metadata
                issues = _write_metadata_once(out_dir / to_folder_name(row["arxiv_id"]), row)
                if "error" in issues:
                    metadata_errors.append({"arxiv_id": row["arxiv_id"], "error": issues["error"]})
                if "revised_dates_missing" in issues:
                    revised_incomplete.append(
                        {
                            "arxiv_id": row["arxiv_id"],
                            "missing_versions": issues["revised_dates_missing"],
                        }
                    )
            for version in range(1, _n_versions(row) + 1):
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                crawl_one(row, version)
            if stopped:
                break
        if stopped:
            logger.warning("Nhan lenh dung: ngung tai, trich xuat not hang doi")
        pipeline.flush()
    finally:
        pipeline.close()  # Ctrl+C/lỗi: vẫn trích xuất nốt những gói đã tải rồi mới thoát
    try:
        spool_dir.rmdir()  # chỉ xóa được khi rỗng; còn file .part thì để lại cho lần sau
    except OSError:
        pass
    summary = summarize(state, papers, out_dir, time.monotonic() - started)
    summary["exit_code"] = 1 if summary["by_status"].get(STATUS_ERROR) else 0
    summary["n_versions_from_api"] = from_api
    summary["stopped"] = stopped
    # Đưa vào summary (và file riêng) thay vì chỉ log ra console
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
    parser.add_argument(
        "--config",
        default=Path("configs/config.yaml"),
        type=Path,
        help="File cau hinh (muc crawler: timeout_read, extract_workers, max_spool_gb)",
    )
    parser.add_argument(
        "--spool",
        default=None,
        type=Path,
        help="Thu muc chua goi da tai cho trich xuat (mac dinh <out>/.spool)",
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
    settings = CrawlerSettings.from_config(args.config)
    fetcher = settings.make_fetcher(args.contact, args.delay)
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
            spool_dir=args.spool,
            extract_workers=settings.extract_workers,
            max_spool_gb=settings.max_spool_gb,
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