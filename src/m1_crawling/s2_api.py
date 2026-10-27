"""Lấy danh sách tham khảo của một bài báo từ Semantic Scholar API (M1, task P2-02).

Mỗi bài được lưu thành hai file trong thư mục của bài:

- ``references_all.json``: MỌI reference S2 trả về, kể cả ref không có arXiv ID (đề M1 yêu cầu
  giữ đầy đủ ở máy; chỉ lọc theo arXiv ID khi xuất bảng ``references`` Parquet).
- ``references.json``: chỉ các reference có arXiv ID, key dạng "yymm-id", đúng định dạng của
  mock để M3 dùng được ngay.

Lưu ý: S2 lưu reference theo BÀI, không phân biệt version, nên chỉ cần gọi một lần cho mỗi bài.

Tham khảo: Semantic Scholar Academic Graph API, https://api.semanticscholar.org/api-docs/
"""

import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, Self

import requests
import yaml
from pydantic import BaseModel, Field

from src.common.ids import to_folder_name, validate_arxiv_id
from src.common.utils_io import read_json, write_json
from src.schemas import Reference

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
# Các field của bài ĐƯỢC TRÍCH DẪN (endpoint /references). paperId luôn được trả sẵn.
S2_FIELDS = "title,authors,year,publicationDate,venue,externalIds"
S2_PAGE_SIZE = 1000  # S2 cho tối đa 1000 phần tử mỗi trang
# Lỗi tạm thời: chờ rồi thử lại. Các lỗi khác (400, 401, 403...) thử lại cũng vô ích.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 60
# Khớp hậu tố version, ví dụ "2310.12345v2" -> "2310.12345". ID cũ như "hep-th/9901001"
# không kết thúc bằng vN nên không bị ảnh hưởng.
VERSION_SUFFIX = re.compile(r"v\d+$")


class S2Error(RuntimeError):
    """Gọi S2 thất bại sau khi đã thử lại hết số lần cho phép."""


class S2Settings(BaseModel):
    """Tham số của S2Client; giá trị mặc định khớp configs/config.yaml."""

    base_url: str = S2_BASE_URL
    api_key: str | None = None  # None: lấy từ biến môi trường S2_API_KEY
    delay_seconds: float = Field(default=1.0, ge=0)
    timeout: float = Field(default=30, gt=0)
    max_retries: int = Field(default=5, ge=1)

    @classmethod
    def from_config(cls, config_path: Path | str = DEFAULT_CONFIG_PATH) -> Self:
        """Đọc mục ``s2_api`` của file cấu hình; thiếu khóa nào thì dùng mặc định.

        Args:
            config_path: Đường dẫn configs/config.yaml.

        Returns:
            Cấu hình đã kiểm tra kiểu.
        """
        with Path(config_path).open(encoding="utf-8") as handle:
            section = (yaml.safe_load(handle) or {}).get("s2_api") or {}
        return cls(**{key: section[key] for key in cls.model_fields if key in section})


class S2Reference(BaseModel):
    """Một tài liệu được trích dẫn, lưu đầy đủ kể cả khi không có arXiv ID."""

    s2_id: str | None = None
    arxiv_id: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    year: int | None = None
    venue: str | None = None


class CrawlResult(BaseModel):
    """Kết quả xử lý một bài, để Profiler và main.py thống kê tỉ lệ thành công.

    status:
        done: gọi S2 và đã ghi file.
        skipped: references.json đã có từ lần chạy trước (Resume); không đọc lại số đếm.
        not_found: S2 không có bài này (khác với bài có mặt nhưng không trích dẫn gì).
        failed: lỗi sau khi đã thử lại (chỉ do run_s2_stage tạo ra, kèm log lỗi).
    """

    arxiv_id: str
    status: Literal["done", "skipped", "not_found", "failed"]
    n_refs: int = 0
    n_refs_co_arxiv_id: int = 0


class S2Client:
    """Client gọi S2 có giãn cách giữa các request, retry và backoff.

    Mọi request tới S2 PHẢI đi qua get_json để không ai vô tình gọi quá nhanh.

    An toàn khi dùng chung giữa nhiều luồng: khoảng cách giữa hai lần BẮT ĐẦU request luôn
    >= delay_seconds, bất kể luồng nào gọi. Tuy vậy cách đơn giản nhất vẫn là mỗi nguồn
    (arXiv, S2) một client riêng và mỗi client chỉ một luồng, vì hai nguồn có giới hạn độc lập.
    """

    def __init__(
        self,
        settings: S2Settings | None = None,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Khởi tạo client.

        Args:
            settings: Tham số gọi API; None thì dùng mặc định.
            session: Đối tượng có phương thức get giống requests.Session (test truyền bản giả).
            sleep: Hàm ngủ; test truyền bản giả để không phải chờ thật.
            clock: Đồng hồ đơn điệu; test truyền bản giả để kiểm tra delay chính xác.
        """
        self._settings = settings or S2Settings()
        # Không hard-code khóa trong code: lấy từ cấu hình hoặc biến môi trường.
        self._api_key = self._settings.api_key or os.environ.get("S2_API_KEY")
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        # -inf để request đầu tiên không bao giờ phải chờ.
        self._last_call = -math.inf
        self._lock = threading.Lock()

    def _wait_turn(self) -> None:
        # Giữ khóa trong lúc ngủ: luồng đến sau phải xếp hàng, nên hai request không bao giờ
        # bắt đầu cách nhau ít hơn delay. Chỉ ngủ phần còn thiếu vì thời gian xử lý giữa hai
        # request cũng được tính vào delay.
        with self._lock:
            remaining = self._settings.delay_seconds - (self._clock() - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
            self._last_call = self._clock()

    @staticmethod
    def _backoff(attempt: int) -> float:
        # Chờ lâu dần: 2s, 4s, 8s, ... tối đa MAX_BACKOFF_SECONDS.
        return float(min(2**attempt, MAX_BACKOFF_SECONDS))

    def _retry_wait(self, response: Any, attempt: int) -> float:
        # Ưu tiên Retry-After của server; thiếu hoặc sai dạng thì dùng backoff mũ.
        try:
            return float(response.headers["Retry-After"])
        except (KeyError, ValueError, TypeError):
            return self._backoff(attempt)

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Gọi GET tới S2 và trả JSON đã giải mã.

        Args:
            path: Đường dẫn sau base_url, ví dụ "/paper/arXiv:1706.03762/references".
            params: Tham số trên URL.

        Returns:
            Dict JSON, hoặc None khi S2 không có dữ liệu (HTTP 404).

        Raises:
            S2Error: Khi vẫn gặp lỗi tạm thời (429, 5xx, timeout) sau max_retries lần.
            requests.HTTPError: Với lỗi không thể thử lại (400, 401, 403...).
        """
        headers = {"x-api-key": self._api_key} if self._api_key else {}
        url = f"{self._settings.base_url}{path}"
        for attempt in range(1, self._settings.max_retries + 1):
            self._wait_turn()
            try:
                response = self._session.get(
                    url, params=params, headers=headers, timeout=self._settings.timeout
                )
            except (requests.Timeout, requests.ConnectionError):
                logger.warning("Lỗi mạng khi gọi %s (lần %d)", path, attempt)
                self._sleep(self._backoff(attempt))
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None
            if response.status_code in RETRY_STATUS:
                logger.warning("S2 trả %d cho %s (lần %d)", response.status_code, path, attempt)
                self._sleep(self._retry_wait(response, attempt))
                continue
            response.raise_for_status()  # 400/401/403: thử lại cũng vô ích
        raise S2Error(f"Hết {self._settings.max_retries} lần thử: {path}")


def parse_reference(cited: dict[str, Any] | None) -> S2Reference | None:
    """Đổi một ``citedPaper`` của S2 thành S2Reference.

    Chịu được dữ liệu thiếu: externalIds là None, authors là None, title là None...

    Args:
        cited: Giá trị khóa ``citedPaper`` trong một phần tử ``data`` của response.

    Returns:
        S2Reference, hoặc None nếu bản ghi rỗng.
    """
    if not cited:
        return None
    # externalIds có thể là None chứ không phải {}, nên bắt buộc có "or {}".
    raw_arxiv = (cited.get("externalIds") or {}).get("ArXiv")
    arxiv_id = None
    if raw_arxiv:
        try:
            arxiv_id = validate_arxiv_id(VERSION_SUFFIX.sub("", str(raw_arxiv).strip()))
        except ValueError:
            logger.warning("ArXiv ID lạ từ S2: %r", raw_arxiv)
    authors = [a["name"] for a in (cited.get("authors") or []) if a.get("name")]
    return S2Reference(
        s2_id=cited.get("paperId"),
        arxiv_id=arxiv_id,
        title=cited.get("title"),
        authors=authors,
        publication_date=cited.get("publicationDate"),
        year=cited.get("year"),
        venue=cited.get("venue"),
    )


def _references_path(arxiv_id: str) -> str:
    # arxiv_id đã qua validate_arxiv_id nên chỉ chứa chữ, số, ".", "-", "/": giữ nguyên dấu "/"
    # của ID cũ ("hep-th/9901001"), không cần mã hóa.
    return f"/paper/arXiv:{arxiv_id}/references"


def fetch_references(arxiv_id: str, client: S2Client | None = None) -> list[S2Reference] | None:
    """Lấy mọi reference của một bài, tự đi qua các trang.

    Args:
        arxiv_id: Mã arXiv chuẩn, không có hậu tố version.
        client: Client dùng chung giữa các bài để delay được tính đúng; None thì tạo mới.

    Returns:
        Danh sách reference (có thể rỗng nếu bài không trích dẫn gì), hoặc None nếu S2 không
        có bài này. Hai trường hợp phải phân biệt khi thống kê "tỉ lệ crawl reference".

    Raises:
        ValueError: Khi arxiv_id sai dạng.
        S2Error: Khi S2 lỗi tạm thời quá số lần thử.
    """
    validate_arxiv_id(arxiv_id)
    client = client or S2Client()
    refs: list[S2Reference] = []
    offset = 0
    while True:
        page = client.get_json(
            _references_path(arxiv_id),
            {"fields": S2_FIELDS, "limit": S2_PAGE_SIZE, "offset": offset},
        )
        if page is None:
            return None
        for item in page.get("data") or []:
            ref = parse_reference(item.get("citedPaper"))
            if ref is not None:
                refs.append(ref)
        next_offset = page.get("next")
        # Hết trang khi không còn "next"; kiểm tra thêm next > offset để không lặp vô hạn.
        if next_offset is None or next_offset <= offset:
            return refs
        offset = next_offset


def to_references_json(arxiv_id: str, refs: list[S2Reference]) -> dict[str, dict[str, Any]]:
    """Dựng nội dung references.json: chỉ ref có arXiv ID, key dạng "yymm-id".

    Args:
        arxiv_id: Bài đang trích dẫn.
        refs: Mọi reference đã crawl của bài đó.

    Returns:
        Dict key -> {ref_title, ref_authors, ref_submission_date, ref_s2_id}. Ref trùng
        arXiv ID thì bản sau ghi đè bản trước.
    """
    result: dict[str, dict[str, Any]] = {}
    for ref in refs:
        if ref.arxiv_id is None:
            continue
        # Đi qua model Reference để định dạng luôn khớp schema chung của dự án.
        row = Reference(
            arxiv_id=arxiv_id,
            ref_arxiv_id=ref.arxiv_id,
            ref_title=ref.title,
            ref_authors=ref.authors,
            ref_submission_date=ref.publication_date,
            ref_s2_id=ref.s2_id,
        )
        result[to_folder_name(ref.arxiv_id)] = row.model_dump(
            mode="json", exclude={"arxiv_id", "ref_arxiv_id"}
        )
    return result


def crawl_references(
    arxiv_id: str,
    paper_dir: Path,
    client: S2Client | None = None,
    resume: bool = True,
) -> CrawlResult:
    """Lấy reference của một bài và ghi references_all.json cùng references.json.

    Args:
        arxiv_id: Mã arXiv chuẩn của bài.
        paper_dir: Thư mục của bài (do crawler quyết định), ví dụ data/raw/2310-12345.
        client: Client dùng chung giữa các bài.
        resume: True thì bỏ qua bài đã có references.json (Checkpoint/Resume).

    Returns:
        CrawlResult cho biết bài đã xong, bị bỏ qua hay S2 không có.
        Nếu muốn thống kê toàn corpus (kể cả bài đã xong từ các lần chạy trước), hãy dùng count_saved_references.
    """
    target = paper_dir / "references.json"
    if resume and target.exists():
        return CrawlResult(arxiv_id=arxiv_id, status="skipped")
    refs = fetch_references(arxiv_id, client)
    if refs is None:
        # Không ghi gì: lần chạy sau sẽ thử lại (tốn một request) thay vì coi như đã xong.
        logger.warning("S2 không có bài %s", arxiv_id)
        return CrawlResult(arxiv_id=arxiv_id, status="not_found")
    by_arxiv = to_references_json(arxiv_id, refs)
    write_json(paper_dir / "references_all.json", [ref.model_dump(mode="json") for ref in refs])
    # Ghi references.json SAU CÙNG: file này tồn tại nghĩa là bài đã xong hoàn toàn.
    write_json(target, by_arxiv)
    logger.info("Xong reference %s: %d ref, %d có arXiv ID", arxiv_id, len(refs), len(by_arxiv))
    return CrawlResult(
        arxiv_id=arxiv_id, status="done", n_refs=len(refs), n_refs_co_arxiv_id=len(by_arxiv)
    )


def run_s2_stage(
    arxiv_ids: Iterable[str],
    root: Path,
    client: S2Client | None = None,
    on_result: Callable[[CrawlResult, float], None] | None = None,
    resume: bool = True,
) -> list[CrawlResult]:
    """Chạy bước S2 cho cả một dải bài; một bài lỗi không làm sập cả dải.

    Hàm này không phụ thuộc vào bước tải source của arXiv (S2 chỉ cần arxiv_id), nên có thể
    chạy trong một luồng riêng song song với crawler arXiv mà không vi phạm giới hạn nào.

    Args:
        arxiv_ids: Các mã arXiv chuẩn (list hay generator đều được).
        root: Thư mục gốc chứa các thư mục bài (tên dạng "yymm-id"), ví dụ data/raw.
        client: Client dùng chung cho cả dải; None thì tạo từ cấu hình mặc định.
        on_result: Hàm gọi sau mỗi bài với (kết quả, số giây xử lý), ví dụ để ghi vào Profiler.
            Bên gọi cần lọc bài "skipped" trước khi ghi vào Profiler để thời gian trung bình
            mỗi bài chính xác.
        resume: True thì bỏ qua bài đã có references.json.

    Returns:
        Kết quả của từng bài theo thứ tự đầu vào.
    """
    client = client or S2Client()
    results: list[CrawlResult] = []
    for arxiv_id in arxiv_ids:
        started = time.perf_counter()
        try:
            result = crawl_references(arxiv_id, root / to_folder_name(arxiv_id), client, resume)
        except (S2Error, requests.RequestException, ValueError):
            # Chỉ bắt lỗi của riêng bài này (mạng, S2, ID sai); lỗi đĩa hay lỗi code vẫn nổ.
            logger.exception("Lỗi khi lấy reference của %s", arxiv_id)
            result = CrawlResult(arxiv_id=arxiv_id, status="failed")
        results.append(result)
        if on_result is not None:
            on_result(result, time.perf_counter() - started)
    return results


def count_saved_references(root: Path) -> tuple[int, int, int]:
    """Đếm số liệu reference của toàn corpus bằng cách quét các file đã lưu.

    Dùng cho thống kê trong báo cáo (số ref trung bình mỗi bài, tỉ lệ ref có arXiv ID). Số
    này không phụ thuộc lịch sử chạy hay Resume, khác với CrawlResult chỉ mô tả MỘT lần gọi.

    Args:
        root: Thư mục gốc chứa các thư mục bài, ví dụ data/raw.

    Returns:
        (số bài đã có references.json, tổng số ref, tổng số ref có arXiv ID). Bài thiếu
        references_all.json (dữ liệu cũ) chỉ góp vào số bài và số ref có arXiv ID.
    """
    n_papers = n_refs = n_arxiv = 0
    for by_arxiv_file in sorted(root.glob("*/references.json")):
        n_papers += 1
        n_arxiv += len(read_json(by_arxiv_file))
        all_file = by_arxiv_file.parent / "references_all.json"
        if all_file.exists():
            n_refs += len(read_json(all_file))
    return n_papers, n_refs, n_arxiv