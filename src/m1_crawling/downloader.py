"""Tải lịch sự từ ``export.arxiv.org`` (tách khỏi ``crawler.py`` trong đợt tối ưu M1).

Nguyên tắc không đổi so với trước:
    - CHỈ ``export.arxiv.org``; redirect sang host khác bị từ chối.
    - Tối đa MỘT request đang chạy tại mọi thời điểm (``threading.Lock`` bao cả lúc gửi lẫn lúc
      đọc body). Hai lần GỬI request cách nhau tối thiểu 3 giây: thời gian truyền được tính vào
      3 giây đó, và request sau chỉ được gửi khi request trước đã đóng xong.
    - HTTP 403 thì dừng ngay (``CrawlBlockedError``); 429/5xx tôn trọng ``Retry-After``.

Điểm mới:
    - ``PoliteFetcher.download(url, dest)`` ghi từng khối 64 KB thẳng xuống file ``.part`` rồi đổi
      tên thành ``dest``, không giữ cả gói trong RAM. Đứt giữa chừng thì tải tiếp bằng ``Range``
      tính theo kích thước file ``.part``, nên tiếp tục được cả sau khi tiến trình bị tắt.
    - Version chỉ có PDF (``Content-Type: application/pdf``) thì đóng kết nối ngay sau header,
      không tải body (chỉ ở ``download``; ``get`` vẫn trả đủ body).
    - Mỗi lần thử được ghi lại (``attempt_log``), kèm tổng thời gian ngủ và thời gian truyền.

``get(url)`` (trả bytes) được giữ cho các lời gọi nhỏ như arXiv API.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from src.common.utils_io import write_json
from src.m1_crawling.limits import MAX_DOWNLOAD_BYTES, limit

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_HOST",
    "DEFAULT_TIMEOUT",
    "MAX_DOWNLOAD_BYTES",
    "MAX_REDIRECTS",
    "MIN_DELAY_S",
    "RETRY_BACKOFF_S",
    "CrawlBlockedError",
    "FetchError",
    "FetchResult",
    "ForbiddenRedirectError",
    "PoliteFetcher",
    "TooLargeError",
    "meta_path",
    "part_path",
]

MIN_DELAY_S = 3.0  # quy định arXiv: tối thiểu 3 giây giữa hai request
RETRY_BACKOFF_S = (4.0, 5.0)  # chờ trước mỗi lần thử lại (lỗi mạng, HTTP 5xx, 429)
# (timeout kết nối, timeout đọc). Đọc treo quá 30 giây thì cắt và tải tiếp bằng Range, thay vì
# ngồi chờ 120 giây như trước (số đo cho thấy phần lớn thời gian thử lại là thời gian treo).
DEFAULT_TIMEOUT = (15.0, 30.0)
CHUNK_BYTES = 1 << 16  # 64 KB mỗi khối ghi xuống đĩa
MAX_ATTEMPT_LOG = 20  # số lần thử tối đa được ghi chi tiết cho một URL
PDF_CONTENT_TYPE = "application/pdf"

# Redirect: chỉ đi theo khi vẫn ở export.arxiv.org (arxiv.org sẽ bị khóa IP).
ALLOWED_HOST = "export.arxiv.org"
MAX_REDIRECTS = 3
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


# ---------------------------------------------------------------------------
# Lỗi
# ---------------------------------------------------------------------------


class CrawlBlockedError(RuntimeError):
    """arXiv trả HTTP 403: có nguy cơ bị khóa IP, phải dừng toàn bộ chương trình."""


class FetchError(RuntimeError):
    """Tải thất bại sau khi đã thử lại (lỗi tạm thời, version sẽ được thử lại sau).

    Attributes:
        attempts: Số request đã gửi.
        attempt_log: Chi tiết từng lần thử (xem ``FetchResult.attempt_log``).
        wait_seconds: Tổng thời gian ngủ (delay + backoff).
        transfer_seconds: Tổng thời gian có kết nối mở.
    """

    attempts: int = 0
    wait_seconds: float = 0.0
    transfer_seconds: float = 0.0

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.attempt_log: list[dict] = []


class TooLargeError(FetchError):
    """Nguồn vượt ``MAX_DOWNLOAD_BYTES``: thử lại cũng vô ích -> record rỗng ``too_large``.

    Attributes:
        downloaded: Số byte đã nhận được trước khi dừng.
    """

    def __init__(self, message: str, downloaded: int = 0) -> None:
        super().__init__(message)
        self.downloaded = downloaded


class ForbiddenRedirectError(FetchError):
    """Server chuyển hướng sang host khác ``export.arxiv.org``: không đi theo."""


# ---------------------------------------------------------------------------
# Kết quả và nơi chứa body
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Kết quả một lần tải (đã gồm các lần thử lại).

    Attributes:
        status_code: Mã HTTP cuối cùng.
        body: Nội dung khi gọi ``get`` (rỗng khi 404/410 và khi gọi ``download``).
        content_type: Header Content-Type nếu có.
        attempts: Số request đã gửi cho URL này.
        seconds: Tổng thời gian tính cả thời gian chờ delay và thử lại.
        path: File đã tải xong (chỉ có khi gọi ``download`` và có body).
        size: Số byte của body đã nhận.
        content_length: Kích thước server công bố (Content-Length/Content-Range), nếu biết.
        body_skipped: True khi cố ý KHÔNG tải body (version chỉ có PDF).
        wait_seconds: Tổng thời gian ngủ (delay + backoff).
        transfer_seconds: Tổng thời gian có kết nối mở.
        attempt_log: Tối đa ``MAX_ATTEMPT_LOG`` phần tử ``{error, bytes, seconds}``, mỗi lần thử
            một phần tử (``error`` là None nếu lần đó thành công).
    """

    status_code: int
    body: bytes
    content_type: str | None
    attempts: int
    seconds: float
    path: Path | None = None
    size: int = 0
    content_length: int | None = None
    body_skipped: bool = False
    wait_seconds: float = 0.0
    transfer_seconds: float = 0.0
    attempt_log: list[dict] = field(default_factory=list)

    def to_meta(self) -> dict:
        """Phần mô tả (không gồm body) để ghi ra ``.meta.json`` cạnh file đã tải."""
        return {
            "http_status": self.status_code,
            "content_type": self.content_type,
            "attempts": self.attempts,
            "seconds": round(self.seconds, 2),
            "content_length": self.content_length,
            "size": self.size,
            "wait_seconds": round(self.wait_seconds, 2),
            "transfer_seconds": round(self.transfer_seconds, 2),
            "attempt_log": self.attempt_log,
        }


@dataclass
class _Telemetry:
    """Số đo cộng dồn qua các lần thử của MỘT URL."""

    attempts: int = 0
    wait_seconds: float = 0.0
    transfer_seconds: float = 0.0
    attempt_log: list[dict] = field(default_factory=list)

    def record(self, error: str | None, n_bytes: int, seconds: float) -> None:
        self.transfer_seconds += seconds
        if len(self.attempt_log) < MAX_ATTEMPT_LOG:
            self.attempt_log.append(
                {"error": error, "bytes": n_bytes, "seconds": round(seconds, 2)}
            )

    def attach(self, exc: FetchError) -> FetchError:
        exc.attempts = self.attempts
        exc.wait_seconds = self.wait_seconds
        exc.transfer_seconds = self.transfer_seconds
        exc.attempt_log = self.attempt_log
        return exc


class _MemorySink:
    """Giữ body trong RAM (cho ``get``: chỉ dùng với response nhỏ như arXiv API)."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def size(self) -> int:
        return len(self.buffer)

    def reset(self) -> None:
        self.buffer.clear()

    def append(self, chunks: "Callable[[], object]", max_bytes: int) -> None:
        for chunk in chunks():
            self.buffer.extend(chunk)
            if len(self.buffer) > max_bytes:
                raise TooLargeError("file tai ve qua lon", downloaded=len(self.buffer))


class _FileSink:
    """Ghi body thẳng xuống file ``.part`` (cho ``download``)."""

    def __init__(self, part: Path) -> None:
        self.part = part

    def size(self) -> int:
        try:
            return self.part.stat().st_size
        except OSError:
            return 0

    def reset(self) -> None:
        self.part.unlink(missing_ok=True)

    def append(self, chunks: "Callable[[], object]", max_bytes: int) -> None:
        total = self.size()
        self.part.parent.mkdir(parents=True, exist_ok=True)
        with self.part.open("ab") as handle:
            for chunk in chunks():
                handle.write(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise TooLargeError("file tai ve qua lon", downloaded=total)


@dataclass
class _Attempt:
    """Kết quả MỘT lần gửi request (chưa quyết định có thử lại hay không)."""

    done: FetchResult | None = None  # có giá trị: xong, trả về luôn
    error: str = "khong ro"
    retry_after: float | None = None
    quiet: bool = False  # lỗi đứt kết nối thường gặp: không cần log ồn ào


def part_path(dest: Path) -> Path:
    """File tải dở tương ứng với ``dest`` (``x.src`` -> ``x.part``)."""
    return dest.with_suffix(".part")


def meta_path(dest: Path) -> Path:
    """File mô tả tương ứng với ``dest`` (``x.src`` -> ``x.meta.json``)."""
    return dest.with_suffix(".meta.json")


# ---------------------------------------------------------------------------
# Bộ tải
# ---------------------------------------------------------------------------


class PoliteFetcher:
    """Tải URL tuần tự với delay tối thiểu giữa các request, đúng quy định arXiv.

    An toàn khi nhiều luồng dùng chung: một ``threading.Lock`` bao trọn mỗi lần thử (chờ đến
    lượt, gửi request, đọc body, đóng kết nối), nên không bao giờ có hai request cùng mở và
    khoảng cách giữa hai lần GỬI luôn >= ``delay``. Request tải lâu hơn ``delay`` thì request sau
    được gửi ngay khi nó đóng xong. Quy tắc này áp dụng cho cả lần thử lại, tải tiếp bằng
    ``Range`` và bước redirect.
    """

    def __init__(
        self,
        user_agent: str,
        delay: float = MIN_DELAY_S,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        backoff: tuple[float, ...] = RETRY_BACKOFF_S,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    ) -> None:
        """Khởi tạo.

        Args:
            user_agent: Chuỗi User-Agent mô tả dự án và email liên hệ.
            delay: Giây tối thiểu giữa hai lần gửi request.
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
        self._last_sent: float | None = None  # thời điểm GỬI request gần nhất
        self._lock = threading.Lock()

    def note_external_request(self) -> None:
        """Báo rằng vừa có request tới arXiv do chương trình KHÁC gửi (ví dụ ``discovery.py``).

        Request đầu tiên của bộ tải này sẽ chờ đủ ``delay`` kể từ thời điểm gọi hàm, để khoảng
        cách 3 giây đúng cho cả hệ thống chứ không chỉ trong nội bộ bộ tải.
        """
        with self._lock:
            self._last_sent = self._clock()

    def _wait_turn(self, telemetry: _Telemetry | None = None) -> None:
        """Ngủ cho đủ ``delay`` giây kể từ lúc request trước được gửi đi."""
        if self._last_sent is None:
            return
        remaining = self.delay - (self._clock() - self._last_sent)
        if remaining > 0:
            self._sleep(remaining)
            if telemetry is not None:
                telemetry.wait_seconds += remaining

    def _request(
        self, url: str, headers: dict | None = None, telemetry: _Telemetry | None = None
    ) -> requests.Response:
        """Gửi GET, KHÔNG để ``requests`` tự đi theo redirect.

        Nếu ``export.arxiv.org`` chuyển hướng sang ``arxiv.org`` thì tự động đi theo sẽ vi phạm
        quy định (bị khóa IP). Ở đây chỉ đi theo redirect khi đích vẫn là ``export.arxiv.org``
        (https); mỗi bước nhảy cũng chờ đủ ``delay`` như một request mới.

        Raises:
            ForbiddenRedirectError: Khi đích redirect nằm ngoài ``export.arxiv.org``.
            FetchError: Khi vượt ``MAX_REDIRECTS`` bước nhảy.
        """
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            if hop:
                self._wait_turn(telemetry)
            self._last_sent = self._clock()
            response = self.session.get(
                current, headers=headers, stream=True, timeout=self._timeout, allow_redirects=False
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

    def _attempt(
        self,
        url: str,
        sink: "_MemorySink | _FileSink",
        telemetry: _Telemetry,
        skip_pdf_body: bool,
    ) -> _Attempt:
        """Gửi MỘT request (gọi khi đang giữ khóa) và đọc body vào ``sink``.

        Returns:
            ``_Attempt`` có ``done`` nếu xong; ngược lại mô tả lỗi tạm thời để vòng ngoài thử lại.

        Raises:
            CrawlBlockedError: HTTP 403.
            FetchError: Lỗi không nên thử lại (mã HTTP lạ, redirect bị cấm, quá lớn).
        """
        have = sink.size()
        headers = {"Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        response = self._request(url, headers, telemetry)
        try:
            status = response.status_code
            content_type = response.headers.get("Content-Type")
            if status == 403:
                raise CrawlBlockedError(f"HTTP 403 khi tai {url}")
            if status in (404, 410):
                sink.reset()
                return _Attempt(done=FetchResult(status, b"", None, 0, 0.0))
            if status == 416 and have:
                # File .part dài hơn/bằng file thật (hoặc server đổi nội dung): bỏ, tải lại từ đầu.
                sink.reset()
                return _Attempt(error="HTTP 416 (Range khong hop le, tai lai tu dau)")
            if status not in (200, 206):
                if status == 429 or status >= 500:
                    header = str(response.headers.get("Retry-After", ""))
                    return _Attempt(
                        error=f"HTTP {status}",
                        retry_after=float(header) if header.isdigit() else None,
                    )
                raise FetchError(f"HTTP {status} khi tai {url}")

            # Body đã được giải nén thì Content-Length (số byte nén) không so được.
            encoding = str(response.headers.get("Content-Encoding", "") or "identity").lower()
            is_raw = encoding == "identity"
            total: int | None = None
            if status == 200:
                length = str(response.headers.get("Content-Length", ""))
                total = int(length) if length.isdigit() and is_raw else None
                if skip_pdf_body and str(content_type or "").lower().startswith(PDF_CONTENT_TYPE):
                    # Version chỉ có PDF: không có gì để giữ lại, đóng luôn, khỏi tải body.
                    sink.reset()
                    return _Attempt(
                        done=FetchResult(
                            200,
                            b"",
                            content_type,
                            0,
                            0.0,
                            content_length=int(length) if length.isdigit() else None,
                            body_skipped=True,
                        )
                    )
                sink.reset()  # server gửi lại từ đầu
            else:
                crange = str(response.headers.get("Content-Range", ""))
                if not crange.startswith(f"bytes {have}-"):
                    sink.reset()
                    raise requests.exceptions.ChunkedEncodingError("Content-Range lech")
                tail = crange.rsplit("/", 1)[-1]
                total = int(tail) if tail.isdigit() and is_raw else None
            sink.append(
                lambda: response.iter_content(chunk_size=CHUNK_BYTES),
                limit("MAX_DOWNLOAD_BYTES"),
            )
            size = sink.size()
            if total is not None and size < total:
                raise requests.exceptions.ChunkedEncodingError(f"tai thieu: {size}/{total} byte")
            return _Attempt(
                done=FetchResult(200, b"", content_type, 0, 0.0, size=size, content_length=total)
            )
        finally:
            response.close()

    def _fetch(
        self,
        url: str,
        sink: "_MemorySink | _FileSink",
        max_total_seconds: float | None,
        skip_pdf_body: bool,
    ) -> FetchResult:
        """Vòng thử lại dùng chung cho ``get`` và ``download``.

        Chỉ bỏ cuộc khi: (a) ``len(backoff) + 1`` lần thử LIÊN TIẾP không tải thêm được byte nào,
        (b) quá ``max_total_seconds`` (None = không giới hạn), (c) vượt ``MAX_DOWNLOAD_BYTES``.
        """
        started = self._clock()
        telemetry = _Telemetry()
        best = sink.size()  # số byte lớn nhất từng đạt (tính cả file .part có sẵn)
        stalls = 0  # số lần thử liên tiếp KHÔNG tiến triển
        while True:
            telemetry.attempts += 1
            before = sink.size()
            outcome = _Attempt()
            with self._lock:
                self._wait_turn(telemetry)
                opened = self._clock()
                try:
                    outcome = self._attempt(url, sink, telemetry, skip_pdf_body)
                except requests.RequestException as exc:
                    outcome = _Attempt(
                        error=f"{type(exc).__name__}: {str(exc)[:120]}",
                        quiet=isinstance(exc, requests.exceptions.ChunkedEncodingError),
                    )
                except TooLargeError as exc:
                    sink.reset()
                    telemetry.record("too_large", exc.downloaded, self._clock() - opened)
                    raise telemetry.attach(exc) from None
                except FetchError as exc:
                    telemetry.record(str(exc)[:160], 0, self._clock() - opened)
                    raise telemetry.attach(exc) from None
                elapsed = self._clock() - opened
            received = max(sink.size() - before, 0)
            if outcome.done is not None:
                result = outcome.done
                telemetry.record(None, result.size if result.size else received, elapsed)
                result.attempts = telemetry.attempts
                result.seconds = self._clock() - started
                result.wait_seconds = telemetry.wait_seconds
                result.transfer_seconds = telemetry.transfer_seconds
                result.attempt_log = telemetry.attempt_log
                return result
            telemetry.record(outcome.error, received, elapsed)

            # --- Lần thử thất bại: quyết định chờ bao lâu hoặc bỏ cuộc ---
            size = sink.size()
            if size > best:
                best, stalls = size, 0
                wait = 0.0
                if not outcome.quiet:
                    logger.info("Dut o %d byte, tiep tuc tu day (%s)", best, outcome.error)
            else:
                stalls += 1
                if stalls > len(self._backoff):
                    raise telemetry.attach(
                        FetchError(
                            f"{outcome.error}: {stalls} lan lien tiep khong tien trien "
                            f"({best} byte): {url}"
                        )
                    )
                wait = max(self._backoff[stalls - 1], outcome.retry_after or 0.0)
                if not outcome.quiet:
                    logger.warning(
                        "Loi tam thoi (%s), khong tien trien, thu lai sau %.0fs",
                        outcome.error,
                        wait,
                    )
            if max_total_seconds is not None and self._clock() - started + wait > max_total_seconds:
                raise telemetry.attach(
                    FetchError(f"qua {max_total_seconds:.0f}s ma chua xong ({best} byte): {url}")
                )
            if wait:
                self._sleep(wait)
                telemetry.wait_seconds += wait

    def get(self, url: str, max_total_seconds: float | None = 3600) -> FetchResult:
        """Tải một URL vào RAM và trả ``body`` (dùng cho response nhỏ như arXiv API).

        Tự tải tiếp bằng ``Range`` khi đứt giữa chừng. Mã nguồn bài báo nên dùng ``download``.

        Raises:
            CrawlBlockedError: HTTP 403.
            FetchError: Hết lượt thử, mã HTTP không thử lại được, redirect bị cấm.
            TooLargeError: Vượt ``MAX_DOWNLOAD_BYTES``.
        """
        sink = _MemorySink()
        result = self._fetch(url, sink, max_total_seconds, skip_pdf_body=False)
        if result.status_code == 200:
            result.body = bytes(sink.buffer)
        return result

    def download(
        self, url: str, dest: Path, max_total_seconds: float | None = 3600
    ) -> FetchResult:
        """Tải một URL thẳng xuống đĩa, không giữ body trong RAM.

        Ghi vào ``part_path(dest)``; đủ file thì ghi ``meta_path(dest)`` rồi mới đổi tên thành
        ``dest``. Vì vậy: có ``dest`` nghĩa là đã tải xong và có kèm meta. File ``.part`` còn
        lại từ lần chạy trước được tải tiếp bằng ``Range`` thay vì tải lại từ đầu. Gặp
        ``Content-Type: application/pdf`` thì đóng kết nối ngay, trả kết quả có
        ``body_skipped=True`` và KHÔNG tạo file.

        Args:
            url: Địa chỉ cần tải.
            dest: File đích (ví dụ ``<spool>/2410-13025v1.src``).
            max_total_seconds: Trần thời gian cho URL này; None là không giới hạn.

        Returns:
            ``FetchResult`` với ``path=dest`` khi có file; ``path=None`` khi 404/410 hoặc bỏ body.

        Raises:
            CrawlBlockedError: HTTP 403.
            FetchError: Hết lượt thử, mã HTTP không thử lại được, redirect bị cấm.
            TooLargeError: Vượt ``MAX_DOWNLOAD_BYTES`` (file ``.part`` đã bị xóa).
        """
        dest = Path(dest)
        part, meta = part_path(dest), meta_path(dest)
        dest.unlink(missing_ok=True)  # được gọi nghĩa là cần bản mới; không dùng lại file cũ
        result = self._fetch(url, _FileSink(part), max_total_seconds, skip_pdf_body=True)
        if result.status_code != 200 or result.body_skipped:
            part.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            return result
        if not part.exists():  # body rỗng (0 byte): vẫn phải có file để bước sau phân loại
            part.parent.mkdir(parents=True, exist_ok=True)
            part.touch()
        write_json(meta, {"url": url, **result.to_meta()})
        part.replace(dest)
        result.path = dest
        return result
