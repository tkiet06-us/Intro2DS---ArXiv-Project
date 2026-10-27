"""Đo thời gian, RAM và đĩa khi chạy pipeline M1 (task P2-02).

Cách đo (để ghi vào báo cáo, đề yêu cầu đo theo cách kiểm chứng được):

- RAM: một thread nền lấy mẫu RSS (RAM thật) của process và các process con mỗi ``interval``
  giây bằng psutil. Peak là mẫu lớn nhất, average là trung bình các mẫu. Lấy mẫu có thể bỏ lỡ
  đỉnh rất ngắn nên peak chỉ là xấp xỉ; khi benchmark hãy đối chiếu thêm với
  ``/usr/bin/time -v`` (dòng "Maximum resident set size").
- Thời gian: ``time.perf_counter`` (đo đều, không bị ảnh hưởng khi đồng hồ hệ thống bị chỉnh).
- Đĩa: tổng kích thước file trong thư mục output (tương đương ``du``), đo thưa hơn RAM vì duyệt
  thư mục tốn kém. Max là mức lớn nhất trong lúc chạy, final là mức ở lần đo cuối.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Self

import psutil
from pydantic import BaseModel

from src.common.utils_io import write_json

logger = logging.getLogger(__name__)

MB = 1024 * 1024


class ProfileSummary(BaseModel):
    """Kết quả đo ở dạng phẳng, sẵn sàng ghi vào summary_stats.json cho Jupyter Book."""

    wall_seconds: float
    n_papers: int
    avg_seconds_per_paper: float | None
    avg_wall_seconds_per_paper: float | None
    peak_ram_mb: float
    avg_ram_mb: float
    n_ram_samples: int
    max_disk_mb: float | None
    final_disk_mb: float | None
    n_refs_total: int
    n_refs_co_arxiv_id_total: int


def dir_size(path: Path) -> int:
    """Tính tổng kích thước (byte) mọi file trong một thư mục, kể cả thư mục con.

    Args:
        path: Thư mục cần đo; không tồn tại thì coi như 0.

    Returns:
        Tổng số byte.
    """
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # file tạm bị xóa giữa lúc đang duyệt
            continue
    return total


class Profiler:
    """Context manager đo toàn bộ một lần chạy.

    Ví dụ:
        with Profiler(watch_dir=Path("data/raw")) as prof:
            ...
            prof.record_paper("2310.12345", seconds=4.2, n_refs=30, n_refs_co_arxiv_id=12)
        prof.save(Path("reports/profile_m1.json"))
    """

    def __init__(
        self, watch_dir: Path | None = None, interval: float = 0.5, disk_every: int = 10
    ) -> None:
        """Khởi tạo profiler.

        Args:
            watch_dir: Thư mục output cần theo dõi dung lượng; None thì không đo đĩa.
            interval: Số giây giữa hai lần lấy mẫu RAM.
            disk_every: Cứ bao nhiêu lần lấy mẫu RAM thì đo đĩa một lần.
        """
        self._watch_dir = watch_dir
        self._interval = interval
        self._disk_every = max(disk_every, 1)
        self._lock = threading.Lock()
        self._ram: list[int] = []
        self._disk: list[int] = []
        # arxiv_id -> [tổng giây các bước, n_refs, n_refs_co_arxiv_id]
        self._papers: dict[str, list[float]] = {}
        self._wall = 0.0
        self._start = 0.0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self._stop.clear()
        self._start = time.perf_counter()
        # Lấy mẫu ngay lúc vào để một khối with rất ngắn vẫn có dữ liệu.
        self._sample(measure_disk=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample(measure_disk=True)  # mẫu cuối, đồng thời là "final disk"
        self._wall = time.perf_counter() - self._start
        return False  # không nuốt lỗi của khối with

    def _run(self) -> None:
        tick = 0
        # wait() vừa là "ngủ interval giây" vừa thoát ngay khi có lệnh dừng.
        while not self._stop.wait(self._interval):
            tick += 1
            self._sample(measure_disk=tick % self._disk_every == 0)

    def _sample(self, measure_disk: bool) -> None:
        rss = self._process.memory_info().rss
        # Tính cả process con (tar, subprocess...) để ra tổng RAM của toàn hệ thống.
        for child in self._process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # Duyệt thư mục ngoài lock để không chặn các luồng khác.
        disk = dir_size(self._watch_dir) if measure_disk and self._watch_dir else None
        with self._lock:
            self._ram.append(rss)
            if disk is not None:
                self._disk.append(disk)

    def record_paper(
        self, arxiv_id: str, seconds: float, n_refs: int = 0, n_refs_co_arxiv_id: int = 0
    ) -> None:
        """Ghi số liệu của một bước xử lý một bài; gọi nhiều lần cho cùng bài thì cộng dồn.

        Khi bước arXiv và bước S2 chạy ở hai luồng, mỗi luồng gọi hàm này cho bài của mình
        (bước arXiv để n_refs = 0). Thời gian của bài là TỔNG thời gian các bước; còn thời gian
        thực tế của cả hệ thống nằm ở wall_seconds và avg_wall_seconds_per_paper.

        Args:
            arxiv_id: Bài vừa xử lý.
            seconds: Thời gian của bước vừa xong.
            n_refs: Số reference S2 trả về (bước S2).
            n_refs_co_arxiv_id: Số reference có arXiv ID (bước S2).
        """
        with self._lock:
            entry = self._papers.setdefault(arxiv_id, [0.0, 0, 0])
            entry[0] += seconds
            entry[1] += n_refs
            entry[2] += n_refs_co_arxiv_id
        logger.info("Đã đo %s: %.2fs, %d refs", arxiv_id, seconds, n_refs)

    def summary(self) -> ProfileSummary:
        """Tổng hợp số liệu đã đo.

        Returns:
            ProfileSummary; các giá trị đo đĩa là None nếu không truyền watch_dir.
        """
        with self._lock:
            n_papers = len(self._papers)
            n_samples = len(self._ram)
            return ProfileSummary(
                wall_seconds=self._wall,
                n_papers=n_papers,
                avg_seconds_per_paper=(
                    sum(p[0] for p in self._papers.values()) / n_papers if n_papers else None
                ),
                avg_wall_seconds_per_paper=(
                    self._wall / n_papers if n_papers and self._wall > 0 else None
                ),
                peak_ram_mb=max(self._ram, default=0) / MB,
                avg_ram_mb=sum(self._ram) / n_samples / MB if n_samples else 0.0,
                n_ram_samples=n_samples,
                max_disk_mb=max(self._disk) / MB if self._disk else None,
                final_disk_mb=self._disk[-1] / MB if self._disk else None,
                n_refs_total=int(sum(p[1] for p in self._papers.values())),
                n_refs_co_arxiv_id_total=int(sum(p[2] for p in self._papers.values())),
            )

    def save(self, path: Path) -> None:
        """Ghi summary() ra file JSON (qua write_json, an toàn khi bị ngắt).

        Args:
            path: File đích, ví dụ reports/profile_m1.json.
        """
        write_json(path, self.summary().model_dump(mode="json"))