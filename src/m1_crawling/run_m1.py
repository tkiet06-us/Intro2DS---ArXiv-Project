"""Chạy toàn bộ Milestone 1 trong một lệnh: discovery -> (crawl arXiv || S2 references) -> thống kê.

Đặt tại ``src/m1_crawling/run_m1.py``, chạy từ thư mục gốc repo::

    python -m src.m1_crawling.run_m1 --name kien --id_start 2506.00788 --id_end 2506.05787 \\
        --kaggle_json data/arxiv-metadata-oai-snapshot.json --max_disk_gb 25

Luồng chạy và chỗ được tăng tốc:

    1. Discovery (script ``discovery.py``, chạy như tiến trình con): quét snapshot Kaggle + gọi
       arXiv API lấy ``n_versions``, rồi vá ``version_dates``. Phải xong TRƯỚC khi crawl vì crawler
       cần ``n_versions``. Bước này chỉ vài chục phút nên không đáng làm phức tạp để chồng lên crawl.
    2. Hai luồng chạy SONG SONG ngay sau đó:
         - luồng ``arxiv``: ``crawler.run`` tải mã nguồn từng version (nút cổ chai: >= 3 giây/request
           và một kết nối, quy định của arXiv, KHÔNG thể song song hóa trong chính luồng này);
         - luồng ``s2``: ``s2_api.run_s2_stage`` lấy references theo lô. S2 là host khác, giới hạn
           khác, và chỉ cần ``arxiv_id`` nên không phải chờ arXiv. Nhờ vậy toàn bộ thời gian S2 được
           "giấu" vào thời gian chờ delay của arXiv thay vì cộng thêm vào.
       Việc giải nén/ghi đĩa của từng version vốn đã chạy trong lúc chờ delay 3 giây kế tiếp
       (``PoliteFetcher`` tính delay từ lúc request trước KẾT THÚC), nên cũng không tốn thêm giờ.
    3. Profiler (RAM đỉnh/trung bình, đĩa, thời gian) bao trùm cả hai luồng, rồi xuất
       ``reports/summary_stats_<name>.json`` cho Jupyter Book và ``reports/profile_m1_<name>.json``.

Checkpoint/Resume: chạy lại CÙNG LỆNH sẽ bỏ qua version/bài đã xong (cả arXiv lẫn S2). Ctrl+C an toàn.

Mã thoát: 0 xong; 1 còn lỗi (chạy lại để thử lại); 2 sai tham số/dữ liệu; 3 bị arXiv chặn (403);
4 dừng vì quá nhiều lỗi liên tiếp; 5 đầy đĩa (--max_disk_gb); 6 discovery thất bại.
Thông điệp in ra màn hình cố ý viết không dấu cho console Windows.
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.utils_io import write_json  # noqa: E402
from src.m1_crawling import crawler, s2_api  # noqa: E402
from src.m1_crawling.profiler import Profiler  # noqa: E402

logger = logging.getLogger(__name__)

DISCOVERY_SCRIPT = Path(__file__).with_name("discovery.py")
EXIT_OK, EXIT_PARTIAL, EXIT_BAD_INPUT, EXIT_BLOCKED, EXIT_ABORTED, EXIT_DISK, EXIT_DISCOVERY = (
    0, 1, 2, 3, 4, 5, 6,
)
ESTIMATED_DOWNLOAD_S = 1.0  # ước lượng thô thời gian tải một version (chưa tính delay 3s)


class DiscoveryFailedError(RuntimeError):
    """Tiến trình con discovery.py thoát với mã lỗi."""


@dataclass
class Paths:
    """Các đường dẫn của một lần chạy, suy ra từ ``--name`` để nhiều người không đè dữ liệu nhau."""

    discovery: Path
    out_dir: Path
    reports: Path

    @classmethod
    def build(cls, name: str, data_root: Path, reports: Path) -> "Paths":
        return cls(data_root / f"discovery_{name}.jsonl", data_root / f"papers_{name}", reports)

    def report(self, stem: str, name: str) -> Path:
        return self.reports / f"{stem}_{name}.json"


@dataclass
class Stage:
    """Kết quả của một luồng (giá trị trả về hoặc lỗi) cùng thời gian chạy."""

    name: str
    value: Any = None
    error: Exception | None = None
    seconds: float = 0.0


def start_stage(stage: Stage, func: Callable[[], Any]) -> threading.Thread:
    """Chạy ``func`` ở luồng nền; mọi lỗi được giữ lại trong ``stage.error`` thay vì mất im lặng."""

    def target() -> None:
        started = time.monotonic()
        try:
            stage.value = func()
        except Exception as exc:  # noqa: BLE001  (cố ý: một luồng lỗi không được giết luồng kia)
            logger.exception("Luong %s that bai", stage.name)
            stage.error = exc
        finally:
            stage.seconds = time.monotonic() - started

    thread = threading.Thread(target=target, name=stage.name, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Discovery (tiến trình con)
# ---------------------------------------------------------------------------


def read_ranges(path: Path) -> list[tuple[str, str]]:
    """Đọc file dải ID (mỗi dòng ``ID_dau ID_cuoi``, ``#`` là comment) giống discovery.py."""
    ranges = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#")[0].strip()
        if line:
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: can 2 ID, gap {line!r}")
            ranges.append((parts[0], parts[1]))
    if not ranges:
        raise ValueError(f"{path} khong co dai nao")
    return sorted(ranges)


def run_discovery(args: argparse.Namespace, paths: Paths, contact: str) -> None:
    """Chạy discovery.py ``--mode full`` rồi ``--mode patch`` (in thẳng ra console).

    Raises:
        DiscoveryFailedError: Khi bước full thất bại. Bước patch lỗi chỉ cảnh báo (idempotent,
            chạy lại được, và thiếu ngày version chỉ làm ``revised_dates`` ngắn đi).
    """
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.discovery.parent.mkdir(parents=True, exist_ok=True)
    full = [
        sys.executable, str(DISCOVERY_SCRIPT), "--mode", "full", "--contact", contact,
        "--kaggle-json", str(args.kaggle_json), "--total-ids", str(args.total_ids),
        "--out", str(paths.discovery), "--report", str(paths.report("discovery_full", args.name)),
    ]  # fmt: skip
    if args.ranges:
        full += ["--ranges", str(args.ranges)]
    else:
        full += ["--id-start", args.id_start, "--id-end", args.id_end]
    if args.fresh_discovery:
        full.append("--fresh")
    logger.info("Discovery (full): %s", " ".join(full).replace(contact, "<email>"))
    if subprocess.run(full, check=False).returncode != 0:
        raise DiscoveryFailedError("discovery.py --mode full that bai (chay lai cung lenh de resume)")

    patch = [
        sys.executable, str(DISCOVERY_SCRIPT), "--mode", "patch", "--contact", contact,
        "--out", str(paths.discovery),
        "--patch-report", str(paths.report("discovery_patch", args.name)),
    ]  # fmt: skip
    if subprocess.run(patch, check=False).returncode != 0:
        logger.warning("discovery.py --mode patch con bai chua va duoc ngay version (xem bao cao)")


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def build_stats(
    args: argparse.Namespace,
    paths: Paths,
    papers: list[dict],
    crawl_summary: dict | None,
    s2_results: list[Any],
    profile: dict,
    stages: dict[str, Stage],
    discovery_seconds: float | None,
    wall_seconds: float,
    interrupted: bool,
) -> dict:
    """Gom mọi số liệu M1 vào một dict phẳng sẵn sàng cho ``summary_stats.json``."""
    n_papers = len(papers)
    s2_papers, n_refs, n_arxiv = s2_api.count_saved_references(paths.out_dir)
    crawl_rates: dict[str, float | None] = {}
    if crawl_summary:
        ok = crawl_summary["by_status"].get(crawler.STATUS_OK, 0)
        crawl_rates = {
            "version_ok_rate": _ratio(ok, crawl_summary["versions_expected"]),
            "paper_with_tex_rate": _ratio(
                n_papers - crawl_summary["papers_with_no_tex_at_all"], n_papers
            ),
            "avg_size_before_bytes_per_paper": _ratio(crawl_summary["size_before_bytes"], n_papers),
            "avg_size_after_bytes_per_paper": _ratio(crawl_summary["size_after_bytes"], n_papers),
        }
    sequential = sum(stage.seconds for stage in stages.values())
    return {
        "name": args.name,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "interrupted": interrupted,
        "n_papers": n_papers,
        "discovery_seconds": discovery_seconds,
        "crawl": crawl_summary,
        "crawl_rates": crawl_rates,
        "s2": {
            "statuses": dict(Counter(result.status for result in s2_results)),
            "papers_with_references": s2_papers,
            "n_refs_total": n_refs,
            "n_refs_with_arxiv_id": n_arxiv,
            "avg_refs_per_paper": _ratio(n_refs, s2_papers),
            "arxiv_id_rate": _ratio(n_arxiv, n_refs),
            "paper_success_rate": _ratio(s2_papers, n_papers),
        },
        "profile": profile,
        "pipeline": {
            "wall_seconds": round(wall_seconds, 1),
            "stage_seconds": {name: round(stage.seconds, 1) for name, stage in stages.items()},
            # Nếu hai luồng chạy nối đuôi thì mất tổng thời gian của chúng; phần chênh là phần được giấu.
            "seconds_saved_by_overlap": round(max(sequential - wall_seconds, 0.0), 1),
            "errors": {name: repr(stage.error) for name, stage in stages.items() if stage.error},
        },
    }


def exit_code_for(stages: dict[str, Stage], crawl_summary: dict | None, s2_results: list[Any]) -> int:
    """Chọn mã thoát theo mức nghiêm trọng của lỗi (chặn > đầy đĩa > dừng > lỗi thường)."""
    error = stages["arxiv"].error
    if isinstance(error, crawler.CrawlBlockedError):
        return EXIT_BLOCKED
    if isinstance(error, crawler.DiskLimitError):
        return EXIT_DISK
    if isinstance(error, crawler.CrawlAbortedError):
        return EXIT_ABORTED
    if isinstance(error, ValueError):
        return EXIT_BAD_INPUT
    if error is not None or stages["s2"].error is not None:
        return EXIT_PARTIAL
    if any(result.status == "failed" for result in s2_results):
        return EXIT_PARTIAL
    return int(crawl_summary["exit_code"]) if crawl_summary else EXIT_OK


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------


def _eta_line(papers: list[dict], delay: float) -> str:
    try:
        versions = sum(crawler._n_versions(row) for row in papers)  # noqa: SLF001
    except ValueError:
        return "Chua tinh duoc ETA (con bai thieu n_versions, crawler se tra API)."
    hours = versions * (delay + ESTIMATED_DOWNLOAD_S) / 3600
    return (
        f"{len(papers)} bai / {versions} version => uoc tinh toi da ~{hours:.1f} gio cho arXiv "
        f"(~{delay + ESTIMATED_DOWNLOAD_S:.0f}s/version; version da xong se duoc bo qua)"
    )


def run_pipeline(
    args: argparse.Namespace,
    contact: str,
    fetcher: crawler.PoliteFetcher | None = None,
    s2_client: s2_api.S2Client | None = None,
) -> int:
    """Chạy discovery (nếu cần) rồi hai luồng arXiv và S2 song song, sau đó xuất thống kê.

    ``fetcher`` và ``s2_client`` cho phép test truyền bản giả, không cần mạng.

    Returns:
        Mã thoát (xem docstring module).
    """
    paths = Paths.build(args.name, args.data_root, args.reports)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    wall_start = time.monotonic()

    discovery_seconds: float | None = None
    if not args.skip_discovery:
        started = time.monotonic()
        run_discovery(args, paths, contact)
        discovery_seconds = time.monotonic() - started
        logger.info("Discovery xong sau %.0fs", discovery_seconds)
    if not paths.discovery.exists():
        print(f"LOI: khong thay file discovery {paths.discovery}", file=sys.stderr)
        return EXIT_BAD_INPUT

    id_start, id_end = args.id_start, args.id_end
    if args.ranges:
        ranges = read_ranges(args.ranges)
        id_start, id_end = ranges[0][0], max(end for _, end in ranges)
    papers = crawler.select_papers(paths.discovery, id_start, id_end, args.limit)
    if not papers:
        print("LOI: khong co bai nao trong dai ID.", file=sys.stderr)
        return EXIT_BAD_INPUT
    logger.info(_eta_line(papers, args.delay))

    fetcher = fetcher or crawler.PoliteFetcher(
        crawler.USER_AGENT.format(contact=contact), args.delay
    )
    s2_client = s2_client or s2_api.S2Client(s2_api.S2Settings.from_config(args.config))
    arxiv_ids = [row["arxiv_id"] for row in papers]
    stages = {"arxiv": Stage("arxiv"), "s2": Stage("s2")}
    interrupted = False

    with Profiler(watch_dir=paths.out_dir) as prof:

        def on_s2_result(result: s2_api.CrawlResult, seconds: float) -> None:
            # Bài skipped (đã xong ở lần chạy trước) không tốn mạng: không đưa vào trung bình.
            if result.status != "skipped":
                prof.record_paper(result.arxiv_id, seconds, result.n_refs, result.n_refs_co_arxiv_id)

        def crawl_arxiv() -> dict:
            return crawler.run(
                papers, paths.out_dir, fetcher,
                write_metadata=True,
                max_consecutive_errors=args.max_consecutive_errors,
                resume=not args.fresh_crawl,
                max_disk_gb=args.max_disk_gb,
            )  # fmt: skip

        def crawl_s2() -> list[s2_api.CrawlResult]:
            return s2_api.run_s2_stage(
                arxiv_ids, paths.out_dir, s2_client, on_result=on_s2_result,
                resume=not args.fresh_s2, batch_size=args.s2_batch,
            )  # fmt: skip

        threads = []
        if not args.skip_crawl:
            threads.append(start_stage(stages["arxiv"], crawl_arxiv))
        if not args.skip_s2:
            threads.append(start_stage(stages["s2"], crawl_s2))
        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=1.0)  # timeout để Ctrl+C vẫn ngắt được
        except KeyboardInterrupt:
            interrupted = True
            print("\nDa dung theo yeu cau. Chay lai CUNG LENH de tiep tuc.")

    wall = time.monotonic() - wall_start
    crawl_summary = stages["arxiv"].value
    if crawl_summary is None and (
        not args.skip_crawl or (paths.out_dir / crawler.LOG_NAME).exists()
    ):
        try:  # luồng arXiv dừng giữa chừng: dựng thống kê từ log đã ghi
            state = crawler.load_log_state(paths.out_dir / crawler.LOG_NAME)
            crawl_summary = crawler.summarize(state, papers, paths.out_dir, stages["arxiv"].seconds)
            crawl_summary["exit_code"] = EXIT_OK if args.skip_crawl else EXIT_PARTIAL
        except ValueError:
            crawl_summary = None
    if crawl_summary:
        crawler.add_coverage(crawl_summary, papers, id_start, id_end, args.limit)
        for arxiv_id, seconds in crawl_summary.get("seconds_per_paper", {}).items():
            prof.record_paper(arxiv_id, seconds)  # cộng dồn với thời gian S2 của cùng bài

    s2_results = stages["s2"].value or []
    stats = build_stats(
        args, paths, papers, crawl_summary, s2_results, prof.summary().model_dump(mode="json"),
        stages, discovery_seconds, wall, interrupted,
    )  # fmt: skip
    write_json(paths.report("summary_stats", args.name), stats)
    prof.save(paths.report("profile_m1", args.name))
    _print_report(stats, paths, args.name)
    return EXIT_PARTIAL if interrupted else exit_code_for(stages, crawl_summary, s2_results)


def _print_report(stats: dict, paths: Paths, name: str) -> None:
    prof, pipe, s2 = stats["profile"], stats["pipeline"], stats["s2"]
    print("\n=== TONG KET M1 ===")
    print(f"Bai: {stats['n_papers']} | wall end-to-end: {pipe['wall_seconds']}s "
          f"(tiet kiem nho chay song song: {pipe['seconds_saved_by_overlap']}s)")
    if stats["crawl"]:
        print(f"arXiv: {stats['crawl']['by_status']} | ti le version ok: "
              f"{stats['crawl_rates'].get('version_ok_rate')}")
    print(f"S2: {s2['statuses']} | ref TB/bai: {s2['avg_refs_per_paper']} | "
          f"ti le ref co arXiv ID: {s2['arxiv_id_rate']}")
    print(f"RAM dinh/TB: {prof['peak_ram_mb']:.0f}/{prof['avg_ram_mb']:.0f} MB | "
          f"dia max/cuoi: {prof['max_disk_mb']}/{prof['final_disk_mb']} MB")
    for name_, error in pipe["errors"].items():
        print(f"LOI luong {name_}: {error}")
    print(f"Bao cao: {paths.report('summary_stats', name)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Bộ phân tích tham số (``--id_start`` và ``--id-start`` đều dùng được)."""
    parser = argparse.ArgumentParser(description="Chay toan bo Milestone 1 (crawl + S2 song song).")

    def add(name: str, **kwargs: Any) -> None:
        parser.add_argument(*dict.fromkeys([name, name.replace("_", "-")]), **kwargs)

    add("--name", required=True, help="Ten ban (vd kien): tao data/raw/papers_<name>/ rieng")
    add("--id_start", help="ID day du dau dai, VD 2506.00788")
    add("--id_end", help="ID day du cuoi dai, VD 2506.05787")
    add("--ranges", type=Path, help="File nhieu dai ID (thay --id_start/--id_end)")
    add("--contact", default=None, help=f"Email User-Agent (hoac bien moi truong {crawler.CONTACT_ENV})")
    add("--kaggle_json", type=Path, help="File snapshot Kaggle (can khi chay discovery)")
    add("--total_ids", type=int, default=5000)
    add("--config", type=Path, default=Path("configs/config.yaml"))
    add("--data_root", type=Path, default=Path("data/raw"))
    add("--reports", type=Path, default=Path("reports"))
    add("--delay", type=float, default=crawler.MIN_DELAY_S, help="Giay giua 2 request arXiv (>=3)")
    add("--limit", type=int, default=None, help="Chi crawl N bai dau (chay thu / benchmark)")
    add("--max_disk_gb", type=float, default=None, help="Dung khi thu muc output vuot nguong (GB)")
    add("--max_consecutive_errors", type=int, default=10)
    add("--s2_batch", type=int, default=100, help="So bai moi lo POST /paper/batch cua S2")
    add("--skip_discovery", action="store_true", help="Dung file discovery co san")
    add("--skip_crawl", action="store_true", help="Khong tai arXiv")
    add("--skip_s2", action="store_true", help="Khong goi Semantic Scholar")
    add("--fresh_discovery", action="store_true", help="Bo checkpoint discovery")
    add("--fresh_crawl", action="store_true", help="Tai lai arXiv tu dau")
    add("--fresh_s2", action="store_true", help="Lay lai references tu dau")
    return parser


def _validate(args: argparse.Namespace, contact: str | None) -> str | None:
    """Trả thông báo lỗi tham số, hoặc None nếu hợp lệ."""
    if not (args.ranges or (args.id_start and args.id_end)):
        return "can --ranges hoac ca --id_start va --id_end"
    if not args.name or not all(c.isalnum() or c in "-_" for c in args.name):
        return "--name chi gom chu, so, '-', '_' (vd kien, bao)"
    if args.ranges and not args.ranges.exists():
        return f"--ranges: khong thay file {args.ranges}"
    if not contact:
        return f"can --contact hoac bien moi truong {crawler.CONTACT_ENV}"
    if args.delay < crawler.MIN_DELAY_S:
        return f"--delay phai >= {crawler.MIN_DELAY_S} giay (quy dinh arXiv)"
    if not args.skip_discovery and not (args.kaggle_json and args.kaggle_json.exists()):
        return "can --kaggle_json tro toi file ton tai (hoac dung --skip_discovery)"
    if args.max_disk_gb is not None and args.max_disk_gb <= 0:
        return "--max_disk_gb phai > 0"
    return None


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI. Trả mã thoát (xem docstring module)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
    )
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()  # đọc S2_API_KEY, ARXIV_CONTACT từ .env
    except ImportError:
        logger.warning("Chua cai python-dotenv: khong doc .env, phai set bien moi truong truc tiep")
    args = build_parser().parse_args(argv)
    contact = args.contact or os.environ.get(crawler.CONTACT_ENV)
    problem = _validate(args, contact)
    if problem:
        print(f"LOI: {problem}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if not args.skip_s2 and not os.environ.get("S2_API_KEY"):
        logger.warning("Chua co S2_API_KEY: S2 se bi gioi han chung va de gap 429")
    try:
        return run_pipeline(args, contact)
    except DiscoveryFailedError as exc:
        print(f"LOI: {exc}", file=sys.stderr)
        return EXIT_DISCOVERY
    except ValueError as exc:
        print(f"LOI du lieu: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())