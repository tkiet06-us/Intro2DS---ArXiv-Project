"""Pipeline Runner tích hợp M1 + M2 (task P3-01).

Chạy từ thư mục gốc repo::

    python -m src.main --id_start 2506.00788 --id_end 2506.05787 --stage m1
    python -m src.main --id_start 2506.00788 --id_end 2506.05787 --stage m2
    python -m src.main --id_start 2506.00788 --id_end 2506.05787 --stage all

Nguyên tắc:

- Import lười theo ``--stage``: chạy M1 KHÔNG import M2 và ngược lại, để số đo RAM/thời gian
  benchmark của mỗi stage không bị nhiễm bởi stage kia.
- M1 ghi ``reports/summary_stats_<name>.json``, M2 ghi ``reports/summary_stats_m2_<name>.json``;
  hai stage có checkpoint riêng nên ``--no_resume`` chỉ tác động stage đang chạy.
- ``--stage all`` (mặc định) chỉ để chạy thử vài bài đầu (``pipeline.all_stage_limit``); nó dùng
  tên ``<name>_sample`` nên KHÔNG ghi đè số liệu benchmark thật.

Mã thoát: 0 thành công (kể cả có bài lỗi đã được log), 1 lỗi cấu hình/đối số/stage, 130 khi Ctrl+C.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from src.common.ids import validate_arxiv_id

logger = logging.getLogger(__name__)

STAGES = ("m1", "m2", "all")
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_NAME = "default"
DEFAULT_ALL_STAGE_LIMIT = 5

# Mã thoát của runner (khớp quy ước M2; run_m1 dùng bộ mã riêng được chuẩn hóa ở đây).
EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_INTERRUPTED = 130

# run_m1: 0 xong, 1 xong một phần (vẫn có dữ liệu để M2 chạy), >= 2 lỗi nặng.
_M1_CONTINUE_CODES = (0, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Bộ phân tích tham số runner (``--id_start`` và ``--id-start`` đều dùng được)."""
    parser = argparse.ArgumentParser(description="arXiv pipeline runner (M1 + M2).")

    def add(name: str, **kwargs: Any) -> None:
        parser.add_argument(*dict.fromkeys([name, name.replace("_", "-")]), **kwargs)

    add("--id_start", help="ID day du dau dai, VD 2506.00788")
    add("--id_end", help="ID day du cuoi dai, VD 2506.05787")
    add("--stage", default="all", help="m1 | m2 | all (mac dinh all, chi de chay thu)")
    add("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    add("--name", default=DEFAULT_NAME, help="Ten bo du lieu (dung chung cho ca hai stage)")
    add("--contact", default=None, help="Email User-Agent (hoac bien moi truong ARXIV_CONTACT)")
    add("--kaggle_json", type=Path, default=None, help="Snapshot Kaggle (can khi M1 chay discovery)")
    add("--data_root", type=Path, default=Path("data/raw"), help="Thu muc goc du lieu M1")
    add("--processed_root", type=Path, default=Path("data/processed"), help="Thu muc xuat M2")
    add("--reports", type=Path, default=Path("reports"))
    add("--limit", type=int, default=None, help="Chi chay N bai dau (chay thu / benchmark)")
    add("--skip_discovery", action="store_true", help="Bo qua discovery, dung file discovery_<name>.jsonl co san")
    add("--no_resume", action="store_true", help="Bo checkpoint, chay lai stage dang chay tu dau")
    add("--no_worker", action="store_true", help="M2 xu ly bai ngay trong tien trinh chinh")
    return parser


def _validate(args: argparse.Namespace) -> str | None:
    """Trả thông báo lỗi tham số, hoặc None nếu hợp lệ."""
    if args.stage not in STAGES:
        return f"--stage phai la mot trong {', '.join(STAGES)}"
    if not args.id_start or not args.id_end:
        return "can ca --id_start va --id_end"
    try:
        validate_arxiv_id(args.id_start)
        validate_arxiv_id(args.id_end)
    except ValueError as exc:
        return str(exc)
    if args.limit is not None and args.limit <= 0:
        return "--limit phai > 0"
    # Chỉ kiểm thứ tự dải khi chạy M2 (M1 tự kiểm theo cách của nó); nhờ vậy `--stage m1`
    # không cần import M2 -> giữ được import lười.
    if args.stage in ("m2", "all"):
        from src.m2_parsing.run_m2 import id_range_key

        if id_range_key(args.id_start) > id_range_key(args.id_end):
            return f"--id_start ({args.id_start}) phai <= --id_end ({args.id_end})"
    return None


# ---------------------------------------------------------------------------
# Adapter cho từng stage
# ---------------------------------------------------------------------------


def _all_stage_limit(config_path: Path) -> int:
    """Đọc ``pipeline.all_stage_limit`` (mặc định 5) cho chế độ ``--stage all``."""
    try:
        with Path(config_path).open(encoding="utf-8") as handle:
            section = (yaml.safe_load(handle) or {}).get("pipeline") or {}
    except FileNotFoundError:
        return DEFAULT_ALL_STAGE_LIMIT
    try:
        limit = int(section.get("all_stage_limit", DEFAULT_ALL_STAGE_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_ALL_STAGE_LIMIT
    return limit if limit > 0 else DEFAULT_ALL_STAGE_LIMIT


def _should_skip_discovery(args: argparse.Namespace, name: str) -> bool:
    """Xác định có bỏ qua discovery không (do cờ tường minh hoặc tự động nhận diện file có sẵn)."""
    if getattr(args, "skip_discovery", False):
        return True
    if getattr(args, "no_resume", False):
        return False

    data_root = Path(args.data_root)
    expected_path = data_root / f"discovery_{name}.jsonl"
    if expected_path.is_file() and expected_path.stat().st_size > 0:
        logger.info(
            "Tu dong nhan dien: tim thay file discovery '%s' -> bo qua discovery (khong can Kaggle snapshot)",
            expected_path,
        )
        return True

    # Khi chạy chế độ thử 'all' (name có dạng '<name>_sample'), kiểm tra file discovery gốc
    if name.endswith("_sample"):
        base_name = name[:-7]
        base_path = data_root / f"discovery_{base_name}.jsonl"
        if base_path.is_file() and base_path.stat().st_size > 0:
            logger.info(
                "Tu dong nhan dien: tim thay file discovery goc '%s' cho mau '%s' -> tai su dung",
                base_path,
                name,
            )
            if not expected_path.exists():
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base_path, expected_path)
            return True

    # Kiểm tra file discovery chung (25k bài của nhóm) nếu có
    for shared_name in ("discovery_25k.jsonl", "discovery.jsonl"):
        shared_path = data_root / shared_name
        if shared_path.is_file() and shared_path.stat().st_size > 0:
            logger.info(
                "Tu dong nhan dien: tim thay file discovery chung '%s' -> sao chep sang '%s' cho '%s'",
                shared_path,
                expected_path,
                name,
            )
            if not expected_path.exists():
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(shared_path, expected_path)
            return True

    return False


def _m1_argv(
    args: argparse.Namespace,
    name: str,
    limit: int | None,
    skip_discovery: bool = False,
) -> list[str]:
    """Dựng argv cho ``run_m1.build_parser`` từ tham số của runner."""
    argv = [
        "--name", name,
        "--config", str(args.config),
        "--data_root", str(args.data_root),
        "--reports", str(args.reports),
    ]  # fmt: skip
    if args.id_start:
        argv += ["--id_start", args.id_start]
    if args.id_end:
        argv += ["--id_end", args.id_end]
    if args.kaggle_json:
        argv += ["--kaggle_json", str(args.kaggle_json)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if args.contact:
        argv += ["--contact", args.contact]
    if args.no_resume:
        argv += ["--fresh_discovery", "--fresh_crawl", "--fresh_s2"]
    if skip_discovery:
        argv += ["--skip_discovery"]
    return argv


def _run_m1(args: argparse.Namespace, name: str, limit: int | None) -> int:
    """Chạy stage M1 qua adapter; trả mã thoát RIÊNG của ``run_m1`` (được chuẩn hóa sau)."""
    from src.m1_crawling import run_m1

    skip_discovery = _should_skip_discovery(args, name)
    stage_args = run_m1.build_parser().parse_args(
        _m1_argv(args, name, limit, skip_discovery=skip_discovery)
    )
    contact = args.contact or os.environ.get(run_m1.crawler.CONTACT_ENV)
    problem = run_m1._validate(stage_args, contact)
    if problem:
        logger.error("LOI M1: %s", problem)
        return run_m1.EXIT_BAD_INPUT
    try:
        code = run_m1.run_pipeline(stage_args, contact)
    except run_m1.DiscoveryFailedError as exc:
        logger.error("M1 discovery that bai: %s", exc)
        return run_m1.EXIT_DISCOVERY
    except ValueError as exc:
        logger.error("M1 du lieu loi: %s", exc)
        return run_m1.EXIT_BAD_INPUT
    if code != run_m1.EXIT_OK:
        logger.error("M1 ket thuc voi ma %d (chay lai cung lenh de tiep tuc)", code)
    return code


def _m2_argv(args: argparse.Namespace, name: str, limit: int | None) -> list[str]:
    """Dựng argv cho ``run_m2.build_parser`` từ tham số của runner."""
    argv = [
        "--name", name,
        "--id_start", args.id_start,
        "--id_end", args.id_end,
        "--config", str(args.config),
        "--data_root", str(args.data_root),
        "--processed_root", str(args.processed_root),
        "--reports", str(args.reports),
    ]  # fmt: skip
    if limit is not None:
        argv += ["--limit", str(limit)]
    if args.no_resume:
        argv += ["--no_resume"]
    if args.no_worker:
        argv += ["--no_worker"]
    return argv


def _run_m2(args: argparse.Namespace, name: str, limit: int | None) -> int:
    """Chạy stage M2 qua adapter; trả mã thoát của ``run_m2`` (0/1/130)."""
    from src.m2_parsing import run_m2

    stage_args = run_m2.build_parser().parse_args(_m2_argv(args, name, limit))
    problem = run_m2._validate(stage_args)
    if problem:
        logger.error("LOI M2: %s", problem)
        return EXIT_BAD_INPUT
    return run_m2.run_stage(stage_args)


def _normalize(code: int) -> int:
    """Chuẩn hóa mã thoát của stage về quy ước runner (0 / 1 / 130)."""
    if code == EXIT_INTERRUPTED:
        return EXIT_INTERRUPTED
    return EXIT_OK if code == EXIT_OK else EXIT_BAD_INPUT


def _dispatch(args: argparse.Namespace) -> int:
    """Điều phối stage theo ``--stage`` và trả mã thoát theo quy ước runner."""
    if args.stage == "m1":
        return _normalize(_run_m1(args, args.name, args.limit))
    if args.stage == "m2":
        return _normalize(_run_m2(args, args.name, args.limit))

    limit = _all_stage_limit(args.config)
    sample_name = f"{args.name}_sample"
    logger.warning(
        "Stage 'all' chi de chay thu (toi da %d bai, ten '%s'), KHONG dung de benchmark.",
        limit, sample_name,
    )
    m1_code = _run_m1(args, sample_name, limit)
    if m1_code not in _M1_CONTINUE_CODES:
        return _normalize(m1_code)
    m2_code = _run_m2(args, sample_name, limit)
    if m2_code == EXIT_INTERRUPTED:
        return EXIT_INTERRUPTED
    return EXIT_OK if (m1_code == EXIT_OK and m2_code == EXIT_OK) else EXIT_BAD_INPUT


def main(argv: list[str] | None = None) -> int:
    """Điểm vào runner. Trả mã thoát (xem docstring module)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    problem = _validate(args)
    if problem:
        logger.error("LOI: %s", problem)
        return EXIT_BAD_INPUT
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        logger.warning("Da dung theo yeu cau (Ctrl+C); checkpoint + summary da duoc flush.")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("Runner gap loi khong mong doi")
        return EXIT_BAD_INPUT


__all__ = [
    "DEFAULT_ALL_STAGE_LIMIT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_NAME",
    "EXIT_BAD_INPUT",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "STAGES",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
