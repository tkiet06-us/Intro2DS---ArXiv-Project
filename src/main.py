"""P3-01 (Owner: Khang) – Pipeline Runner M1 + M2.

Ví dụ: python -m src.main --id_start 2506.00788 --id_end 2506.05787
Tự xuất summary_stats.json cho Jupyter Book; defensive programming (missing/null).
"""
from __future__ import annotations
import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="arXiv pipeline runner")
    p.add_argument("--id_start", required=True)
    p.add_argument("--id_end", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--no_resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: crawler -> s2_api -> parser -> cleanup -> bibtex/dedup -> summary_stats.json
    print(f"Pipeline: {args.id_start} -> {args.id_end}")


if __name__ == "__main__":
    main()
