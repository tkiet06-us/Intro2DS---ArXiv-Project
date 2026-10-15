"""Tải 2-3 bài arXiv thật làm mẫu cho parser (task P1-02, bước "real-sample mocking").

Chạy: ARXIV_USER_AGENT="DS-Project-HCMUS/1.0 (mailto:you@student.hcmus.edu.vn)" \
      python tests/fetch_real_samples.py 1706.03762 <id khác> ...

Tuân thủ quy tắc crawler của đề (README mục 6.9.10): chỉ export.arxiv.org, delay >= 3 giây,
một kết nối một lúc, có User-Agent mô tả rõ, mọi request đều có timeout và retry.
"""

import argparse
import gzip
import io
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.ids import to_folder_name, validate_arxiv_id  # noqa: E402

logger = logging.getLogger(__name__)

SRC_URL = "https://export.arxiv.org/src/{arxiv_id}v{version}"
DELAY_SECONDS = 3  # luật của arXiv: >= 3 giây giữa 2 request
TIMEOUT_SECONDS = 60  # không để request treo vô hạn
MAX_RETRIES = 3
# Chỉ giữ file văn bản. Ảnh bị loại để repo nhẹ; lệnh \includegraphics vẫn nằm trong .tex.
TEXT_SUFFIXES = {".tex", ".bib", ".bbl", ".sty", ".cls", ".bst", ".def", ".cfg", ".txt"}


def polite_get(url: str, user_agent: str) -> bytes:
    """Tải một URL arXiv với delay, timeout và retry tăng dần.

    Mọi request tới arXiv PHẢI đi qua hàm này để không ai vô tình gọi quá nhanh.

    Args:
        url: URL trên export.arxiv.org.
        user_agent: Chuỗi định danh có email liên hệ.

    Returns:
        Nội dung phản hồi.

    Raises:
        requests.RequestException: Khi vẫn lỗi sau MAX_RETRIES lần thử.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers={"User-Agent": user_agent}, timeout=TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            logger.warning("Lỗi lần %d khi tải %s, thử lại", attempt, url)
            time.sleep(DELAY_SECONDS * 2**attempt)  # chờ tăng dần: 6s, 12s, ...
        finally:
            time.sleep(DELAY_SECONDS)  # luôn nghỉ, kể cả khi thành công
    raise AssertionError("không tới được đây")  # để type checker yên tâm


def extract_text_files(blob: bytes) -> dict[str, bytes]:
    """Lấy các file văn bản từ nội dung /src/ (tar.gz, .gz đơn, hoặc PDF-only).

    Args:
        blob: Byte tải về.

    Returns:
        Dict tên file -> nội dung. Rỗng nếu không có TeX (bài chỉ có PDF/Word).
    """
    try:
        files: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name)
                # Chặn path traversal ("../../x") trong tar không đáng tin.
                if not member.isfile() or name.is_absolute() or ".." in name.parts:
                    continue
                if name.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                handle = tar.extractfile(member)
                if handle is not None:
                    files[name.as_posix()] = handle.read()
        return files
    except tarfile.ReadError:
        pass
    # Không phải tar: arXiv trả .gz đơn cho bài chỉ có một file .tex.
    try:
        return {"main.tex": gzip.decompress(blob)}
    except gzip.BadGzipFile:
        return {}  # PDF/Word-only: caller giữ record rỗng


def main() -> None:
    """Tải các bài được chỉ định (version 1) vào tests/mock_data/real/."""
    parser = argparse.ArgumentParser(description="Tải mẫu thật từ export.arxiv.org.")
    parser.add_argument("ids", nargs="*", default=["1706.03762"], help="arxiv_id dạng chuẩn")
    parser.add_argument("--out", default="tests/mock_data/real")
    args = parser.parse_args()

    user_agent = os.environ.get("ARXIV_USER_AGENT")
    if not user_agent:
        # Không hard-code User-Agent: mỗi người dùng email thật của mình.
        raise SystemExit("Thiếu biến môi trường ARXIV_USER_AGENT (xem README mục 8).")

    for arxiv_id in args.ids:
        validate_arxiv_id(arxiv_id)
        blob = polite_get(SRC_URL.format(arxiv_id=arxiv_id, version=1), user_agent)
        files = extract_text_files(blob)
        if not files:
            logger.warning("%s không có mã nguồn TeX (PDF/Word-only)", arxiv_id)
        folder = Path(args.out) / to_folder_name(arxiv_id) / "v1"
        folder.mkdir(parents=True, exist_ok=True)  # PDF-only vẫn tạo folder rỗng
        for filename, content in files.items():
            target = folder / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        logger.info("Xong %s: %d file", arxiv_id, len(files))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()