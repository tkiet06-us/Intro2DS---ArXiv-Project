"""Hàm tiện ích I/O dùng chung: đổi tên folder theo arxiv_id, Tiện ích I/O: đọc/ghi JSON, checkpoint, logging."""

import json
import os
from pathlib import Path
from typing import Any


def to_folder_name(arxiv_id: str) -> str:
    """Đổi arxiv_id chuẩn sang tên folder (dạng "yymm-id" của đề).

    Đây là hàm DUY NHẤT được phép đổi dạng ID (README mục 6.9.2). Nếu mỗi người tự viết một
    kiểu thì "2310.12345" và "2310-12345" sẽ lẫn lộn giữa các module và join dữ liệu bị lệch.

    Args:
        arxiv_id: Mã chuẩn, ví dụ "2310.12345" hoặc "hep-th/9901001".

    Returns:
        "2310-12345" hoặc "hep-th_9901001" (dấu "/" không hợp lệ trong tên folder).
    """
    if "/" in arxiv_id:
        return arxiv_id.replace("/", "_")
    return arxiv_id.replace(".", "-")


def from_folder_name(folder_name: str) -> str:
    """Đổi ngược tên folder về arxiv_id chuẩn (dùng cho key của references.json).

    Phép đổi này không mơ hồ: ID cũ luôn chứa "_" (từ "/"), còn ID mới thì không.

    Args:
        folder_name: "2310-12345" hoặc "hep-th_9901001".

    Returns:
        "2310.12345" hoặc "hep-th/9901001".
    """
    if "_" in folder_name:
        return folder_name.replace("_", "/")
    return folder_name.replace("-", ".", 1)


def write_json(path: Path, data: Any) -> None:
    """Ghi JSON an toàn khi bị ngắt giữa chừng (ghi file tạm rồi đổi tên).

    Nếu ghi thẳng vào file đích mà bị Ctrl+C giữa chừng, ta để lại file JSON cụt và lần
    resume sau sẽ đọc lỗi. Đổi tên (replace) là thao tác nguyên tử nên file đích luôn nguyên vẹn.

    Args:
        path: File đích; thư mục cha sẽ được tạo nếu chưa có.
        data: Dữ liệu tuần tự hóa được thành JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # ensure_ascii=False để giữ tiếng Việt/tên tác giả có dấu, indent=2 cho dễ đọc diff.
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def read_json(path: Path) -> Any:
    """Đọc một file JSON UTF-8.

    Args:
        path: File cần đọc.

    Returns:
        Dữ liệu đã giải mã.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)