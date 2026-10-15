"""Hàm tiện ích I/O dùng chung: đọc/ghi JSON an toàn, và cửa vào của hàm đổi dạng ID."""

import json
import os
from pathlib import Path
from typing import Any

# Logic đổi ID nằm ở ids.py (một nguồn sự thật); re-export để code gọi qua utils_io như README
# mục 6.9.2 mô tả vẫn chạy.
from src.common.ids import from_folder_name, to_folder_name

__all__ = ["from_folder_name", "read_json", "to_folder_name", "write_json"]


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