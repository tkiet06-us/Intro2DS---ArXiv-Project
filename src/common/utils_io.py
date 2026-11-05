"""Hàm tiện ích I/O dùng chung: đọc/ghi JSON an toàn, nhật ký .jsonl, cửa vào hàm đổi dạng ID."""

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.common.ids import from_folder_name, to_folder_name

__all__ = [
    "append_jsonl",
    "from_folder_name",
    "iter_jsonl",
    "read_json",
    "to_folder_name",
    "write_json",
]


def write_json(path: Path, data: Any) -> None:
    """Ghi JSON an toàn khi bị ngắt giữa chừng (ghi file tạm rồi đổi tên).

    Args:
        path: File đích; thư mục cha sẽ được tạo nếu chưa có.
        data: Dữ liệu tuần tự hóa được thành JSON (không chứa NaN/Infinity).

    Raises:
        ValueError: Khi data chứa NaN hoặc Infinity (không hợp lệ theo chuẩn JSON).
        TypeError: Khi data chứa kiểu không tuần tự hóa được.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # File tạm có tên duy nhất, nằm cùng thư mục để os.replace là rename trong cùng ổ.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            # ensure_ascii=False để giữ tên tác giả có dấu, indent=2 cho dễ đọc diff.
            # allow_nan=False để NaN báo lỗi NGAY ở đây, thay vì ghi ra JSON sai chuẩn và nổ
            # ở bước đọc sau (ví dụ Pydantic từ chối year = NaN) khi đã khó truy ra nguồn.
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)

            # flush + fsync: ép dữ liệu xuống đĩa trước khi đổi tên, để mất điện hay VM bị thu
            # hồi không để lại file đích rỗng/cụt (checkpoint mất còn nguy hiểm hơn chạy chậm).
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp tạo file quyền 0600; đặt lại 0644 để file đích có quyền như file thường,
        # để dùng chung thư mục (Drive, nhóm) đọc được.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        # Bắt cả KeyboardInterrupt (Ctrl+C): dọn file tạm rồi ném lại lỗi, không nuốt lỗi.
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    """Đọc một file JSON UTF-8.

    Args:
        path: File cần đọc.

    Returns:
        Dữ liệu đã giải mã.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path | str, record: dict) -> None:
    """Ghi nối MỘT bản ghi JSON vào cuối file ``.jsonl`` và ép xuống đĩa.

    Gọi sau mỗi đơn vị công việc để checkpoint: nếu chương trình bị ngắt thì tối đa mất
    bản ghi đang ghi dở, các bản ghi trước đó vẫn an toàn.

    Args:
        path: File ``.jsonl`` (tự tạo thư mục cha nếu chưa có).
        record: Dict JSON-hóa được; ghi thành đúng một dòng.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    """Đọc từng bản ghi của file ``.jsonl``, bỏ qua dòng trống và dòng hỏng.

    Dòng hỏng thường là dòng cuối bị cắt ngang khi chương trình bị ngắt giữa chừng.

    Args:
        path: File ``.jsonl``. Không tồn tại thì không sinh bản ghi nào.

    Yields:
        Từng bản ghi dạng dict.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record