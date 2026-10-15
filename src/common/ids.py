"""Mọi thứ liên quan đến dạng của arXiv ID: kiểm tra và đổi qua lại với tên folder.

Gom vào một module nhẹ (không import thư viện nặng) để schemas.py, utils_io.py, crawler và
exporter cùng dùng chung, tránh mỗi nơi tự viết regex một kiểu rồi lệch nhau.
"""

import re

# ID mới: YYMM.NNNN (trước 2015) hoặc YYMM.NNNNN (từ 2015). Ví dụ "2310.12345".
NEW_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")
# ID cũ (trước 2007): archive[.XX]/YYMMNNN. Ví dụ "hep-th/9901001", "math.GT/0309136".
OLD_ID_PATTERN = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}$")
# Dạng folder tương ứng: "." -> "-" (ID mới), "/" -> "_" (ID cũ).
NEW_FOLDER_PATTERN = re.compile(r"^\d{4}-\d{4,5}$")
OLD_FOLDER_PATTERN = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?_\d{7}$")


def validate_arxiv_id(value: str) -> str:
    """Kiểm tra arxiv_id đúng dạng chuẩn (không có hậu tố version).

    Dùng fullmatch thay vì match: "^...$" với match vẫn chấp nhận "2310.12345\n" (hay gặp
    khi đọc ID từ file văn bản), còn fullmatch thì chặn được.

    Args:
        value: Chuỗi ID cần kiểm tra.

    Returns:
        Chính chuỗi đó nếu hợp lệ.

    Raises:
        ValueError: Khi ID không khớp dạng mới lẫn dạng cũ.
    """
    if not (NEW_ID_PATTERN.fullmatch(value) or OLD_ID_PATTERN.fullmatch(value)):
        raise ValueError(f"arXiv ID không hợp lệ: {value!r}")
    return value


def to_folder_name(arxiv_id: str) -> str:
    """Đổi arxiv_id chuẩn sang tên folder (dạng "yymm-id" của đề).

    Validate đầu vào để lỗi nổ ngay tại đây, thay vì âm thầm tạo folder sai ("", "../x",
    "2310-12345v2"...) rồi làm lệch phép join dữ liệu ở bước sau.

    Args:
        arxiv_id: Mã chuẩn, ví dụ "2310.12345" hoặc "hep-th/9901001".

    Returns:
        "2310-12345" hoặc "hep-th_9901001" (dấu "/" không hợp lệ trong tên folder).

    Raises:
        ValueError: Khi arxiv_id không hợp lệ.
    """
    validate_arxiv_id(arxiv_id)
    if "/" in arxiv_id:
        return arxiv_id.replace("/", "_")
    return arxiv_id.replace(".", "-")


def from_folder_name(folder_name: str) -> str:
    """Đổi ngược tên folder về arxiv_id chuẩn (dùng cho key của references.json).

    Kiểm tra DẠNG folder bằng regex thay vì đoán theo ký tự "_": nhờ vậy "2310.12345"
    (đã là ID chuẩn) hay "a_b_c" (rác) bị từ chối thay vì "đổi" ra kết quả sai.

    Args:
        folder_name: "2310-12345" hoặc "hep-th_9901001".

    Returns:
        "2310.12345" hoặc "hep-th/9901001".

    Raises:
        ValueError: Khi tên folder không đúng dạng.
    """
    if OLD_FOLDER_PATTERN.fullmatch(folder_name):
        return folder_name.replace("_", "/")
    if NEW_FOLDER_PATTERN.fullmatch(folder_name):
        return folder_name.replace("-", ".", 1)
    raise ValueError(f"Tên folder không hợp lệ: {folder_name!r}")