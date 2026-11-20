"""Các giới hạn kích thước dùng chung của M1 (tải về và giải nén).

Tách riêng để ``downloader.py`` và ``extractor.py`` cùng dùng mà không phải import lẫn nhau.

``crawler.py`` là "mặt tiền" cũ: code và test lâu nay chỉnh các hằng số này qua
``crawler.MAX_DOWNLOAD_BYTES``... Hàm ``limit`` vì vậy ưu tiên giá trị đang đặt trên module
``crawler`` (nếu module đó đã được nạp), để chỉnh ở đó vẫn có tác dụng sau khi tách file.
"""

import sys

MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024  # chặn file tải về quá lớn
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # chặn "zip bomb"
MAX_KEPT_FILE_BYTES = 50 * 1024 * 1024  # một file .tex/.bib lớn hơn mức này bị bỏ qua

_FACADE = "src.m1_crawling.crawler"


def limit(name: str) -> int:
    """Giá trị hiện hành của một giới hạn.

    Args:
        name: Tên hằng số, ví dụ ``"MAX_DOWNLOAD_BYTES"``.

    Returns:
        Giá trị trên module ``crawler`` nếu module đó đã nạp và có hằng số này, ngược lại là
        giá trị mặc định khai báo ở file này.
    """
    default = globals()[name]
    facade = sys.modules.get(_FACADE)
    return int(getattr(facade, name, default)) if facade is not None else int(default)
