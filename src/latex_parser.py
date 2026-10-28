"""Gộp mã nguồn LaTeX của một version thành một chuỗi.

Theo đề M2: gốc biên dịch là file .tex chứa \documentclass; từ đó
lần theo \input / \include một cách bắc cầu. Lệnh nằm trong comment chỉ là văn bản thường.
File không nằm trên đường biên dịch bị bỏ qua. Đầu ra là chuỗi LaTeX gộp, vẫn còn phần
References để bibtex_extractor dùng; tree_builder sẽ tự cắt phần này khi dựng cây.

Dùng:
    text = resolve_paper(Path("data/2310-12345/v1"))
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Mỗi dòng: gom phần trước dấu % chưa bị escape
_COMMENT_LINE = re.compile(r"^((?:[^%\\\n]|\\.)*)%.*$", re.MULTILINE)
_DOCUMENTCLASS = re.compile(r"\\documentclass\b")
_BEGIN_DOCUMENT = re.compile(r"\\begin\{document\}")
# \input{a}, \include{a}, \subfile{a} hoặc dạng không ngoặc "\input a.tex".
_INCLUDE = re.compile(r"\\(?:input|include|subfile)(?![A-Za-z])\s*(?:\{([^{}]+)\}|([^\s{}\\]+))")


def strip_comments(text: str) -> str:
    """Xóa comment `%...` đến hết dòng, giữ nguyên `\\%`.

    Args:
        text: Mã nguồn LaTeX thô.

    Returns:
        Văn bản không còn comment (ký tự xuống dòng được giữ lại).
    """
    return _COMMENT_LINE.sub(r"\1", text)


def _read(path: Path) -> str:
    """Đọc file .tex chịu lỗi mã hóa (nhiều bài cũ dùng latin-1)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _locate(base: Path, name: str) -> Path | None:
    """Tìm file được include, thử có/không đuôi .tex, tính từ thư mục `base`.

    Args:
        base: Thư mục chứa file gốc (LaTeX tính đường dẫn tương đối từ đây).
        name: Tên trong \\input{...}.

    Returns:
        Đường dẫn file tồn tại, hoặc None.
    """
    name = name.strip()
    for candidate in (base / name, base / f"{name}.tex"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_main_tex(src_dir: Path) -> Path | None:
    """Chọn file gốc biên dịch trong thư mục nguồn của một version.

    Ưu tiên: có \\begin{document} > không bị file nào khác include > ở nông hơn > dài hơn.

    Args:
        src_dir: Thư mục chứa các .tex đã giải nén của một version.

    Returns:
        File gốc, hoặc None nếu không có .tex nào chứa \\documentclass.
    """
    texts: dict[Path, str] = {}
    for path in sorted(src_dir.rglob("*.tex")):
        text = strip_comments(_read(path))
        if _DOCUMENTCLASS.search(text):
            texts[path.resolve()] = text
    if not texts:
        return None

    included: set[str] = set()
    for text in texts.values():
        for match in _INCLUDE.finditer(text):
            included.add(Path((match.group(1) or match.group(2)).strip()).stem)

    def rank(item: tuple[Path, str]) -> tuple:
        path, text = item
        return (
            not _BEGIN_DOCUMENT.search(text),  # False (có document) xếp trước
            path.stem in included,
            len(path.relative_to(src_dir.resolve()).parts),
            -len(text),
        )

    return min(texts.items(), key=rank)[0]


def _expand(path: Path, base: Path, stack: frozenset[Path]) -> str:
    """Gộp đệ quy một file, chặn include vòng bằng tập `stack` các file đang mở.

    Args:
        path: File đang xử lý.
        base: Thư mục gốc để phân giải đường dẫn include.
        stack: Các file trên đường đi hiện tại (để phát hiện chu trình).

    Returns:
        Nội dung file với mọi include hợp lệ đã được thay bằng nội dung file con.
    """
    text = strip_comments(_read(path))

    def replace(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        target = _locate(base, name)
        if target is None:
            logger.warning("%s: \\input{%s} trỏ tới file không tồn tại, bỏ qua", path.name, name)
            return " "
        if target in stack:
            logger.warning("%s: include vòng tới %s, bỏ qua", path.name, target.name)
            return " "
        return "\n" + _expand(target, base, stack | {target}) + "\n"

    return _INCLUDE.sub(replace, text)


def resolve_paper(src_dir: Path) -> str:
    """Gộp toàn bộ nguồn LaTeX của một version thành một chuỗi duy nhất.

    Args:
        src_dir: Thư mục version, ví dụ data/2310-12345/v1. Thư mục rỗng (bài PDF-only)
            là "record rỗng" hợp lệ.

    Returns:
        Chuỗi LaTeX đã gộp; chuỗi rỗng nếu version không có TeX.
    """
    main = find_main_tex(src_dir)
    if main is None:
        logger.warning("%s: không có file .tex nào chứa \\documentclass", src_dir)
        return ""
    return _expand(main, main.parent, frozenset({main}))