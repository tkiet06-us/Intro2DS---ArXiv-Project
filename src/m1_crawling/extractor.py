"""Trích xuất ``.tex``/``.bib`` từ gói nguồn arXiv (tách khỏi ``crawler.py`` trong đợt tối ưu M1).

Hai đường cùng một bộ quy tắc:

- ``extract_from_path(path, dest_tmp)``: đường CHÍNH. Đọc gói theo luồng từ đĩa (tar mở ở chế độ
  tuần tự ``r|``, không seek) và ghi từng file giữ lại THẲNG vào thư mục tạm, nên RAM không phụ
  thuộc kích thước gói. File ``.gz`` đơn được giải nén theo khối.
- ``extract_tex_sources(data)``: đường cũ, nhận cả gói trong RAM và trả nội dung trong dict. Giữ
  lại cho code/test cũ và làm chuẩn đối chiếu cho đường mới (test hồi quy).

Quy tắc dữ liệu (không đổi):
    - Nhận diện theo NỘI DUNG: PDF, ``.tar.gz``, ``.tar``, ``.gz`` đơn, zip/Word, văn bản thô.
    - Chỉ giữ ``.tex`` và ``.bib``; ảnh và mọi file khác không bao giờ vào corpus. Nội dung file
      được chép NGUYÊN VĂN từng byte.
    - Bỏ symlink/hardlink/thiết bị, đường dẫn tuyệt đối hay có ``..``, rác macOS, file trùng tên
      không phân biệt hoa thường, file giữ lại lớn hơn ``MAX_KEPT_FILE_BYTES``.
    - Chỉ có ``.bib`` mà không có ``.tex`` thì là record rỗng ``no_tex``.
    - Tổng dung lượng giải nén vượt ``MAX_UNCOMPRESSED_BYTES`` thì ``too_large``.
    - Tar hỏng giữa chừng hoặc gzip hỏng thì ``corrupt_archive`` (crawler sẽ thử tải lại).
"""

import gzip
import io
import logging
import os
import re
import shutil
import tarfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from src.m1_crawling.limits import MAX_KEPT_FILE_BYTES, MAX_UNCOMPRESSED_BYTES, limit

logger = logging.getLogger(__name__)

__all__ = [
    "KEEP_EXTENSIONS",
    "MAX_KEPT_FILE_BYTES",
    "MAX_UNCOMPRESSED_BYTES",
    "SINGLE_FILE_NAME",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "STATUS_OK",
    "Extracted",
    "extract_from_path",
    "extract_tex_sources",
    "staging_dir",
    "write_version",
]

KEEP_EXTENSIONS = frozenset({".tex", ".bib"})  # ảnh và mọi loại khác không vào corpus
SINGLE_FILE_NAME = "main.tex"  # tên file khi nguồn là một .gz đơn
COPY_BYTES = 1 << 20  # kích thước khối khi chép/giải nén theo luồng

STATUS_OK = "ok"  # có ít nhất một .tex
STATUS_EMPTY = "empty"  # record rỗng: version không có TeX
STATUS_ERROR = "error"  # lỗi tạm thời

# Nhận diện "trông giống TeX" bằng marker thật, không chỉ có ký tự "\".
_TEX_MARKERS = re.compile(
    rb"\\(documentclass|documentstyle|begin|end|input|include|section|usepackage|def|"
    rb"newcommand|magnification|font|hsize|catcode|bye)\b"
)
_BINARY_MAGIC = (b"%!PS", b"%PDF", b"\x89PNG", b"GIF8", b"\xff\xd8")  # PS, PDF, PNG, GIF, JPEG
_MARKER_SCAN_BYTES = 200_000  # chỉ tìm marker trong 200KB đầu
_BAD_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')  # ký tự Windows không cho phép
# Lỗi khi ĐỌC gói (tar/gzip hỏng). Tách bạch với lỗi GHI đĩa: đĩa đầy không được coi là gói hỏng.
_READ_ERRORS = (tarfile.ReadError, EOFError, OSError, zlib.error)


class _BrokenArchiveError(Exception):
    """Gói hỏng khi đang đọc (đã đọc được một phần)."""


@dataclass
class Extracted:
    """Kết quả phân tích một nguồn đã tải về.

    Attributes:
        status: ``ok`` nếu có ít nhất một ``.tex``, ngược lại ``empty``.
        reason: Lý do khi ``empty`` (``pdf_only``, ``no_tex``, ``not_tex``...), ``None`` khi ok.
        files: Đường dẫn tương đối (dấu ``/``) -> nội dung THÔ (bytes). Chỉ đường cũ
            (``extract_tex_sources``) điền trường này.
        size_before: Tổng dung lượng mọi file trong archive (đã giải nén), gồm cả ảnh.
        n_files_before: Số file thường trong archive.
        skipped: Các file ``.tex``/``.bib`` bị bỏ vì không an toàn, quá lớn hoặc trùng tên.
        sizes: Đường dẫn tương đối -> số byte của file ĐÃ GHI XUỐNG ĐĨA. Chỉ đường mới
            (``extract_from_path``) điền trường này.
    """

    status: str
    reason: str | None = None
    files: dict[str, bytes] = field(default_factory=dict)
    size_before: int = 0
    n_files_before: int = 0
    skipped: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def kept(self) -> dict[str, int]:
        """Đường dẫn tương đối -> số byte của các file giữ lại (dùng được cho cả hai đường)."""
        return self.sizes or {name: len(content) for name, content in self.files.items()}

    @property
    def size_after(self) -> int:
        """Tổng dung lượng các file đã giữ lại."""
        return sum(self.kept.values())


# ---------------------------------------------------------------------------
# Quy tắc dùng chung
# ---------------------------------------------------------------------------


def _safe_relpath(name: str) -> str | None:
    """Chuẩn hóa tên file trong archive thành đường dẫn tương đối an toàn.

    Chặn đường dẫn tuyệt đối, ổ đĩa (``C:``), ``..`` (path traversal) và file rác của macOS
    (``__MACOSX``, ``._x``). Thay ký tự Windows không hợp lệ bằng ``_`` để ghi được trên
    máy Windows.

    Args:
        name: Tên member trong archive.

    Returns:
        Đường dẫn tương đối dùng dấu ``/``, hoặc None nếu phải bỏ file này.
    """
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute():
        return None
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    if parts[0] == "__MACOSX" or any(part.startswith("._") for part in parts):
        return None
    cleaned = [_BAD_FILENAME_CHARS.sub("_", part).rstrip(". ") for part in parts]
    if any(not part for part in cleaned):
        return None
    original, renamed = "/".join(parts), "/".join(cleaned)
    if renamed != original:
        # File bị đổi tên trên đĩa nhưng \input{...} trong .tex vẫn trỏ tên gốc: M2 sẽ không
        # resolve được file này. Chỉ cảnh báo để truy vết, không tự sửa nội dung .tex.
        logger.warning(
            "Doi ten file co ky tu khong hop le: %r -> %r (co the lam hong \\input/\\include)",
            original,
            renamed,
        )
    return renamed


def _single_file_reason(prefix: bytes) -> str | None:
    """Phân loại nguồn là MỘT file dựa trên phần đầu (tối đa ``_MARKER_SCAN_BYTES`` byte).

    Returns:
        None nếu trông giống LaTeX (giữ lại thành ``main.tex``), ngược lại là lý do record rỗng.
    """
    head = prefix[:4096]
    stripped = head.lstrip()[:15].lower()
    if stripped.startswith((b"<!doctype", b"<html", b"<?xml")):
        # Nghi trang lỗi của server, KHÔNG phải bài thật: crawler sẽ thử lại thay vì chốt ngay.
        return "html_response"
    if head.startswith(b"%PDF-"):  # PDF bọc trong gzip
        return "pdf_only"
    # Loại theo magic bytes VÀ bắt buộc có ít nhất một marker TeX/LaTeX trong phần đầu file
    # (PostScript hay file text bất kỳ có dấu "\" không được ghi thành main.tex).
    if (
        b"\x00" in head
        or head.startswith(_BINARY_MAGIC)
        or not _TEX_MARKERS.search(prefix[:_MARKER_SCAN_BYTES])
    ):
        return "not_tex"
    return None


def _member_decision(name: str, size: int, seen: set[str]) -> tuple[str | None, bool]:
    """Quyết định số phận một member ``.tex``/``.bib`` của tar.

    Returns:
        (relpath, True) nếu giữ; (None, True) nếu phải ghi vào ``skipped``; (None, False) nếu
        phần mở rộng không thuộc ``KEEP_EXTENSIONS`` (bỏ qua im lặng).
    """
    if PurePosixPath(name).suffix.lower() not in KEEP_EXTENSIONS:
        return None, False  # ảnh, .sty, .cls, .pdf... không bao giờ vào corpus
    relpath = _safe_relpath(name)
    if relpath is None or size > limit("MAX_KEPT_FILE_BYTES") or relpath.lower() in seen:
        return None, True
    return relpath, True


def _has_tex(names: "dict[str, object]") -> bool:
    return any(PurePosixPath(name).suffix.lower() == ".tex" for name in names)


# ---------------------------------------------------------------------------
# Đường cũ: cả gói trong RAM
# ---------------------------------------------------------------------------


def _from_single_file(raw: bytes) -> Extracted:
    """Nguồn là MỘT file (``.gz`` đơn đã giải nén, hoặc văn bản thô).

    Returns:
        ``ok`` với file ``main.tex`` nếu trông giống LaTeX, ngược lại record rỗng.
    """
    reason = _single_file_reason(raw[:_MARKER_SCAN_BYTES])
    if reason is not None:
        return Extracted(STATUS_EMPTY, reason, size_before=len(raw), n_files_before=1)
    return Extracted(
        STATUS_OK, files={SINGLE_FILE_NAME: raw}, size_before=len(raw), n_files_before=1
    )


def _from_tar(tar: tarfile.TarFile) -> Extracted:
    """Duyệt archive tar, chỉ đọc nội dung ``.tex`` và ``.bib`` (vào RAM).

    Returns:
        Kết quả đã lọc. Archive hỏng GIỮA CHỪNG (đã đọc được ít nhất một member) trả
        ``corrupt_archive``.

    Raises:
        tarfile.ReadError, EOFError, OSError, zlib.error: Khi không đọc nổi member nào.
    """
    result = Extracted(STATUS_EMPTY, "no_tex")
    seen: set[str] = set()
    members_read = 0
    try:
        for member in tar:
            members_read += 1
            if not member.isfile():
                continue
            result.n_files_before += 1
            result.size_before += member.size
            if result.size_before > limit("MAX_UNCOMPRESSED_BYTES"):
                return Extracted(STATUS_EMPTY, "too_large", size_before=result.size_before)
            relpath, wanted = _member_decision(member.name, member.size, seen)
            if relpath is None:
                if wanted:
                    result.skipped.append(member.name)
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            seen.add(relpath.lower())  # so khớp không phân biệt hoa thường (Windows)
            result.files[relpath] = handle.read()
    except _READ_ERRORS as exc:
        if members_read == 0:
            raise  # chưa đọc được gì: không phải tar (hoặc hỏng từ đầu), để caller phân loại
        logger.warning("Tar hong giua chung (da doc %d member): %s", members_read, exc)
        return Extracted(
            STATUS_EMPTY,
            "corrupt_archive",
            size_before=result.size_before,
            n_files_before=result.n_files_before,
        )
    if _has_tex(result.files):
        result.status, result.reason = STATUS_OK, None
    else:
        # Chỉ có .bib mà không có .tex: không phải bài LaTeX dùng được -> record rỗng.
        result.files.clear()
    return result


def _gunzip_limited(data: bytes) -> bytes | None:
    """Giải nén gzip nhưng dừng khi vượt giới hạn (chống bom nén).

    Returns:
        Nội dung đã giải nén, hoặc None nếu vượt ``MAX_UNCOMPRESSED_BYTES``.
    """
    cap = limit("MAX_UNCOMPRESSED_BYTES")
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        out = handle.read(cap + 1)
    return None if len(out) > cap else out


def extract_tex_sources(data: bytes) -> Extracted:
    """Phân tích nguồn arXiv nằm trong RAM và lọc ra ``.tex``/``.bib`` (đường cũ).

    Nhận diện theo NỘI DUNG chứ không theo header HTTP. Không bao giờ raise vì dữ liệu hỏng:
    trả record rỗng kèm lý do. Pipeline crawl dùng ``extract_from_path``; hàm này dành cho dữ
    liệu nhỏ đã có sẵn trong RAM và làm chuẩn đối chiếu trong test.

    Args:
        data: Toàn bộ byte của gói.

    Returns:
        Kết quả đã lọc, nội dung nằm trong ``files``, chưa ghi ra đĩa.
    """
    if data[:5] == b"%PDF-":
        return Extracted(STATUS_EMPTY, "pdf_only", size_before=len(data), n_files_before=1)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            return _from_tar(tar)
    except tarfile.ReadError:
        pass  # không phải tar: thử .gz đơn / văn bản thô bên dưới
    except (EOFError, OSError, zlib.error) as exc:
        logger.warning("Archive hong: %s", exc)
        return Extracted(STATUS_EMPTY, "corrupt_archive", size_before=len(data))
    if data[:2] == b"\x1f\x8b":
        try:
            raw = _gunzip_limited(data)
        except (EOFError, OSError, zlib.error) as exc:
            logger.warning("File gz hong: %s", exc)
            return Extracted(STATUS_EMPTY, "corrupt_archive", size_before=len(data))
        if raw is None:
            return Extracted(STATUS_EMPTY, "too_large", size_before=len(data))
        return _from_single_file(raw)
    if data[:2] == b"PK":  # zip/docx: bài nộp dạng Word
        return Extracted(STATUS_EMPTY, "word_or_zip", size_before=len(data), n_files_before=1)
    return _from_single_file(data)


# ---------------------------------------------------------------------------
# Đường mới: đọc theo luồng từ đĩa, ghi thẳng vào thư mục tạm
# ---------------------------------------------------------------------------


def _clear_dir(folder: Path) -> None:
    """Xóa mọi thứ BÊN TRONG ``folder`` (giữ lại chính thư mục, tạo nếu chưa có)."""
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _read_block(reader: BinaryIO, size: int) -> bytes:
    """Đọc một khối; lỗi đọc gói được đổi thành ``_BrokenArchiveError``."""
    try:
        return reader.read(size)
    except _READ_ERRORS as exc:
        raise _BrokenArchiveError(str(exc)) from exc


def _read_prefix(reader: BinaryIO, size: int) -> bytes:
    """Đọc đúng ``size`` byte đầu (ít hơn nếu hết file)."""
    blocks: list[bytes] = []
    remaining = size
    while remaining > 0:
        block = _read_block(reader, remaining)
        if not block:
            break
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _tar_to_dir(tar: tarfile.TarFile, dest_tmp: Path) -> Extracted:
    """Duyệt tar TUẦN TỰ, ghi từng ``.tex``/``.bib`` thẳng vào ``dest_tmp``.

    Cùng quy tắc với ``_from_tar``. Lỗi ghi đĩa (đĩa đầy, tên file xung đột...) được ném ra
    nguyên vẹn để crawler ghi ``error`` kèm tên lỗi; chỉ lỗi ĐỌC gói mới thành ``corrupt_archive``.
    """
    result = Extracted(STATUS_EMPTY, "no_tex")
    seen: set[str] = set()
    members = iter(tar)
    try:
        while True:
            try:
                member = next(members)
            except StopIteration:
                break
            except _READ_ERRORS as exc:
                raise _BrokenArchiveError(str(exc)) from exc
            if not member.isfile():
                continue
            result.n_files_before += 1
            result.size_before += member.size
            if result.size_before > limit("MAX_UNCOMPRESSED_BYTES"):
                _clear_dir(dest_tmp)
                return Extracted(STATUS_EMPTY, "too_large", size_before=result.size_before)
            relpath, wanted = _member_decision(member.name, member.size, seen)
            if relpath is None:
                if wanted:
                    result.skipped.append(member.name)
                continue
            try:
                handle = tar.extractfile(member)
            except _READ_ERRORS as exc:
                raise _BrokenArchiveError(str(exc)) from exc
            if handle is None:
                continue
            seen.add(relpath.lower())  # so khớp không phân biệt hoa thường (Windows)
            target = dest_tmp / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with target.open("wb") as out:
                while True:
                    block = _read_block(handle, COPY_BYTES)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
            if written != member.size:
                raise _BrokenArchiveError(f"{member.name}: doc {written}/{member.size} byte")
            result.sizes[relpath] = written
    except _BrokenArchiveError as exc:
        logger.warning("Tar hong giua chung (%d file da doc): %s", result.n_files_before, exc)
        _clear_dir(dest_tmp)
        return Extracted(
            STATUS_EMPTY,
            "corrupt_archive",
            size_before=result.size_before,
            n_files_before=result.n_files_before,
        )
    if _has_tex(result.sizes):
        result.status, result.reason = STATUS_OK, None
    else:
        # Chỉ có .bib mà không có .tex: không phải bài LaTeX dùng được -> record rỗng.
        result.sizes.clear()
        _clear_dir(dest_tmp)
    return result


def _single_to_dir(
    reader: BinaryIO, dest_tmp: Path, *, source_size: int, max_bytes: int | None
) -> Extracted:
    """Nguồn là MỘT file: phân loại bằng phần đầu rồi chép theo khối vào ``main.tex``.

    Args:
        reader: Luồng nội dung (đã giải nén nếu nguồn là ``.gz``).
        dest_tmp: Thư mục tạm nhận ``main.tex``.
        source_size: Kích thước file nguồn trên đĩa (ghi vào ``size_before`` khi hỏng/quá lớn).
        max_bytes: Trần số byte sau giải nén; None là không giới hạn (nguồn không nén).
    """
    try:
        prefix = _read_prefix(reader, _MARKER_SCAN_BYTES)
        reason = _single_file_reason(prefix)
        total = len(prefix)
        target = dest_tmp / SINGLE_FILE_NAME
        out = target.open("wb") if reason is None else None
        try:
            if out is not None:
                out.write(prefix)
            while max_bytes is None or total <= max_bytes:
                block = _read_block(reader, COPY_BYTES)
                if not block:
                    break
                total += len(block)
                if out is not None:
                    out.write(block)
        finally:
            if out is not None:
                out.close()
    except _BrokenArchiveError as exc:
        logger.warning("File gz hong: %s", exc)
        _clear_dir(dest_tmp)
        return Extracted(STATUS_EMPTY, "corrupt_archive", size_before=source_size)
    if max_bytes is not None and total > max_bytes:
        _clear_dir(dest_tmp)
        return Extracted(STATUS_EMPTY, "too_large", size_before=source_size)
    if reason is not None:
        return Extracted(STATUS_EMPTY, reason, size_before=total, n_files_before=1)
    return Extracted(
        STATUS_OK, sizes={SINGLE_FILE_NAME: total}, size_before=total, n_files_before=1
    )


def extract_from_path(path: Path, dest_tmp: Path) -> Extracted:
    """Phân tích gói nguồn nằm trên đĩa và ghi ``.tex``/``.bib`` thẳng vào ``dest_tmp``.

    Đọc theo luồng nên RAM không phụ thuộc kích thước gói. Kết quả (``status``, ``reason``, tên
    file, nội dung, ``size_before``, ``n_files_before``, ``skipped``) giống hệt
    ``extract_tex_sources`` trên cùng dãy byte. Không raise vì GÓI hỏng; lỗi ghi đĩa thì raise.

    Args:
        path: File gói đã tải xong.
        dest_tmp: Thư mục tạm nhận các file giữ lại (tạo nếu chưa có). Khi kết quả không phải
            ``ok`` thì thư mục này rỗng.

    Returns:
        Kết quả đã lọc; tên và kích thước file giữ lại nằm trong ``sizes``.
    """
    path = Path(path)
    dest_tmp.mkdir(parents=True, exist_ok=True)
    source_size = path.stat().st_size
    with path.open("rb") as handle:
        magic = handle.read(5)
    if magic == b"%PDF-":
        return Extracted(STATUS_EMPTY, "pdf_only", size_before=source_size, n_files_before=1)
    is_gzip = magic[:2] == b"\x1f\x8b"
    try:
        if is_gzip:
            # Tự giải nén bằng GzipFile (kiểm CRC và dấu kết thúc) rồi mới đưa cho tarfile đọc
            # tuần tự: lớp giải nén riêng của chế độ "r|gz" âm thầm chấp nhận file bị cắt cụt.
            with gzip.GzipFile(filename=path) as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
                return _tar_to_dir(tar, dest_tmp)
        with tarfile.open(name=path, mode="r|*") as tar:
            return _tar_to_dir(tar, dest_tmp)
    except tarfile.ReadError:
        pass  # không phải tar: thử .gz đơn / văn bản thô bên dưới
    except (EOFError, zlib.error, gzip.BadGzipFile) as exc:
        logger.warning("Archive hong: %s", exc)
        _clear_dir(dest_tmp)
        return Extracted(STATUS_EMPTY, "corrupt_archive", size_before=source_size)
    _clear_dir(dest_tmp)
    if is_gzip:
        with gzip.GzipFile(filename=path) as raw:
            return _single_to_dir(
                raw,
                dest_tmp,
                source_size=source_size,
                max_bytes=limit("MAX_UNCOMPRESSED_BYTES"),
            )
    if magic[:2] == b"PK":  # zip/docx: bài nộp dạng Word
        return Extracted(STATUS_EMPTY, "word_or_zip", size_before=source_size, n_files_before=1)
    with path.open("rb") as raw:
        return _single_to_dir(raw, dest_tmp, source_size=source_size, max_bytes=None)


# ---------------------------------------------------------------------------
# Ghi một version ra đĩa (nguyên tử)
# ---------------------------------------------------------------------------


def staging_dir(folder: Path, version: int) -> Path:
    """Tạo thư mục tạm cho ``folder/v{n}`` (cùng ổ đĩa để đổi tên là thao tác nguyên tử).

    Tên UNIQUE mỗi lần, nên không bao giờ phải xóa tmp cũ (tránh vòng lặp "tmp cũ bị
    Defender/Indexer lock -> rmtree fail -> lần sau lại fail").
    """
    tmp = folder / f".v{version}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    tmp.mkdir(parents=True)
    return tmp


def write_version(folder: Path, version: int, files: "dict[str, bytes] | Path") -> None:
    """Ghi một version ra ``folder/v{n}/`` theo kiểu nguyên tử.

    Nội dung nằm trong thư mục tạm rồi mới đổi tên, nên chương trình bị ngắt giữa chừng không
    để lại thư mục ``v{n}`` ghi dở.

    Args:
        folder: Thư mục của bài.
        version: Số version.
        files: Dict đường dẫn tương đối -> nội dung (rỗng thì tạo thư mục rỗng, tức "record
            rỗng"); HOẶC đường dẫn một thư mục tạm đã chứa sẵn nội dung (tạo bằng
            ``staging_dir`` và điền bằng ``extract_from_path``), khi đó hàm chỉ việc đổi tên.

    Raises:
        OSError: Khi ghi/đổi tên thất bại sau khi đã thử lại. Thư mục tạm được dọn.
    """
    final = folder / f"v{version}"
    tmp = files if isinstance(files, Path) else staging_dir(folder, version)
    try:
        if not isinstance(files, Path):
            for relpath, content in files.items():
                target = tmp / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        if final.exists():
            _rmtree_with_retry(final)
        # Retry os.replace: Windows Defender/Indexer có thể lock tmp trong vài giây.
        last_error: Exception | None = None
        for wait in (0.5, 1.0, 2.0, 4.0):
            try:
                tmp.replace(final)
                return
            except PermissionError as exc:
                last_error = exc
                logger.warning(
                    "PermissionError khi doi ten %s -> %s, thu lai sau %.1fs", tmp, final, wait
                )
                time.sleep(wait)
        raise last_error  # type: ignore[misc]
    except Exception:
        _rmtree_with_retry(tmp, ignore_errors=True)
        raise


def _rmtree_with_retry(path: Path, *, ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` có retry cho Windows/Defender/Indexer lock.

    Args:
        path: Thư mục cần xóa (không tồn tại thì bỏ qua).
        ignore_errors: True thì nuốt lỗi cuối cùng (dùng khi dọn tmp trong except).
    """
    if not path.exists():
        return
    last_error: Exception | None = None
    for wait in (0.5, 1.0, 2.0, 4.0):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(wait)
    if not ignore_errors:
        raise last_error  # type: ignore[misc]
