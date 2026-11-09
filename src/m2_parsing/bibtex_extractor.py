"""P2-04 – Trích xuất và chuẩn hóa BibTeX từ ``\\bibitem`` hoặc file ``.bib``.

Module nhận chuỗi LaTeX đã gộp (còn nguyên phần References) và/hoặc thư mục nguồn của
một version, rồi trả về danh sách entry phẳng đúng các trường của ``src.schemas.Bibtex``
(trừ ``arxiv_id`` do caller gắn). Ngoài ra module có helper ghi ``refs.bib`` chuẩn hóa.

Hai dạng nguồn theo ``Docs/latex_spec.md`` mục 1.2 và giả thuyết H6:

1. ``\\begin{thebibliography} ... \\bibitem{key} ... \\end{thebibliography}``
2. ``\\bibliography{refs}`` / ``\\addbibresource{refs.bib}`` + file ``.bib`` đi kèm

Tham khảo cú pháp BibTeX: ``@type{key, field = {value}, ...}`` với giá trị có thể dùng
ngoặc nhọn lồng nhau hoặc nháy kép.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from src.m2_parsing.latex_parser import strip_comments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex và hằng số
# ---------------------------------------------------------------------------
_BIBITEM_RE = re.compile(
    r"\\bibitem\b\s*(?:\[[^\]]*\]\s*)?\{(?P<key>[^{}]+)\}",
    re.IGNORECASE,
)
_THEBIBLIOGRAPHY_END_RE = re.compile(r"\\end\s*\{\s*thebibliography\s*\}", re.IGNORECASE)
_DOCUMENT_END_RE = re.compile(r"\\end\s*\{\s*document\s*\}", re.IGNORECASE)
_BIB_RESOURCE_RE = re.compile(r"\\(?:bibliography|addbibresource)\s*\{([^{}]+)\}", re.IGNORECASE)

# Năm 4 chữ số, không dính vào chữ số khác (tránh cắt "20201").
_YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20|21)\d{2})[a-z]?(?!\d)", re.IGNORECASE)

_DOI_RE = re.compile(
    r"(?:\bdoi\s*:\s*|https?://(?:dx\.)?doi\.org/|\\doi\s*\{\s*)"
    r"(?P<doi>10\.\d{4,9}/[^\s{}<>\"']+)",
    re.IGNORECASE,
)
_BARE_DOI_RE = re.compile(r"(?<![\w/])(?P<doi>10\.\d{4,9}/[^\s{}<>\"']+)", re.IGNORECASE)
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", re.IGNORECASE)

_URL_COMMAND_RE = re.compile(r"\\url\s*\{([^{}]*)\}", re.IGNORECASE)
_HREF_COMMAND_RE = re.compile(r"\\href\s*\{([^{}]*)\}\s*\{([^{}]*)\}", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"https?://[^\s{}<>\"']+", re.IGNORECASE)

_BIBINFO_RE = re.compile(r"\\(?:bibinfo|bibfield)\s*\{([^{}]+)\}\s*\{", re.IGNORECASE)
_EMPHASIS_START_RE = re.compile(
    r"\\(?:emph|textit|textbf|textsc|texttt|textrm|textsf|underline)\s*\{",
    re.IGNORECASE,
)
_TEX_EM_START_RE = re.compile(r"\{\s*\\(?:em|it|bf|sl|sc)\s+", re.IGNORECASE)
_FORMAT_START_RE = re.compile(
    r"\\(?:emph|textit|textbf|textsc|texttt|textrm|textsf|underline|mbox|url)\s*\{",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(r"\\[A-Za-z@]+\*?")
_LATEX_ESCAPES = {
    r"\&": "&",
    r"\%": "%",
    r"\$": "$",
    r"\#": "#",
    r"\_": "_",
    r"\{": "{",
    r"\}": "}",
    r"\~": " ",
    r"\ ": " ",
    "~": " ",
    "---": "—",
    "--": "–",
}

# Entry trong file .bib: @article{key, ...} hoặc @article(key, ...)
_ENTRY_START_RE = re.compile(
    r"@\s*(?P<type>[A-Za-z]+)\s*(?P<open>[{(])\s*(?P<key>[^,\s{}()]+)\s*,",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(r"(?P<name>[A-Za-z][\w-]*)\s*=\s*", re.IGNORECASE)

# Dạng APA thường gặp ở \bibitem không có \emph: "Lee, B. (2019). Graph nets."
_APA_RE = re.compile(
    r"^(?P<authors>.+?)\s*\(\s*(?:18|19|20|21)\d{2}[a-z]?\s*\)\s*[.,]?\s*(?P<rest>.*)$",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Tiện ích cú pháp
# ---------------------------------------------------------------------------
def _read_delimited(text: str, start: int, open_char: str = "{", close_char: str = "}") -> tuple[str, int]:
    """Đọc nội dung giữa cặp ngoặc, tính cả ngoặc lồng nhau.

    Args:
        text: Chuỗi nguồn.
        start: Vị trí ngay sau ngoặc mở.
        open_char: Ký tự mở (mặc định ``{``).
        close_char: Ký tự đóng (mặc định ``}``).

    Returns:
        (nội dung bên trong, vị trí ngay sau ngoặc đóng). Thiếu ngoặc đóng thì lấy đến hết.
    """
    depth, index = 1, start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start:index], index + 1
        index += 1
    return text[start:], len(text)


def _read_braced(text: str, start: int) -> tuple[str, int]:
    """Đọc nội dung nhóm ``{...}`` có lồng nhau; ``start`` nằm ngay sau ``{``."""
    return _read_delimited(text, start, "{", "}")


def _read_quoted(text: str, start: int) -> tuple[str, int]:
    """Đọc chuỗi trong nháy kép kiểu BibTeX; ``start`` nằm ngay sau ``"``."""
    pieces: list[str] = []
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            pieces.append(text[index : index + 2])
            index += 2
            continue
        if char == '"':
            return "".join(pieces), index + 1
        pieces.append(char)
        index += 1
    return "".join(pieces), len(text)


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    """Tách theo ``separator`` ở độ sâu ngoặc 0, bỏ qua dấu đã escape."""
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return parts


def _strip_comments(text: str) -> str:
    """Xóa comment nhưng giữ ``\\%`` (dùng chung quy tắc với latex_parser)."""
    return strip_comments(text)


def _latex_to_text(text: str, *, strip_period: bool = True) -> str:
    """Chuyển các macro định dạng thường gặp và escape TeX thành văn bản đọc được.

    Args:
        text: Chuỗi LaTeX.
        strip_period: Xóa dấu chấm cuối (đúng cho title/journal). Với tên tác giả phải
            giữ lại vì ``J.``/``A.`` là viết tắt, không phải dấu hết câu.
    """
    # \bibinfo{field}{value}: giữ value, bỏ tên field (IEEEtran hay dùng).
    while match := _BIBINFO_RE.search(text):
        value, end = _read_braced(text, match.end())
        text = text[: match.start()] + value + text[end:]

    # \href{url}{text}: giữ text hiển thị.
    text = _HREF_COMMAND_RE.sub(r"\2", text)

    # \emph{...}, \textit{...}: bóc ngoặc, lặp để xử lý macro lồng nhau.
    while match := _FORMAT_START_RE.search(text):
        inner, end = _read_braced(text, match.end())
        text = text[: match.start()] + inner + text[end:]

    text = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", text)
    for latex, plain in _LATEX_ESCAPES.items():
        text = text.replace(latex, plain)
    text = _COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n,;." if strip_period else " \t\r\n,;")


def _as_optional(value: object) -> str | None:
    """Chuẩn hóa một giá trị bất kỳ thành chuỗi non-empty hoặc ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# Trích xuất từng trường
# ---------------------------------------------------------------------------
def _protect_spans(text: str) -> str:
    """Thay URL/DOI/arXiv ID bằng khoảng trắng để không nhặt năm sai từ chúng.

    Giữ nguyên độ dài chuỗi nên vị trí các match sau đó vẫn dùng được trên văn bản gốc.
    """
    chars = list(text)
    patterns = (
        re.compile(r"\\(?:url|href)\s*\{[^{}]*\}", re.IGNORECASE),
        re.compile(r"https?://[^\s{}<>\"']+", re.IGNORECASE),
        re.compile(r"\\doi\s*\{[^{}]*\}", re.IGNORECASE),
        re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/[^\s{}]+", re.IGNORECASE),
        re.compile(r"\b10\.\d{4,9}/[^\s{}]+", re.IGNORECASE),
        re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE),
        re.compile(r"\barxiv\.org/(?:abs|pdf)/\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            for index in range(match.start(), match.end()):
                chars[index] = " "
    return "".join(chars)


def _find_year(raw: str) -> tuple[int | None, tuple[int, int] | None]:
    """Tìm năm xuất bản trong phần văn bản không phải URL/DOI/arXiv ID.

    Returns:
        (năm, span) hoặc (None, None). Ưu tiên năm trong ngoặc hoặc đứng trước dấu câu.
    """
    masked = _protect_spans(raw)
    candidates = list(_YEAR_RE.finditer(masked))
    if not candidates:
        return None, None

    def score(match: re.Match) -> int:
        before = masked[match.start() - 1 : match.start()]
        after = masked[match.end() : match.end() + 1]
        points = 0
        if before == "(":
            points += 3
        if after in ".,;:)":
            points += 2
        if before == "-":
            points -= 2
        return points

    best = max(candidates, key=score)
    return int(best.group(1)), best.span()


def _clean_doi(value: str) -> str | None:
    """Bỏ tiền tố URL/``doi:``/đuôi câu khỏi DOI, giữ dấu ngoặc cân bằng bên trong."""
    doi = _DOI_PREFIX.sub("", value.strip()).rstrip(".,;:")
    while doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1].rstrip(".,;:")
    return doi or None


def _find_doi(raw: str) -> str | None:
    """Tìm DOI ở các dạng ``doi:``, ``doi.org``, ``\\doi{}`` hoặc DOI trần."""
    match = _DOI_RE.search(raw)
    if match:
        return _clean_doi(match.group("doi"))
    match = _BARE_DOI_RE.search(raw)
    if match:
        return _clean_doi(match.group("doi"))
    return None


def _find_url(raw: str) -> str | None:
    """Tìm URL ở ``\\url{}``, ``\\href{}{}`` hoặc URL trần."""
    match = _URL_COMMAND_RE.search(raw)
    if match:
        return _as_optional(match.group(1))
    match = _HREF_COMMAND_RE.search(raw)
    if match:
        return _as_optional(match.group(1))
    match = _BARE_URL_RE.search(raw)
    if match:
        return _as_optional(match.group(0).rstrip(".,;:)]}"))
    return None


def _bibinfo_fields(raw: str) -> dict[str, str]:
    """Gom các cặp ``\\bibinfo{field}{value}`` thành dict (giữ lần xuất hiện đầu)."""
    fields: dict[str, str] = {}
    for match in _BIBINFO_RE.finditer(raw):
        name = match.group(1).strip().lower()
        value, _ = _read_braced(raw, match.end())
        fields.setdefault(name, value)
    return fields


def _find_emphasis(raw: str) -> tuple[str, int, int] | None:
    """Tìm tiêu đề được đánh dấu ``\\emph{}`` / ``\\textit{}`` / ``{\\em ...}``.

    Returns:
        (nội dung, vị trí bắt đầu, vị trí kết thúc) hoặc ``None``.
    """
    match = _EMPHASIS_START_RE.search(raw)
    if match:
        inner, end = _read_braced(raw, match.end())
        return inner, match.start(), end

    match = _TEX_EM_START_RE.search(raw)
    if match:
        # {\em Title}: nội dung bắt đầu ngay sau "\em " và kết thúc ở "}" của nhóm.
        inner, end = _read_braced(raw, match.end())
        return inner, match.start(), end
    return None


def _author_key(name: str) -> str:
    """Lấy khóa so khớp tác giả: họ nếu có dạng ``Họ, Tên``, ngược lại token cuối."""
    cleaned = _latex_to_text(name)
    part = cleaned.split(",", 1)[0] if "," in cleaned else (cleaned.split()[-1] if cleaned.split() else "")
    return re.sub(r"[^\w]+", "", part.casefold(), flags=re.UNICODE)


def _authors_from_text(text: str) -> list[str]:
    text = _latex_to_text(text, strip_period=False)
    if not text:
        return []
    text = re.sub(r"[,;]?\s*\bet\s+al\.?", "", text, flags=re.IGNORECASE)
    parts = [p for p in re.split(r"\s+\band\b\s+|\s*&\s*|\s*;\s*", text, flags=re.IGNORECASE) if p.strip(" ,")]
    out: list[str] = []
    for part in parts:
        tokens = [t.strip() for t in part.split(",") if t.strip()]
        if len(tokens) > 1 and all(" " in t for t in tokens):   # "J. Smith, A. Doe"
            out.extend(tokens)
        elif len(tokens) >= 4 and len(tokens) % 2 == 0:          # "Smith, J., Doe, A."
            out.extend(f"{tokens[i]}, {tokens[i+1]}" for i in range(0, len(tokens), 2))
        else:
            out.append(part)
    return [a for a in map(_tidy_author, out) if a]

def _tidy_author(name: str) -> str:
    name = name.strip(" ,;")
    return name[:-1] if re.search(r"[A-Za-z]{3,}\.$", name) else name  # "Lee." -> "Lee", giữ "J."


def _journal_from_suffix(suffix: str) -> str | None:
    """Lấy tên tạp chí/hội nghị từ phần sau tiêu đề, bỏ năm, trang và URL/DOI."""
    masked = _protect_spans(suffix)
    masked = _YEAR_RE.sub(" ", masked)

    candidates: list[str] = []
    for part in _split_top_level(masked, ","):
        text = _latex_to_text(part).strip(" ,;:.")
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        if re.fullmatch(r"(?:pp?\.?\s*)?\d+(?:\s*[-–]\s*\d+)?", text, re.IGNORECASE):
            continue
        candidates.append(text)
    return ", ".join(candidates) or None


def _split_plain(raw: str) -> tuple[str, str, str]:
    """Tách ``(authors, title, suffix)`` cho ``\\bibitem`` không có ``\\emph``.

    Nhận diện dạng APA (``Lee, B. (2019). Title. Venue.``) và dạng
    ``A. Author. Title. Journal, 2020.``; trả authors rỗng khi không nhận ra.
    """
    raw = raw.strip()
    match = _APA_RE.match(raw)
    if match:
        parts = re.split(r"(?<=[a-z0-9\)])\.\s+", match["rest"], maxsplit=1)
        return match["authors"], parts[0], parts[1] if len(parts) > 1 else ""

    parts = re.split(r"(?<![\s.][A-Z])(?<=[a-z0-9\)])\.\s+", raw)
    if len(parts) >= 3:  # "A. Author. Title. Journal, 2020."
        return parts[0], parts[1], " ".join(parts[2:])
    
    if len(parts) == 2:  # cắt tại chữ viết tắt cuối cùng của danh sách tác giả
        m = re.match(r"^(.*\b[A-Z]\.)\s+(\S.*)$", parts[0], re.S)
        if m:
            return m[1], m[2], parts[1]
    return "", raw, ""


def _parse_entry(citation_key: str, raw_text: str, entry_type: str = "misc") -> dict:
    """Phân tích một entry từ thân ``\\bibitem`` hoặc một entry ``.bib`` đã tách field."""
    raw_text = _strip_comments(raw_text).strip()
    bibinfo = _bibinfo_fields(raw_text)
    year, year_span = _find_year(raw_text)

    doi = _find_doi(raw_text) or _clean_doi(bibinfo.get("doi", ""))
    url = _find_url(raw_text) or _as_optional(bibinfo.get("url"))
    journal = _latex_to_text(bibinfo.get("journal", "")) or None

    emphasis = _find_emphasis(raw_text)
    if _APA_RE.match(raw_text):  # APA: chữ nghiêng là venue, không phải title
        authors_prefix, plain_title, suffix = _split_plain(raw_text)
        title = _latex_to_text(plain_title) or None
    elif emphasis is not None:
        inner, start, end = emphasis
        title = _latex_to_text(inner) or None
        authors_prefix = raw_text[:start]
        suffix = raw_text[end:]
    elif bibinfo.get("title"):
        title = _latex_to_text(bibinfo["title"]) or None
        authors_prefix = ""
        suffix = raw_text
    else:
        authors_prefix, title_text, suffix = _split_plain(raw_text)
        if not authors_prefix and not suffix:
            # Dạng phẩy "A. Author, Title, Venue, 2020." chưa nhận diện ở trên.
            before_year = raw_text[: year_span[0]] if year_span else raw_text
            segments = _split_top_level(before_year, ",")
            if len(segments) >= 2:
                authors_prefix = segments[0]
                title_text = segments[1]
                suffix = ",".join(segments[2:]) + (raw_text[year_span[1] :] if year_span else "")
        title = _latex_to_text(title_text) or None

    authors = _authors_from_text(bibinfo.get("author") or authors_prefix)
    if journal is None:
        journal = _journal_from_suffix(suffix)

    return {
        "citation_key": citation_key.strip(),
        "entry_type": entry_type or "misc",
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "journal": journal,
        "url": url,
    }


# ---------------------------------------------------------------------------
# Nguồn 1: \bibitem trong chuỗi LaTeX
# ---------------------------------------------------------------------------
def extract_bibitems(latex: str) -> list[dict]:
    """Trích xuất toàn bộ ``\\bibitem`` từ chuỗi LaTeX đã gộp.

    Args:
        latex: Chuỗi LaTeX (còn nguyên phần References).

    Returns:
        Danh sách entry phẳng theo thứ tự xuất hiện. Mỗi dict có các khóa
        ``citation_key``, ``entry_type``, ``title``, ``authors``, ``year``, ``doi``,
        ``journal``, ``url`` (đúng ``src.schemas.Bibtex`` trừ ``arxiv_id``).
    """
    if not latex:
        return []

    source = _strip_comments(latex.replace("\r\n", "\n").replace("\r", "\n"))
    matches = list(_BIBITEM_RE.finditer(source))
    entries: list[dict] = []
    for index, match in enumerate(matches):
        body_start = match.end()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        boundary = next_start
        for end_pattern in (_THEBIBLIOGRAPHY_END_RE, _DOCUMENT_END_RE):
            found = end_pattern.search(source, body_start, next_start)
            if found is not None:
                boundary = min(boundary, found.start())
        citation_key = match.group("key").strip()
        if citation_key:
            entries.append(_parse_entry(citation_key, source[body_start:boundary]))
    return entries


# ---------------------------------------------------------------------------
# Nguồn 2: file .bib đi kèm
# ---------------------------------------------------------------------------
def _parse_bib_fields(body: str) -> dict[str, str]:
    """Tách các cặp ``field = value`` trong thân một entry ``.bib``."""
    fields: dict[str, str] = {}
    position = 0
    while match := _FIELD_RE.search(body, position):
        name = match.group("name").lower()
        value_start = match.end()
        opener = body[value_start : value_start + 1]
        if opener == "{":
            value, end = _read_braced(body, value_start + 1)
        elif opener == '"':
            value, end = _read_quoted(body, value_start + 1)
        else:
            end = value_start
            while end < len(body) and body[end] != ",":
                end += 1
            value = body[value_start:end].strip()
        fields[name] = value.strip()
        position = max(end, value_start + 1)
    return fields


def _entry_from_bib_fields(citation_key: str, fields: dict[str, str], entry_type: str) -> dict:
    """Chuẩn hóa một entry ``.bib`` đã tách field thành dict phẳng."""
    authors = _authors_from_text(fields.get("author") or fields.get("editor") or "")
    title = _latex_to_text(fields.get("title", "")) or None
    journal = _latex_to_text(fields.get("journal") or fields.get("booktitle") or "") or None
    year_text = fields.get("year", "")
    year_match = _YEAR_RE.search(year_text) if year_text else None
    year = int(year_match.group(1)) if year_match else None
    doi = _clean_doi(fields.get("doi", "")) or _find_doi(fields.get("url", ""))
    url = _as_optional(fields.get("url"))

    return {
        "citation_key": citation_key.strip(),
        "entry_type": entry_type or "misc",
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "journal": journal,
        "url": url,
    }


def parse_bib_file(path: str | Path) -> list[dict]:
    """Đọc một file ``.bib`` và trả về danh sách entry đã chuẩn hóa.

    Args:
        path: Đường dẫn file ``.bib``.

    Returns:
        Danh sách entry theo thứ tự trong file. Entry ``@comment``/``@string``/
        ``@preamble`` bị bỏ qua.
    """
    bib_path = Path(path)
    raw = bib_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = _strip_comments(text)

    entries: list[dict] = []
    position = 0
    while match := _ENTRY_START_RE.search(text, position):
        entry_type = match.group("type").lower()
        open_char = match.group("open")
        close_char = "}" if open_char == "{" else ")"
        body, end = _read_delimited(text, match.end("open"), open_char, close_char)
        position = max(end, match.end())
        if entry_type in {"comment", "string", "preamble"}:
            continue
        fields = _parse_bib_fields(body)
        entries.append(_entry_from_bib_fields(match.group("key"), fields, entry_type))
    return entries


def _find_bib_files(latex: str, version_dir: Path) -> list[Path]:
    """Tìm các file ``.bib`` được ``\\bibliography{}`` / ``\\addbibresource{}`` chỉ định."""
    names: list[str] = []
    for match in _BIB_RESOURCE_RE.finditer(latex):
        names.extend(name.strip() for name in match.group(1).split(",") if name.strip())

    paths: list[Path] = []
    for name in names:
        candidate = version_dir / (name if name.endswith(".bib") else f"{name}.bib")
        if not candidate.is_file():
            candidate = version_dir / name
        if candidate.is_file():
            paths.append(candidate.resolve())
        else:
            logger.warning("%s: không tìm thấy file .bib được tham chiếu: %s", version_dir, name)

    if not paths and version_dir.is_dir():
        # Một số bài không ghi \bibliography nhưng vẫn có refs.bib trong thư mục nguồn.
        paths = sorted(path.resolve() for path in version_dir.rglob("*.bib"))
    return paths


def extract_bibtex(latex: str, version_dir: str | Path | None = None) -> list[dict]:
    """Trích xuất BibTeX từ cả ``\\bibitem`` và file ``.bib`` đi kèm (nếu có).

    Args:
        latex: Chuỗi LaTeX đã gộp.
        version_dir: Thư mục nguồn của version (chứa ``main.tex`` và ``refs.bib``).
            ``None`` thì chỉ đọc ``\\bibitem``.

    Returns:
        Danh sách entry từ ``\\bibitem`` trước, sau đó từ các file ``.bib``.
    """
    entries = extract_bibitems(latex)
    if version_dir is None:
        return entries

    for bib_path in _find_bib_files(latex, Path(version_dir)):
        entries.extend(parse_bib_file(bib_path))
    return entries


# ---------------------------------------------------------------------------
# Ghi refs.bib chuẩn hóa
# ---------------------------------------------------------------------------
def _escape_bib_value(value: object) -> str:
    """Escape tối thiểu cho giá trị đã là văn bản thuần sau ``_latex_to_text``."""
    return re.sub(r"(?<!\\)%", r"\\%", str(value)).replace("{", "").replace("}", "")


def format_bibtex_entry(entry: dict, fallback_key: str = "cite1") -> str:
    """Dựng chuỗi một entry BibTeX từ dict đã chuẩn hóa."""
    key = _as_optional(entry.get("citation_key")) or fallback_key
    entry_type = _as_optional(entry.get("entry_type")) or "misc"
    authors = entry.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    author_text = " and ".join(str(author) for author in authors if _as_optional(author))

    fields = (
        ("title", entry.get("title")),
        ("author", author_text or None),
        ("year", entry.get("year")),
        ("journal", entry.get("journal")),
        ("doi", entry.get("doi")),
        ("url", entry.get("url")),
    )
    body = ",\n".join(
        f"  {name} = {{{_escape_bib_value(value)}}}" for name, value in fields if value not in (None, "")
    )
    return f"@{entry_type}{{{key},\n{body}\n}}\n"


def write_bibtex(entries: list[dict], path: str | Path) -> None:
    """Ghi danh sách entry ra file ``refs.bib`` (UTF-8, ghi tạm rồi đổi tên).

    Args:
        entries: Các dict entry đã chuẩn hóa (cùng dạng ``extract_bibitems`` trả về).
        path: File đích, thường là ``<version_dir>/refs.bib`` hoặc ``refs.bib`` cấp bài.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        format_bibtex_entry(entry, fallback_key=f"cite{index + 1}") for index, entry in enumerate(entries)
    )

    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


__all__ = [
    "extract_bibitems",
    "extract_bibtex",
    "format_bibtex_entry",
    "parse_bib_file",
    "write_bibtex",
]
