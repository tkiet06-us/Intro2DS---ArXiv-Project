"""P2-04 – Khử trùng lặp 2 cấp: BibTeX (Field Union) và full-text giữa các version.

Hai bài toán tách biệt nhưng dùng chung một ý tưởng "canonical id":

1. ``dedup_bibtex`` gộp các entry trùng nhau (theo DOI, hoặc theo title có kiểm tra
   tác giả/năm), hợp nhất mọi trường (Field Union), cấp lại citation key duy nhất và
   trả về bảng ``citation_key_map`` để ``rewrite_citations`` sửa các lệnh ``\\cite{}``.
2. ``dedup_fulltext`` nhận ``hierarchy.json`` (format phẳng của M2), so khớp nội dung
   từng element giữa các version, đổi các id dạng ``pub#v2-s1`` về canonical
   ``pub#s1`` và ánh xạ lại toàn bộ cạnh cha-con.

Thiết kế bám theo ``Docs/latex_spec.md`` mục 1.2, 3.5, 6.1 và README mục 13.2.
"""

from __future__ import annotations

import copy
import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from src.m2_parsing.latex_cleanup import cleanup_latex
from src.schemas import Kind

logger = logging.getLogger(__name__)

_BIBTEX_FIELDS = (
    "citation_key",
    "entry_type",
    "title",
    "authors",
    "year",
    "doi",
    "journal",
    "url",
)

# ---------------------------------------------------------------------------
# Tiện ích chung
# ---------------------------------------------------------------------------


def _as_text(value: object) -> str | None:
    """Chuẩn hóa giá trị bất kỳ thành chuỗi non-empty hoặc ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_doi(value: object) -> str | None:
    """Bỏ tiền tố ``doi:`` / ``https://doi.org/`` và dấu câu cuối DOI."""
    doi = _as_text(value)
    if not doi:
        return None
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi\s*:\s*", "", doi, flags=re.IGNORECASE)
    return doi.rstrip(".,;") or None


def _normalize_doi(value: object) -> str | None:
    """Chuẩn hóa DOI để so khớp (lowercase, không tiền tố URL)."""
    doi = _clean_doi(value)
    return doi.casefold() if doi else None


def _normalize_title(value: object) -> str | None:
    """Chuẩn hóa title để so khớp: NFKC, casefold, bỏ dấu câu."""
    title = _as_text(value)
    if not title:
        return None
    title = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[^\w]+", " ", title, flags=re.UNICODE).strip() or None


def _author_key(name: str) -> str:
    """Lấy khóa so khớp tác giả: họ nếu có dạng ``Họ, Tên``, ngược lại token cuối."""
    cleaned = unicodedata.normalize("NFKC", name).casefold()
    part = cleaned.split(",", 1)[0] if "," in cleaned else (cleaned.split()[-1] if cleaned.split() else "")
    return re.sub(r"[^\w]+", "", part, flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Dedup BibTeX
# ---------------------------------------------------------------------------


def _normalize_entry(entry: Mapping[str, object], index: int) -> dict:
    """Chuẩn hóa một entry thô về đúng 8 trường của bảng ``bibtex``."""
    normalized = {field: entry.get(field) for field in _BIBTEX_FIELDS}
    normalized["citation_key"] = _as_text(normalized["citation_key"]) or f"cite{index + 1}"
    normalized["entry_type"] = (_as_text(normalized["entry_type"]) or "misc").lower()
    for field in ("title", "journal", "url"):
        normalized[field] = _as_text(normalized[field])
    normalized["doi"] = _clean_doi(normalized["doi"])

    raw_authors = normalized["authors"]
    if raw_authors is None:
        normalized["authors"] = []
    elif isinstance(raw_authors, str):
        normalized["authors"] = [raw_authors.strip()] if raw_authors.strip() else []
    else:
        normalized["authors"] = [
            author for value in raw_authors if (author := _as_text(value)) is not None
        ]

    raw_year = normalized["year"]
    try:
        normalized["year"] = (
            int(raw_year) if raw_year is not None and str(raw_year).strip() else None
        )
    except (TypeError, ValueError):
        normalized["year"] = None
    return normalized


class _UnionFind:
    """Union-Find tối giản, nén đường đi + gộp theo rank."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        """Trả về đại diện của tập chứa ``node`` (nén đường đi)."""
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        """Gộp hai tập; không làm gì nếu đã cùng tập."""
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _compatible(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Kiểm tra hai entry cùng title có thực sự là cùng một công trình không.

    Chỉ dùng khi khớp title (khóa yếu). Năm lệch quá 1 hoặc hai danh sách tác giả
    hoàn toàn rời nhau là dấu hiệu trùng tên chứ không trùng bài.
    """
    left_year, right_year = left.get("year"), right.get("year")
    if isinstance(left_year, int) and isinstance(right_year, int) and abs(left_year - right_year) > 1:
        return False

    left_authors = {_author_key(author) for author in (left.get("authors") or []) if _author_key(author)}
    right_authors = {_author_key(author) for author in (right.get("authors") or []) if _author_key(author)}
    return not (left_authors and right_authors) or bool(left_authors & right_authors)


def _prefer_longer(current: object, incoming: object) -> object:
    """Chọn giá trị dài hơn; coi chuỗi rỗng/None là thiếu."""
    current_text, incoming_text = _as_text(current), _as_text(incoming)
    if current_text is None:
        return incoming
    if incoming_text is None:
        return current
    return incoming if len(incoming_text) > len(current_text) else current


def _merge_entry(target: dict, incoming: Mapping[str, object]) -> None:
    """Field Union: điền chỗ thiếu và chọn giá trị đầy đủ hơn cho ``target``."""
    target["title"] = _prefer_longer(target.get("title"), incoming.get("title"))
    target["journal"] = _prefer_longer(target.get("journal"), incoming.get("journal"))
    target["url"] = _prefer_longer(target.get("url"), incoming.get("url"))
    target["doi"] = target.get("doi") or incoming.get("doi")
    if target.get("year") is None:
        target["year"] = incoming.get("year")

    current_type = str(target.get("entry_type") or "misc").lower()
    incoming_type = str(incoming.get("entry_type") or "misc").lower()
    if current_type == "misc" and incoming_type != "misc":
        target["entry_type"] = incoming.get("entry_type")

    authors = target.get("authors") or []
    seen = {author.casefold() for author in authors}
    for author in incoming.get("authors") or []:
        if author.casefold() not in seen:
            authors.append(author)
            seen.add(author.casefold())
    target["authors"] = authors


def _unused_key(base: str, used_keys: Iterable[str]) -> str:
    """Sinh key duy nhất từ ``base`` bằng hậu tố ``_2``, ``_3``,..."""
    used = {key.casefold() for key in used_keys}
    candidate, suffix = base, 2
    while candidate.casefold() in used:
        candidate, suffix = f"{base}_{suffix}", suffix + 1
    return candidate


def dedup_bibtex(entries: list[dict]) -> tuple[list[dict], dict]:
    """Gộp các entry BibTeX trùng nhau và trả về map để sửa ``\\cite{}``.

    Các entry được nối bằng union-find theo hai khóa:

    - DOI giống nhau (sau khi chuẩn hóa) luôn là cùng một công trình.
    - Title giống nhau chỉ gộp khi tác giả không mâu thuẫn và năm không lệch quá 1.

    Mọi trường được hợp nhất theo kiểu Field Union: title/journal/url chọn bản dài hơn,
    DOI và năm lấy giá trị đầu tiên có được, danh sách tác giả hợp nhất theo thứ tự.
    ``citation_key`` của entry đầu tiên trong nhóm được giữ làm canonical; các key
    trùng nhưng trỏ tới công trình khác được đổi thành ``key_2``, ``key_3``,...

    Args:
        entries: Danh sách entry phẳng (cùng dạng ``bibtex_extractor`` trả về).

    Returns:
        ``(deduplicated, citation_key_map)``. ``citation_key_map`` ánh xạ mọi key cũ
        không nhập nhằng sang key canonical. Với key bị dùng lại cho hai công trình
        khác nhau, map chứa khóa định danh lần xuất hiện, ví dụ
        ``"same#duplicate-2" -> "same_2"``.
    """
    normalized = [_normalize_entry(entry, index) for index, entry in enumerate(entries)]
    if not normalized:
        return [], {}

    forest = _UnionFind(len(normalized))
    doi_index: dict[str, int] = {}
    title_buckets: dict[str, list[int]] = defaultdict(list)

    for index, entry in enumerate(normalized):
        doi = _normalize_doi(entry.get("doi"))
        if doi:
            if doi in doi_index:
                forest.union(index, doi_index[doi])
            else:
                doi_index[doi] = index

        title = _normalize_title(entry.get("title"))
        if title:
            for other_index in title_buckets[title]:
                if _compatible(entry, normalized[other_index]):
                    forest.union(index, other_index)
            title_buckets[title].append(index)

    groups: dict[int, list[dict]] = defaultdict(list)
    group_order: list[int] = []
    for index, entry in enumerate(normalized):
        root = forest.find(index)
        if root not in groups:
            group_order.append(root)
        groups[root].append(entry)

    deduplicated: list[dict] = []
    used_keys: list[str] = []
    occupied_by_group: dict[str, tuple[int, str]] = {}
    source_occurrences: list[tuple[str, int, str]] = []

    for group_index, root in enumerate(group_order):
        group = groups[root]
        merged = copy.deepcopy(group[0])
        for other in group[1:]:
            _merge_entry(merged, other)

        source_key = str(merged["citation_key"])
        canonical_key = source_key
        if source_key.casefold() in {key.casefold() for key in used_keys}:
            canonical_key = _unused_key(source_key, used_keys)
        merged["citation_key"] = canonical_key
        used_keys.append(canonical_key)
        occupied_by_group.setdefault(canonical_key.casefold(), (group_index, canonical_key))

        for entry in group:
            key = str(entry["citation_key"])
            source_occurrences.append((key, group_index, canonical_key))
        deduplicated.append(merged)

    citation_key_map: dict[str, str] = {}
    ambiguous: set[str] = set()
    occurrence_counts: dict[str, int] = defaultdict(int)

    for source_key, group_index, canonical_key in source_occurrences:
        occurrence_counts[source_key.casefold()] += 1
        count = occurrence_counts[source_key.casefold()]
        if count > 1:
            citation_key_map[f"{source_key}#duplicate-{count}"] = canonical_key

        if source_key.casefold() == canonical_key.casefold():
            continue

        owner = occupied_by_group.get(source_key.casefold())
        if owner is not None:
            owner_group, owner_key = owner
            if owner_group == group_index:
                citation_key_map.setdefault(source_key, owner_key)
            else:
                ambiguous.add(source_key.casefold())
            continue

        if source_key.casefold() in ambiguous:
            continue
        previous = citation_key_map.get(source_key)
        if previous is not None and previous.casefold() != canonical_key.casefold():
            ambiguous.add(source_key.casefold())
            citation_key_map.pop(source_key, None)
        else:
            citation_key_map[source_key] = canonical_key

    return deduplicated, citation_key_map


# ``\cite``, ``\citep``, ``\citet``, ``\parencite``, ``\nocite``, ...; tối đa 2 optional arg.
_CITE_RE = re.compile(
    r"\\(?P<command>[A-Za-z]*cite[A-Za-z]*|nocite)(?P<star>\*?)"
    r"(?P<options>(?:\s*\[[^\]]*\]){0,2})(?P<open>\{)(?P<keys>[^{}]*)(?P<close>\})",
    re.IGNORECASE,
)


def rewrite_citations(latex: str, citation_key_map: Mapping[str, str]) -> str:
    """Đổi key trong mọi lệnh ``\\cite{}`` / ``\\citep{}`` / ... theo map.

    Args:
        latex: Chuỗi LaTeX (có thể còn nhiều lệnh cite).
        citation_key_map: Map key cũ -> key canonical do ``dedup_bibtex`` trả về.

    Returns:
        Chuỗi LaTeX đã sửa key; key không có trong map được giữ nguyên. Khoảng trắng
        và optional argument của lệnh gốc được bảo toàn.
    """
    if not citation_key_map:
        return latex

    casefold_map: dict[str, str] = {}
    for old_key, new_key in citation_key_map.items():
        casefold_map.setdefault(old_key.casefold(), new_key)

    def replacement(match: re.Match[str]) -> str:
        raw_keys = match.group("keys")
        if not raw_keys.strip():
            return match.group(0)

        rewritten: list[str] = []
        for part in raw_keys.split(","):
            stripped = part.strip()
            if not stripped:
                rewritten.append(part)
                continue
            leading = part[: len(part) - len(part.lstrip())]
            trailing = part[len(part.rstrip()) :]
            new_key = citation_key_map.get(stripped) or casefold_map.get(stripped.casefold())
            rewritten.append(f"{leading}{new_key or stripped}{trailing}")

        command = match.group("command")
        star = match.group("star") or ""
        options = match.group("options") or ""
        return f"\\{command}{star}{options}{{{','.join(rewritten)}}}"

    return _CITE_RE.sub(replacement, latex)


# ---------------------------------------------------------------------------
# Dedup full-text trên hierarchy.json
# ---------------------------------------------------------------------------

_VERSION_PREFIX = re.compile(r"^v(\d+)-(.+)$", re.IGNORECASE)
_ROOT_ID = re.compile(r"^root(\d+)$", re.IGNORECASE)
_PREFIX_KINDS: tuple[tuple[str, Kind], ...] = (
    ("root", Kind.DOCUMENT),
    ("doc", Kind.DOCUMENT),
    ("sentence", Kind.SENTENCE),
    ("sent", Kind.SENTENCE),
    ("heading", Kind.HEADING),
    ("sec", Kind.HEADING),
    ("figure", Kind.FIGURE),
    ("fig", Kind.FIGURE),
    ("formula", Kind.FORMULA),
    ("eq", Kind.FORMULA),
    ("itemize", Kind.ITEMIZE),
    ("list", Kind.ITEMIZE),
    ("item", Kind.ITEM),
    ("it", Kind.ITEM),
    ("h", Kind.HEADING),
    ("f", Kind.FORMULA),
    ("s", Kind.SENTENCE),
)


def _anchor_of(element_id: str) -> str:
    """Lấy phần sau dấu ``#`` (anchor) của element_id."""
    return element_id.rsplit("#", 1)[-1]


def _strip_version_prefix(element_id: str) -> str:
    """Bỏ tiền tố ``vN-`` khỏi anchor: ``pub#v2-s1`` -> ``pub#s1``."""
    prefix, separator, anchor = element_id.rpartition("#")
    match = _VERSION_PREFIX.match(anchor)
    if separator and match:
        return f"{prefix}#{match.group(2)}"
    return element_id


def _id_version(element_id: str) -> int | None:
    """Suy ra số version từ id dạng ``...v2-s1`` hoặc ``...root2``."""
    anchor = _anchor_of(element_id)
    match = _VERSION_PREFIX.match(anchor)
    if match:
        return int(match.group(1))
    root = _ROOT_ID.match(anchor)
    return int(root.group(1)) if root else None


def _kind_from_content(content: str) -> Kind:
    """Suy ra kind từ nội dung khi id không có tiền tố quy ước."""
    if re.search(r"\\begin\{(?:figure|table|tabular|wrapfigure|wraptable)\*?\}", content):
        return Kind.FIGURE
    if content.lstrip().startswith(("$$", r"\[", r"\begin{equation", r"\begin{align")):
        return Kind.FORMULA
    if not content.strip():
        return Kind.SENTENCE
    return Kind.SENTENCE


def _infer_kind(element_id: str, content: str) -> Kind:
    """Suy ra ``kind`` từ tiền tố id (``s``, ``f``, ``fig``, ``h``, ``list``...)."""
    anchor = _VERSION_PREFIX.sub(r"\2", _anchor_of(element_id)).casefold()
    for prefix, kind in _PREFIX_KINDS:
        if anchor.startswith(prefix):
            return kind
    return _kind_from_content(content)


def _normalize_content(content: str) -> str:
    """Chuẩn hóa khoảng trắng để so khớp nội dung giữa các version."""
    return " ".join(content.split())


def _dedup_texts(versions: Iterable[str]) -> list[str]:
    """Fallback tương thích API cũ: nhận list văn bản version, trả list duy nhất."""
    unique: list[str] = []
    seen: set[str] = set()
    for version in versions:
        cleaned = cleanup_latex(version)
        key = cleaned.strip()
        if key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _normalize_hierarchy(data: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Kiểm tra và chuẩn hóa đầu vào ``{"elements": ..., "hierarchy": ...}``."""
    elements = data.get("elements")
    if not isinstance(elements, Mapping):
        # Đây là sai hình dạng cả payload, không phải sai kiểu của một đối số đơn lẻ;
        # ValueError khớp với cách các validator khác của pipeline báo lỗi.
        raise ValueError(  # noqa: TRY004
            "hierarchy.json phải có 'elements' là mapping element_id -> content"
        )
    hierarchy = data.get("hierarchy") or {}
    if not isinstance(hierarchy, Mapping):
        raise ValueError(  # noqa: TRY004
            "hierarchy.json phải có 'hierarchy' là mapping version -> cạnh cha-con"
        )

    normalized_elements = {str(key): str(value) for key, value in elements.items()}
    normalized_hierarchy: dict[str, dict[str, str]] = {}
    for version, edges in hierarchy.items():
        if not isinstance(edges, Mapping):
            raise ValueError(  # noqa: TRY004
                f"hierarchy['{version}'] phải là mapping element_id -> parent_id"
            )
        normalized_hierarchy[str(version)] = {
            str(child): str(parent) for child, parent in edges.items()
        }
    return normalized_elements, normalized_hierarchy


def dedup_fulltext(
    data: Mapping[str, Any] | list[str],
    citation_key_map: Mapping[str, str] | None = None,
) -> dict | list[str]:
    """Khử trùng lặp nội dung giữa các version từ ``hierarchy.json``.

    Đầu vào là format phẳng của M2::

        {
          "elements": {"2310-12345#v1-s1": "We study matching.", ...},
          "hierarchy": {"1": {"...#v1-s1": "...#v1-h1"}, "2": {...}}
        }

    Cùng một nội dung (sau khi chuẩn hóa khoảng trắng) ở nhiều version sẽ dùng chung
    một ``element_id`` canonical. Tiền tố version ``vN-`` bị bỏ đi; nếu canonical id
    bị chiếm bởi nội dung khác, hậu tố ``-2``, ``-3``,... được thêm vào. Nút gốc mỗi
    version (``root1``, ``root2``) và nút rỗng (``itemize``/``item``) không bị gộp bừa.

    Args:
        data: ``hierarchy.json`` đã đọc, hoặc list văn bản version (API cũ).
        citation_key_map: Tùy chọn map key cũ -> key canonical; nếu có, ``\\cite{}``
            trong content được sửa trước khi so khớp.

    Returns:
        ``hierarchy.json`` đã dedup cùng hình dạng đầu vào. Với list đầu vào, trả list
        văn bản duy nhất (hành vi cũ).

    Raises:
        ValueError: Khi dict đầu vào không đúng hình dạng ``hierarchy.json``.
    """
    if isinstance(data, (list, tuple)):
        return _dedup_texts(data)

    elements, hierarchy = _normalize_hierarchy(data)
    if not elements:
        return {"elements": {}, "hierarchy": hierarchy}

    # Một element có thể xuất hiện trong nhiều version; chỉ coi là "thuộc version X"
    # khi nó chỉ xuất hiện ở đúng một version (id canonical dùng chung thì không).
    id_versions: dict[str, set[str]] = defaultdict(set)
    for version, edges in hierarchy.items():
        for child, parent in edges.items():
            id_versions[child].add(version)
            id_versions[parent].add(version)

    def version_of(element_id: str) -> str | None:
        number = _id_version(element_id)
        if number is not None:
            return str(number)
        known = id_versions.get(element_id, set())
        return next(iter(known)) if len(known) == 1 else None

    def sort_key(item: tuple[int, str]) -> tuple[int, int]:
        index, element_id = item
        number = _id_version(element_id)
        if number is None:
            known = id_versions.get(element_id, set())
            if len(known) == 1:
                value = next(iter(known))
                number = int(value) if value.isdigit() else None
        return (number if number is not None else 10**9, index)

    content_by_old = {
        element_id: (
            rewrite_citations(content, citation_key_map) if citation_key_map else content
        )
        for element_id, content in elements.items()
    }

    old_to_new: dict[str, str] = {}
    identity_of_new: dict[str, tuple[str, str] | None] = {}
    ids_by_identity: dict[tuple[str, str], list[str]] = defaultdict(list)
    used_in_version: dict[str, dict[str, str]] = defaultdict(dict)

    for _, old_id in sorted(enumerate(elements), key=sort_key):
        content = content_by_old[old_id]
        kind = _infer_kind(old_id, content)
        # Cleanup chạy TRƯỚC dedup trong pipeline; gọi lại ở đây chỉ để so khớp
        # (idempotent, không đổi content xuất ra) nên nội dung chưa sạch vẫn gộp đúng.
        normalized = _normalize_content(cleanup_latex(content))
        # Nút gốc mỗi version phải giữ id riêng; nút rỗng (itemize/item) không đủ
        # thông tin nội dung để kết luận là trùng, nên không dedup theo content.
        identity = None if kind is Kind.DOCUMENT or not normalized else (str(kind), normalized)
        version = version_of(old_id)
        used = used_in_version[version] if version is not None else None

        canonical: str | None = None
        if identity is not None:
            for candidate in ids_by_identity[identity]:
                if used is None or candidate not in used:
                    canonical = candidate
                    break

        if canonical is None:
            base = _strip_version_prefix(old_id)
            canonical = base
            suffix = 2
            while canonical in identity_of_new or (used is not None and canonical in used):
                canonical = f"{base}-{suffix}"
                suffix += 1
            identity_of_new[canonical] = identity
            if identity is not None:
                ids_by_identity[identity].append(canonical)

        old_to_new[old_id] = canonical
        if used is not None:
            used[canonical] = old_id

    def map_reference(element_id: str, version: str) -> str:
        """Ánh xạ id xuất hiện trong hierarchy nhưng không có trong ``elements``."""
        if element_id in old_to_new:
            return old_to_new[element_id]

        used = used_in_version[version]
        base = _strip_version_prefix(element_id)
        canonical = base
        suffix = 2
        while canonical in identity_of_new or canonical in used:
            canonical = f"{base}-{suffix}"
            suffix += 1
        identity_of_new[canonical] = None
        used[canonical] = element_id
        old_to_new[element_id] = canonical
        content_by_old.setdefault(element_id, "")
        return canonical

    new_hierarchy: dict[str, dict[str, str]] = {}
    for version, edges in hierarchy.items():
        remapped: dict[str, str] = {}
        for child, parent in edges.items():
            new_child = map_reference(child, version)
            new_parent = map_reference(parent, version)
            remapped[new_child] = new_parent
        new_hierarchy[version] = remapped

    new_elements: dict[str, str] = {}
    for old_id, content in content_by_old.items():
        canonical = old_to_new.get(old_id)
        if canonical is None:
            continue
        if canonical not in new_elements:
            new_elements[canonical] = content

    return {"elements": new_elements, "hierarchy": new_hierarchy}


__all__ = ["dedup_bibtex", "dedup_fulltext", "rewrite_citations"]
