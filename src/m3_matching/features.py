"""P2-06 (Owner: Kiệt & Bảo) – Tiền xử lý văn bản và đặc trưng cho cặp (BibTeX, ứng viên).

Bài toán (M3): với mỗi mục BibTeX của một bài báo, xếp hạng các reference arXiv của CHÍNH bài
đó. Mỗi cặp (mục BibTeX, ứng viên) được biến thành một vector số (xem FEATURE_NAMES).

Cách dùng:
    tensor = build_feature_matrix(entries, candidates)  # shape (m, n, len(FEATURE_NAMES))
    tensor[i, j] là vector đặc trưng của cặp (entries[i], candidates[j]).

Tham khảo:
    - Levenshtein: "Binary codes capable of correcting deletions, insertions and reversals",
      1966. Dùng thư viện rapidfuzz (https://github.com/rapidfuzz/RapidFuzz).
    - TF-IDF: scikit-learn TfidfVectorizer
      (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from src.schemas import Bibtex, Reference

# Thứ tự cột của vector đặc trưng. Mô hình và test đều dựa vào thứ tự này.
FEATURE_NAMES: tuple[str, ...] = (
    "arxiv_id_in_fields",  # ID arXiv của ứng viên xuất hiện trong url/journal/doi/chuỗi thô
    "title_lev_sim",  # Levenshtein chuẩn hóa trên tiêu đề
    "title_word_jaccard",  # Jaccard trên tập từ của tiêu đề
    "title_char3_jaccard",  # Jaccard trên trigram ký tự của tiêu đề
    "title_token_set",  # rapidfuzz token_set_ratio / 100 (chịu thiếu/thừa từ)
    "title_containment",  # rapidfuzz partial_ratio / 100 (tiêu đề bị cắt ngắn)
    "title_tfidf_cos",  # cosine TF-IDF, idf học trên các ứng viên của chính bài đó
    "first_author_match",  # họ tác giả đầu trùng nhau
    "author_surname_jaccard",  # Jaccard trên tập họ tác giả
    "year_match",  # cùng năm
    "year_diff",  # chênh lệch năm tuyệt đối, chuẩn hóa về 0..1
    "year_signed_diff",  # (năm ứng viên - năm mục) chuẩn hóa về -1..1; preprint thường <= 0
    "has_title",  # cả hai bên đều có tiêu đề
    "has_authors",  # cả hai bên đều có tác giả
    "has_year",  # cả hai bên đều có năm
)

# Năm arXiv (preprint) thường sớm hơn năm xuất bản vài năm; chênh quá 5 năm coi như "rất xa".
YEAR_DIFF_CAP = 5
# Khi thiếu năm: giá trị trung tính, kèm cờ has_year = 0 để mô hình biết không nên tin.
MISSING_YEAR_DIFF = 0.5

# Khớp arXiv ID dạng mới "2310.12345" và dạng cũ "hep-th/9901001", "math.GT/0309136", kèm
# hậu tố version tùy chọn. Không cho chữ số dính ngay trước/sau để "2310.1234" không bị cắt ra
# từ "2310.12345". Ví dụ: "arXiv:2310.12345v2" -> "2310.12345v2".
ARXIV_ID_PATTERN = re.compile(
    r"(?<!\d)(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?(?!\d)", re.IGNORECASE
)
VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$")

# Khớp "\'e", "\"u", "\^o", "\~n", "\=a", "\.z" (ngoặc nhọn tùy chọn) và "\c{c}", "\v{s}",
# "\H{o}" (bắt buộc có ngoặc nhọn để không nhầm với \cite{...}); giữ lại chữ cái gốc.
# Ví dụ: r'M{\"u}ller' -> 'M{u}ller'; r"Fran\c{c}ois" -> "Francois".
LATEX_ACCENT_PATTERN = re.compile(r"""\\(?:["'`^~=.]\s*\{?\s*|[hcvukrdb]\s*\{\s*)([a-z])\s*\}?""")
# Khớp "\i", "\j" (chữ i, j không chấm). Ví dụ: r"Mart\'{\i}nez" -> r"Mart\'{i}nez".
LATEX_DOTLESS_PATTERN = re.compile(r"\\([ij])\b")
# Khớp chữ đặc biệt của LaTeX như "\o", "\ss", "\l". Ví dụ: r"Erd\H{o}s" không dính (xem trên).
LATEX_SPECIAL_PATTERN = re.compile(r"\\(ss|ae|oe|aa|o|l)\b")
LATEX_SPECIAL_LETTERS = {"ss": "ss", "ae": "ae", "oe": "oe", "aa": "a", "o": "o", "l": "l"}
# Lệnh LaTeX còn lại như "\textbf", "\emph": bỏ tên lệnh, giữ nội dung trong ngoặc.
LATEX_COMMAND_PATTERN = re.compile(r"\\[a-z]+\*?")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
NON_NAME_PATTERN = re.compile(r"[^a-z,\s-]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
# Khớp năm 1800-2099 đứng riêng. Ví dụ: "2017-06-12" -> 2017.
YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# Ký tự Unicode mà NFKD không tách được dấu.
UNICODE_LETTERS = str.maketrans(
    {"ø": "o", "ł": "l", "đ": "d", "ß": "ss", "æ": "ae", "œ": "oe", "ı": "i"}
)
STOPWORDS = frozenset(
    {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "by", "from", "at", "as",
     "is", "are", "via", "using", "towards", "toward"}
)  # fmt: skip
ET_AL_NAMES = frozenset({"et al", "al", "others"})


# --------------------------------------------------------------------------- tiền xử lý
def ascii_fold(text: str) -> str:
    """Chữ thường, đổi mã dấu LaTeX và ký tự có dấu về ASCII.

    Args:
        text: Chuỗi thô, ví dụ r'M{\\"u}ller' hoặc "Müller".

    Returns:
        Chuỗi chữ thường, không dấu: "m{uller}" cho ví dụ đầu (ngoặc nhọn do preprocess bỏ).
    """
    text = text.lower()
    text = LATEX_DOTLESS_PATTERN.sub(r"\1", text)
    text = LATEX_ACCENT_PATTERN.sub(r"\1", text)
    text = LATEX_SPECIAL_PATTERN.sub(lambda m: LATEX_SPECIAL_LETTERS[m.group(1)], text)
    text = text.translate(UNICODE_LETTERS)
    # NFKD tách "ü" thành "u" + dấu rời, rồi bỏ dấu rời đi.
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def preprocess(text: str | None, remove_stopwords: bool = False) -> str:
    """Làm sạch chuỗi để so sánh: bỏ LaTeX, bỏ dấu, chữ thường, bỏ dấu câu, gộp khoảng trắng.

    Args:
        text: Chuỗi thô (có thể None).
        remove_stopwords: True để bỏ các từ như "the", "of" (nên thử cả hai và so sánh).

    Returns:
        Chuỗi đã làm sạch; chuỗi rỗng nếu đầu vào rỗng/None.
    """
    if not text:
        return ""
    text = ascii_fold(text)
    text = LATEX_COMMAND_PATTERN.sub(" ", text)
    # Ngoặc nhọn bỏ hẳn (không đổi thành khoảng trắng) để M{\"u}ller không bị cắt đôi.
    text = text.replace("{", "").replace("}", "")
    text = NON_ALNUM_PATTERN.sub(" ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return " ".join(tokens)


def parse_year(raw: str | None) -> int | None:
    """Lấy năm đầu tiên trong chuỗi, ví dụ "2017-06-12" -> 2017.

    Args:
        raw: Chuỗi ngày/năm (có thể None).

    Returns:
        Năm, hoặc None nếu không tìm thấy.
    """
    if not raw:
        return None
    match = YEAR_PATTERN.search(raw)
    return int(match.group()) if match else None


def extract_arxiv_ids(text: str | None) -> set[str]:
    """Lấy mọi arXiv ID trong một chuỗi, viết thường và đã bỏ hậu tố version.

    Args:
        text: Chuỗi bất kỳ (url, journal, chuỗi \\bibitem thô...).

    Returns:
        Tập ID, ví dụ {"2310.12345", "solv-int/9901001"}; rỗng nếu không có.
    """
    if not text:
        return set()
    found = (m.group(0).lower() for m in ARXIV_ID_PATTERN.finditer(text))
    return {VERSION_SUFFIX_PATTERN.sub("", raw_id) for raw_id in found}


def author_key(name: str) -> tuple[str, str] | None:
    """Chuẩn hóa tên tác giả về (họ, chữ cái đầu của tên) viết thường.

    "Kaiming He", "He, K." và "K. He" đều cho ("he", "k"). Họ là từ cuối của phần họ để
    "van der Waals, J." và "J. van der Waals" cùng cho ("waals", "j"). Giới hạn: tên viết
    kiểu "He Kaiming" (họ trước, không dấu phẩy) sẽ bị đọc ngược.

    Args:
        name: Tên tác giả ở bất kỳ dạng phổ biến nào.

    Returns:
        (họ, chữ đầu của tên; "" nếu không có), hoặc None với "et al."/"others"/rỗng.
    """
    folded = NON_NAME_PATTERN.sub(" ", ascii_fold(name))
    folded = MULTI_SPACE_PATTERN.sub(" ", folded).strip(" ,-")
    if not folded or folded in ET_AL_NAMES:
        return None
    if "," in folded:
        surname_part, _, given_part = folded.partition(",")
    else:
        tokens = folded.split()
        surname_part, given_part = tokens[-1], " ".join(tokens[:-1])
    surname_tokens = surname_part.split()
    if not surname_tokens:
        return None
    given = given_part.replace(",", " ").strip()
    return surname_tokens[-1], given[:1]


def author_keys(names: list[str]) -> list[tuple[str, str]]:
    """Chuẩn hóa cả danh sách tác giả, bỏ các phần tử không phải tên người (et al.)."""
    keys = (author_key(name) for name in names)
    return [key for key in keys if key is not None]


# --------------------------------------------------------------------------- độ giống
def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Tập n-gram ký tự liên tiếp, ví dụ char_ngrams("night", 2) = {"ni","ig","gh","ht"}."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(set_a: set, set_b: set) -> float:
    """Tỉ lệ |A ∩ B| / |A ∪ B|; 0.0 nếu một trong hai tập rỗng."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def title_levenshtein(title_a: str, title_b: str) -> float:
    """Độ giống Levenshtein chuẩn hóa 0..1: 1 - khoảng_cách / độ_dài_chuỗi_dài_hơn."""
    if not title_a or not title_b:
        return 0.0
    return Levenshtein.normalized_similarity(title_a, title_b)


# --------------------------------------------------------------------------- vector đặc trưng
@dataclass(frozen=True)
class _Prepared:
    """Dữ liệu đã làm sạch của một bên (mục BibTeX hoặc ứng viên), tính một lần dùng nhiều lần."""

    title: str
    words: frozenset[str]
    char3: frozenset[str]
    authors: tuple[tuple[str, str], ...]
    year: int | None
    arxiv_ids: frozenset[str] = frozenset()  # chỉ phía BibTeX: mọi ID tìm thấy trong các trường
    arxiv_id: str = ""  # chỉ phía ứng viên


def _prepare_entry(entry: Bibtex, raw_text: str | None = None) -> _Prepared:
    """Làm sạch một mục BibTeX; raw_text (nếu có) bổ sung ID arXiv và năm khi trường bị thiếu."""
    title = preprocess(entry.title)
    raw_text = raw_text or getattr(entry, "raw", None)  # sẵn sàng nếu schema thêm trường raw
    text_fields = (entry.url, entry.journal, entry.doi, raw_text)
    return _Prepared(
        title=title,
        words=frozenset(title.split()),
        char3=frozenset(char_ngrams(title)),
        authors=tuple(author_keys(entry.authors)),
        year=entry.year if entry.year is not None else parse_year(raw_text),
        arxiv_ids=frozenset(extract_arxiv_ids(" ".join(f for f in text_fields if f))),
    )


def _prepare_candidate(candidate: Reference) -> _Prepared:
    title = preprocess(candidate.ref_title)
    return _Prepared(
        title=title,
        words=frozenset(title.split()),
        char3=frozenset(char_ngrams(title)),
        authors=tuple(author_keys(candidate.ref_authors)),
        year=parse_year(candidate.ref_submission_date),
        arxiv_id=candidate.ref_arxiv_id.lower(),
    )


def _arxiv_id_in_fields(entry: _Prepared, candidate: _Prepared) -> float:
    """1.0 nếu ID của ứng viên nằm trong số ID tìm thấy ở url/journal/doi/chuỗi thô."""
    return float(bool(candidate.arxiv_id) and candidate.arxiv_id in entry.arxiv_ids)


def _year_features(entry: _Prepared, candidate: _Prepared) -> tuple[float, float, float, float]:
    """Trả về (year_match, year_diff, year_signed_diff, has_year)."""
    if entry.year is None or candidate.year is None:
        return 0.0, MISSING_YEAR_DIFF, 0.0, 0.0
    signed = candidate.year - entry.year
    # Chặn hai đầu để một năm sai/lạ (ví dụ 2030) không tạo giá trị ngoại lai.
    capped = max(-YEAR_DIFF_CAP, min(YEAR_DIFF_CAP, signed))
    return float(signed == 0), abs(capped) / YEAR_DIFF_CAP, capped / YEAR_DIFF_CAP, 1.0


def _pair_vector(entry: _Prepared, candidate: _Prepared, tfidf_cosine: float) -> list[float]:
    has_title = bool(entry.title and candidate.title)
    has_authors = bool(entry.authors and candidate.authors)
    year_match, year_diff, year_signed_diff, has_year = _year_features(entry, candidate)
    if has_title:
        token_set = fuzz.token_set_ratio(entry.title, candidate.title) / 100
        containment = fuzz.partial_ratio(entry.title, candidate.title) / 100
    else:
        token_set = containment = 0.0
    if has_authors:
        first_match = float(entry.authors[0][0] == candidate.authors[0][0])
    else:
        first_match = 0.0
    surname_jaccard = jaccard({a[0] for a in entry.authors}, {a[0] for a in candidate.authors})
    return [
        _arxiv_id_in_fields(entry, candidate),
        title_levenshtein(entry.title, candidate.title),
        jaccard(entry.words, candidate.words),
        jaccard(entry.char3, candidate.char3),
        token_set,
        containment,
        tfidf_cosine,
        first_match,
        surname_jaccard,
        year_match,
        year_diff,
        year_signed_diff,
        float(has_title),
        float(has_authors),
        has_year,
    ]


def build_features(
    entry: Bibtex,
    candidate: Reference,
    tfidf_cosine: float = 0.0,
    raw_text: str | None = None,
) -> list[float]:
    """Tạo vector đặc trưng cho MỘT cặp (mục BibTeX, ứng viên).

    TF-IDF cần cả tập ứng viên của bài để tính idf nên không tính được cho một cặp riêng lẻ;
    dùng build_feature_matrix để có đủ cột, hoặc truyền sẵn giá trị vào `tfidf_cosine`.

    Args:
        entry: Mục BibTeX (đã chuẩn hóa ở M2).
        candidate: Reference arXiv của cùng bài báo (từ M1).
        tfidf_cosine: Giá trị cosine TF-IDF đã tính sẵn (mặc định 0.0).
        raw_text: Chuỗi \\bibitem gốc của mục (tùy chọn); dùng để tìm ID arXiv và năm.

    Returns:
        Danh sách số thực, đúng thứ tự FEATURE_NAMES. Thiếu dữ liệu thì trả giá trị trung tính
        kèm cờ has_* = 0, không ném lỗi.
    """
    prepared_entry = _prepare_entry(entry, raw_text)
    return _pair_vector(prepared_entry, _prepare_candidate(candidate), tfidf_cosine)


def tfidf_cosine_matrix(entry_titles: list[str], candidate_titles: list[str]) -> np.ndarray:
    """Ma trận cosine TF-IDF (m, n) giữa tiêu đề các mục BibTeX và tiêu đề các ứng viên.

    idf chỉ học trên tiêu đề ứng viên của chính bài đó (không dùng Valid/Test của bài khác nên
    không rò rỉ dữ liệu): từ càng hiếm trong "thư viện" ứng viên thì càng có trọng số cao.

    Args:
        entry_titles: Tiêu đề đã làm sạch của các mục BibTeX.
        candidate_titles: Tiêu đề đã làm sạch của các ứng viên.

    Returns:
        Mảng (len(entry_titles), len(candidate_titles)); toàn 0 nếu không có từ nào để học.
    """
    zeros = np.zeros((len(entry_titles), len(candidate_titles)))
    corpus = [title for title in candidate_titles if title]
    if not corpus:
        return zeros
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
    try:
        vectorizer.fit(corpus)
    except ValueError:  # từ điển rỗng (chỉ toàn ký tự bị lọc)
        return zeros
    # Hàng đã chuẩn hóa L2 nên tích vô hướng chính là cosine.
    entry_vectors = vectorizer.transform(entry_titles)
    candidate_vectors = vectorizer.transform(candidate_titles)
    return (entry_vectors @ candidate_vectors.T).toarray()


def build_feature_matrix(
    entries: list[Bibtex],
    candidates: list[Reference],
    raw_texts: list[str | None] | None = None,
) -> np.ndarray:
    """Tạo tensor đặc trưng cho ma trận m x n của MỘT bài báo.

    Args:
        entries: m mục BibTeX của bài.
        candidates: n reference arXiv của cùng bài.
        raw_texts: Chuỗi \\bibitem gốc của từng mục, cùng thứ tự với entries (tùy chọn).

    Returns:
        Mảng float shape (m, n, len(FEATURE_NAMES)); phần tử [i, j] ứng với cặp
        (entries[i], candidates[j]).

    Raises:
        ValueError: Khi raw_texts có độ dài khác entries.
    """
    if raw_texts is not None and len(raw_texts) != len(entries):
        raise ValueError(f"Có {len(raw_texts)} chuỗi thô nhưng {len(entries)} mục BibTeX")
    shape = (len(entries), len(candidates), len(FEATURE_NAMES))
    if 0 in shape:
        return np.zeros(shape)
    raws = raw_texts if raw_texts is not None else [None] * len(entries)
    prepared_entries = [_prepare_entry(entry, raw) for entry, raw in zip(entries, raws)]
    prepared_candidates = [_prepare_candidate(candidate) for candidate in candidates]
    cosine = tfidf_cosine_matrix(
        [p.title for p in prepared_entries], [p.title for p in prepared_candidates]
    )
    tensor = np.zeros(shape)
    for i, prepared_entry in enumerate(prepared_entries):
        for j, prepared_candidate in enumerate(prepared_candidates):
            tensor[i, j] = _pair_vector(prepared_entry, prepared_candidate, cosine[i, j])
    return tensor
