"""P2-06 (Owner: Kiệt & Bảo) – Tiền xử lý văn bản và đặc trưng cho cặp (BibTeX, ứng viên).

Bài toán (M3): với mỗi mục BibTeX của một bài báo, xếp hạng các reference arXiv của CHÍNH bài
đó. Mỗi cặp (mục BibTeX, ứng viên) được biến thành một vector số (xem FEATURE_NAMES).

Cách dùng:
    tensor = build_feature_matrix(entries, candidates)  # shape (m, n, len(FEATURE_NAMES))
    tensor[i, j] là vector đặc trưng của cặp (entries[i], candidates[j]).

Quy ước thiếu dữ liệu (hai tầng; riêng ba cột composite_* thì tự xử lý NaN ngay khi tính, không chờ tầng 2):
    1. Khi TÍNH feature: feature so sánh hai phía mà thiếu đầu vào thì trả NaN, KHÔNG trả 0.
       Giá trị 0 chỉ dành cho "có dữ liệu nhưng không giống". Các cờ has_* luôn là 0/1. Tầng
       này cần giữ NaN vì composite_score dựa vào NaN để bỏ tín hiệu thiếu và chia lại trọng số.
    2. Sau khi tính xong (kể cả ba cột composite_*): fill_missing_features điền NaN bằng giá trị
       CÓ Ý NGHĨA theo từng feature (bảng MISSING_FILL_VALUES), ví dụ "không rõ năm" -> year_diff
       trung tính 0.5 chứ không phải 0 (0 nghĩa là trùng năm). Bật bằng build_feature_matrix(...,
       fill_missing=True); mô hình nhận tensor này không cần imputer nữa.

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
    "has_title",  # cả hai bên đều có tiêu đề
    "has_authors",  # cả hai bên đều có tác giả
    "has_year",  # cả hai bên đều có năm
    # Ba feature "ngữ cảnh" so sánh một cặp với các cặp khác trong cùng bài; chỉ tính được ở
    # build_feature_matrix (single-pair build_features để NaN). Điểm gốc: composite_score.
    # Theo spec của Kiệt, ở build_feature_matrix ba cột này KHÔNG bao giờ NaN: thiếu tiêu đề
    # thì hạng = 1.0 (tệ nhất), margin = 0.0 (trung tính); pool chỉ có 1 phần tử thì hạng = 0.0.
    "composite_rank_in_pool",  # hạng phân vị theo HÀNG: 0 = ref khớp nhất với entry, 1 = tệ nhất
    "composite_margin",  # điểm - điểm cao nhất của các ref còn lại cùng hàng (>0 nếu đứng nhất)
    "composite_col_rank_in_paper",  # hạng phân vị theo CỘT: 0 = entry khớp nhất với ref này
)

# Trọng số của composite_score (tổng = 1). Chỉ gồm các tín hiệu ĐÃ có trong FEATURE_NAMES.
# Thiếu tín hiệu nào (NaN) thì bỏ nó và chia lại cho tổng trọng số còn lại (xem
# composite_score_matrix).
#
# Nguồn gốc: giá trị ban đầu chọn theo heuristic (title là tín hiệu mạnh nhất sau ID,
# author trung bình, year yếu). CHƯA tune bằng ablation — sẽ kiểm chứng trên Train/Valid
# ở P4-01/P4-04 với 4 cấu hình:
#   A. Có composite_*, trọng số hiện tại (0.35/0.25/0.15/0.10/0.10/0.05)
#   B. Có composite_*, trọng số đều (1/6 mỗi tín hiệu)
#   C. KHÔNG có composite_* (bỏ 3 cột, model chỉ học từ F01–F15)
#   D. Có composite_*, trọng số hiện tại nhưng trên bài manual only (đối chứng)
# Chọn cấu hình cuối bằng MRR@5 trên Valid (manual, ưu tiên) trước khi báo cáo Test.
# KHÔNG chọn bằng Test.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "title_lev_sim": 0.35,
    "title_tfidf_cos": 0.25,
    "title_token_set": 0.15,
    "first_author_match": 0.10,
    "author_surname_jaccard": 0.10,
    "year_match": 0.05,
}
# Giá trị điền khi một feature bị NaN (thiếu dữ liệu), chọn theo Ý NGHĨA của từng feature chứ
# không phải median. Nguyên tắc: "thiếu" không được giả làm "rất giống" hay "rất khác"; việc thiếu
# vốn đã được cờ has_title/has_authors/has_year báo cho mô hình. Hằng số cố định (không fit từ
# dữ liệu) nên dùng được nguyên xi cho Train, Valid, Test.
MISSING_FILL_VALUES: dict[str, float] = {
    # Không có trường nào để dò ID, hoặc ứng viên không có ID: không có bằng chứng -> 0.
    "arxiv_id_in_fields": 0.5,
    # Độ giống tiêu đề/tác giả: thiếu dữ liệu thì không có bằng chứng giống -> 0.
    "title_lev_sim": 0.5,
    "title_word_jaccard": 0.5,
    "title_char3_jaccard": 0.5,
    "title_token_set": 0.5,
    "title_containment": 0.5,
    "title_tfidf_cos": 0.5,
    "first_author_match": 0.5,
    "author_surname_jaccard": 0.5,
    "year_match": 0.5,
    # year_diff là KHOẢNG CÁCH (0 = trùng năm, tốt nhất) nên không thể điền 0: điền giữa dải.
    "year_diff": 0.5,
    # Cờ có/không: không bao giờ NaN, điền 0 chỉ để bảng đầy đủ cột.
    "has_title": 0.0,
    "has_authors": 0.0,
    "has_year": 0.0,
    # Ba cột ngữ cảnh: build_feature_matrix đã tự xử lý thiếu theo spec của Kiệt nên không còn NaN;
    # giá trị dưới đây chỉ dùng cho đường single-pair build_features (không có ma trận để xếp
    # hạng). Cùng quy ước: không tính được điểm -> hạng 1.0 (tệ nhất), margin 0.0 (trung tính).
    "composite_rank_in_pool": 1.0,
    "composite_margin": 0.0,
    "composite_col_rank_in_paper": 1.0,
}
assert set(MISSING_FILL_VALUES) == set(FEATURE_NAMES), "mỗi feature phải có giá trị điền"

# Làm tròn điểm trước khi xếp hạng để hai điểm "bằng nhau" sai khác ~1e-16 vẫn được coi là hòa.
RANK_DECIMALS = 9
NAN = float("nan")

# Năm arXiv (preprint) thường sớm hơn năm xuất bản vài năm; chênh quá 5 năm coi như "rất xa".
YEAR_DIFF_CAP = 5

# Archive prefix hợp lệ của arXiv ID style cũ (trước 2007), theo
# https://arxiv.org/help/arxiv_identifier và danh sách archive chính thức của arXiv.
# Không có whitelist thì nhánh [a-z\-]+/\d{7} quá permissive: "foo/1234567" (URL lạ, page
# number...) cũng bị nhận là ID. Sắp xếp để archive dài (math-ph, nucl-ex) đứng trước
# archive ngắn (math, nlin) tránh backtrack không cần thiết.
_OLD_ARCHIVES = (
    "astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|nlin|nucl-ex|nucl-th|physics|"
    "quant-ph|math|cs|q-bio|q-fin|stat|eess|econ|solv-int|chao-dyn|adap-org|alg-geom|comp-gas|"
    "dg-ga|funct-an|patt-sol|supr-con|acc-phys|ao-sci|atom-ph|bayes-an|chem-ph|cmp-lg|mtrl-th|"
    "plasm-ph|q-alg"
)
# Khớp arXiv ID dạng mới "2310.12345" và dạng cũ "hep-th/9901001", "math.GT/0309136", kèm
# hậu tố version tùy chọn. Không cho chữ số dính ngay trước/sau để "2310.1234" không bị cắt ra
# từ "2310.12345". Lookbehind (?<![a-z\-]) trước _OLD_ARCHIVES ngăn match inside longer word:
# "xhep-th/9901001" không bị nhận nhầm thành "hep-th/9901001". Ví dụ:
# "arXiv:2310.12345v2" -> "2310.12345v2"; "see hep-th/9901001" -> "hep-th/9901001".
ARXIV_ID_PATTERN = re.compile(
    rf"(?<!\d)(?:\d{{4}}\.\d{{4,5}}|(?<![a-z\-])(?:{_OLD_ARCHIVES})(?:\.[a-z]{{2}})?/\d{{7}})"
    r"(?:v\d+)?(?!\d)",
    re.IGNORECASE,
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
    # Loại bỏ arXiv ID trước (có thể có version suffix) để "1806.01234" không bị
    # coi là năm 1806; sau đó mới tìm năm trong phần còn lại của chuỗi.
    cleaned = ARXIV_ID_PATTERN.sub(" ", raw)
    match = YEAR_PATTERN.search(cleaned)
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
    folded = ascii_fold(name).replace("{", "").replace("}", "")  # M{\"u}ller -> muller
    folded = NON_NAME_PATTERN.sub(" ", folded)
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
    """Tỉ lệ |A ∩ B| / |A ∪ B|; NaN nếu một trong hai tập rỗng (thiếu dữ liệu)."""
    if not set_a or not set_b:
        return NAN
    return len(set_a & set_b) / len(set_a | set_b)


def title_levenshtein(title_a: str, title_b: str) -> float:
    """Độ giống Levenshtein chuẩn hóa 0..1: 1 - khoảng_cách / độ_dài_chuỗi_dài_hơn.

    NaN nếu một trong hai chuỗi rỗng (rapidfuzz sẽ trả 1.0 cho hai chuỗi rỗng, sai ngữ nghĩa).
    """
    if not title_a or not title_b:
        return NAN
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
    has_text: bool = False  # chỉ phía BibTeX: có ít nhất một trường (url/journal/doi/raw) để dò ID


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
        has_text=any(text_fields),
    )


def _normalize_arxiv_id(raw: str | None) -> str:
    """Chuẩn hóa ID arXiv về dạng canonical: bỏ prefix "arXiv:", version suffix, hạ chữ thường.

    Dùng cùng logic với phía entry (`extract_arxiv_ids`) để hai bên so sánh được với nhau.
    Chuỗi không nhận ra là ID hợp lệ thì chỉ hạ chữ thường và strip version (fallback).

    Args:
        raw: ID thô từ S2, ví dụ "1706.03762v5", "arXiv:1706.03762", "hep-th/9901001".

    Returns:
        ID canonical, ví dụ "1706.03762", "hep-th/9901001"; rỗng nếu raw rỗng/None.
    """
    ids = extract_arxiv_ids(raw)
    if ids:
        # ref_arxiv_id là một trường đơn nên chỉ có một ID; sorted để kết quả
        # deterministic nếu dữ liệu lỗi có nhiều ID trong một chuỗi.
        return min(ids)
    return VERSION_SUFFIX_PATTERN.sub("", (raw or "").strip().lower())


def _prepare_candidate(candidate: Reference) -> _Prepared:
    title = preprocess(candidate.ref_title)
    return _Prepared(
        title=title,
        words=frozenset(title.split()),
        char3=frozenset(char_ngrams(title)),
        authors=tuple(author_keys(candidate.ref_authors)),
        year=parse_year(candidate.ref_submission_date),
        arxiv_id=_normalize_arxiv_id(candidate.ref_arxiv_id),
    )


def _arxiv_id_in_fields(entry: _Prepared, candidate: _Prepared) -> float:
    """1.0/0.0 nếu ID của ứng viên có/không nằm trong ID tìm thấy ở url/journal/doi/chuỗi thô.

    NaN khi mục BibTeX không có trường nào để dò (không thể kết luận "không có ID").
    """
    if not entry.has_text or not candidate.arxiv_id:
        return NAN
    return float(candidate.arxiv_id in entry.arxiv_ids)


def _year_features(entry: _Prepared, candidate: _Prepared) -> tuple[float, float, float]:
    """Trả về (year_match, year_diff, has_year); hai giá trị đầu NaN nếu thiếu."""
    if entry.year is None or candidate.year is None:
        return NAN, NAN, 0.0
    signed = candidate.year - entry.year
    # Chặn hai đầu để một năm sai/lạ (ví dụ 2030) không tạo giá trị ngoại lai.
    capped = max(-YEAR_DIFF_CAP, min(YEAR_DIFF_CAP, signed))
    return float(signed == 0), abs(capped) / YEAR_DIFF_CAP, 1.0


def _pair_vector(entry: _Prepared, candidate: _Prepared, tfidf_cosine: float) -> list[float]:
    """Vector đặc trưng của một cặp; feature thiếu đầu vào là NaN, ba cột ngữ cảnh để NaN."""
    has_title = bool(entry.title and candidate.title)
    has_authors = bool(entry.authors and candidate.authors)
    year_match, year_diff, has_year = _year_features(entry, candidate)
    if has_title:
        lev = title_levenshtein(entry.title, candidate.title)
        word_jaccard = jaccard(entry.words, candidate.words)
        char3_jaccard = jaccard(entry.char3, candidate.char3)
        token_set = fuzz.token_set_ratio(entry.title, candidate.title) / 100
        containment = fuzz.partial_ratio(entry.title, candidate.title) / 100
        tfidf = tfidf_cosine
    else:
        lev = word_jaccard = char3_jaccard = token_set = containment = tfidf = NAN
    if has_authors:
        first_match = float(entry.authors[0][0] == candidate.authors[0][0])
        surname_jaccard = jaccard({a[0] for a in entry.authors}, {a[0] for a in candidate.authors})
    else:
        first_match = surname_jaccard = NAN
    return [
        _arxiv_id_in_fields(entry, candidate),
        lev,
        word_jaccard,
        char3_jaccard,
        token_set,
        containment,
        tfidf,
        first_match,
        surname_jaccard,
        year_match,
        year_diff,
        float(has_title),
        float(has_authors),
        has_year,
        NAN,  # composite_rank_in_pool
        NAN,  # composite_margin
        NAN,  # composite_col_rank_in_paper
    ]


def build_features(
    entry: Bibtex,
    candidate: Reference,
    tfidf_cosine: float = NAN,
    raw_text: str | None = None,
) -> list[float]:
    """Tạo vector đặc trưng cho MỘT cặp (mục BibTeX, ứng viên).

    TF-IDF cần cả tập ứng viên của bài để tính idf nên không tính được cho một cặp riêng lẻ;
    dùng build_feature_matrix để có đủ cột, hoặc truyền sẵn giá trị vào `tfidf_cosine`.

    Args:
        entry: Mục BibTeX (đã chuẩn hóa ở M2).
        candidate: Reference arXiv của cùng bài báo (từ M1).
        tfidf_cosine: Giá trị cosine TF-IDF đã tính sẵn (mặc định NaN = chưa tính).
        raw_text: Chuỗi \\bibitem gốc của mục (tùy chọn); dùng để tìm ID arXiv và năm.

    Returns:
        Danh sách số thực, đúng thứ tự FEATURE_NAMES. Feature thiếu đầu vào là NaN (cờ has_*
        vẫn là 0/1), không ném lỗi. Ba cột composite_* luôn NaN vì cần cả ma trận của bài.
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
        Mảng (len(entry_titles), len(candidate_titles)). Ô nào có tiêu đề rỗng ở một trong hai
        bên là NaN; toàn NaN nếu không ứng viên nào có tiêu đề.
    """
    shape = (len(entry_titles), len(candidate_titles))
    nan_matrix = np.full(shape, NAN)
    corpus = [title for title in candidate_titles if title]
    if 0 in shape or not corpus:
        return nan_matrix
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
    try:
        vectorizer.fit(corpus)
    except ValueError:  # từ điển rỗng (chỉ toàn ký tự bị lọc)
        return nan_matrix
    # Hàng đã chuẩn hóa L2 nên tích vô hướng chính là cosine.
    entry_vectors = vectorizer.transform(entry_titles)
    candidate_vectors = vectorizer.transform(candidate_titles)
    result = (entry_vectors @ candidate_vectors.T).toarray()
    result[[not title for title in entry_titles], :] = NAN
    result[:, [not title for title in candidate_titles]] = NAN
    return result

# Ba cột composite_* được tính từ COMPOSITE_WEIGHTS. Khi che feature trùng luật
# (ml_design.md mục 3.5), che một tín hiệu gốc của composite là chưa đủ — phải che cả
# ba cột composite_* nếu không model vẫn đọc lại luật qua hạng/margin.
_COMPOSITE_COLUMNS: tuple[str, ...] = (
    "composite_rank_in_pool",
    "composite_margin",
    "composite_col_rank_in_paper",
)


def columns_to_mask(rule_features: list[str]) -> list[str]:
    """Cột cần che khi huấn luyện, nếu luật gán nhãn tự động dùng ``rule_features``.

    Args:
        rule_features: Tên các feature mà luật auto dùng (ví dụ ["title_lev_sim"]).

    Returns:
        Danh sách feature cần che, đã gồm các cột composite_* liên quan. Không trùng phần tử.

    Raises:
        KeyError: Khi một tên không có trong FEATURE_NAMES (thường do gõ sai).
    """
    unknown = [name for name in rule_features if name not in FEATURE_NAMES]
    if unknown:
        raise KeyError(f"feature không tồn tại: {unknown}")
    # dict.fromkeys để giữ thứ tự và khử trùng nếu caller truyền trùng.
    cols = list(dict.fromkeys(rule_features))
    if any(name in COMPOSITE_WEIGHTS for name in rule_features):
        cols += [name for name in _COMPOSITE_COLUMNS if name not in cols]
    return cols

def composite_score_matrix(tensor: np.ndarray) -> np.ndarray:
    """Điểm tương đồng tổng hợp S[i, j] cho từng cặp, làm gốc cho ba feature ngữ cảnh.

    S là trung bình có trọng số (COMPOSITE_WEIGHTS) của các tín hiệu còn dùng được; tín hiệu
    NaN bị bỏ và trọng số được chia lại, để một cặp thiếu tác giả/năm không bị "phạt" chỉ vì
    thiếu. S là NaN khi cặp thiếu tiêu đề (has_title = 0), vì tiêu đề chiếm 75% trọng số.

    Args:
        tensor: Mảng (m, n, F) đã có đủ các cột trong COMPOSITE_WEIGHTS và has_title.

    Returns:
        Mảng (m, n) điểm S trong [0, 1], hoặc NaN.
    """
    names = list(COMPOSITE_WEIGHTS)
    columns = [FEATURE_NAMES.index(name) for name in names]
    weights = np.array([COMPOSITE_WEIGHTS[name] for name in names])
    values = tensor[..., columns]
    available = ~np.isnan(values)
    total_weight = (available * weights).sum(axis=-1)
    weighted_sum = (np.where(available, values, 0.0) * weights).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = weighted_sum / total_weight
    has_title = tensor[..., FEATURE_NAMES.index("has_title")] == 1.0
    return np.where(has_title & (total_weight > 0), score, NAN)


def _dense_percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Hạng dense giảm dần chuẩn hóa về [0, 1] cho một vector điểm; 0 = tốt nhất.

    Theo spec của Kiệt, với n = số phần tử của vector (cả pool hàng hoặc cả cột bài báo):
        - điểm NaN (thiếu tiêu đề / không tính được): 1.0, tệ nhất;
        - điểm hợp lệ: (dense_rank - 1) / max(1, n - 1). Vector chỉ có n = 1 phần tử hợp lệ
          cho 0.0 (mặc định đứng đầu); một điểm hợp lệ duy nhất giữa các NaN cũng được 0.0.
    Mẫu số là n (kích thước pool) chứ không phải số điểm hợp lệ, nên ô hợp lệ tệ nhất vẫn
    nhỏ hơn 1.0 khi pool có ô NaN: "thiếu dữ liệu" luôn tệ hơn "có dữ liệu nhưng kém".
    Không bao giờ trả NaN.
    """
    result = np.ones(scores.shape)
    valid = ~np.isnan(scores)
    if valid.any():
        rounded = np.round(scores[valid], RANK_DECIMALS)
        # Điểm cao nhất -> chỉ số 0; điểm bằng nhau cùng chỉ số (dense rank, tính từ 0).
        dense_index = np.searchsorted(np.unique(-rounded), -rounded)
        result[valid] = dense_index / max(1, scores.size - 1)
    return result


def _margin_by_row(scores: np.ndarray) -> np.ndarray:
    """Mỗi ô trừ điểm cao nhất của các ô KHÁC cùng hàng: S[i,j] - max_{k != j} S[i,k].

    Dương nếu ref j đứng nhất hàng, 0 nếu hòa nhất, âm nếu thua. Theo spec của Kiệt, ô NaN
    (thiếu tiêu đề) hoặc hàng có dưới 2 điểm hợp lệ (không có đối thủ) nhận 0.0 (trung tính,
    không có sự áp đảo). Không bao giờ trả NaN.
    """
    result = np.zeros(scores.shape)
    for i, row in enumerate(scores):
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        rounded = np.round(row[valid], RANK_DECIMALS)
        second, first = np.sort(rounded)[-2:]
        # Ô đứng nhất so với á quân; mọi ô còn lại so với quán quân. Hòa nhất: second == first.
        best_of_others = np.where(rounded == first, second, first)
        result[i, valid] = rounded - best_of_others
    return result


def _fill_context_features(tensor: np.ndarray) -> None:
    """Điền ba cột composite_* vào tensor (sửa tại chỗ) từ composite_score_matrix."""
    score = composite_score_matrix(tensor)
    tensor[..., FEATURE_NAMES.index("composite_rank_in_pool")] = np.apply_along_axis(
        _dense_percentile_rank, 1, score
    )
    tensor[..., FEATURE_NAMES.index("composite_margin")] = _margin_by_row(score)
    tensor[..., FEATURE_NAMES.index("composite_col_rank_in_paper")] = np.apply_along_axis(
        _dense_percentile_rank, 0, score
    )


def fill_missing_features(features: np.ndarray) -> np.ndarray:
    """Điền mọi NaN bằng giá trị có ý nghĩa theo từng feature (MISSING_FILL_VALUES).

    Phải gọi SAU khi đã tính xong ba cột composite_*: composite_score cần NaN để biết tín hiệu
    nào thiếu mà bỏ đi. Nếu điền trước, feature thiếu sẽ bị tính như điểm 0 và kéo thấp điểm
    tổng hợp một cách vô lý.

    Args:
        features: Mảng bất kỳ có trục cuối là len(FEATURE_NAMES), ví dụ (m, n, F) từ
            build_feature_matrix, (n_pairs, F) hoặc một vector (F,).

    Returns:
        Mảng mới cùng shape, không còn NaN (mảng đầu vào không bị sửa).

    Raises:
        ValueError: Khi trục cuối không khớp số feature (thường do sai thứ tự/thiếu cột).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 0 or features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"Trục cuối phải có {len(FEATURE_NAMES)} cột, nhận shape {features.shape}")
    fill = np.array([MISSING_FILL_VALUES[name] for name in FEATURE_NAMES])
    # fill có shape (F,) nên tự broadcast theo trục cuối của features.
    return np.where(np.isnan(features), fill, features)


def build_feature_matrix(
    entries: list[Bibtex],
    candidates: list[Reference],
    raw_texts: list[str | None] | None = None,
    fill_missing: bool = False,
) -> np.ndarray:
    """Tạo tensor đặc trưng cho ma trận m x n của MỘT bài báo.

    Hạng theo cột (composite_col_rank_in_paper) so entry với các entry khác của bài, nên
    `entries` phải là TOÀN BỘ entry của bài sau dedup, kể cả entry chưa gán nhãn; nếu chỉ
    truyền entry đã gán nhãn thì phân phối feature này lệch giữa bài manual và bài auto.
    `candidates` phải là toàn bộ pool ref của bài ở cả Train/Valid/Test (không lấy mẫu).

    Args:
        entries: m mục BibTeX của bài.
        candidates: n reference arXiv của cùng bài.
        raw_texts: Chuỗi \\bibitem gốc của từng mục, cùng thứ tự với entries (tùy chọn).
        fill_missing: True để điền NaN bằng fill_missing_features (khuyên dùng khi đưa vào mô
            hình); False giữ NaN thô để debug và để các hàm composite_* tự dùng.

    Returns:
        Mảng float shape (m, n, len(FEATURE_NAMES)); phần tử [i, j] ứng với cặp
        (entries[i], candidates[j]). Không còn NaN nếu fill_missing=True.

    Raises:
        ValueError: Khi raw_texts có độ dài khác entries.
    """
    if raw_texts is not None and len(raw_texts) != len(entries):
        raise ValueError(f"Có {len(raw_texts)} chuỗi thô nhưng {len(entries)} mục BibTeX")
    shape = (len(entries), len(candidates), len(FEATURE_NAMES))
    if 0 in shape:
        return np.full(shape, NAN)
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
    _fill_context_features(tensor)
    # Điền SAU _fill_context_features: composite_score cần NaN gốc để bỏ tín hiệu thiếu.
    return fill_missing_features(tensor) if fill_missing else tensor
