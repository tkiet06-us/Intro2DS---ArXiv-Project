"""P2-06 – Tiền xử lý văn bản và tạo đặc trưng cho cặp (BibTeX, ứng viên)."""

import re
import unicodedata
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# --- BẢNG ÁNH XÁ VÀ PATTERN TIỀN XỬ LÝ LATEX ---

# Ánh xá các ký tự có dấu LaTeX đặc biệt sang ký tự Unicode chuẩn trước khi xóa lệnh
LATEX_ACCENTS = [
    (re.compile(r'\\(?:["\'`^~]|v|u|H|r|c)\{?([a-zA-Z])\}?'), r"\1"),  # \"{u}, \'e, \v{r}, \c{c}
    (re.compile(r"\\ss\{?\}?"), "ss"),  # \ss -> ss
    (re.compile(r"\\ae\{?\}?"), "ae"),  # \ae -> ae
    (re.compile(r"\\oe\{?\}?"), "oe"),  # \oe -> oe
    (re.compile(r"\\aa\{?\}?"), "a"),   # \aa -> a
    (re.compile(r"\\o\{?\}?"), "o"),    # \o -> o
]

# Khớp công thức toán inline $...$ hoặc \( ... \)
MATH_INLINE_PATTERN = re.compile(r"\$.*?\$|\\\(.*?\\\)")

# Khớp các lệnh LaTeX tổng quát dạng \command hoặc \command*
LATEX_COMMAND_PATTERN = re.compile(r"\\[a-zA-Z]+\*?")

# Ký tự không phải chữ cái, chữ số hoặc khoảng trắng
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")

# Khoảng trắng thừa
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Biểu thức chính quy trích xuất năm (4 chữ số từ 1900 đến 2099)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Biểu thức chính quy nhận diện arXiv ID (dạng mới 2106.01234 hoặc cũ hep-th/9901001)
ARXIV_ID_PATTERN = re.compile(
    r"\b(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]+)?/\d{7})(?:v\d+)?\b", re.IGNORECASE
)

# Danh sách stop-words tiếng Anh cơ bản cho tiêu đề (tùy chọn thử nghiệm)
COMMON_STOP_WORDS = {
    "a", "an", "the", "on", "in", "at", "for", "of", "with", "by", "from",
    "to", "and", "or", "is", "are", "was", "were", "be", "been", "using",
    "based", "via", "through", "towards", "with", "into"
}


def preprocess(text: str | None, remove_stopwords: bool = False) -> str:
    """
    Làm sạch chuỗi văn bản thô để so sánh.

    Quy trình xử lý các tình huống biên:
    1. Trả về chuỗi rỗng nếu text là None hoặc rỗng.
    2. Giải mã các lệnh ký tự đặc biệt/dấu LaTeX (\"{u} -> u, \c{c} -> c, \ss -> ss).
    3. Loại bỏ công thức toán inline $...$ để tránh nhiễu ký tự toán học.
    4. Bỏ các lệnh LaTeX còn lại (\textbf, \emph, \url...).
    5. Khử dấu Unicode (NFKD decomposition: Müller -> Muller).
    6. Hạ chữ thường (lowercasing).
    7. Loại bỏ dấu câu và ký tự đặc biệt.
    8. Gộp khoảng trắng thừa và trim.
    9. (Tùy chọn) Loại bỏ stop-words.
    """
    if not text:
        return ""

    # 1. Chuyển đổi các dấu/ký tự đặc biệt LaTeX về ký tự cơ sở
    for pattern, repl in LATEX_ACCENTS:
        text = pattern.sub(repl, text)

    # 2. Xóa toán inline
    text = MATH_INLINE_PATTERN.sub(" ", text)

    # 3. Xóa các lệnh LaTeX còn lại và ngoặc nhọn
    text = LATEX_COMMAND_PATTERN.sub(" ", text)
    text = text.replace("{", "").replace("}", "")

    # 4. Chuẩn hóa NFKD để tách dấu khỏi ký tự gốc và xóa dấu
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # 5. Hạ chữ thường và xóa ký tự không phải alnum
    text = NON_ALNUM_PATTERN.sub(" ", text.lower())
    cleaned = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    # 6. Loại bỏ stop-words nếu được yêu cầu
    if remove_stopwords and cleaned:
        words = [w for w in cleaned.split() if w not in COMMON_STOP_WORDS]
        return " ".join(words)

    return cleaned


def extract_surnames(author_str: str | None) -> list[str]:
    """Trích xuất và làm sạch danh sách họ (surname) từ chuỗi tác giả.
    
    Xử lý các định dạng:
    - "Vaswani, Ashish and Shazeer, Noam" (dạng BibTeX chuẩn với phẩy)
    - "Ashish Vaswani, Noam Shazeer" (dạng viết xuôi)
    - "Vaswani et al." hoặc "Vaswani and others"
    """
    if not author_str:
        return []

    # Bỏ cụm "et al" hoặc "and others"
    raw_str = re.sub(r"\bet\s+al\b|\band\s+others\b", "", author_str, flags=re.IGNORECASE)
    
    # Phân tách các tác giả theo từ khóa "and" hoặc dấu chấm phẩy
    raw_authors = re.split(r"\band\b|;", raw_str, flags=re.IGNORECASE)
    surnames = []

    for author in raw_authors:
        cleaned_author = preprocess(author)
        if not cleaned_author:
            continue
        
        parts = cleaned_author.split()
        if "," in author:
            # Nếu có dấu phẩy: "Surname, Firstname" -> lấy phần tử đầu tiên
            surname = preprocess(author.split(",")[0])
            if surname:
                surnames.append(surname)
        else:
            # Viết xuôi: "Firstname Surname" -> lấy từ cuối cùng
            if parts:
                surnames.append(parts[-1])

    return surnames


def extract_year(text: str | None) -> int | None:
    """Trích xuất năm xuất bản (4 chữ số) từ chuỗi văn bản thô."""
    if not text:
        return None
    match = YEAR_PATTERN.search(text)
    return int(match.group(1)) if match else None


def extract_arxiv_id(text: str | None) -> str | None:
    """Trích xuất arXiv ID từ chuỗi thô nếu tác giả có ghi kèm."""
    if not text:
        return None
    match = ARXIV_ID_PATTERN.search(text)
    if match:
        return match.group(0).lower().replace("v", "")  # Bỏ suffix version nếu có
    return None


# --- CÁC HÀM TÍNH ĐỘ TƯƠNG ĐỒNG VÀ ĐẶC TRƯNG ---


def title_levenshtein(title_a: str, title_b: str) -> float:
    """Độ giống Levenshtein chuẩn hóa (0.0 .. 1.0) dùng C++ rapidfuzz."""
    if not title_a or not title_b:
        return 0.0
    return Levenshtein.normalized_similarity(title_a, title_b)


def word_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity trên tập từ."""
    set_a, set_b = set(text_a.split()), set(text_b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def char_ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Jaccard similarity trên tập character n-gram (mặc định trigram)."""
    if len(text_a) < n or len(text_b) < n:
        return 0.0
    ngrams_a = {text_a[i : i + n] for i in range(len(text_a) - n + 1)}
    ngrams_b = {text_b[i : i + n] for i in range(len(text_b) - n + 1)}
    
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union if union > 0 else 0.0


def containment_ratio(short_text: str, long_text: str) -> float:
    """Đo mức độ chứa đựng (containment) bằng partial ratio.
    
    Rất hiệu quả khi một bên bị viết tắt (ví dụ chứa 'et al.' hoặc tiêu đề bị cắt ngắn).
    """
    if not short_text or not long_text:
        return 0.0
    return fuzz.partial_ratio(short_text, long_text) / 100.0


def build_feature_vector(
    bib_entry: dict[str, Any], candidate_ref: dict[str, Any]
) -> dict[str, float]:
    """Tạo vector đặc trưng (feature vector) cho một cặp (BibTeX entry, ứng viên arXiv).
    
    Đầu vào:
    - bib_entry: Dict chứa 'raw_text', 'title', 'authors', 'year', 'citation_key'
    - candidate_ref: Dict chứa 'ref_title', 'ref_authors', 'ref_submission_date', 'ref_arxiv_id'
    
    Trả về Dict gồm các features từ F01 đến F12 kèm cờ thiếu dữ liệu defensive.
    """
    features: dict[str, float] = {}

    # Preprocessing các trường cơ bản
    raw_text = bib_entry.get("raw_text", "")
    bib_title = preprocess(bib_entry.get("title"))
    cand_title = preprocess(candidate_ref.get("ref_title"))
    
    # 1. Cờ thiếu dữ liệu
    has_title = float(bool(bib_title and cand_title))
    features["has_title"] = has_title

    # 2. Độ tương đồng tiêu đề (Title Features)
    if has_title:
        features["title_lev"] = title_levenshtein(bib_title, cand_title)
        features["title_word_jaccard"] = word_jaccard(bib_title, cand_title)
        features["title_char3_jaccard"] = char_ngram_jaccard(bib_title, cand_title, n=3)
        features["title_containment"] = containment_ratio(bib_title, cand_title)
    else:
        features["title_lev"] = 0.0
        features["title_word_jaccard"] = 0.0
        features["title_char3_jaccard"] = 0.0
        features["title_containment"] = 0.0

    # 3. Kiểm tra sự xuất hiện của arXiv ID trong chuỗi BibTeX thô
    cand_arxiv_id = candidate_ref.get("ref_arxiv_id", "")
    extracted_id = extract_arxiv_id(raw_text)
    if cand_arxiv_id and extracted_id and (cand_arxiv_id.lower() in extracted_id.lower()):
        features["arxiv_id_in_raw"] = 1.0
    else:
        features["arxiv_id_in_raw"] = 0.0

    # 4. Đặc trưng tác giả (Author Features)
    bib_authors_raw = bib_entry.get("authors")
    cand_authors_raw = candidate_ref.get("ref_authors")

    bib_surnames = extract_surnames(
        " ".join(bib_authors_raw) if isinstance(bib_authors_raw, list) else bib_authors_raw
    )
    cand_surnames = extract_surnames(
        " ".join(cand_authors_raw) if isinstance(cand_authors_raw, list) else cand_authors_raw
    )

    has_authors = float(bool(bib_surnames and cand_surnames))
    features["has_authors"] = has_authors

    if has_authors:
        # Tác giả đầu tiên có trùng họ không
        features["first_author_match"] = float(bib_surnames[0] == cand_surnames[0])
        # Jaccard trên tập họ
        features["author_surname_jaccard"] = word_jaccard(
            " ".join(bib_surnames), " ".join(cand_surnames)
        )
    else:
        features["first_author_match"] = 0.0
        features["author_surname_jaccard"] = 0.0

    # 5. Đặc trưng năm xuất bản (Year Features)
    bib_year = bib_entry.get("year") or extract_year(raw_text)
    cand_year = candidate_ref.get("year") or extract_year(candidate_ref.get("ref_submission_date"))

    has_year = float(bool(bib_year and cand_year))
    features["has_year"] = has_year

    if has_year and bib_year is not None and cand_year is not None:
        features["year_match"] = float(bib_year == cand_year)
        # Hiệu số năm có dấu: (cand_year - bib_year). Năm arXiv thường <= năm xuất bản chính thức
        features["year_signed_diff"] = float(cand_year - bib_year)
        features["year_abs_diff"] = float(abs(cand_year - bib_year))
    else:
        features["year_match"] = 0.0
        features["year_signed_diff"] = 0.0
        features["year_abs_diff"] = 99.0  # Giá trị phạt khi thiếu thông tin năm

    return features