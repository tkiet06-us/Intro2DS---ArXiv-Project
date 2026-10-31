"""Module LaTeX Cleanup & Math Normalization (task P2-05).

Vị trí trong pipeline (latex_spec.md Mục 1.2): chạy SAU latex_parser (đã gộp \\input) và
TRƯỚC tree_builder và dedup:

    latex_parser -> latex_cleanup -> tree_builder / bibtex_extractor -> dedup -> export

Việc module làm (đúng phạm vi "Được phép" của Mục 1.2):
1. Xóa lệnh định dạng hiển thị thừa: ``\\centering``, ``\\raggedright``, ``\\bf``, ``\\it``,
       ``\\vspace{..}``, ``\\hspace{..}``, kích cỡ chữ (``\\small``...), đường kẻ booktabs
       (``\\midrule``...) và tùy chọn vị trí float ``[htbp]`` của figure/table.
    2. Chuẩn hóa toán (Mục 5.2): khối ``$$..$$`` và ``\\[..\\]`` ->
       ``\\begin{equation}..\\end{equation}``;
       toán nội dòng ``\\(..\\)`` và ``\\begin{math}..\\end{math}`` -> ``$..$``.
    3. Xóa comment ``%`` (Mục 6.5) và dọn khoảng trắng thừa, để hai version của một bài
       không khác nhau chỉ vì định dạng khi dedup so khớp nội dung.

Điều CẤM (Mục 1.2): không xóa heading, ``figure``/``table``/``equation`` (và các môi trường
toán khác), ``\\caption``, ``\\label``, ``\\cite``, ``\\appendix``, ``thebibliography``. Vì vậy:
- Nội dung bên trong môi trường toán, ``$..$`` và verbatim luôn được giữ NGUYÊN VĂN
      (Mục 3.3: ``content`` của formula là khối TeX thô).
- Lệnh ``\\\\`` (xuống dòng trong tabular/align) tuyệt đối không bị đụng tới.

Cách làm: MỘT regex duy nhất quét văn bản từ trái sang phải. Các vùng "bảo vệ" (verbatim,
comment, toán, escape như ``\\%`` ``\\$`` ``\\\\``) được nhận diện trước nên lệnh định dạng nằm
bên trong chúng không bị xóa nhầm. Không dùng thư viện LaTeX ngoài (Mục 1.3: chỉ regex).
"""
import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Danh sách lệnh bị xóa. Muốn thêm/bớt lệnh thì sửa ở đây, không cần đụng regex bên dưới.
# ---------------------------------------------------------------------------
# Lệnh đổi font kiểu cũ (\bf, \it...) và kiểu mới (\bfseries...). KHÔNG gồm \em, \emph,
# \textbf{...}: đó là nhấn mạnh có nội dung bọc bên trong, spec không cho phép xóa.
_FONT_SWITCHES = (
    "bf", "it", "tt", "sf", "rm", "sc", "sl",
    "bfseries", "itshape", "ttfamily", "sffamily", "rmfamily",
    "scshape", "slshape", "mdseries", "upshape", "normalfont",
)  # fmt: skip
_SIZE_SWITCHES = (
    "tiny", "scriptsize", "footnotesize", "small", "normalsize",
    "large", "Large", "LARGE", "huge", "Huge",
)  # fmt: skip
# Canh lề, ngắt trang, khoảng cách dọc cố định: chỉ ảnh hưởng hiển thị.
_LAYOUT_SWITCHES = (
    "centering", "raggedright", "raggedleft", "noindent",
    "newpage", "clearpage", "cleardoublepage", "bigskip", "medskip", "smallskip",
)  # fmt: skip
# Ký tự cho phép trong tùy chọn vị trí float: [htbp], [!ht], [H], [t]... Giới hạn như vậy để
# không xóa nhầm một "[" mở đầu văn bản thật ngay sau \begin{figure}.
_FLOAT_ENVS = r"(?:figure|table|sidewaysfigure|sidewaystable)\*?"

# ---------------------------------------------------------------------------
# Các mảnh regex. Mỗi mảnh nhận diện MỘT loại token; thứ tự ghép ở _TOKEN_RE là thứ tự ưu
# tiên khi hai mảnh khớp cùng một vị trí.
# ---------------------------------------------------------------------------
_NAME_END = r"(?![A-Za-z@])"  # \bf không được khớp \bfseries hay \bfnew
_SP = r"[ \t]*"  # TeX nuốt khoảng trắng sau control word nên ta cũng nuốt theo

def _names(names: tuple[str, ...]) -> str:
    return "|".join(names)

# Lệnh xóa hẳn (thay bằng chuỗi rỗng).
_R_SWITCH = rf"\\(?:{_names(_FONT_SWITCHES + _SIZE_SWITCHES + _LAYOUT_SWITCHES)}){_NAME_END}{_SP}"
# Khoảng trắng tường minh: thay bằng MỘT dấu cách để "a\hspace{1em}b" không dính thành "ab".
_R_SPACING = rf"\\(?:[vh]space\*?{_SP}\{{[^{{}}\n]*\}}|[vh]fill{_NAME_END}){_SP}"
# Đường kẻ booktabs (task P2-05 nêu rõ \midrule). \hline KHÔNG xóa vì thuộc cấu trúc tabular.
_R_RULE = (
    rf"\\(?:(?:top|mid|bottom)rule{_NAME_END}(?:{_SP}\[[^\]\n]*\])?"
    rf"|cmidrule{_SP}(?:\([^)\n]*\))?{_SP}\{{[^{{}}\n]*\}}){_SP}"
)
_R_ANY = rf"(?:{_R_SWITCH}|{_R_SPACING}|{_R_RULE})"

_ENV_VERBATIM = r"verbatim\*?|lstlisting|minted|rawhtml|comment"
# Các môi trường toán bắt buộc nhận diện ở Mục 5.2 (giữ nguyên, không chuẩn hóa thêm).
_ENV_MATH = r"equation|align|gather|multline|flalign|eqnarray|displaymath"

_TOKEN_RE = re.compile(
    "|".join(
        [
            # --- Vùng BẢO VỆ khối: trả về nguyên văn ---
            rf"(?P<verbatim>\\begin\{{(?P<vb_name>{_ENV_VERBATIM})\}}.*?\\end\{{(?P=vb_name)\}})",
            r"(?P<iffalse>\\iffalse\b.*?\\fi\b)",
            rf"(?P<mathenv>\\begin\{{(?P<me_name>(?:{_ENV_MATH})\*?)\}}.*?\\end\{{(?P=me_name)\}})",

            # --- Vùng BẢO VỆ nội dòng (URL, Code) chứa ký tự đặc biệt như % ---
            r"(?P<verb>\\verb\*?(?P<vdelim>[^a-zA-Z0-9]).*?(?P=vdelim))",
            r"(?P<url>\\(?:url|path)\{[^{}\n]*\})",
            r"(?P<href>\\href\{[^{}\n]*\}\{[^{}\n]*\})",
            r"(?P<inlinecode>\\lstinline(?:\[[^\]\n]*\])?(?:\{[^{}\n]*\}|(?P<cdelim1>[^a-zA-Z0-9{]).*?(?P=cdelim1))|\\mintinline(?:\[[^\]\n]*\])?\{[^{}\n]*\}(?:\{[^{}\n]*\}|(?P<cdelim2>[^a-zA-Z0-9{]).*?(?P=cdelim2)))",

            # --- Toán khối/nội dòng cần CHUẨN HÓA ---
            r"(?P<dollars>\$\$(?P<dd_body>.*?)\$\$)",
            r"(?P<bracket>\\\[(?P<br_body>.*?)\\\])",
            r"(?P<paren>\\\((?P<pa_body>.*?)\\\))",
            r"(?P<mathbegin>\\begin\{math\}(?P<mb_body>.*?)\\end\{math\})",

            # --- Escape: phải đứng TRƯỚC các mảnh dùng \ hoặc $ hoặc % ---
            r"(?P<linebreak>\\\\\*?(?:\[[^\]\n]*\])?)",
            r"(?P<escape>\\[%$&#_{}])",
            
            # --- Comment ---
            r"(?P<comment>%[^\n]*(?:\n[ \t]*)?)",  # TeX: % nuốt cả xuống dòng + thụt đầu dòng sau
            r"(?P<inline>\$(?:[^$\\\n]|\\.|\n(?![ \t]*\n))+\$)",  # $..$ không vượt qua dòng trống

            # --- Dọn định dạng ---
            rf"(?P<float>(?P<fl_begin>\\begin\{{{_FLOAT_ENVS}\}}){_SP}\[[!htbpHT]*\])",
            rf"(?P<fmtline>^[ \t]*{_R_ANY}+(?:\n|\Z))",  # dòng chỉ chứa lệnh định dạng: xóa cả dòng
            rf"(?P<spacing>{_R_SPACING})",
            rf"(?P<fmt>{_R_SWITCH}|{_R_RULE})",

            # --- Dọn khoảng trắng ---
            r"(?P<trailing>[ \t]+(?=\n|\Z))",
            r"(?P<blank>\n(?:[ \t]*\n){2,})",  # 3+ xuống dòng liên tiếp -> 1 dòng trống
            r"(?P<multispace>(?<=[^ \t\n])[ \t]{2,})",  # nhiều dấu cách giữa dòng -> 1
        ]
    ),
    re.DOTALL | re.MULTILINE,
)

# Các môi trường cần cân bằng \begin/\end; chỉ để CẢNH BÁO (Mục 7.2: không được crash).
_BALANCE_ENVS = ("figure", "table", "tabular", "equation", "align", "itemize", "thebibliography")
_KEEP = frozenset({
    "verbatim", "iffalse", "mathenv", "linebreak", "escape", "inline",
    "verb", "url", "href", "inlinecode"
})

def _display_to_equation(body: str) -> str | None:
    """Gói thân công thức khối vào môi trường ``equation`` (Mục 5.2).

    Ép về 1 canonical duy nhất trên một dòng (single-line) để thuật toán
    deduplication không nhận diện sai lệch do khác biệt dòng trống / thụt lề.

    Args:
        body: Phần thân giữa ``$$...$$`` hoặc ``\\[...\\]``.

    Returns:
        Khối TeX hoàn chỉnh, hoặc None khi thân rỗng.
    """
    body = body.strip()
    if not body:
        return None
    return f"\\begin{{equation}} {body} \\end{{equation}}"

def _make_replacer(strip_comments: bool, math_only: bool = False) -> Callable[[re.Match[str]], str]:
    """Tạo hàm thay thế cho ``_TOKEN_RE.sub``, quyết định số phận từng loại token."""
    def replace(match: re.Match[str]) -> str:
        kind = match.lastgroup
        text = match.group(0)

        if kind in _KEEP:
            return text

        if kind == "comment":
            # Nếu đang ở chế độ chỉ chuẩn hóa toán, không xóa comment (trừ khi có spec khác)
            return "" if (strip_comments and not math_only) else text

        if kind == "dollars":
            return _display_to_equation(match.group("dd_body")) or text
        if kind == "bracket":
            return _display_to_equation(match.group("br_body")) or text
        if kind in ("paren", "mathbegin"):
            body = match.group("pa_body" if kind == "paren" else "mb_body").strip()
            return f"${body}$" if body else text
            
        # Nếu chỉ chạy normalize_math, trả về nguyên văn mọi định dạng
        if math_only:
            return text

        if kind == "float":
            return match.group("fl_begin")  # bỏ [htbp], giữ \begin{figure}
        if kind in ("fmtline", "fmt", "trailing"):
            return ""
        if kind == "spacing":
            return " "
        if kind == "blank":
            return "\n\n"
        if kind == "multispace":
            return " "

        return text  # không tới được; giữ nguyên cho an toàn
    return replace

def _warn_unbalanced(text: str) -> None:
    """Ghi cảnh báo khi số ``\\begin`` và ``\\end`` của một môi trường không khớp."""
    for env in _BALANCE_ENVS:
        n_begin = len(re.findall(rf"\\begin\{{{env}\*?\}}", text))
        n_end = len(re.findall(rf"\\end\{{{env}\*?\}}", text))
        if n_begin != n_end:
            logger.warning("Môi trường %s lệch thẻ: %d \\begin vs %d \\end", env, n_begin, n_end)
    if text.count("$$") % 2:
        logger.warning("Số lượng '$$' lẻ: có thể có công thức khối thiếu thẻ đóng")

def normalize_math(text: str) -> str:
    """Chuẩn hóa riêng các công thức toán khối và nội dòng.
    
    Hàm này được cung cấp để tương thích ngược với API đã định nghĩa. Nó chỉ can thiệp
    vào các tag liên quan đến toán và giữ nguyên toàn bộ định dạng, comment khác.
    """
    if not text:
        return text
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    replacer = _make_replacer(strip_comments=False, math_only=True)
    return _TOKEN_RE.sub(replacer, cleaned)

def cleanup_latex(text: str, *, strip_comments: bool = True) -> str:
    """Làm sạch chuỗi LaTeX đã gộp: xóa định dạng thừa, chuẩn hóa toán, dọn khoảng trắng.

    Hàm là thuần túy (không I/O), xác định (cùng đầu vào luôn cho cùng đầu ra) và lũy đẳng:
    ``cleanup_latex(cleanup_latex(x)) == cleanup_latex(x)``. Không bao giờ raise vì nội dung
    LaTeX hỏng (Mục 7): gặp thẻ lệch thì chỉ ghi warning rồi trả văn bản đã xử lý được.
    """
    if not text:
        return text
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    replacer = _make_replacer(strip_comments)
    
    # Một lượt có thể sinh ra khoảng trắng mới (vd xóa cả dòng làm hai dòng trống dính nhau),
    # nên lặp tới khi ổn định; lượt đầu xử lý hầu hết, lượt sau chỉ dọn phần dư.
    for _ in range(5):
        new = _TOKEN_RE.sub(replacer, cleaned)
        if new == cleaned:
            break
        cleaned = new
        
    _warn_unbalanced(cleaned)
    return cleaned