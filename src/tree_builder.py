"""Tree Builder: dựng cây phân cấp của một version từ chuỗi LaTeX đã gộp.

Đầu vào là chuỗi do latex_parser.resolve_paper trả về (lý tưởng là đã qua latex_cleanup, nhưng
module này không phụ thuộc vào việc đó). Đầu ra là danh sách nút theo thứ tự duyệt tiền thứ tự,
cùng hình dạng với `Node` trong tests/make_mock.py dedup và exporter dùng lại được.

Quyết định thiết kế bám theo src/schemas.py:
- Kind chỉ có document/heading/itemize/item/sentence/formula/figure. Table cũng là FIGURE,
  mọi cấp tiêu đề (kể cả \\paragraph) đều là HEADING; độ sâu nằm ở quan hệ cha-con.
- Bảng elements chỉ có `content`: tiêu đề section là content của nút heading; caption/label
  của figure và label của công thức nằm trong chuỗi content thô, không tách cột riêng.
- Không có cột level/role/order, nên role (abstract, appendix...) chỉ dùng nội bộ.

Tham khảo: quy tắc tách câu theo docs/latex_spec.md mục 5.1; luật dựng cây bằng ngăn xếp theo mục 4.2.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from src.latex_parser import resolve_paper, strip_comments
from src.schemas import Element, Hierarchy, Kind, make_element_id

logger = logging.getLogger(__name__)

# Hạng chuẩn của các lệnh heading. Level thật của từng bài được tính động (spec mục 4.1).
HEADING_RANK = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
REFERENCE_TITLES = {"references", "reference", "bibliography", "literature cited", "references and notes"}

MATH_ENVS = {"equation", "align", "gather", "multline", "flalign", "eqnarray", "displaymath"}
FLOAT_ENVS = {"figure", "table", "wrapfigure", "wraptable", "tabular"}
LIST_ENVS = {"itemize", "enumerate", "description"}
SPECIAL_ENVS = {
    "abstract": "Abstract",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "acknowledgment": "Acknowledgments",
    "acknowledgement": "Acknowledgments",
}
# Vùng nguyên văn / không phải văn xuôi: bỏ hẳn, không phân tích bên trong.
DROP_ENVS = {
    "verbatim", "lstlisting", "minted", "rawhtml", "comment", "thebibliography",
    "tikzpicture", "algorithm", "algorithmic", "filecontents",
}  # fmt: skip
# Metadata đã nằm ở bảng papers nên không đi vào cây.
_METADATA_CMDS = ("title", "author", "date", "thanks", "affiliation", "email", "address", "institute", "keywords", "pacs")

_HEADING_CMD = "|".join(sorted(HEADING_RANK, key=len, reverse=True))
_HEADING = re.compile(rf"\\(?:{_HEADING_CMD})(?![A-Za-z])")
_TOKEN = re.compile(
    rf"""
    (?P<heading>\\(?P<hcmd>{_HEADING_CMD})(?![A-Za-z])\*?\s*(?:\[[^\]]*\])?\s*\{{)
  | (?P<begin>\\begin\{{(?P<env>[A-Za-z]+\*?)\}})
  | (?P<end>\\end\{{[A-Za-z]+\*?\}})
  | (?P<dd>\$\$)
  | (?P<ob>\\\[)
  | (?P<appendix>\\appendix(?![A-Za-z]))
  | (?P<bibcmd>\\(?:bibliography|printbibliography)(?![A-Za-z]))
    """,
    re.VERBOSE,
)
_ITEM_TOKEN = re.compile(r"\\begin\{(?:itemize|enumerate|description)\}|\\end\{(?:itemize|enumerate|description)\}|\\item(?![A-Za-z])")

# --- Tách câu (spec mục 5.1) ---
_PROTECT = re.compile(
    r"\$(?:\\.|[^$\\])+\$"  # toán nội dòng $...$
    r"|\\\(.*?\\\)"  # toán nội dòng \( ... \)
    r"|\\href\s*\{[^{}]*\}\s*\{[^{}]*\}"
    r"|\\(?:cite[a-zA-Z]*|ref|eqref|autoref|cref|Cref|url|footnote)\*?(?:\[[^\]]*\])*\{[^{}]*\}",
    re.DOTALL,
)
_ABBREV = re.compile(
    r"\b(?:i\.e|e\.g|et al|cf|vs|Figs?|Eqs?|Sec|Secs|Ref|Refs|Thm|Lem|Prop|Def|Cor|Tab|Alg|No|Vol|pp|etc|resp|approx)\."
)
_INITIAL = re.compile(r"(?<![A-Za-z$])[A-Z]\.(?=\s[A-Z])")  # "J. D. Watson"
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)|(?<![\w)])\.(?=\d)")  # 3.14, .05
_BOUNDARY = re.compile(r"(?<=[.?!])((?:''|[)\]}\"'])*)\s+(?=[A-Z\x00\\$])")
_DOT = "\x01"  # thay tạm cho dấu chấm không phải hết câu


class ParsedNode(NamedTuple):
    """Một nút của cây một version (chưa gắn element_id toàn cục)."""

    local_id: str
    kind: Kind
    content: str
    parent: str | None  # None chỉ ở nút gốc


@dataclass
class _Block:
    """Khối trung gian sau bước quét: text | heading | formula | figure | list | special | appendix."""

    type: str
    text: str = ""
    rank: int = -1
    children: list = field(default_factory=list)


# Tiện ích đọc cú pháp
def _read_braced(text: str, start: int) -> tuple[str, int]:
    """Đọc nội dung nhóm `{...}` (có lồng nhau) mà ký tự đầu nằm tại `start`.

    Args:
        text: Chuỗi nguồn.
        start: Vị trí ngay sau dấu `{` mở.

    Returns:
        (nội dung bên trong, vị trí ngay sau dấu `}` đóng). Thiếu `}` thì lấy đến hết chuỗi.
    """
    depth, i = 1, start
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return text[start:], len(text)


def _find_env_end(text: str, pos: int, name: str) -> tuple[int, int, bool]:
    """Tìm `\\end{name}` tương ứng, tính cả môi trường cùng tên lồng nhau.

    Args:
        text: Chuỗi nguồn.
        pos: Vị trí ngay sau `\\begin{name}`.
        name: Tên môi trường (có thể kèm dấu `*`).

    Returns:
        (cuối phần nội dung, vị trí tiếp theo, có_thẻ_đóng). Thiếu thẻ đóng thì tự đóng tại
        heading kế tiếp hoặc cuối tài liệu (spec mục 7).
    """
    pattern = re.compile(r"\\(begin|end)\{" + re.escape(name) + r"\}")
    depth = 1
    for match in pattern.finditer(text, pos):
        depth += 1 if match.group(1) == "begin" else -1
        if depth == 0:
            return match.start(), match.end(), True
    logger.warning("môi trường %s thiếu \\end, tự đóng", name)
    heading = _HEADING.search(text, pos)
    end = heading.start() if heading else len(text)
    return end, end, False


def _normalize_display_math(inner: str) -> str:
    """Chuẩn hóa khối toán $$...$$ hoặc \\[...\\] về môi trường equation chuẩn."""
    return rf"\begin{{equation}} {inner.strip()} \end{{equation}}"


def _remove_command(text: str, command: str) -> str:
    r"""Xóa mọi lệnh `\command[...]{...}` cùng đối số (đối số có thể có ngoặc lồng nhau)."""
    pattern = re.compile(rf"\\{command}(?![A-Za-z])\s*(?:\[[^\]]*\])?\s*\{{")
    while match := pattern.search(text):
        _, end = _read_braced(text, match.end())
        text = text[: match.start()] + text[end:]
    return text


def _prepare(latex: str) -> str:
    """Lấy phần thân văn bản: bỏ comment, preamble, phần sau \\end{document}, metadata tác giả."""
    text = strip_comments(latex)
    text = re.sub(r"\\iffalse.*?\\fi(?![A-Za-z])", " ", text, flags=re.DOTALL)
    begin = re.search(r"\\begin\{document\}", text)
    if begin:
        text = text[begin.end() :]
    end = re.search(r"\\end\{document\}", text)
    if end:
        text = text[: end.start()]
    for command in _METADATA_CMDS:
        text = _remove_command(text, command)
    text = _remove_command(text, "bibliographystyle")
    return re.sub(r"\\maketitle(?![A-Za-z])", " ", text)


def _split_items(inner: str) -> list[str]:
    """Cắt phần thân một danh sách thành các mục theo `\\item` ở độ sâu 0 (bỏ qua danh sách con)."""
    depth, cuts = 0, []
    for match in _ITEM_TOKEN.finditer(inner):
        token = match.group(0)
        if token.startswith("\\begin"):
            depth += 1
        elif token.startswith("\\end"):
            depth -= 1
        elif depth == 0:
            cuts.append(match)
    items = []
    for i, match in enumerate(cuts):
        stop = cuts[i + 1].start() if i + 1 < len(cuts) else len(inner)
        body = inner[match.end() : stop]
        # \item[nhãn tùy biến]: nhãn không có cột riêng trong schema nên bỏ.
        items.append(re.sub(r"^\s*\[[^\]]*\]", "", body))
    return items



# Quét văn bản thành các khối
def _parse_blocks(text: str) -> list[_Block]:
    """Quét tuần tự văn bản thành chuỗi khối theo đúng thứ tự xuất hiện."""
    blocks: list[_Block] = []
    pending: list[str] = []  # các mảnh văn xuôi chưa đóng gói (môi trường "trong suốt" bị nối liền)
    start = pos = 0

    def take(upto: int) -> None:
        pending.append(text[start:upto])

    def flush() -> None:
        chunk = "".join(pending)
        pending.clear()
        if chunk.strip():
            blocks.append(_Block("text", text=chunk))

    while match := _TOKEN.search(text, pos):
        if match["heading"]:
            take(match.start())
            flush()
            title, end = _read_braced(text, match.end())
            blocks.append(_Block("heading", text=title, rank=HEADING_RANK[match["hcmd"]]))
            start = pos = end
        elif match["begin"]:
            env = match["env"]
            base = env.rstrip("*")
            inner_end, after, closed = _find_env_end(text, match.end(), env)
            inner = text[match.end() : inner_end]
            raw = text[match.start() : after]
            if not closed:
                raw = f"{raw.rstrip()}\n\\end{{{env}}}"
            if base in DROP_ENVS:
                take(match.start())
                pending.append("\n\n")
                start = pos = after
            elif base in MATH_ENVS or base in FLOAT_ENVS or base in LIST_ENVS or base in SPECIAL_ENVS:
                take(match.start())
                flush()
                if base in MATH_ENVS:
                    blocks.append(_Block("formula", text=raw))
                elif base in FLOAT_ENVS:
                    blocks.append(_Block("figure", text=raw))
                elif base in LIST_ENVS:
                    blocks.append(_Block("list", children=[_parse_blocks(item) for item in _split_items(inner)]))
                else:
                    blocks.append(_Block("special", text=SPECIAL_ENVS[base], children=_parse_blocks(inner)))
                start = pos = after
            elif base == "appendices":
                take(match.start())
                flush()
                blocks.append(_Block("appendix"))
                start = pos = match.end()
            else:  # theorem, proof, center, minipage...: "trong suốt", chỉ bỏ thẻ và đối số đi kèm
                take(match.start())
                tail = re.match(r"\s*(?:\[[^\]]*\])?(?:\s*\{[^{}]*\})?" if base == "minipage" else r"\s*(?:\[[^\]]*\])?", text[match.end() :])
                start = pos = match.end() + (tail.end() if tail else 0)
        elif match["end"]:  # \end{...} lạc (env trong suốt hoặc không khớp)
            take(match.start())
            start = pos = match.end()
        elif match["dd"] or match["ob"]:
            closer = "$$" if match["dd"] else r"\]"
            close = text.find(closer, match.end())
            if close < 0:  # không có thẻ đóng: coi như văn bản thường
                pos = match.end()
                continue
            take(match.start())
            flush()
            # Đưa về dạng chuẩn \begin{equation}...\end{equation} như mock và đề M2 yêu cầu.
            blocks.append(_Block("formula", text=_normalize_display_math(text[match.end() : close])))
            start = pos = close + len(closer)
        elif match["appendix"]:
            take(match.start())
            flush()
            blocks.append(_Block("appendix"))
            start = pos = match.end()
        else:  # \bibliography{...} / \printbibliography: chỉ là lệnh, bỏ đi
            take(match.start())
            tail = re.match(r"\s*\{[^{}]*\}", text[match.end() :])
            start = pos = match.end() + (tail.end() if tail else 0)
    take(len(text))
    flush()
    return blocks


# Làm sạch tiêu đề và tách câu
def _clean_title(raw: str) -> str:
    """Chuẩn hóa tiêu đề heading (spec mục 4.3), giữ nguyên toán nội dòng."""
    text = re.sub(r"\\label\s*\{[^{}]*\}", "", raw)
    while match := re.search(r"\\texorpdfstring\s*\{", text):
        _, after_first = _read_braced(text, match.end())
        second_start = text.find("{", after_first)
        if second_start < 0:
            break
        plain, after_second = _read_braced(text, second_start + 1)
        text = text[: match.start()] + plain + text[after_second:]
    text = re.sub(r"\\\\|\\newline(?![A-Za-z])", " ", text)
    previous = None
    while previous != text:  # bóc lệnh định dạng, có thể lồng nhau
        previous = text
        text = re.sub(r"\\(?:textbf|textit|emph|textsc|texttt|textrm|textsf|underline|mbox)\s*\{([^{}]*)\}", r"\1", text)
    return " ".join(text.replace("~", " ").split())


def _split_paragraph(paragraph: str) -> list[str]:
    """Tách một đoạn (đã gộp dòng) thành các câu, không cắt trong toán, \\cite, viết tắt, số thập phân."""
    tokens: list[str] = []

    def protect(match: re.Match) -> str:
        tokens.append(match.group(0))
        return f"\x00{len(tokens) - 1}\x00"

    masked = _PROTECT.sub(protect, paragraph)
    masked = _ABBREV.sub(lambda m: m.group(0).replace(".", _DOT), masked)
    masked = _INITIAL.sub(_DOT, masked)
    masked = _DECIMAL.sub(_DOT, masked)

    pieces, last = [], 0
    for match in _BOUNDARY.finditer(masked):
        pieces.append(masked[last : match.end(1)])
        last = match.end()
    pieces.append(masked[last:])

    sentences = []
    for piece in pieces:
        piece = re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], piece.replace(_DOT, "."))
        piece = piece.strip()
        if re.search(r"[A-Za-z0-9\\$]", piece):
            sentences.append(piece)
    return sentences


def split_sentences(chunk: str) -> list[str]:
    """Tách một khối văn xuôi thành các câu; ngắt đoạn (dòng trống, \\par) luôn ngắt câu."""
    chunk = re.sub(r"\\label\s*\{[^{}]*\}", "", chunk)
    chunk = re.sub(r"\\(?:noindent|medskip|bigskip|smallskip|newpage|clearpage|tableofcontents)(?![A-Za-z])", " ", chunk)
    chunk = chunk.replace("~", " ")
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\\par(?![A-Za-z])", chunk):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            sentences.extend(_split_paragraph(paragraph))
    return sentences


# Dựng cây
def _collect_ranks(blocks: list[_Block]) -> set[int]:
    """Gom hạng của mọi lệnh heading thực sự xuất hiện (kể cả trong mục danh sách)."""
    ranks: set[int] = set()
    for block in blocks:
        if block.type == "heading":
            ranks.add(block.rank)
        elif block.type == "special":
            ranks |= _collect_ranks(block.children)
        elif block.type == "list":
            for item in block.children:
                ranks |= _collect_ranks(item)
    return ranks


class _Builder:
    """Dựng danh sách ParsedNode từ chuỗi khối bằng ngăn xếp (spec mục 4.2)."""

    _PREFIX = {
        Kind.HEADING: "h",
        Kind.SENTENCE: "s",
        Kind.FORMULA: "f",
        Kind.FIGURE: "fig",
        Kind.ITEMIZE: "list",
        Kind.ITEM: "it",
    }

    def __init__(self, version: int, levels: dict[int, int]):
        self.version = version
        self.levels = levels
        self.counts: dict[Kind, int] = defaultdict(int)
        self.skipping = False  # đang ở trong phần References
        self.skip_level = 0
        self.root = f"root{version}"
        self.nodes = [ParsedNode(self.root, Kind.DOCUMENT, f"Version {version} root", None)]

    def add(self, kind: Kind, content: str, parent: str) -> str:
        """Thêm nút mới. Id có tiền tố version để không đụng id của version khác trước khi dedup."""
        self.counts[kind] += 1
        local_id = f"v{self.version}-{self._PREFIX[kind]}{self.counts[kind]}"
        self.nodes.append(ParsedNode(local_id, kind, content, parent))
        return local_id

    def emit(self, blocks: list[_Block], stack: list[tuple[int, str]]) -> None:
        """Đưa các khối vào cây. `stack` là các (level, id) của section đang mở; phần tử 0 là cha gốc."""
        for block in blocks:
            if block.type == "appendix":
                self.skipping = False  # phụ lục sau References vẫn được giữ (spec mục 6.3)
            elif block.type == "heading":
                self._heading(block, stack)
            elif block.type == "special":  # abstract / acknowledgments: section cấp cao nhất
                del stack[1:]
                node = self.add(Kind.HEADING, block.text, stack[0][1])
                self.emit(block.children, [(0, node)])
            elif self.skipping:
                continue
            elif block.type == "text":
                for sentence in split_sentences(block.text):
                    self.add(Kind.SENTENCE, sentence, stack[-1][1])
            elif block.type in ("formula", "figure"):
                kind = Kind.FORMULA if block.type == "formula" else Kind.FIGURE
                self.add(kind, block.text, stack[-1][1])
            elif block.type == "list":
                list_id = self.add(Kind.ITEMIZE, "", stack[-1][1])
                for item_blocks in block.children:
                    item_id = self.add(Kind.ITEM, "", list_id)
                    self.emit(item_blocks, [(0, item_id)])

    def _heading(self, block: _Block, stack: list[tuple[int, str]]) -> None:
        level = self.levels[block.rank]
        if self.skipping and level > self.skip_level:
            return  # heading con nằm trong phạm vi References
        self.skipping = False
        title = _clean_title(block.text)
        if title.lower().strip(" .:") in REFERENCE_TITLES:
            self.skipping, self.skip_level = True, level
            return
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, self.add(Kind.HEADING, title, stack[-1][1])))


def build_tree(latex: str, version: int = 1) -> list[ParsedNode]:
    """Dựng cây phân cấp của một version.

    Args:
        latex: Chuỗi LaTeX đã gộp (đầu ra của latex_parser.resolve_paper). Chuỗi rỗng nghĩa là
            version không có TeX: trả về cây chỉ có nút gốc ("record rỗng").
        version: Số thứ tự version (>= 1), dùng đặt id gốc `root{version}`.

    Returns:
        Danh sách ParsedNode theo thứ tự duyệt tiền thứ tự; phần tử đầu là nút DOCUMENT.
        Id cục bộ có dạng `v{version}-{kind}{i}`; bước dedup (P2-04) sẽ đổi sang id dùng chung
        cho nội dung trùng giữa các version.
    """
    blocks = _parse_blocks(_prepare(latex)) if latex.strip() else []
    levels = {rank: i + 1 for i, rank in enumerate(sorted(_collect_ranks(blocks)))}
    builder = _Builder(version, levels)
    builder.emit(blocks, [(0, builder.root)])
    return builder.nodes


# Chuyển đổi đầu ra
def to_rows(arxiv_id: str, version: int, nodes: list[ParsedNode]) -> tuple[list[Element], list[Hierarchy]]:
    """Chuyển cây một version thành dòng bảng elements và hierarchy (qua make_element_id).

    Args:
        arxiv_id: Mã arXiv chuẩn của bài.
        version: Version của cây.
        nodes: Kết quả của build_tree.

    Returns:
        (elements, hierarchy). Nút gốc không sinh dòng hierarchy; thứ tự dòng là thứ tự văn bản.
    """
    elements = [
        Element(element_id=make_element_id(arxiv_id, n.local_id), arxiv_id=arxiv_id, kind=n.kind, content=n.content)
        for n in nodes
    ]
    hierarchy = [
        Hierarchy(
            arxiv_id=arxiv_id,
            version=version,
            element_id=make_element_id(arxiv_id, n.local_id),
            parent_id=make_element_id(arxiv_id, n.parent),
        )
        for n in nodes
        if n.parent is not None
    ]
    return elements, hierarchy


def to_nested(nodes: list[ParsedNode]) -> dict:
    """Dựng dạng JSON lồng nhau (hierarchy.json trung gian) để đọc/kiểm tra bằng mắt."""
    by_id = {n.local_id: {"id": n.local_id, "kind": str(n.kind), "content": n.content, "children": []} for n in nodes}
    for node in nodes:
        if node.parent is not None:
            by_id[node.parent]["children"].append(by_id[node.local_id])
    return by_id[nodes[0].local_id]


def parse_version_dir(version_dir: Path, version: int) -> list[ParsedNode]:
    """Tiện ích đầu-cuối: thư mục nguồn của một version -> cây (resolve rồi build)."""
    return build_tree(resolve_paper(version_dir), version)