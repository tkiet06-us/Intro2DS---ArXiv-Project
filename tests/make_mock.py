"""Sinh mock data nhẹ cho M2/M3 (task P1-02).

Chạy từ thư mục gốc repo: python tests/make_mock.py --out tests/mock_data

Mock sinh ra CẢ HAI dạng để mỗi thành viên dùng ngay, không chờ M1 crawl xong (IDD):
- Dạng làm việc (cho M2, M3): <yymm-id>/v1/, v2/, metadata.json, references.json, ...
- Dạng Parquet 7 bảng (cho exporter ở P4-02): parquet/<table>/part-000.parquet.

Mock cố tình KHÔNG "đẹp": có ID cũ, bài PDF-only, entry thiếu năm, ref thiếu tiêu đề...
vì dữ liệu quá sạch sẽ che mất bug của parser/matcher (xem bước 3 trong hướng dẫn P1-02).
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

# Python chỉ thêm thư mục chứa script (tests/) vào sys.path, nên "python tests/make_mock.py"
# sẽ không thấy package `src`. Thêm thư mục gốc repo để lệnh trong README chạy được.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.utils_io import to_folder_name, write_json  # noqa: E402
from src.schemas import (  # noqa: E402
    MODELS,
    PARQUET_SCHEMAS,
    BibtexEntry,
    Element,
    ElementParent,
    Kind,
    Paper,
    PaperReference,
    Prediction,
    Source,
    make_element_id,
)

logger = logging.getLogger(__name__)

PARQUET_FILE_NAME = "part-000.parquet"

# --- Nội dung LaTeX mẫu. Dùng r"""...""" để dấu "\" không bị Python hiểu là escape. ---
TEX_A_V1 = r"""\documentclass{article}
\begin{document}
\section{Introduction}
We study matching. It works well.
\begin{equation} E = mc^2 \end{equation}
\begin{thebibliography}{9}
\bibitem{foo} A. Author, Foo Bar, 2020.
\end{thebibliography}
\end{document}"""
# v2 chỉ sửa MỘT câu: câu còn lại giống hệt để test full-text dedup (P2-04).
TEX_A_V2 = TEX_A_V1.replace("It works well.", "It works very well.")

TEX_B_MAIN = r"""\documentclass{article}
\begin{document}
\section{Method}
Our method is simple.
\begin{figure}[htpb]
\centering
\includegraphics{fig1.png}
\caption{Overview.}
\end{figure}
\bibliography{refs}
\end{document}"""

# Bài dùng \input nhiều file để parser của Nhi phải lần theo đường biên dịch (P2-03).
TEX_C_MAIN = r"""\documentclass{article}
\begin{document}
\input{intro}
\end{document}"""
TEX_C_INTRO = r"""\section{Introduction}
Strings are fun.
\begin{thebibliography}{9}
\bibitem{witten} E. Witten, String theory dynamics, 1995.
\end{thebibliography}"""


class Node(NamedTuple):
    """Một nút trong cây phân cấp của mock (trước khi gắn element_id toàn cục)."""

    local_id: str
    kind: Kind
    content: str
    parent: str | None  # None nghĩa là nút gốc


@dataclass
class MockPaper:
    """Toàn bộ dữ liệu giả của một bài, đủ để sinh cả dạng làm việc lẫn Parquet."""

    paper: Paper
    sources: dict[int, dict[str, str]] = field(default_factory=dict)  # version -> {file: text}
    trees: dict[int, list[Node]] = field(default_factory=dict)  # version -> các nút
    bib_entries: list[BibtexEntry] = field(default_factory=list)
    references: list[PaperReference] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)


def build_mock_papers() -> list[MockPaper]:
    """Tạo 4 bài mock, mỗi bài đại diện cho một tình huống khó.

    Returns:
        Danh sách bài: (A) 2 version, (B) có .bib + figure, (C) ID cũ + \\input, (D) PDF-only.
    """
    # --- Bài A: 2 version, một câu giống hệt, một câu khác ---
    a_id = "2310.12345"
    doc, sec = Node("doc", Kind.DOCUMENT, "Mock Paper 1", None), "sec1"
    shared = [
        Node(sec, Kind.SECTION, "Introduction", "doc"),
        Node("s1", Kind.SENTENCE, "We study matching.", sec),
        Node("f1", Kind.FORMULA, r"\begin{equation} E = mc^2 \end{equation}", sec),
    ]
    paper_a = MockPaper(
        paper=Paper(
            arxiv_id=a_id,
            title="Mock Paper 1",
            authors=["Alice", "Bob"],
            submission_date="2023-10-05",
            revised_dates=["2023-11-20"],
        ),
        sources={1: {"main.tex": TEX_A_V1}, 2: {"main.tex": TEX_A_V2}},
        trees={
            1: [doc, *shared, Node("s2", Kind.SENTENCE, "It works well.", sec)],
            2: [doc, *shared, Node("s2b", Kind.SENTENCE, "It works very well.", sec)],
        },
        bib_entries=[
            BibtexEntry(
                arxiv_id=a_id,
                citation_key="foo",
                entry_type="misc",
                title="Foo Bar",
                authors=["A. Author"],
                year=2020,
            )
        ],
        references=[
            PaperReference(
                arxiv_id=a_id,
                ref_arxiv_id="1706.03762",
                ref_title="Attention Is All You Need",
                ref_authors=["Vaswani"],
                ref_submission_date="2017-06-12",
                ref_s2_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            )
        ],
        predictions=[
            Prediction(
                arxiv_id=a_id,
                partition="train",
                citation_key="foo",
                groundtruth_arxiv_id="1706.03762",
                cand_1="1706.03762",
                cand_2="1810.04805",
            )
        ],
    )

    # --- Bài B: có file .bib, có figure; entry thiếu năm; ref thiếu tiêu đề/tác giả ---
    b_id = "2401.00001"
    paper_b = MockPaper(
        paper=Paper(
            arxiv_id=b_id,
            title="Mock Paper 2",
            authors=["Carol"],
            submission_date="2024-01-02",
            venue=None,  # nhiều bài không có venue
        ),
        sources={1: {"main.tex": TEX_B_MAIN, "refs.bib": ""}},  # refs.bib điền ở dưới
        trees={
            1: [
                Node("doc", Kind.DOCUMENT, "Mock Paper 2", None),
                Node("sec1", Kind.SECTION, "Method", "doc"),
                Node("s1", Kind.SENTENCE, "Our method is simple.", "sec1"),
                Node("fig1", Kind.FIGURE, r"\begin{figure}\caption{Overview.}\end{figure}", "sec1"),
            ]
        },
        bib_entries=[
            BibtexEntry(
                arxiv_id=b_id,
                citation_key="bert",
                entry_type="article",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["J. Devlin", "M. Chang"],
                year=2019,
                journal="NAACL",
            ),
            BibtexEntry(  # thiếu year/title: kiểm tra Optional hoạt động
                arxiv_id=b_id, citation_key="noyear", entry_type="misc", authors=["Unknown"]
            ),
        ],
        references=[
            PaperReference(
                arxiv_id=b_id,
                ref_arxiv_id="1810.04805",
                ref_title="BERT: Pre-training of Deep Bidirectional Transformers",
                ref_authors=["Jacob Devlin"],
                ref_submission_date="2018-10-11",
                ref_s2_id="df2b0e26d0599ce3e70df8a9da02e51594e0e992",
            ),
            PaperReference(  # ref thiếu tiêu đề/tác giả/ngày: S2 hay trả về như vậy
                arxiv_id=b_id, ref_arxiv_id="1301.3781", ref_s2_id="330da625c15427c6e42ccfa3b747fb"
            ),
        ],
        predictions=[
            Prediction(
                arxiv_id=b_id,
                partition="valid",
                citation_key="bert",
                groundtruth_arxiv_id="1810.04805",
                cand_1="1301.3781",  # đáp án đúng ở hạng 2 -> MRR = 0.5
                cand_2="1810.04805",
            ),
            Prediction(  # chưa biết đáp án: groundtruth null
                arxiv_id=b_id, partition="valid", citation_key="noyear", cand_1="1301.3781"
            ),
        ],
    )
    # refs.bib trong source là text thật, sinh từ chính các entry để hai bên luôn khớp.
    paper_b.sources[1]["refs.bib"] = "\n".join(to_bibtex(e) for e in paper_b.bib_entries)

    # --- Bài C: ID cũ trước 2007, \input nhiều file ---
    c_id = "hep-th/9901001"
    paper_c = MockPaper(
        paper=Paper(
            arxiv_id=c_id,
            title="Mock Paper 3 (old ID)",
            authors=["Dave"],
            submission_date="1999-01-04",
        ),
        sources={1: {"main.tex": TEX_C_MAIN, "intro.tex": TEX_C_INTRO}},
        trees={
            1: [
                Node("doc", Kind.DOCUMENT, "Mock Paper 3 (old ID)", None),
                Node("sec1", Kind.SECTION, "Introduction", "doc"),
                Node("s1", Kind.SENTENCE, "Strings are fun.", "sec1"),
            ]
        },
        bib_entries=[
            BibtexEntry(
                arxiv_id=c_id,
                citation_key="witten",
                entry_type="misc",
                title="String theory dynamics",
                authors=["E. Witten"],
                year=1995,
            )
        ],
        references=[
            PaperReference(
                arxiv_id=c_id,
                ref_arxiv_id="hep-th/9503124",  # ID cũ cũng xuất hiện ở phía được trích dẫn
                ref_title="String Theory Dynamics In Various Dimensions",
                ref_authors=["Edward Witten"],
                ref_submission_date="1995-03-17",
                ref_s2_id="0000000000000000000000000000000000000000",
            )
        ],
        predictions=[
            Prediction(
                arxiv_id=c_id,
                partition="test",
                citation_key="witten",
                groundtruth_arxiv_id="hep-th/9503124",
                cand_1="hep-th/9503124",
            )
        ],
    )

    # --- Bài D: PDF-only. Giữ "record rỗng": có metadata, KHÔNG có source/tree/bib ---
    paper_d = MockPaper(
        paper=Paper(
            arxiv_id="2402.99999",
            title="Mock Paper 4 (PDF only)",
            authors=["Eve"],
            submission_date="2024-02-10",
        )
    )
    return [paper_a, paper_b, paper_c, paper_d]


def to_bibtex(entry: BibtexEntry) -> str:
    """Dựng chuỗi BibTeX từ một entry (chỉ ghi các trường có giá trị).

    Args:
        entry: Entry đã chuẩn hóa.

    Returns:
        Văn bản dạng "@type{key, field = {value}, ...}".
    """
    fields = {
        "title": entry.title,
        "author": " and ".join(entry.authors),
        "year": entry.year,
        "journal": entry.journal,
        "doi": entry.doi,
        "url": entry.url,
    }
    # Bỏ trường rỗng thay vì ghi "year = {}", đúng như entry thiếu năm ngoài đời thật.
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items() if value)
    return f"@{entry.entry_type}{{{entry.citation_key},\n{body}\n}}\n"


def flatten_trees(
    arxiv_id: str, trees: dict[int, list[Node]]
) -> tuple[list[Element], list[ElementParent]]:
    """Biến các cây theo version thành dòng của bảng elements và element_parents.

    Cùng element_id thì chỉ giữ MỘT dòng trong elements (full-text dedup giữa các version),
    còn quan hệ cha-con vẫn ghi riêng cho từng version.

    Args:
        arxiv_id: Bài chứa các cây.
        trees: version -> danh sách nút.

    Returns:
        (elements đã khử trùng, element_parents của mọi version).

    Raises:
        ValueError: Khi hai nút cùng id nhưng khác nội dung (mock tự mâu thuẫn).
    """
    elements: dict[str, Element] = {}
    parents: list[ElementParent] = []
    for version, nodes in trees.items():
        for node in nodes:
            element_id = make_element_id(arxiv_id, node.local_id)
            known = elements.get(element_id)
            if known is not None and known.content != node.content:
                raise ValueError(f"{element_id} bị dùng cho hai nội dung khác nhau")
            elements[element_id] = Element(
                element_id=element_id, arxiv_id=arxiv_id, kind=node.kind, content=node.content
            )
            # Nút gốc không có cha nên không sinh dòng (quyết định số 3 trong schemas.py).
            if node.parent is not None:
                parents.append(
                    ElementParent(
                        arxiv_id=arxiv_id,
                        version=version,
                        element_id=element_id,
                        parent_id=make_element_id(arxiv_id, node.parent),
                    )
                )
    return list(elements.values()), parents


def build_hierarchy_json(elements: list[Element], parents: list[ElementParent]) -> dict:
    """Dựng hierarchy.json theo định dạng gợi ý của M2 từ các dòng bảng.

    Sinh từ cùng danh sách dòng với Parquet nên hai dạng dữ liệu không thể lệch nhau.

    Args:
        elements: Các element đã khử trùng của một bài.
        parents: Các dòng cha-con của bài đó.

    Returns:
        {"elements": {id: content}, "hierarchy": {"1": {id: parent_id}, ...}}.
    """
    hierarchy: dict[str, dict[str, str]] = {}
    for row in parents:
        hierarchy.setdefault(str(row.version), {})[row.element_id] = row.parent_id
    return {"elements": {e.element_id: e.content for e in elements}, "hierarchy": hierarchy}


def references_to_json(references: list[PaperReference]) -> dict[str, dict]:
    """Dựng references.json: key là ID dạng "yymm-id", value là metadata (theo đề M1).

    Args:
        references: Các dòng paper_references của một bài.

    Returns:
        Dict key -> metadata, không lặp lại arxiv_id/ref_arxiv_id vì đã nằm ở key và thư mục.
    """
    return {
        to_folder_name(ref.ref_arxiv_id): ref.model_dump(
            mode="json", exclude={"arxiv_id", "ref_arxiv_id"}
        )
        for ref in references
    }


def write_working_format(out: Path, mock: MockPaper, rows: tuple[list, list]) -> None:
    """Ghi một bài ra thư mục theo quy ước làm việc của M1/M2.

    Args:
        out: Thư mục mock gốc.
        mock: Dữ liệu giả của bài.
        rows: (elements, parents) đã tính sẵn bởi flatten_trees.
    """
    folder = out / to_folder_name(mock.paper.arxiv_id)
    folder.mkdir(parents=True, exist_ok=True)  # bài PDF-only vẫn có folder
    for version, files in mock.sources.items():
        for filename, text in files.items():
            path = folder / f"v{version}" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    write_json(folder / "metadata.json", mock.paper.model_dump(mode="json"))
    write_json(folder / "references.json", references_to_json(mock.references))
    if mock.trees:
        write_json(folder / "hierarchy.json", build_hierarchy_json(*rows))
    if mock.bib_entries:
        bib_text = "\n".join(to_bibtex(e) for e in mock.bib_entries)
        (folder / "refs.bib").write_text(bib_text, encoding="utf-8")


def write_parquet(out: Path, name: str, rows: list[BaseModel]) -> None:
    """Ghi một bảng Parquet theo schema tường minh.

    Args:
        out: Thư mục mock gốc.
        name: Tên bảng, khóa của PARQUET_SCHEMAS.
        rows: Các dòng đã là model Pydantic (đã được kiểm tra khi khởi tạo).
    """
    expected = MODELS[name]
    for row in rows:
        # Nổ lỗi sớm nếu ai đó nhét nhầm model vào bảng khác.
        if not isinstance(row, expected):
            raise TypeError(f"Bảng {name} cần {expected.__name__}, nhận {type(row).__name__}")
    # mode="json" biến Enum thành chuỗi thường để pyarrow nhận được cho cột dictionary.
    dumped = [row.model_dump(mode="json") for row in rows]
    # Truyền schema tường minh để kiểu cột (int32, dictionary) không phụ thuộc suy luận.
    table = pa.Table.from_pylist(dumped, schema=PARQUET_SCHEMAS[name])
    directory = out / "parquet" / name
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / PARQUET_FILE_NAME)


def generate(out: Path) -> None:
    """Sinh toàn bộ mock (dạng làm việc + 7 bảng Parquet) vào thư mục `out`.

    Args:
        out: Thư mục đích, ví dụ tests/mock_data.
    """
    tables: dict[str, list[BaseModel]] = {name: [] for name in MODELS}
    for mock in build_mock_papers():
        arxiv_id = mock.paper.arxiv_id
        elements, parents = flatten_trees(arxiv_id, mock.trees)
        write_working_format(out, mock, (elements, parents))
        tables["papers"].append(mock.paper)
        tables["elements"].extend(elements)
        tables["element_parents"].extend(parents)
        tables["bibtex_entries"].extend(mock.bib_entries)
        tables["paper_references"].extend(mock.references)
        tables["predictions"].extend(mock.predictions)
        for version, files in mock.sources.items():
            tables["sources"].extend(
                Source(arxiv_id=arxiv_id, version=version, filename=name, content=text)
                for name, text in files.items()
            )
    for name, rows in tables.items():
        write_parquet(out, name, rows)
        logger.info("Đã ghi bảng %s (%d dòng)", name, len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sinh mock data cho dự án arXiv pipeline.")
    parser.add_argument("--out", default="tests/mock_data", help="Thư mục đầu ra")
    generate(Path(parser.parse_args().out))