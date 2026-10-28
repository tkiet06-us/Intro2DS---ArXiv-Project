"""Unit tests cho src/tree_builder.py."""

from src.schemas import Kind
from src.tree_builder import _normalize_display_math, build_tree


ARXIV_ID = "2411.03799" 
 
# build_tree trả về dict lồng nhau, nên cần hàm duyệt cây thay cho list node cũ.
def _walk(node):
    yield node
    for child in node["children"]:
        yield from _walk(child)
 
 
def _outline(node, depth=0):
    """Làm phẳng cây thành list (độ sâu, id, kind, content) theo thứ tự tiền thứ tự."""
    out = [(depth, node["id"], node["kind"], node["content"])]
    for child in node["children"]:
        out.extend(_outline(child, depth + 1))
    return out
 

def _sentences(node):
    return [n["content"] for n in _walk(node) if n["kind"] == Kind.SENTENCE]
 

def _headings(node):
    return [n["content"] for n in _walk(node) if n["kind"] == Kind.HEADING]
 
 
def test_normalize_display_math():
    assert _normalize_display_math("E = mc^2") == r"\begin{equation} E = mc^2 \end{equation}"
    assert _normalize_display_math("  a + b = c  ") == r"\begin{equation} a + b = c \end{equation}"
 
 
def test_display_math_conversion():
    tex = r"""\begin{document}
$$x^2 + y^2 = z^2$$
\[a = b\]
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    formulas = [n for n in _walk(tree) if n["kind"] == Kind.FORMULA]
    assert len(formulas) == 2
    assert formulas[0]["content"] == r"\begin{equation} x^2 + y^2 = z^2 \end{equation}"
    assert formulas[1]["content"] == r"\begin{equation} a = b \end{equation}"
 
 
def test_auto_close_missing_end():
    tex = r"""\begin{document}
\section{First}
\begin{figure}[ht]
\caption{My figure}
\section{Second}
Some text.
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    figures = [n for n in _walk(tree) if n["kind"] == Kind.FIGURE]
    assert len(figures) == 1
    assert figures[0]["content"].endswith(r"\end{figure}")
    # Hình thiếu \end phải dừng tại heading kế tiếp, không nuốt "Second" và câu sau nó
    assert _headings(tree) == ["First", "Second"]
    assert _sentences(tree) == ["Some text."]
 
 
# Ví dụ hoàn chỉnh ở mục 8 của latex_spec.md: đầu vào 8.1 -> cây 8.2.
SPEC_TEX = r"""\documentclass[11pt]{article}
\usepackage{amsmath,graphicx}
 
\begin{document}
 
\begin{abstract}
Graph coloring is a fundamental problem in computer science. We present a linear-time heuristic bound.
\end{abstract}
 
\section{Introduction}
Graph coloring has been studied extensively~\cite{brooks1941colouring}. Let $G=(V, E)$ be a connected graph.
\begin{itemize}
    \item Planar graphs are 4-colorable.
    \item Bipartite graphs are 2-colorable.
\end{itemize}
 
\subsection{Brooks' Bound}
The chromatic number satisfies the following upper bound:
\begin{equation}
\chi(G) \leq \Delta(G) + 1 \label{eq:brooks}
\end{equation}
where $\Delta(G)$ denotes the maximum degree.
 
\begin{figure}[ht]
\caption{Visualization of 4-coloring on a planar triangulation.}
\label{fig:planar_graph}
\end{figure}
 
\section*{Acknowledgments}
This research was supported by the VNU-HCM Research Grant.
 
\begin{thebibliography}{99}
\bibitem{brooks1941colouring}
R.~L. Brooks, \emph{On colouring the nodes of a network}, Math. Proc. Cambridge, 1941.
\end{thebibliography}
 
\appendix
\section{Omitted Proofs}
The detailed proof of correctness follows by induction on $|V|$.
 
\end{document}
"""
 
SPEC_OUTLINE = [
    (0, "root1", "document", "Version 1 root"),
    (1, "v1-h1", "heading", "Abstract"),
    (2, "v1-s1", "sentence", "Graph coloring is a fundamental problem in computer science."),
    (2, "v1-s2", "sentence", "We present a linear-time heuristic bound."),
    (1, "v1-h2", "heading", "Introduction"),
    (2, "v1-s3", "sentence", "Graph coloring has been studied extensively \\cite{brooks1941colouring}."),
    (2, "v1-s4", "sentence", "Let $G=(V, E)$ be a connected graph."),
    (2, "v1-list1", "itemize", ""),
    (3, "v1-it1", "item", ""),
    (4, "v1-s5", "sentence", "Planar graphs are 4-colorable."),
    (3, "v1-it2", "item", ""),
    (4, "v1-s6", "sentence", "Bipartite graphs are 2-colorable."),
    (2, "v1-h3", "heading", "Brooks' Bound"),
    (3, "v1-s7", "sentence", "The chromatic number satisfies the following upper bound:"),
    (
        3,
        "v1-f1",
        "formula",
        "\\begin{equation}\n\\chi(G) \\leq \\Delta(G) + 1 \\label{eq:brooks}\n\\end{equation}",
    ),
    (3, "v1-s8", "sentence", "where $\\Delta(G)$ denotes the maximum degree."),
    (
        3,
        "v1-fig1",
        "figure",
        "\\begin{figure}[ht]\n\\caption{Visualization of 4-coloring on a planar triangulation.}\n"
        "\\label{fig:planar_graph}\n\\end{figure}",
    ),
    (1, "v1-h4", "heading", "Acknowledgments"),
    (2, "v1-s9", "sentence", "This research was supported by the VNU-HCM Research Grant."),
    (1, "v1-h5", "heading", "Omitted Proofs"),
    (2, "v1-s10", "sentence", "The detailed proof of correctness follows by induction on $|V|$."),
]


def test_spec_full_example():
    assert _outline(build_tree(SPEC_TEX, ARXIV_ID)) == SPEC_OUTLINE


# Mục 4.1: level động.
def test_dynamic_levels():
    tex = r"""\begin{document}
\chapter{C}
\section{S}
\subsection{T}
Deep text.
\section{S2}
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    assert [(d, c) for d, _, k, c in _outline(tree) if k == "heading"] == [
        (1, "C"), (2, "S"), (3, "T"), (2, "S2"),
    ]
 
    # Nhảy cấp section -> subsubsection: không sinh nút rỗng trung gian.
    tex = r"\begin{document}\section{A}\subsubsection{B}Text here.\end{document}"
    tree = build_tree(tex, ARXIV_ID)
    assert [(d, c) for d, _, k, c in _outline(tree) if k == "heading"] == [(1, "A"), (2, "B")]
 
 
# Mục 4.2: văn bản trước heading đầu tiên và bài không có heading gắn vào root.
def test_text_without_heading_attaches_to_root():
    tree = build_tree(r"\begin{document}Intro text. More text.\end{document}", ARXIV_ID)
    assert [c["kind"] for c in tree["children"]] == ["sentence", "sentence"]
 
 
# Mục 4.3.
def test_heading_title_cleanup():
    tex = (
        r"\begin{document}"
        r"\section{A \textbf{bold} \label{sec:a} and $\Delta$-Coloring}"
        r"\section{\texorpdfstring{$\alpha$ test}{Alpha test}}"
        r"\end{document}"
    )
    assert _headings(build_tree(tex, ARXIV_ID)) == ["A bold and $\\Delta$-Coloring", "Alpha test"]
 
 
# Mục 5.1: không tách sai ở viết tắt, số thập phân, URL, toán nội dòng, tên viết tắt.
def test_sentence_splitting_protects_entities():
    tex = (
        r"\begin{document}"
        r"Smith et al. showed the bound. The value is 3.14 and .05 here. "
        r"See Fig. 2 for details, i.e. the plot. Visit \url{http://a.b/c.d} now. "
        r"Let $x = 1. 5$ hold. J. D. Watson wrote it."
        r"\end{document}"
    )
    assert _sentences(build_tree(tex, ARXIV_ID)) == [
        "Smith et al. showed the bound.",
        "The value is 3.14 and .05 here.",
        "See Fig. 2 for details, i.e. the plot.",
        r"Visit \url{http://a.b/c.d} now.",
        "Let $x = 1. 5$ hold.",
        "J. D. Watson wrote it.",
    ]
 
 
# Mục 5.2: công thức giữa câu -> chuỗi 3 nút sentence, formula, sentence.
def test_formula_splits_sentence():
    tex = r"""\begin{document}
Friction is defined by
\begin{equation}
f = \mu N
\end{equation}
where $\mu$ is the coefficient.
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    assert [c["kind"] for c in tree["children"]] == ["sentence", "formula", "sentence"]
 
 
# Mục 5.2, 5.3: align giữ nguyên raw, bảng là figure, toán nội dòng không thành formula.
def test_block_environments_kinds():
    tex = r"""\begin{document}
\begin{align}
a &= b \label{eq:1}
\end{align}
\begin{table}[t]
\begin{tabular}{ll} a & b \end{tabular}
\caption{T}
\end{table}
Inline $x^2$ stays.
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    kinds = [c["kind"] for c in tree["children"]]
    assert kinds == ["formula", "figure", "sentence"]  # tabular nằm trong table, không tách riêng
    assert tree["children"][0]["content"] == "\\begin{align}\na &= b \\label{eq:1}\n\\end{align}"
    assert tree["children"][2]["content"] == "Inline $x^2$ stays."
 
 
# Mục 5.4: danh sách lồng nhau.
def test_nested_lists():
    tex = r"""\begin{document}
\section{S}
\begin{itemize}
\item Outer one.
\item Outer two.
  \begin{enumerate}
  \item Inner a.
  \end{enumerate}
\end{itemize}
\end{document}"""
    got = [(d, k, c) for d, _, k, c in _outline(build_tree(tex, ARXIV_ID))][2:]
    assert got == [
        (2, "itemize", ""),
        (3, "item", ""),
        (4, "sentence", "Outer one."),
        (3, "item", ""),
        (4, "sentence", "Outer two."),
        (4, "itemize", ""),
        (5, "item", ""),
        (6, "sentence", "Inner a."),
    ]
 
 
# Mục 6.1, 6.2, 6.3: bỏ References, giữ Ack và Appendix.
def test_references_removed_ack_and_appendix_kept():
    tex = r"""\begin{document}
\section{Intro}
Hello world.
\section*{Acknowledgments}
Thanks.
\section*{References}
\begin{thebibliography}{9}
\bibitem{a} A. Author, Title.
\end{thebibliography}
\appendix
\section{Proof}
Done.
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    assert _headings(tree) == ["Intro", "Acknowledgments", "Proof"]
    assert _sentences(tree) == ["Hello world.", "Thanks.", "Done."]
 
 
# Mục 6.1: References dạng heading thường, phạm vi kết thúc ở heading cùng cấp.
def test_references_heading_scope_ends_at_next_heading():
    tex = r"""\begin{document}
\section{Intro}
Hello.
\section{References}
Some entry text. Another entry.
\subsection{Part of refs}
Nested entry.
\section{Appendix A}
Proof.
\bibliography{refs}
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    assert _headings(tree) == ["Intro", "Appendix A"]
    assert _sentences(tree) == ["Hello.", "Proof."]
 
 
# Mục 6.3: phụ lục đứng sau References vẫn được giữ (\appendix kết thúc phạm vi References).
def test_appendix_after_references_heading():
    tex = r"""\begin{document}
\section{Intro}
Hello.
\section{References}
Entry one.
\appendix
Appendix intro.
\section{Extra}
Details.
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    assert _headings(tree) == ["Intro", "Extra"]
    assert _sentences(tree) == ["Hello.", "Appendix intro.", "Details."]
 
 
# Mục 6.4: abstract và môi trường acknowledgments (revtex) thành heading con của root.
def test_abstract_and_acknowledgments_environments():
    tex = r"""\begin{document}
\begin{abstract}
We prove it.
\end{abstract}
\section{Intro}
Body.
\begin{acknowledgments}
We thank everyone.
\end{acknowledgments}
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    assert [(c["kind"], c["content"]) for c in tree["children"]] == [
        ("heading", "Abstract"),
        ("heading", "Intro"),
        ("heading", "Acknowledgments"),
    ]
 
 
# Mục 6.5: comment bị bỏ, `\%` không phải comment.
def test_comments_removed():
    tex = (
        "\\begin{document}\nReal sentence. % Hidden \\section{Nope}\n"
        "Next one. Gain was 50\\% higher.\n\\end{document}"
    )
    tree = build_tree(tex, ARXIV_ID)
    assert _headings(tree) == []
    assert _sentences(tree) == ["Real sentence.", "Next one.", "Gain was 50\\% higher."]
 
 
# Mục 6.6: môi trường định lý trong suốt, nội dung gắn vào section hiện tại.
def test_transparent_environments():
    tex = r"""\begin{document}
\section{S}
\begin{theorem}[Brooks]
Every graph is colorable.
\end{theorem}
\end{document}"""
    tree = build_tree(tex, ARXIV_ID)
    section = tree["children"][0]
    assert [c["content"] for c in section["children"]] == ["Every graph is colorable."]
 
 
# Mục 7.1 và 7.5.
def test_empty_input_and_empty_section():
    tree = build_tree("", ARXIV_ID)
    assert tree == {"id": "root1", "kind": "document", "content": "Version 1 root", "children": []}
 
    tree = build_tree(r"\begin{document}\section{Empty}\end{document}", ARXIV_ID)
    assert tree["children"][0]["content"] == "Empty"
    assert tree["children"][0]["children"] == []