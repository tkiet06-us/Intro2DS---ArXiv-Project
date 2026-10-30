"""Test cho src/m2_parsing/latex_cleanup.py (task P2-05).

Chia theo nhóm: (1) xóa định dạng thừa, (2) chuẩn hóa toán, (3) vùng phải giữ NGUYÊN
(đúng danh sách "Điều cấm" ở latex_spec.md Mục 1.2), (4) phòng thủ, (5) tương thích mock.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.m2_parsing.latex_cleanup import cleanup_latex


# ---------------------------------------------------------------- (1) Xóa định dạng thừa
def test_removes_centering_and_float_option_inside_figure():
    src = (
        "\\begin{figure}[htpb]\n\\centering\n\\includegraphics{fig1.png}\n"
        "\\caption{Overview.}\n\\end{figure}"
    )
    out = cleanup_latex(src)
    assert (
        out == "\\begin{figure}\n\\includegraphics{fig1.png}\n\\caption{Overview.}\n\\end{figure}"
    )


@pytest.mark.parametrize("opt", ["[htbp]", "[!ht]", "[H]", "[t]"])
def test_removes_float_option_variants(opt):
    assert (
        cleanup_latex(f"\\begin{{table}}{opt}\nx\n\\end{{table}}")
        == "\\begin{table}\nx\n\\end{table}"
    )


def test_removes_float_option_on_starred_env():
    assert (
        cleanup_latex("\\begin{figure*}[t]\nx\n\\end{figure*}")
        == "\\begin{figure*}\nx\n\\end{figure*}"
    )


def test_keeps_tabular_column_spec():
    # {ll} và [t] của tabular KHÔNG phải tùy chọn float.
    src = "\\begin{tabular}[t]{ll}\na & b\n\\end{tabular}"
    assert cleanup_latex(src) == src


def test_removes_font_switches_but_not_wrapped_commands():
    out = cleanup_latex("{\\bf Bold} and {\\it slanted} and \\textbf{kept} and \\emph{kept}")
    assert out == "{Bold} and {slanted} and \\textbf{kept} and \\emph{kept}"


def test_font_switch_does_not_eat_longer_command_names():
    src = "\\item one \\itshape x \\bfseries y \\itemize"
    out = cleanup_latex(src)
    assert "\\item one" in out  # \it không được cắn \item
    assert "\\itemize" in out
    assert "\\itshape" not in out and "\\bfseries" not in out


def test_removes_spacing_commands_without_gluing_words():
    out = cleanup_latex("Hello\\hspace{1em}world and\\vspace*{2pt} more.")
    assert out == "Hello world and more."


def test_removes_size_and_alignment_switches():
    assert cleanup_latex("{\\small \\raggedright text}") == "{text}"


def test_removes_booktabs_rules_but_keeps_hline_and_row_breaks():
    src = (
        "\\begin{tabular}{ll}\n\\toprule\nA & B \\\\\n\\midrule\n"
        "C & D \\\\\n\\bottomrule\n\\end{tabular}"
    )
    out = cleanup_latex(src)
    assert "rule" not in out
    assert out.count("\\\\") == 2  # \\ phân dòng của tabular PHẢI còn
    assert cleanup_latex("a \\\\ \\hline b") == "a \\\\ \\hline b"


def test_removes_cmidrule_with_trim_argument():
    assert "cmidrule" not in cleanup_latex("A \\\\ \\cmidrule(lr){2-3}\nB")


def test_line_with_only_commands_is_dropped_entirely():
    assert cleanup_latex("a\n\\centering\n\\vspace{1em}\nb") == "a\nb"


# ---------------------------------------------------------------- (2) Chuẩn hóa toán
def test_dollar_display_becomes_equation_single_line_like_mock():
    # Dạng đầu ra khớp đúng nút formula của tests/make_mock.py.
    assert cleanup_latex("Before $$E = mc^2$$ after.") == (
        "Before \\begin{equation} E = mc^2 \\end{equation} after."
    )


def test_bracket_display_becomes_equation():
    assert cleanup_latex("x \\[ a+b \\] y") == "x \\begin{equation} a+b \\end{equation} y"


def test_multiline_display_keeps_newline_style_of_spec_example():
    out = cleanup_latex("$$\na = b \\label{eq:1}\n$$")
    assert out == "\\begin{equation}\na = b \\label{eq:1}\n\\end{equation}"


def test_paren_inline_and_math_env_become_dollars():
    assert cleanup_latex("Let \\( x^2 \\) and \\begin{math}y\\end{math}.") == "Let $x^2$ and $y$."


def test_inline_dollar_is_untouched():
    src = "Let $x^2 + y^2 = z^2$ be true and $\\Delta(G)$ small."
    assert cleanup_latex(src) == src


def test_existing_math_environments_are_untouched():
    for env in [
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "multline",
        "flalign",
        "eqnarray",
        "displaymath",
    ]:
        src = f"\\begin{{{env}}}\n\\bf x \\\\ y \\vspace{{1em}}\n\\end{{{env}}}"
        assert cleanup_latex(src) == src, env


def test_nested_math_substructures_are_untouched():
    src = "\\begin{equation}\n\\begin{cases} a & b \\\\ c & d \\end{cases}\n\\end{equation}"
    assert cleanup_latex(src) == src


def test_formatting_commands_inside_formula_are_kept():
    src = "$$ {\\rm d}x \\, \\bf R $$"
    assert cleanup_latex(src) == "\\begin{equation} {\\rm d}x \\, \\bf R \\end{equation}"
    assert cleanup_latex("$ {\\rm d}x $") == "$ {\\rm d}x $"


def test_row_separator_with_optional_length_is_not_display_math():
    # "\\[2pt]" là xuống dòng + khoảng cách, KHÔNG phải "\[" mở công thức khối.
    src = "\\begin{tabular}{ll}\na & b \\\\[2pt]\nc & d \\\\[1ex]\n\\end{tabular}"
    assert cleanup_latex(src) == src


def test_escaped_dollar_and_percent_are_plain_text():
    src = "It costs \\$5 and \\$7, a 50\\% gain."
    assert cleanup_latex(src) == src


def test_empty_display_math_is_left_alone():
    assert cleanup_latex("a $$ $$ b") == "a $$ $$ b"


# ---------------------------------------------------------------- (3) Vùng phải giữ nguyên
def test_headings_labels_cites_appendix_survive():
    src = (
        "\\section{Introduction}\\label{sec:intro}\nSee~\\cite{foo}.\n\\appendix\n\\section{Proofs}"
    )
    assert cleanup_latex(src) == src


def test_caption_and_label_in_figure_survive():
    out = cleanup_latex("\\begin{figure}[ht]\n\\caption{Cap.}\n\\label{fig:a}\n\\end{figure}")
    assert "\\caption{Cap.}" in out and "\\label{fig:a}" in out and "[ht]" not in out


def test_thebibliography_untouched():
    src = (
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{a} R.~Brooks, \\emph{On colouring}, 1941.\n"
        "\\end{thebibliography}"
    )
    assert cleanup_latex(src) == src


@pytest.mark.parametrize("env", ["verbatim", "lstlisting", "minted", "rawhtml"])
def test_verbatim_like_regions_are_untouched(env):
    src = f"\\begin{{{env}}}\n\\centering $$x$$ % keep \\bf\n\\end{{{env}}}"
    assert cleanup_latex(src) == src


def test_iffalse_region_is_untouched():
    src = "a\n\\iffalse\n\\centering $$x$$\n\\fi\nb"
    assert cleanup_latex(src) == src


def test_itemize_untouched():
    src = (
        "\\begin{itemize}\n    \\item Planar graphs.\n    \\item Bipartite graphs.\n\\end{itemize}"
    )
    assert cleanup_latex(src) == src


# ---------------------------------------------------------------- comment + khoảng trắng
def test_comments_are_removed_like_tex():
    assert cleanup_latex("a % note \\section{Gone}\nb") == "a b"
    assert cleanup_latex("line one\n% whole-line comment\nline two") == "line one\nline two"
    assert cleanup_latex("100\\% sure % trailing\nnext") == "100\\% sure next"


def test_comments_can_be_kept():
    assert cleanup_latex("a % keep\nb", strip_comments=False) == "a % keep\nb"


def test_whitespace_is_normalised():
    assert cleanup_latex("a   b  \n\n\n\n\nc") == "a b\n\nc"
    assert cleanup_latex("a\r\nb") == "a\nb"
    assert cleanup_latex("    indented\n        deeper") == "    indented\n        deeper"


# ---------------------------------------------------------------- (4) Tính chất chung, phòng thủ
def test_empty_input():
    assert cleanup_latex("") == ""


def test_idempotent_on_messy_input():
    src = (
        "\\centering % c\n\n\n\n$$a$$ \\vspace{1em} \\[b\\] \\( c \\)\n"
        "\\begin{figure}[htbp]\n\\centering\n\\caption{x}\n\\end{figure}\n"
        "\\begin{tabular}{l}\n\\toprule\nA \\\\[2pt]\n\\midrule\n\\end{tabular}   \n"
    )
    once = cleanup_latex(src)
    assert cleanup_latex(once) == once


def test_unclosed_environment_does_not_crash_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = cleanup_latex("\\begin{figure}[ht]\n\\centering\n\\caption{no end}")
    assert out == "\\begin{figure}\n\\caption{no end}"
    assert "figure" in caplog.text


def test_unbalanced_dollars_do_not_crash(caplog):
    with caplog.at_level(logging.WARNING):
        out = cleanup_latex("price $$ 5 and \\centering text")
    assert "\\centering" not in out
    assert "$$" in caplog.text


def test_large_document_is_fast():
    block = (
        "\\section{S}\nWe study $x$ here \\cite{a}.\n$$a=b$$\n"
        "\\begin{figure}[ht]\n\\centering\n\\end{figure}\n"
    )
    out = cleanup_latex(block * 20000)  # khoảng 1 MB
    assert out.count("\\begin{equation}") == 20000


# ---------------------------------------------------------------- (5) Tương thích mock
def _find_mock() -> Path:
    """Tìm tests/make_mock.py dù file test này nằm ở tests/ hay tests/m2_parsing/."""
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / "make_mock.py", parent / "tests" / "make_mock.py"):
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Không tìm thấy tests/make_mock.py")


MOCK_PATH = _find_mock()


@pytest.fixture(scope="module")
def mock() -> SimpleNamespace:
    """Đọc các hằng ``TEX_*`` từ tests/make_mock.py mà KHÔNG import cả file.

    make_mock.py kéo theo pyarrow, pydantic và src.schemas; test cleanup chỉ cần vài chuỗi TeX
    mẫu nên ta chỉ chạy các câu gán ``TEX_* = ...`` (lấy ra bằng ast), nhờ vậy test chạy được
    ngay cả khi môi trường chưa cài các thư viện đó.
    """
    tree = ast.parse(MOCK_PATH.read_text(encoding="utf-8"))
    assigns = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("TEX_")
    ]
    namespace: dict[str, str] = {}
    exec(compile(ast.Module(body=assigns, type_ignores=[]), str(MOCK_PATH), "exec"), namespace)  # noqa: S102
    return SimpleNamespace(**{k: v for k, v in namespace.items() if k.startswith("TEX_")})


def test_mock_tex_with_nothing_to_clean_is_unchanged(mock):
    # Bài A (cả hai version) và bài C (intro) đã sạch: cleanup không được làm đổi gì.
    for tex in (mock.TEX_A_V1, mock.TEX_A_V2, mock.TEX_C_MAIN, mock.TEX_C_INTRO):
        assert cleanup_latex(tex) == tex


def test_mock_figure_loses_option_and_centering_but_keeps_caption(mock):
    out = cleanup_latex(mock.TEX_B_MAIN)
    assert "[htpb]" not in out and "\\centering" not in out
    assert (
        "\\begin{figure}\n\\includegraphics{fig1.png}\n\\caption{Overview.}\n\\end{figure}" in out
    )
    assert "\\section{Method}" in out and "\\bibliography{refs}" in out


def test_mock_equation_is_exactly_the_formula_node_content(mock):
    # Nút formula "f1" của bài A trong mock phải xuất hiện nguyên văn sau cleanup, và
    # "$$E = mc^2$$" phải chuẩn hóa về đúng chuỗi đó.
    formula = "\\begin{equation} E = mc^2 \\end{equation}"
    assert formula in cleanup_latex(mock.TEX_A_V1)
    assert cleanup_latex("$$E = mc^2$$") == formula