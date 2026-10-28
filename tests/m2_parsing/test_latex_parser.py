"""Unit tests cho src/latex_parser.py (dùng tests/mock_data)."""

import logging
from pathlib import Path

import pytest

from src.latex_parser import find_main_tex, resolve_includes

PREAMBLE = "\\documentclass{article}\n\\begin{document}\n"


def _write(root: Path, rel: str, text: str) -> str:
    """Tạo file (kèm thư mục cha) và trả về đường dẫn dạng str."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


# find_main_tex: file chính là file chứa \documentclass


def test_find_main_single_file(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "Hello.\n\\end{document}\n")
    assert Path(find_main_tex(str(tmp_path))).resolve() == Path(main).resolve()


def test_find_main_among_many_files(tmp_path):
    # File phụ đứng trước và sau file chính theo thứ tự bảng chữ cái: không được chọn nhầm.
    _write(tmp_path, "a_intro.tex", "\\section{Intro}\nText.\n")
    main = _write(tmp_path, "paper.tex", PREAMBLE + "\\input{a_intro}\n\\input{z_end}\n")
    _write(tmp_path, "z_end.tex", "\\section{End}\n")
    assert Path(find_main_tex(str(tmp_path))).resolve() == Path(main).resolve()


# Giả định: \documentclass bị comment thì không tính là file chính (cùng tinh thần với \input
# bị comment trong đề).
def test_find_main_ignores_commented_documentclass(tmp_path):
    _write(tmp_path, "old.tex", "% \\documentclass{article}\nLeftover draft.\n")
    main = _write(tmp_path, "real.tex", PREAMBLE + "Body.\n")
    assert Path(find_main_tex(str(tmp_path))).resolve() == Path(main).resolve()


# Giả định: Không có file nào chứa \documentclass thì báo lỗi rõ ràng, không trả bừa một file.
# Nếu nhóm chọn ném lỗi khác (hoặc trả None) thì sửa tuple ngoại lệ bên dưới.
def test_find_main_without_documentclass_raises(tmp_path):
    _write(tmp_path, "a.tex", "\\section{Only a fragment}\n")
    with pytest.raises((FileNotFoundError, ValueError)):
        find_main_tex(str(tmp_path))


# resolve_includes: gộp \input / \include theo đường biên dịch thực tế

def test_single_file_without_includes_is_unchanged(tmp_path):
    text = PREAMBLE + "Just text.\n\\end{document}\n"
    main = _write(tmp_path, "main.tex", text)
    assert resolve_includes(main).strip() == text.strip()


def test_input_is_inlined_in_place(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "BEFORE\n\\input{intro}\nAFTER\n")
    _write(tmp_path, "intro.tex", "CHILD_TEXT\n")
    out = resolve_includes(main)
    assert out.index("BEFORE") < out.index("CHILD_TEXT") < out.index("AFTER")
    assert "\\input{intro}" not in out  # lệnh được thay bằng nội dung file


def test_include_is_inlined(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\include{chap1}\n")
    _write(tmp_path, "chap1.tex", "CHAPTER_ONE\n")
    out = resolve_includes(main)
    assert "CHAPTER_ONE" in out
    assert "\\include{chap1}" not in out


def test_extension_is_optional(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\input{a}\n\\input{b.tex}\n")
    _write(tmp_path, "a.tex", "FROM_A\n")
    _write(tmp_path, "b.tex", "FROM_B\n")
    out = resolve_includes(main)
    assert "FROM_A" in out
    assert "FROM_B" in out


def test_include_from_subdirectory(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\input{sections/intro}\n\\input{./sections/method}\n")
    _write(tmp_path, "sections/intro.tex", "INTRO_TEXT\n")
    _write(tmp_path, "sections/method.tex", "METHOD_TEXT\n")
    out = resolve_includes(main)
    assert "INTRO_TEXT" in out
    assert "METHOD_TEXT" in out


def test_transitive_includes_keep_order(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "M_START\n\\input{a}\nM_END\n")
    _write(tmp_path, "a.tex", "A_START\n\\input{b}\nA_END\n")
    _write(tmp_path, "b.tex", "B_ONLY\n")
    out = resolve_includes(main)
    order = [out.index(x) for x in ("M_START", "A_START", "B_ONLY", "A_END", "M_END")]
    assert order == sorted(order)


# LaTeX tìm file con theo thư mục biên dịch (thư mục của file chính), không phải thư mục của
# file chứa lệnh \input. Vì vậy trong sections/a.tex, \input{sections/b} vẫn đúng.
def test_nested_include_path_is_relative_to_main_directory(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\input{sections/a}\n")
    _write(tmp_path, "sections/a.tex", "A_TEXT\n\\input{sections/b}\n")
    _write(tmp_path, "sections/b.tex", "B_TEXT\n")
    out = resolve_includes(main)
    assert "A_TEXT" in out
    assert "B_TEXT" in out


def test_unused_files_are_ignored(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\input{used}\n")
    _write(tmp_path, "used.tex", "USED_TEXT\n")
    _write(tmp_path, "unused.tex", "UNUSED_TEXT\n")
    _write(tmp_path, "old/draft.tex", "DRAFT_TEXT\n")
    out = resolve_includes(main)
    assert "USED_TEXT" in out
    assert "UNUSED_TEXT" not in out
    assert "DRAFT_TEXT" not in out


def test_commented_out_include_is_not_followed(tmp_path):
    text = PREAMBLE + "% \\input{hidden}\nVisible. % \\include{hidden2}\n"
    main = _write(tmp_path, "main.tex", text)
    _write(tmp_path, "hidden.tex", "HIDDEN_TEXT\n")
    _write(tmp_path, "hidden2.tex", "HIDDEN_TEXT_2\n")
    out = resolve_includes(main)
    assert "HIDDEN_TEXT" not in out
    assert "HIDDEN_TEXT_2" not in out
    assert "Visible." in out


# `\%` là dấu phần trăm thường, không mở comment, nên \input phía sau vẫn có hiệu lực.
def test_escaped_percent_does_not_start_comment(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "Gain of 50\\% \\input{extra}\n")
    _write(tmp_path, "extra.tex", "EXTRA_TEXT\n")
    assert "EXTRA_TEXT" in resolve_includes(main)


# \includegraphics (và các lệnh bắt đầu bằng "include") không phải \include.
def test_includegraphics_is_not_an_include(tmp_path, caplog):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\includegraphics[width=3cm]{fig}\n")
    with caplog.at_level(logging.WARNING):
        out = resolve_includes(main)
    assert "\\includegraphics[width=3cm]{fig}" in out
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# Cùng một file được include hai lần (không phải chu trình): LaTeX biên dịch cả hai lần.
def test_same_file_included_twice_appears_twice(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\input{note}\nMIDDLE\n\\input{note}\n")
    _write(tmp_path, "note.tex", "NOTE_TEXT\n")
    assert resolve_includes(main).count("NOTE_TEXT") == 2


# Chu trình include (a -> b -> a, hoặc file tự include chính nó) không được làm treo hay tràn
# stack; kết quả vẫn chứa nội dung của cả hai file.
def test_include_cycle_terminates(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "A_MARK\n\\input{b}\n")
    _write(tmp_path, "b.tex", "B_MARK\n\\input{main}\n")
    out = resolve_includes(main)
    assert "A_MARK" in out
    assert "B_MARK" in out


def test_self_include_terminates(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "SELF_MARK\n\\input{main}\n")
    assert "SELF_MARK" in resolve_includes(main)


# Đề: lệnh include trỏ tới file không tồn tại thì ghi warning, thay bằng khoảng trắng và tiếp tục.
def test_missing_include_warns_and_continues(tmp_path, caplog):
    main = _write(tmp_path, "main.tex", PREAMBLE + "BEFORE\n\\input{missing}\nAFTER\n")
    with caplog.at_level(logging.WARNING):
        out = resolve_includes(main)
    assert "BEFORE" in out
    assert "AFTER" in out
    assert "\\input{missing}" not in out
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# Giả định: Nguồn arXiv cũ hay dùng latin-1: một file con sai UTF-8 không được làm hỏng cả bài.
def test_non_utf8_file_does_not_crash(tmp_path):
    main = _write(tmp_path, "main.tex", PREAMBLE + "\\input{old}\nTAIL_MARK\n")
    (tmp_path / "old.tex").write_bytes("caf\xe9 LATIN_MARK\n".encode("latin-1"))
    out = resolve_includes(main)
    assert "LATIN_MARK" in out
    assert "TAIL_MARK" in out


# Ghép hai bước như pipeline thật: thư mục -> file chính -> chuỗi gộp

def test_folder_to_merged_string(tmp_path):
    _write(tmp_path, "paper.tex", PREAMBLE + "\\input{sec/intro}\n\\include{sec/method}\n")
    _write(tmp_path, "sec/intro.tex", "INTRO_TEXT\n")
    _write(tmp_path, "sec/method.tex", "METHOD_TEXT\n")
    _write(tmp_path, "unused.tex", "UNUSED_TEXT\n")
    out = resolve_includes(find_main_tex(str(tmp_path)))
    assert "INTRO_TEXT" in out
    assert "METHOD_TEXT" in out
    assert "UNUSED_TEXT" not in out