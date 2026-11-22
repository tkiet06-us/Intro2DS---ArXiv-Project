"""Unit tests cho ``src.m2_parsing.bibtex_extractor`` (dùng tests/mock_data)."""

from pathlib import Path

import pytest

from src.m2_parsing.bibtex_extractor import (
    _is_valid_title,
    convert_bbl_to_bib,
    extract_bibitems,
    extract_bibtex,
    format_bibtex_entry,
    parse_bbl_file,
    parse_bib_file,
    write_bibtex,
)
from src.m2_parsing.latex_parser import resolve_paper

MOCK_DATA = Path(__file__).resolve().parents[1] / "mock_data"


def test_extracts_bibitem_from_mock_version():
    """Nguồn thật của mock dùng \\begin{thebibliography} + \\bibitem."""
    latex = resolve_paper(MOCK_DATA / "2310-12345" / "v1")
    entries = extract_bibitems(latex)

    assert len(entries) == 1
    entry = entries[0]
    assert entry == {
        "citation_key": "foo",
        "entry_type": "misc",
        "title": "Foo Bar",
        "authors": ["A. Author"],
        "year": 2020,
        "doi": None,
        "journal": None,
        "url": None,
    }


def test_parses_bib_file_from_mock():
    """Nguồn thứ hai: file .bib đi kèm do \\bibliography{refs} chỉ định."""
    entries = parse_bib_file(MOCK_DATA / "2401-00001" / "v1" / "refs.bib")

    assert [entry["citation_key"] for entry in entries] == ["bert", "noyear"]
    bert = entries[0]
    assert bert["entry_type"] == "article"
    assert bert["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
    assert bert["authors"] == ["J. Devlin", "M. Chang"]
    assert bert["year"] == 2019
    assert bert["journal"] == "NAACL"

    # Entry thiếu năm là chuyện thường gặp: phải là None, không được bịa 0.
    assert entries[1]["year"] is None
    assert entries[1]["authors"] == ["Unknown"]


def test_extract_bibtex_reads_bibliography_command():
    """\\bibliography{refs} phải được phân giải thành file refs.bib trong thư mục version."""
    version_dir = MOCK_DATA / "2401-00001" / "v1"
    latex = resolve_paper(version_dir)

    assert extract_bibitems(latex) == []
    entries = extract_bibtex(latex, version_dir)
    assert [entry["citation_key"] for entry in entries] == ["bert", "noyear"]


def test_extract_bibtex_without_version_dir_only_reads_bibitems():
    latex = resolve_paper(MOCK_DATA / "2310-12345" / "v1")
    assert [entry["citation_key"] for entry in extract_bibtex(latex)] == ["foo"]


def test_bibitem_with_doi_url_and_journal():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{smith2021}
    J. Smith and A. Doe, \emph{A Study of Things}, Journal of Tests, 2021.
    \url{https://example.com/paper} \doi{10.1234/ABC.567}
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["citation_key"] == "smith2021"
    assert entry["title"] == "A Study of Things"
    assert entry["authors"] == ["J. Smith", "A. Doe"]
    assert entry["year"] == 2021
    assert entry["journal"] == "Journal of Tests"
    assert entry["doi"] == "10.1234/ABC.567"
    assert entry["url"] == "https://example.com/paper"


def test_year_is_not_taken_from_url_doi_or_arxiv_id():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{x} A. Author, \emph{Title}, 2024.
    \url{https://arxiv.org/abs/2023.12345} \doi{10.2022/abc.123}
    \end{thebibliography}
    """
    assert extract_bibitems(latex)[0]["year"] == 2024


def test_ieee_bibinfo_fields_and_nested_braces():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{ieee}
    \bibinfo{author}{A. B. C.} \bibinfo{title}{A {Nested} Title}
    \bibinfo{journal}{Conference on Tests} \bibinfo{year}{2018}
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["authors"] == ["A. B. C."]
    assert entry["title"] == "A Nested Title"
    assert entry["journal"] == "Conference on Tests"
    assert entry["year"] == 2018


def test_apa_bibitem_with_textit_venue_keeps_title_and_authors():
    """APA ``\\bibitem`` dùng ``\\textit{}`` cho venue: title/authors phải đúng."""
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{lee2019} Lee, B. (2019). Graph nets. In \textit{NeurIPS}.
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["title"] == "Graph nets"
    assert entry["authors"] == ["Lee, B."]
    assert entry["journal"] == "In NeurIPS"
    assert entry["year"] == 2019


def test_plain_bibitem_with_initials_after_surname_splits_author_list():
    """``Họ, X., Họ, Y. Title. Venue, năm.`` không được cắt title thành 'A'."""
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{vaswani2017} Vaswani, A., Shazeer, N. Attention is all you need. NeurIPS, 2017.
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["title"] == "Attention is all you need"
    assert entry["authors"] == ["Vaswani, A.", "Shazeer, N."]
    assert entry["journal"] == "NeurIPS"
    assert entry["year"] == 2017


def test_multiple_bibitems_are_split_at_the_right_boundary():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{first} A. One, \emph{First Title}, 2001.
    \bibitem{second} B. Two, \emph{Second Title}, 2002.
    \end{thebibliography}
    """
    entries = extract_bibitems(latex)

    assert [entry["citation_key"] for entry in entries] == ["first", "second"]
    assert entries[0]["title"] == "First Title"
    assert entries[1]["title"] == "Second Title"
    assert entries[1]["year"] == 2002


def test_author_lists_with_and_ampersand_and_semicolon():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{a} Alice A. and Bob B.; Carol C., \emph{Title}, 2020.
    \bibitem{b} Dan D. & Eve E., \emph{Other}, 2021.
    \end{thebibliography}
    """
    entries = extract_bibitems(latex)

    assert entries[0]["authors"] == ["Alice A.", "Bob B.", "Carol C."]
    assert entries[1]["authors"] == ["Dan D.", "Eve E."]


def test_parse_bib_file_skips_string_and_comment_entries(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        '@string{jan = "January"}\n'
        "@comment{not an entry}\n"
        "@article{real,\n"
        "  title = {A Real Title},\n"
        "  author = {Real Author},\n"
        "  year = {2020},\n"
        "}\n",
        encoding="utf-8",
    )

    entries = parse_bib_file(bib)
    assert [entry["citation_key"] for entry in entries] == ["real"]
    assert entries[0]["title"] == "A Real Title"
    assert entries[0]["authors"] == ["Real Author"]
    assert entries[0]["year"] == 2020


def test_write_bibtex_round_trip(tmp_path):
    entries = [
        {
            "citation_key": "k1",
            "entry_type": "article",
            "title": "A Title",
            "authors": ["Alice", "Bob"],
            "year": 2020,
            "doi": "10.1234/xyz",
            "journal": "Journal",
            "url": "https://example.com/a",
        }
    ]
    target = tmp_path / "nested" / "refs.bib"
    write_bibtex(entries, target)

    text = target.read_text(encoding="utf-8")
    assert text.startswith("@article{k1,")
    assert "author = {Alice and Bob}" in text
    assert parse_bib_file(target) == entries


def test_format_bibtex_entry_strips_braces_from_values():
    text = format_bibtex_entry(
        {
            "citation_key": "x",
            "entry_type": "misc",
            "title": "A {weird} title",
            "authors": [],
            "year": None,
            "doi": None,
            "journal": None,
            "url": None,
        }
    )

    assert "@misc{x," in text
    assert "title = {A weird title}" in text
    assert "year" not in text


@pytest.mark.parametrize(
    ("raw_authors", "expected"),
    [
        ("J. Devlin and M. Chang", ["J. Devlin", "M. Chang"]),
        ("Smith, J., Doe, A.", ["Smith, J.", "Doe, A."]),
        ("A. One; B. Two", ["A. One", "B. Two"]),
    ],
)
def test_author_parsing_variants(raw_authors, expected):
    latex = (
        "\\begin{thebibliography}{9}\n"
        f"\\bibitem{{k}} {raw_authors}, \\emph{{Title}}, 2020.\n"
        "\\end{thebibliography}\n"
    )
    assert extract_bibitems(latex)[0]["authors"] == expected


def test_braced_capital_letters_not_split():
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{k} A. Author, \\emph{{V}ariational {A}utoencoders and {BERT}}, 2020.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["title"] == "Variational Autoencoders and BERT"


def test_volume_number_not_extracted_as_title():
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{revtex}\n"
        "\\bibinfo{author}{\\bibfnamefont{A.~M.} \\bibnamefont{Childs}},\n"
        "\\bibinfo{journal}{Physical review letters} \\textbf{\\bibinfo{volume}{123}},\n"
        "\\bibinfo{pages}{050503} (\\bibinfo{year}{2019}).\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["title"] is None
    assert entries[0]["journal"] == "Physical review letters"
    assert entries[0]["year"] == 2019


def test_quoted_title_extracted_correctly():
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{quote} A. Vaswani et al., ``Attention is all you need'', \\textit{NeurIPS}, 2017.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["title"] == "Attention is all you need"
    assert entries[0]["journal"] == "NeurIPS"
    assert entries[0]["year"] == 2017


def test_bib_file_with_eprint_populates_arxiv_id(tmp_path):
    bib = tmp_path / "eprint.bib"
    bib.write_text(
        "@article{bert,\n"
        "  title = {BERT},\n"
        "  author = {J. Devlin},\n"
        "  year = {2018},\n"
        "  eprint = {1810.04805},\n"
        "}\n",
        encoding="utf-8",
    )
    entries = parse_bib_file(bib)
    assert entries[0]["url"] == "https://arxiv.org/abs/1810.04805"
    assert "1810.04805" in entries[0]["journal"]


def test_parse_bib_file_filters_by_cited_keys(tmp_path):
    bib = tmp_path / "multi.bib"
    bib.write_text(
        "@article{cited1, title = {Cited 1}, author = {A. One}, year = {2020}}\n"
        "@article{uncited, title = {Uncited}, author = {B. Two}, year = {2021}}\n"
        "@article{cited2, title = {Cited 2}, author = {C. Three}, year = {2022}}\n",
        encoding="utf-8",
    )
    entries = parse_bib_file(bib, filter_keys={"cited1", "cited2"})
    assert [e["citation_key"] for e in entries] == ["cited1", "cited2"]


def test_extract_bibtex_fallback_other_tex(tmp_path):
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    other_tex = version_dir / "refs_standalone.tex"
    other_tex.write_text(
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{from_other} Other Author, \\emph{Standalone Title}, 2024.\n"
        "\\end{thebibliography}\n",
        encoding="utf-8",
    )
    # LaTeX đã gộp không có bibitem, nhưng version_dir có refs_standalone.tex
    entries = extract_bibtex("Some text without bibitems", version_dir)
    assert [e["citation_key"] for e in entries] == ["from_other"]
    assert entries[0]["title"] == "Standalone Title"


def test_valid_title_containing_journal_name():
    # R1: Tên tạp chí là chuỗi con của tiêu đề (ví dụ "Nature" trong "The Quantum Nature of Light")
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{q} A. Einstein, \\emph{The Quantum Nature of Light}, \\textit{Nature}, 2020.\n"
        "\\bibitem{ds} B. Smith, \\emph{Advances in Data Science}, \\textit{Science}, 2021.\n"
        "\\bibitem{ml} C. Brown, \\emph{Pattern Recognition and Machine Learning}, \\textit{Machine Learning}, 2022.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["title"] == "The Quantum Nature of Light"
    assert entries[0]["journal"] == "Nature"
    assert entries[1]["title"] == "Advances in Data Science"
    assert entries[1]["journal"] == "Science"
    assert entries[2]["title"] == "Pattern Recognition and Machine Learning"
    assert entries[2]["journal"] == "Machine Learning"


def test_invalid_title_journal_plus_digits_rejected():
    # R1: Tiêu đề bóc nhầm journal + volume/trang (ví dụ "Physical review letters 123") phải bị loại bỏ
    assert not _is_valid_title("Physical review letters 123", journal="Physical review letters")
    assert not _is_valid_title("Physical review", journal="Physical review letters")  # prefix < 15 ký tự
    assert not _is_valid_title("Nature", journal="Nature")  # trùng hoàn toàn


def test_revtex_bibinfo_without_title_not_guessed_from_volume():
    # R1: REVTeX không có \bibinfo{title} -> title = None, không rơi vào _split_plain đoán tiêu đề
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{revtex}\n"
        "\\bibinfo{author}{\\bibfnamefont{J.} \\bibnamefont{Preskill}},\n"
        "\\bibinfo{journal}{Quantum} \\textbf{\\bibinfo{volume}{2}},\n"
        "\\bibinfo{pages}{79} (\\bibinfo{year}{2018}).\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["title"] is None
    assert entries[0]["journal"] == "Quantum"
    assert entries[0]["year"] == 2018


def test_parse_bib_file_filters_case_insensitive(tmp_path):
    # R2: So khớp citation_key case-insensitive giữa .tex và .bib
    bib = tmp_path / "case.bib"
    bib.write_text(
        "@article{devlin2018,\n"
        "  title = {BERT},\n"
        "  author = {J. Devlin},\n"
        "  year = {2018}\n"
        "}\n",
        encoding="utf-8",
    )
    # Trong .tex viết hoa Devlin2018, trong .bib viết thường devlin2018
    entries = parse_bib_file(bib, filter_keys={"Devlin2018"})
    assert len(entries) == 1
    assert entries[0]["citation_key"] == "devlin2018"
    assert entries[0]["title"] == "BERT"


def test_extract_bibtex_case_insensitive_dedup(tmp_path):
    # R2: Khử trùng lặp case-insensitive giữa \bibitem và .bib
    vdir = tmp_path / "v1"
    vdir.mkdir()
    bib = vdir / "refs.bib"
    bib.write_text(
        "@article{devlin2018,\n"
        "  title = {BERT},\n"
        "  author = {J. Devlin},\n"
        "  year = {2018}\n"
        "}\n",
        encoding="utf-8",
    )
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{Devlin2018} J. Devlin, \\emph{BERT}, 2018.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibtex(latex, vdir)
    assert len(entries) == 1
    assert entries[0]["citation_key"] == "Devlin2018"


def test_protect_spans_and_year_regex_avoids_eprint_digits():
    # R3: Không nhận nhầm năm từ arXiv ID trong \eprint hoặc URL
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{paper} A. Author, \\emph{Deep Learning}, \\eprint{2004.12345}, 2019.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["year"] == 2019


def test_bibitem_eprint_populates_url_and_journal():
    # R4: \bibitem có \eprint hoặc arXiv:... bổ sung url và journal giống logic .bib
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{bert} J. Devlin et al., \\emph{BERT}, \\eprint{1810.04805}, 2018.\n"
        "\\bibitem{gpt} A. Rad et al., \\emph{GPT}, arXiv:2005.14165, 2020.\n"
        "\\bibitem{venue} C. Chen, \\emph{Vision}, \\textit{CVPR}, \\eprint[arXiv]{2106.00001}, 2021.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert entries[0]["url"] == "https://arxiv.org/abs/1810.04805"
    assert entries[0]["journal"] == "arXiv:1810.04805"

    assert entries[1]["url"] == "https://arxiv.org/abs/2005.14165"
    assert entries[1]["journal"] == "arXiv:2005.14165"

    assert entries[2]["url"] == "https://arxiv.org/abs/2106.00001"
    assert entries[2]["journal"] == "CVPR, arXiv:2106.00001"


def test_latex_accents_converted_to_unicode():
    # R5: Chuyển các dấu TeX \"o, \'e, \~n, \v{s}, \c{c}, \H{o} sang Unicode, tránh Peña biến thành "Pe na"
    latex = (
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{godel} K. G\\\"odel, \\emph{On Formally Undecidable Propositions}, 1931.\n"
        "\\bibitem{rene} Ren\\'e Descartes, \\emph{Discourse on the Method}, 1637.\n"
        "\\bibitem{pena} F. Pe\\~na, \\emph{Matrix Analysis}, 2015.\n"
        "\\bibitem{dvorak} A. Dvo\\v{r}\\'{a}k, \\emph{New World Symphony}, 1893.\n"
        "\\bibitem{erdos} P. Erd\\H{o}s, \\emph{Combinatorics}, 1975.\n"
        "\\bibitem{cedilla} Fran\\c{c}ois Bourdon, \\emph{Mechanics}, 2010.\n"
        "\\end{thebibliography}\n"
    )
    entries = extract_bibitems(latex)
    assert "Gödel" in entries[0]["authors"][0]
    assert "René" in entries[1]["authors"][0]
    assert "Peña" in entries[2]["authors"][0]
    assert "Dvořák" in entries[3]["authors"][0]
    assert "Erdős" in entries[4]["authors"][0]
    assert "François" in entries[5]["authors"][0]


def test_extract_bibtex_falls_back_to_bbl_and_converts_to_bib(tmp_path):
    """Khi không có .bib và .tex không có \\bibitem, phải fallback sang .bbl và giữ nguyên .bbl gốc."""
    vdir = tmp_path / "v1"
    vdir.mkdir()
    bbl_file = vdir / "main.bbl"
    bbl_file.write_text(
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{doe2023} J. Doe, \\emph{Fallback Paper}, NeurIPS, 2023.\n"
        "\\end{thebibliography}\n",
        encoding="utf-8",
    )
    latex = "\\documentclass{article}\\begin{document}Hello \\cite{doe2023}\\end{document}"

    entries = extract_bibtex(latex, vdir)
    assert len(entries) == 1
    assert entries[0]["citation_key"] == "doe2023"
    assert entries[0]["title"] == "Fallback Paper"
    assert entries[0]["year"] == 2023

    out_bib = tmp_path / "processed" / "refs.bib"
    converted = convert_bbl_to_bib(vdir, out_bib_path=out_bib)
    assert len(converted) == 1
    assert out_bib.is_file()
    assert not (vdir / "refs.bib").exists()  # Không ghi bẩn thư mục nguồn raw
    assert bbl_file.is_file()  # file .bbl gốc vẫn được giữ nguyên


def test_parse_bbl_supports_amsrefs_and_keeps_unparseable_bbl(tmp_path):
    ams_bbl = tmp_path / "ams.bbl"
    ams_bbl.write_text(
        "\\begin{bibdiv}\\begin{biblist}\n"
        "\\bib{k1}{article}{author={A. Smith}, title={AMS Title}, year={2022}}\n"
        "\\end{biblist}\\end{bibdiv}\n",
        encoding="utf-8",
    )
    entries = parse_bbl_file(ams_bbl)
    assert len(entries) == 1
    assert entries[0]["citation_key"] == "k1"
    assert entries[0]["title"] == "AMS Title"
    assert entries[0]["year"] == 2022

    weird_dir = tmp_path / "weird"
    weird_dir.mkdir()
    weird_bbl = weird_dir / "custom.bbl"
    weird_bbl.write_text("Custom non-standard bibliography format without bibitem", encoding="utf-8")
    assert convert_bbl_to_bib(weird_dir) == []
    assert not (weird_dir / "refs.bib").exists()
    assert weird_bbl.is_file()


def _write_amsrefs_bbl(tmp_path, date_line):
    path = tmp_path / "a.bbl"
    path.write_text(
        "\\begin{biblist}\n"
        "\\bib{k1}{article}{\n"
        "  author={A. Smith and B. Jones},\n"
        "  title={AMS Title},\n"
        "  journal={J. Tests},\n"
        f"{date_line}"
        "}\n"
        "\\end{biblist}\n"
    )
    return path


def test_amsrefs_date_with_month_gives_year(tmp_path):
    bbl = _write_amsrefs_bbl(tmp_path, "  date={2022-05},\n")
    assert parse_bbl_file(bbl)[0]["year"] == 2022


def test_amsrefs_date_year_only_gives_year(tmp_path):
    bbl = _write_amsrefs_bbl(tmp_path, "  date={2022},\n")
    assert parse_bbl_file(bbl)[0]["year"] == 2022


def test_amsrefs_without_year_or_date_gives_none(tmp_path):
    bbl = _write_amsrefs_bbl(tmp_path, "")
    assert parse_bbl_file(bbl)[0]["year"] is None


_BBL = (
    "\\begin{thebibliography}{1}\n"
    "\\bibitem{k1} A. Smith, Title, J. Tests, 2022.\n"
    "\\end{thebibliography}\n"
)


def test_convert_bbl_does_not_overwrite_existing_refs_bib(tmp_path):
    (tmp_path / "main.bbl").write_text(_BBL)
    original = "@inproceedings{k1,\n  title = {T},\n  booktitle = {Proc},\n  pages = {1--2}\n}\n"
    (tmp_path / "refs.bib").write_text(original)

    entries = convert_bbl_to_bib(tmp_path)

    assert entries  # vẫn trả về entry đã convert
    assert (tmp_path / "refs.bib").read_text() == original


def test_convert_bbl_without_out_path_does_not_write_to_source_dir(tmp_path):
    (tmp_path / "main.bbl").write_text(_BBL)
    entries = convert_bbl_to_bib(tmp_path)
    assert len(entries) == 1
    assert not (tmp_path / "refs.bib").exists()


def test_convert_bbl_explicit_out_path_leaves_source_refs_bib_alone(tmp_path):
    (tmp_path / "main.bbl").write_text(_BBL)
    (tmp_path / "refs.bib").write_text("orig")
    out = tmp_path / "converted.bib"

    convert_bbl_to_bib(tmp_path, out)

    assert out.is_file()
    assert (tmp_path / "refs.bib").read_text() == "orig"


def test_convert_bbl_to_bib_no_bbl_returns_empty_list(tmp_path: Path) -> None:
    """0 file .bbl phải trả về [] (không phải None) và không tạo file."""
    out_bib = tmp_path / "processed" / "refs.bib"
    result = convert_bbl_to_bib(tmp_path, out_bib_path=out_bib)
    assert result == []
    assert not out_bib.exists()


def test_convert_bbl_to_bib_multiple_files_and_broken_first(tmp_path: Path) -> None:
    """File .bbl đầu hỏng vẫn đọc tiếp file sau; 2 file .bbl lấy đủ cả 2 key."""
    (tmp_path / "a_broken.bbl").write_text("noi dung rac khong co bibitem", encoding="utf-8")
    (tmp_path / "b_first.bbl").write_text(
        r"\begin{thebibliography}{1}\bibitem{k1} A. One. \emph{Title 1}. 2020.\end{thebibliography}",
        encoding="utf-8",
    )
    (tmp_path / "c_second.bbl").write_text(
        r"\begin{thebibliography}{1}\bibitem{k2} B. Two. \emph{Title 2}. 2021.\end{thebibliography}",
        encoding="utf-8",
    )

    out_bib = tmp_path / "processed" / "refs.bib"
    result = convert_bbl_to_bib(tmp_path, out_bib_path=out_bib)

    assert [e["citation_key"] for e in result] == ["k1", "k2"]
    assert out_bib.is_file()
    assert not (tmp_path / "refs.bib").exists()


def test_extract_bibtex_deduplicates_citation_keys(tmp_path: Path) -> None:
    """Trùng citation_key giữa các file .bbl thì giữ lại lần xuất hiện đầu tiên."""
    (tmp_path / "part1.bbl").write_text(
        r"\begin{thebibliography}{1}\bibitem{k1} A. One. \emph{First Title}. 2020.\end{thebibliography}",
        encoding="utf-8",
    )
    (tmp_path / "part2.bbl").write_text(
        r"\begin{thebibliography}{1}\bibitem{k1} B. Two. \emph{Duplicate Title}. 2021.\end{thebibliography}",
        encoding="utf-8",
    )

    result = extract_bibtex("", version_dir=tmp_path)
    assert len(result) == 1
    assert result[0]["citation_key"] == "k1"
    assert result[0]["title"] == "First Title"


def test_parse_bbl_amsrefs_multiple_author_fields(tmp_path: Path) -> None:
    """amsrefs có nhiều dòng author={...} phải được gộp đủ danh sách tác giả."""
    bbl_file = tmp_path / "ams.bbl"
    bbl_file.write_text(
        r"""
        \begin{bibdiv}
        \begin{biblist}
        \bib{ams1}{article}{
          author={Smith, A.},
          author={Jones, B.},
          title={Sample AMS Paper},
          date={2022},
        }
        \end{biblist}
        \end{bibdiv}
        """,
        encoding="utf-8",
    )

    result = parse_bbl_file(bbl_file)
    assert len(result) == 1
    assert result[0]["authors"] == ["Smith, A.", "Jones, B."]
